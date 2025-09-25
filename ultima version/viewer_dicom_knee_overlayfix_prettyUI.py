import os
import numpy as np
import SimpleITK as sitk
import vtk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox, RangeSlider
# ---- Estilo de figura / UI ----
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.titlesize': 12,
    'font.size': 10,
    'axes.edgecolor': '#cccccc',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
})

from SimpleITK.utilities.vtk import sitk2vtk
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog

# Silenciar warnings de VTK
try:
    vtk.vtkObject.GlobalWarningDisplayOff()
except Exception:
    try:
        vtk.vtkOutputWindow.SetGlobalWarningDisplay(False)
    except Exception:
        pass
# si la barra de Matplotlib está en “pan/zoom”, la apaga para que no interfiera con los clics/arrastres.
def _deactivate_toolbar(fig):
    try:
        tb = fig.canvas.manager.toolbar
    except Exception:
        return
    try:
        mode = getattr(tb, "mode", "")
        if mode in ("pan/zoom", "zoom rect", "zoom"):
            try:
                if "pan" in mode:
                    tb.pan()
                elif "zoom" in mode:
                    tb.zoom()
            except Exception:
                pass
    except Exception:
        pass
    try:
        if getattr(tb, "_active", None) is not None:
            if tb._active == "PAN":
                tb.pan()
            elif tb._active == "ZOOM":
                tb.zoom()
    except Exception:
        pass

# detectan si el backend de Matplotlib es Tk y obtienen un parent seguro para los diálogos (mejora el foco/posición).
def _mpl_uses_tk():
    try:
        return "tk" in matplotlib.get_backend().lower()
    except Exception:
        return False

def _safe_parent_for_tk(fig):
    if not _mpl_uses_tk():
        return None
    try:
        return fig.canvas.manager.window
    except Exception:
        return None

# ---------- Popups ----------
def _open_loading_popup(parent, message="cargando..."):
    try:
        if parent is None:
            win = tk.Tk()
        else:
            win = tk.Toplevel(parent)
    except Exception:
        win = tk.Tk()
        parent = None

    win.title("")
    try: win.attributes("-topmost", True)
    except Exception: pass
    win.resizable(False, False)

    ww, wh = 520, 200
    try:
        sw = win.winfo_screenwidth(); sh = win.winfo_screenheight()
        x = int((sw - ww) / 2); y = int((sh - wh) / 2)
        win.geometry(f"{ww}x{wh}+{max(0,x)}+{max(0,y)}")
    except Exception:
        pass

    frm = ttk.Frame(win, padding=24); frm.pack(fill="both", expand=True)
    style = ttk.Style(win)
    try: style.configure("Big.TLabel", font=("Helvetica", 24))
    except Exception: pass
    lbl = ttk.Label(frm, text=message, anchor="center", justify="center", style="Big.TLabel")
    lbl.pack(fill="both", expand=True)

    try: win.lift(); win.focus_force()
    except Exception: pass

    win.update_idletasks(); win.update()
    return win, None

def _close_loading_popup(win, _unused=None):
    try:
        win.destroy()
    except Exception:
        pass

def _show_info_message(msg_text: str, title: str = "Aviso"):
    try:
        root = tk.Tk()
        root.withdraw()  # no mostrar ventana raíz
        root.attributes("-topmost", True)
    except Exception:
        # Fallback si Tk falla por alguna razón
        print(f"[INFO] {title}: {msg_text}")
        return
    try:
        messagebox.showinfo(title, msg_text, parent=root)
    except Exception:
        # sin parent
        messagebox.showinfo(title, msg_text)
    try:
        root.destroy()
    except Exception:
        pass

# ============================
#  Métricas / VTK utils
# ============================

# calculo de parametros: convierte la mascara, porcentaje de los HU altos que quedaron cubiertos por la máscara, porcentaje de los HU bajos que quedaron cubiertos por la máscara,
# hu_p5_mask,hu_p50_mask, hu_p95_mask, cantidad de voxeles en el volumen, cantidad de voxeles que caen dentro de la mascara}

def hu_coverage_metrics(hu_img: sitk.Image, mask_sitk: sitk.Image, t3_hu: float = 300.0, soft_hu_max: float = 150.0):
    hu_np = sitk.GetArrayFromImage(hu_img).astype("float32")
    m_np  = sitk.GetArrayFromImage(mask_sitk).astype("uint8") > 0
    total_mask = int(m_np.sum())
    if total_mask == 0:
        return {"voxels_in_mask": 0, "coverage_highHU_%": 0.0, "leakage_soft_%": 0.0,
                "hu_p5_mask": float("nan"), "hu_p50_mask": float("nan"), "hu_p95_mask": float("nan"),
                "highHU_voxels": 0, "highHU_in_mask": 0}
    highHU = hu_np >= float(t3_hu)
    highHU_vox = int(highHU.sum())
    highHU_in_mask = int((highHU & m_np).sum())
    coverage = (100.0 * highHU_in_mask / max(highHU_vox, 1)) if highHU_vox else 0.0
    soft_in_mask = int(((hu_np < float(soft_hu_max)) & m_np).sum())
    leakage = 100.0 * soft_in_mask / total_mask
    hu_vals_mask = hu_np[m_np]
    p5, p50, p95 = [float(np.percentile(hu_vals_mask, p)) for p in (5, 50, 95)]
    return {"voxels_in_mask": total_mask, "coverage_highHU_%": coverage, "leakage_soft_%": leakage,
            "hu_p5_mask": p5, "hu_p50_mask": p50, "hu_p95_mask": p95,
            "highHU_voxels": highHU_vox, "highHU_in_mask": highHU_in_mask}

# calculo de volumen en mm3, dimensiones en x,y y z
def mask_geometric_metrics(mask_sitk: sitk.Image):
    m_np = sitk.GetArrayFromImage(mask_sitk).astype("uint8") > 0
    sx, sy, sz = mask_sitk.GetSpacing()
    if not m_np.any():
        return {"vol_mm3": 0.0, "dim_x_mm": 0.0, "dim_y_mm": 0.0, "dim_z_mm": 0.0}
    z, y, x = np.nonzero(m_np)
    dim_x_mm = (x.max() - x.min() + 1) * sx
    dim_y_mm = (y.max() - y.min() + 1) * sy
    dim_z_mm = (z.max() - z.min() + 1) * sz
    vol_mm3 = float(m_np.sum()) * (sx * sy * sz)
    return {"vol_mm3": float(vol_mm3), "dim_x_mm": float(dim_x_mm),
            "dim_y_mm": float(dim_y_mm), "dim_z_mm": float(dim_z_mm)}

# procesamiento de la isosuperficie: convierte a triangulos y calcula volumen y superficie
def vtk_mesh_report(poly):
    tri = vtk.vtkTriangleFilter(); tri.SetInputData(poly); tri.Update()
    clean = vtk.vtkCleanPolyData(); clean.SetInputConnection(tri.GetOutputPort()); clean.Update()
    fe = vtk.vtkFeatureEdges(); fe.SetInputConnection(clean.GetOutputPort())
    fe.BoundaryEdgesOn(); fe.NonManifoldEdgesOn(); fe.ManifoldEdgesOff(); fe.FeatureEdgesOff(); fe.Update()
    boundary_edges = fe.GetOutput().GetNumberOfCells()
    mp = vtk.vtkMassProperties(); mp.SetInputConnection(clean.GetOutputPort()); mp.Update()
    return {"boundary_edges": int(boundary_edges),
            "watertight": bool(boundary_edges == 0),
            "volume_mm3": float(mp.GetVolume()),
            "surface_mm2": float(mp.GetSurfaceArea()),
            "n_points": int(clean.GetOutput().GetNumberOfPoints()),
            "n_polys": int(clean.GetOutput().GetNumberOfPolys())}

# comparacion de volumenes
def compare_volumes(mask_sitk: sitk.Image, poly):
    tri = vtk.vtkTriangleFilter(); tri.SetInputData(poly); tri.Update()
    clean = vtk.vtkCleanPolyData(); clean.SetInputConnection(tri.GetOutputPort()); clean.Update()
    mp = vtk.vtkMassProperties(); mp.SetInputConnection(clean.GetOutputPort()); mp.Update()
    stl_vol = float(mp.GetVolume())
    vox = sitk.GetArrayFromImage(mask_sitk).astype(bool).sum()
    vox_mm3 = float(vox) * float(np.prod(mask_sitk.GetSpacing()))
    return {"vox_mm3": vox_mm3, "stl_mm3": stl_vol, "delta_mm3": stl_vol - vox_mm3}

# ============================
#  Lectura DICOMs / Conversión HU
# ============================
try:
    import pydicom
except Exception:
    pydicom = None

def read_best_series(folder: str, require_modality_ct=True, require_ct_sopclass=False):
    reader = sitk.ImageSeriesReader()
    folder = os.path.abspath(folder)
    candidates = {folder}
    for root, _dirs, files in os.walk(folder):
        if files:
            candidates.add(root)

    def is_ct_file(path):
        if not require_modality_ct and not require_ct_sopclass:
            return True
        modality_ok = True
        sop_ok = True
        if require_modality_ct:
            if pydicom is not None:
                try:
                    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
                    modality_ok = (str(getattr(ds, "Modality", "")).upper() == "CT")
                except Exception:
                    modality_ok = False
            else:
                try:
                    r = sitk.ImageFileReader(); r.SetFileName(path); r.ReadImageInformation()
                    modality_ok = r.HasMetaDataKey("0008|0060") and r.GetMetaData("0008|0060").upper() == "CT"
                except Exception:
                    modality_ok = False
        if require_ct_sopclass and pydicom is not None:
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
                sop_uid = str(getattr(ds, "SOPClassUID", ""))
                sop_ok = sop_uid in {"1.2.840.10008.5.1.4.1.1.2","1.2.840.10008.5.1.4.1.1.2.1"}
            except Exception:
                sop_ok = False
        return modality_ok and sop_ok

    best_files, best_len = None, -1
    for path in sorted(candidates):
        try:
            series_ids = reader.GetGDCMSeriesIDs(path) or []
            for uid in series_ids:
                files = reader.GetGDCMSeriesFileNames(path, uid) or []
                if not files: continue
                if not is_ct_file(files[0]): continue
                if len(files) > best_len:
                    best_len = len(files); best_files = files
        except Exception:
            continue
    if not best_files:
        raise RuntimeError(f"NoSeriesFound: No se encontró ninguna serie DICOM en: {folder}")
    reader.SetFileNames(best_files)
    return reader.Execute(), best_files

def to_hu(img: sitk.Image, first_file: str, file_list=None) -> sitk.Image:
    def read_sitk_tags(p):
        r = sitk.ImageFileReader(); r.SetFileName(p); r.LoadPrivateTagsOn(); r.ReadImageInformation()
        si = r.GetMetaData("0028|1052") if r.HasMetaDataKey("0028|1052") else None
        ss = r.GetMetaData("0028|1053") if r.HasMetaDataKey("0028|1053") else None
        return ss, si
    def read_dcm_tags(p):
        if pydicom is None: return (None, None)
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            return getattr(ds, "RescaleSlope", None), getattr(ds, "RescaleIntercept", None)
        except Exception:
            return (None, None)

    ss, si = read_sitk_tags(first_file)
    if ss is None or si is None:
        ss, si = read_dcm_tags(first_file)
    try: slope = float(ss) if ss is not None else 1.0
    except Exception: slope = 1.0
    try: intercept = float(si) if si is not None else 0.0
    except Exception: intercept = 0.0
    if slope == 0.0: slope = 1.0

    img = sitk.Cast(img, sitk.sitkFloat32)
    return sitk.ShiftScale(img, shift=intercept, scale=slope)

# ============================
#  Segmentación
# ============================

# máscara del hueso, doble umbralado
def double_threshold_mask(hu_img: sitk.Image, T1, T2, T3, T4, use_second=True):
    mask255 = sitk.DoubleThreshold(hu_img, T1, T2, T3 if use_second else T2, T4 if use_second else T2, 255, 0)
    mask_total = sitk.Cast(mask255 > 0, sitk.sitkUInt8)
    #trab = sitk.BinaryThreshold(hu_img, lowerThreshold=T1, upperThreshold=np.nextafter(T2, -np.inf), insideValue=1, outsideValue=0)
    #cort = sitk.BinaryThreshold(hu_img, lowerThreshold=T3 if use_second else T2, upperThreshold=T4 if use_second else T2, insideValue=1, outsideValue=0)
    return mask_total#, trab, cort

# máscara del paciente
def build_patient_mask(hu_img: sitk.Image, hu_threshold: float = -300, xy_margin_mm: float = 6.0,
                       keep_components='auto', min_volume_mm3: float = 3e5):
    patient = sitk.BinaryThreshold(hu_img, lowerThreshold=hu_threshold, upperThreshold=3000, insideValue=1, outsideValue=0)
    patient = sitk.Cast(patient, sitk.sitkUInt8)
    if xy_margin_mm > 0:
        sx, sy, _ = hu_img.GetSpacing()
        rx = max(1, int(round(xy_margin_mm / sx))); ry = max(1, int(round(xy_margin_mm / sy)))
        arr = sitk.GetArrayFromImage(patient).astype(np.uint8)
        arr[:, :, :rx]  = 0; arr[:, :, -rx:] = 0
        arr[:, :ry, :]  = 0; arr[:, -ry:, :] = 0
        patient = sitk.GetImageFromArray(arr); patient.CopyInformation(hu_img)

    
    cc = sitk.ConnectedComponent(patient)
    stats = sitk.LabelShapeStatisticsImageFilter(); stats.Execute(cc)
    labels = list(stats.GetLabels())
    if not labels:
        return patient
    labels.sort(key=lambda l: stats.GetPhysicalSize(l), reverse=True)
    
    # --- AUTO keep_components (detect 1 vs 2 knees) ---
    big_labels = [lab for lab in labels if stats.GetPhysicalSize(lab) >= float(min_volume_mm3)]
    if keep_components in (None, 0, 'auto', 'AUTO', 'Auto'):
        k_auto = 2 if len(big_labels) >= 2 else 1
        try:
            print(f"[AUTO] Detectados {len(big_labels)} componentes grandes → keep_components={k_auto}", flush=True)
        except Exception:
            pass
        k_to_keep = k_auto
    else:
        try:
            k_to_keep = int(keep_components)
        except Exception:
            k_to_keep = 1
        k_to_keep = max(1, min(20, k_to_keep))
    
    # --- Logging del keep_components efectivo ---
    try:
        _mode = "auto" if keep_components in (None, 0, 'auto', 'AUTO', 'Auto') else "manual"
        print(f"[PATIENT] keep_components (efectivo) = {int(k_to_keep)} [{_mode}] | componentes_grandes={len(big_labels)}", flush=True)
    except Exception:
        pass
    
    out = sitk.Image(patient.GetSize(), sitk.sitkUInt8); out.CopyInformation(hu_img)
    kept = 0
    # Priorizamos componentes grandes primero
    for lab in big_labels:
        comp = sitk.BinaryThreshold(cc, lab, lab, 1, 0)
        out = sitk.Cast(out | comp, sitk.sitkUInt8)
        kept += 1
        if kept >= int(k_to_keep):
            break
    # Si no alcanzó (p.ej., ninguna supera min_volume), caer al más grande disponible
    if kept == 0:
        max_lab = max(labels, key=lambda l: stats.GetPhysicalSize(l))
        out = sitk.BinaryThreshold(cc, max_lab, max_lab, 1, 0)
    return sitk.Cast(out>0, sitk.sitkUInt8)

# intersecion de las máscaras
def combine_bone_with_patient(bone_mask: sitk.Image, patient_mask: sitk.Image) -> sitk.Image:
    return sitk.Cast(sitk.Cast(bone_mask>0, sitk.sitkUInt8) & sitk.Cast(patient_mask>0, sitk.sitkUInt8), sitk.sitkUInt8)

# ============================
#  VTK
# ============================

def prepare_mask_for_vtk(mask_sitk: sitk.Image) -> sitk.Image:
    pad = sitk.ConstantPad(mask_sitk, [1]*3, [1]*3, 0)
    orienter = sitk.DICOMOrientImageFilter(); orienter.SetDesiredCoordinateOrientation("LPS")
    return orienter.Execute(pad)

def vtk_surface_from_mask(mask_sitk_for_vtk: sitk.Image, smooth_iterations=50, passband=0.01, feature_angle=60.0, decimate_ratio=0.0):
    vtk_img = sitk2vtk(mask_sitk_for_vtk)
    cast = vtk.vtkImageCast(); cast.SetInputData(vtk_img); cast.SetOutputScalarTypeToFloat(); cast.Update()
    try: mc = vtk.vtkFlyingEdges3D()
    except AttributeError: mc = vtk.vtkMarchingCubes()
    mc.SetInputConnection(cast.GetOutputPort()); mc.SetValue(0, 0.5); mc.Update()
    tri0 = vtk.vtkTriangleFilter(); tri0.SetInputConnection(mc.GetOutputPort()); tri0.Update()
    src = tri0
    if decimate_ratio > 0:
        dec = vtk.vtkDecimatePro(); dec.SetInputConnection(src.GetOutputPort())
        dec.SetTargetReduction(float(np.clip(decimate_ratio, 0.0, 0.9)))
        dec.PreserveTopologyOn(); dec.SplittingOff(); dec.BoundaryVertexDeletionOff(); dec.Update()
        src = dec
    ws = vtk.vtkWindowedSincPolyDataFilter(); ws.SetInputConnection(src.GetOutputPort())
    ws.SetNumberOfIterations(int(np.clip(smooth_iterations, 0, 200)))
    ws.BoundarySmoothingOff(); ws.FeatureEdgeSmoothingOff(); ws.SetFeatureAngle(feature_angle)
    ws.SetPassBand(float(np.clip(passband, 1e-4, 0.5))); ws.NonManifoldSmoothingOn(); ws.NormalizeCoordinatesOn(); ws.Update()
    norms = vtk.vtkPolyDataNormals(); norms.SetInputConnection(ws.GetOutputPort())
    norms.ConsistencyOn(); norms.AutoOrientNormalsOn(); norms.SplittingOff(); norms.Update()
    tri_final = vtk.vtkTriangleFilter(); tri_final.SetInputConnection(norms.GetOutputPort()); tri_final.Update()
    return tri_final.GetOutput()

def vtk_render_polydata(poly, title="Modelo óseo (VTK)"):
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(poly); mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor(); actor.SetMapper(mapper)
    ren = vtk.vtkRenderer(); ren.AddActor(actor); ren.SetBackground(0.1, 0.1, 0.12)
    rw = vtk.vtkRenderWindow(); rw.AddRenderer(ren); rw.SetSize(900, 700); rw.SetWindowName(title)
    iren = vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(rw)
    ren.ResetCamera(); rw.Render(); iren.Initialize(); iren.Start()

def vtk_write_stl(poly, path_out: str):
    w = vtk.vtkSTLWriter(); w.SetFileName(path_out); w.SetInputData(poly); w.SetFileTypeToBinary(); w.Write()

# ============================
#  VIEWER
# ============================

def launch_viewer():
    state = {
        "hu": None, "vol_hu": None, "vmin": None, "vmax": None, "current_z": 0,
        "mask_bone": None, "mask_patient": None, "mask_final": None,
        "axes": {}, "widgets": {}, "overlay": {"bone": True, "patient": True},
        "hist_win_fig": None, "hist_win_ax": None,
        "patient_hu_thr": 150.0,
        "keep_components": 2,
        "min_volume_mm3": 3e5,
    }

    fig, ax = plt.subplots(figsize=(7.5, 7.5)); _deactivate_toolbar(fig)
    fig.patch.set_facecolor('white')
    # Panel lateral de controles (fondo suave)
    ax_panel_bg = fig.add_axes([0.80, 0.02, 0.19, 0.94])
    ax_panel_bg.add_patch(Rectangle((0,0), 1, 1, transform=ax_panel_bg.transAxes,
                                    facecolor='#f5f5f7', edgecolor='#e6e6ea'))
    ax_panel_bg.axis('off')
    # Encabezado con leyenda rápida de colores
    ax_hdr = fig.add_axes([0.05, 0.93, 0.74, 0.05]); ax_hdr.axis('off')
    ax_hdr.text(0.00, 0.5, 'Bone', color='red', va='center', fontsize=11)
    ax_hdr.text(0.10, 0.5, '∧', color='#444', va='center', fontsize=11)
    ax_hdr.text(0.15, 0.5, 'Patient', color='green', va='center', fontsize=11)
    ax_hdr.text(0.30, 0.5, '→  STL 3D cerrado', color='#444', va='center', fontsize=11)

    ax.axis('off')
    ax.text(0.5, 0.5, "Cargá una carpeta DICOM\ncon el botón 'Cargar DICOM'",
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    plt.subplots_adjust(left=0.05, right=0.82, bottom=0.26, top=0.92)
    fig.suptitle('Segmentación de rodilla: visualización y STL', fontsize=13)
    state["fig"] = fig; state["axes"]["img"] = ax

    # Barra de estado inferior
    ax_status = fig.add_axes([0.05, 0.01, 0.74, 0.03]); ax_status.axis('off')
    state['axes']['status'] = ax_status
    def _update_status(msg=None):
        ax_status.clear(); ax_status.axis('off')
        if state['vol_hu'] is not None:
            z = int(state['current_z'])
            overlay_txt = f"Bone={'ON' if state['overlay']['bone'] else 'OFF'} | Patient={'ON' if state['overlay']['patient'] else 'OFF'}"
            base = f"Z={z}   |   {overlay_txt}"
        else:
            base = 'Listo para cargar un estudio DICOM.'
        if msg:
            base += '   |   ' + str(msg)
        ax_status.text(0.0, 0.5, base, va='center', fontsize=9, color='#333')
        fig.canvas.draw_idle()


    # Botones fijos
    ax_btnload = plt.axes([0.82, 0.90, 0.15, 0.05]); btnload = Button(ax_btnload, "Cargar DICOM")
    ax_btnhist = plt.axes([0.82, 0.83, 0.15, 0.05]); btnhist = Button(ax_btnhist, "Histograma")

    state["widgets"]["btnload"] = btnload
    state["widgets"]["btnhist"] = btnhist

    
    def _format_slider_labels():
        try:
            state['widgets']['s_sit'].valtext.set_text(f"{int(state['widgets']['s_sit'].val)}")
            state['widgets']['s_pbd'].valtext.set_text(f"{float(state['widgets']['s_pbd'].val):.3f}")
            state['widgets']['s_dec'].valtext.set_text(f"{float(state['widgets']['s_dec'].val):.1f}%")
            lo, hi = state['widgets']['s_T23'].val
            state['widgets']['s_T23'].valtext.set_text(f"({int(lo)}, {int(hi)})")
            lo2, hi2 = state['widgets']['s_T14'].val
            state['widgets']['s_T14'].valtext.set_text(f"({int(lo2)}, {int(hi2)})")
            state['widgets']['s_z'].valtext.set_text(f"{int(state['widgets']['s_z'].val)}")
        except Exception:
            pass

    def show_main_image():
        if state["vol_hu"] is None: return
        aximg = state["axes"]["img"]; aximg.clear()
        z = int(np.clip(state["current_z"], 0, state["vol_hu"].shape[0]-1))
        sl = state["vol_hu"][z]
        aximg.imshow(sl, cmap="gray", vmin=state["vmin"], vmax=state["vmax"])
        if state["overlay"]["patient"] and state["mask_patient"] is not None:
            aximg.imshow((state["mask_patient"][z] > 0.5).astype(float), cmap="Greens", vmin=0, vmax=1, alpha=0.25)
        if state["overlay"]["bone"] and state["mask_bone"] is not None:
            aximg.imshow((state["mask_bone"][z] > 0.5).astype(float), cmap="Reds", vmin=0, vmax=1, alpha=0.35)
        aximg.set_title(f"Z = {z}")
        _update_status()
        aximg.axis("off"); fig.canvas.draw_idle()

    def update_hist_window():
        if state["hist_win_fig"] is None or not plt.fignum_exists(state["hist_win_fig"].number):
            return
        axh = state["hist_win_ax"]; axh.clear()
        if state["vol_hu"] is None:
            axh.text(0.5, 0.5, "No hay volumen cargado", ha="center", va="center")
        else:
            z = int(np.clip(state["current_z"], 0, state["vol_hu"].shape[0]-1))
            sl = state["vol_hu"][z].ravel()
            axh.hist(sl, bins=100)
            axh.set_title(f"Slice Z={z}")
            axh.set_xlabel("Valores HU"); axh.set_ylabel("Frecuencia")
        state["hist_win_fig"].canvas.draw_idle()

    def recompute_all(_=None):
        hu = state["hu"]
        if hu is None: return

        # Tomar estado actual de los checkboxes de overlay (si existen)
        try:
            ov_bone, ov_pat = state["widgets"]["chk_ovl"].get_status()
            state["overlay"]["bone"]    = bool(ov_bone)
            state["overlay"]["patient"] = bool(ov_pat)
        except Exception:
            pass
        use2 = state["widgets"]["chk_2nd"].get_status()[0] if "chk_2nd" in state["widgets"] else True
        (T1v, T4v) = state['widgets']['s_T14'].val
        (T2v, T3v) = state['widgets']['s_T23'].val
        T1v, T2v, T3v, T4v = int(T1v), int(T2v), int(T3v), int(T4v)

        # --- Enforce DoubleThreshold ordering: T1 <= T2 <= T3 <= T4 ---
        if state.get("fixing_T"):
            # Salida rápida para evitar recursión por set_val()
            state["fixing_T"] = False
            return
        changed = False
        if T2v < T1v:
            T2v = T1v; changed = True
        if T3v < T2v:
            T3v = T2v; changed = True
        if T4v < T3v:
            T4v = T3v; changed = True
        if changed:
            try:
                state["fixing_T"] = True
                state['widgets']['s_T14'].set_val((T1v, T4v))
                state['widgets']['s_T23'].set_val((T2v, T3v))
                print(f"[DT] Corrección T1..T4 => ({T1v}, {T2v}, {T3v}, {T4v})", flush=True)
            finally:
                pass
        bone_s = double_threshold_mask(hu, T1v, T2v, T3v, T4v, use2)
        patient_s = build_patient_mask(
            hu,
            hu_threshold=float(state["patient_hu_thr"]),
            xy_margin_mm=6.0,  # fijo por simplicidad
            keep_components='auto',
            min_volume_mm3=1.5e5,
        )
        final_s = combine_bone_with_patient(bone_s, patient_s)
        state["mask_bone"]    = sitk.GetArrayFromImage(bone_s).astype(np.float32)
        state["mask_patient"] = sitk.GetArrayFromImage(patient_s).astype(np.float32)
        state["mask_final"]   = sitk.GetArrayFromImage(final_s).astype(np.float32)

        # Sincronizar overlays según lo tildado en la UI (antes del primer dibujado)
        try:
            ov_bone, ov_pat = state["widgets"]["chk_ovl"].get_status()
            state["overlay"]["bone"]    = bool(ov_bone)
            state["overlay"]["patient"] = bool(ov_pat)
        except Exception:
            pass

        show_main_image(); update_hist_window()

    def build_poly_from_current_mask():
        if state["mask_final"] is None or state["mask_final"].sum() == 0:
            raise RuntimeError("Máscara final vacía: ajustá T1..T4 o 'Patient HU >'.")
        mt_np = (state["mask_final"] > 0.5).astype(np.uint8)
        mt_sitk = sitk.GetImageFromArray(mt_np); mt_sitk.CopyInformation(state["hu"])
        mt_sitk_capped = prepare_mask_for_vtk(mt_sitk)
        poly = vtk_surface_from_mask(
            mt_sitk_capped,
            smooth_iterations=int(state["widgets"]["s_sit"].val),
            passband=float(state["widgets"]["s_pbd"].val),
            decimate_ratio=float(state["widgets"]["s_dec"].val)/100.0,
            feature_angle=60.0
        )
        return poly, mt_sitk

    def on_3d(_e):
        loading_win, _ = _open_loading_popup(_safe_parent_for_tk(fig), "cargando...")
        poly = None
        try:
            poly, _ = build_poly_from_current_mask()
        except Exception as e:
            print(f"[VTK] Error en render: {e}")
        finally:
            _close_loading_popup(loading_win)
        if poly is not None:
            try:
                vtk_render_polydata(poly, title="Modelo óseo (VTK) – bone ∧ patient")
            except Exception as e:
                print(f"[VTK] Error abriendo viewer 3D: {e}")

    def on_stl(_e):
        root2 = tk.Tk(); root2.withdraw()
        out_path = filedialog.asksaveasfilename(title="Guardar STL", defaultextension=".stl",
                                                filetypes=[("STL files","*.stl")], initialfile="bone_surface.stl")
        root2.destroy()
        if not out_path:
            print("Guardado cancelado."); return
        loading_win, _ = _open_loading_popup(_safe_parent_for_tk(fig), "cargando...")
        try:
            poly, _ = build_poly_from_current_mask()
            vtk_write_stl(poly, out_path)
            print(f"STL guardado en: {out_path}")
        except Exception as e:
            print(f"[VTK] Error guardando STL: {e}")
        finally:
            _close_loading_popup(loading_win)

    def on_validate(_e):
        loading_win, _ = _open_loading_popup(_safe_parent_for_tk(fig), "cargando...")
        try:
            if state["mask_final"] is None or state["mask_final"].sum() == 0:
                print("[VAL] Máscara final vacía, no se puede validar."); return
            poly, mt_sitk = build_poly_from_current_mask()
            T3v = int(state['widgets']['s_T23'].val[1])
            hu_met = hu_coverage_metrics(state["hu"], mt_sitk, t3_hu=T3v, soft_hu_max=150.0)
            geo = mask_geometric_metrics(mt_sitk)
            mesh_rep = vtk_mesh_report(poly)
            vol_cmp = compare_volumes(mt_sitk, poly)
        except Exception as e:
            print(f"[VAL] Error en validación sin GT: {e}")
            return
        finally:
            _close_loading_popup(loading_win)
        print("\n[VALIDACIÓN SIN GT]")
        print(f"- Cobertura cortical (HU >= T3={T3v}): {hu_met['coverage_highHU_%']:.1f}% (en máscara)")
        print(f"- Leakage a blandos (HU<150) dentro de máscara: {hu_met['leakage_soft_%']:.1f}%")
        print(f"- HU en máscara (p5/50/95): {hu_met['hu_p5_mask']:.0f} / {hu_met['hu_p50_mask']:.0f} / {hu_met['hu_p95_mask']:.0f}")
        print(f"- Dimensiones (mm) X×Y×Z: {geo['dim_x_mm']:.1f} × {geo['dim_y_mm']:.1f} × {geo['dim_z_mm']:.1f}")
        print(f"- Volumen vóxel: {geo['vol_mm3']:.0f} mm^3 | Volumen STL: {vol_cmp['stl_mm3']:.0f} mm^3 | Δ: {vol_cmp['delta_mm3']:.0f} mm^3")
        print(f"- Superficie STL: {mesh_rep['surface_mm2']:.0f} mm^2")
        print(f"- Malla watertight: {mesh_rep['watertight']} (edges abiertos: {mesh_rep['boundary_edges']}) | Triángulos: {mesh_rep['n_polys']}")
        print(f"- Vóxeles en máscara final: {hu_met['voxels_in_mask']}")
    def on_hist(_e):
        loading_win, _ = _open_loading_popup(_safe_parent_for_tk(fig), "cargando...")
        try:
            if state["hist_win_fig"] is None or not plt.fignum_exists(state["hist_win_fig"].number):
                fig_h, ax_h = plt.subplots(figsize=(6,4))
                fig_h.suptitle("Histograma (ventana separada)", fontsize=12)
                state["hist_win_fig"] = fig_h; state["hist_win_ax"] = ax_h
                try: fig_h.show(); plt.show(block=False)
                except Exception: pass
            update_hist_window()
        finally:
            _close_loading_popup(loading_win)

    def on_slice_change(val):
        state["current_z"] = int(val); _update_status(); show_main_image(); update_hist_window()

    def on_overlay_clicked(label):
        if label == "Ver Bone":
            state["overlay"]["bone"] = not state["overlay"]["bone"]
        elif label == "Ver Patient":
            state["overlay"]["patient"] = not state["overlay"]["patient"]
        show_main_image()
        _update_status()

    def on_apply_patient_hu(_e):
        txt = state["widgets"]["tb_pmh"].text
        try:
            val = float(txt)
        except Exception:
            print(f"[UI] Valor inválido para 'Patient HU >': {txt}. Use un número (ej. -300, 150)."); return
        if val < -3000: val = -3000
        if val >  4000: val = 4000
        state["patient_hu_thr"] = float(val)
        print(f"[UI] 'Patient HU >' = {val}")
        loading_win, _ = _open_loading_popup(_safe_parent_for_tk(fig), "cargando...")
        try:
            recompute_all()
        finally:
            _close_loading_popup(loading_win)

    def on_apply_keep(_e):
        raw = state["widgets"]["tb_keep"].text.strip()
        try:
            k = int(float(raw))
        except Exception:
            print(f"[UI] keep_components inválido: {raw}"); return
        k = max(1, min(20, k))
        state["keep_components"] = int(k)
        print(f"[UI] keep_components = {k}")
        loading_win, _ = _open_loading_popup(_safe_parent_for_tk(fig), "cargando...")
        try:
            recompute_all()
        finally:
            _close_loading_popup(loading_win)

    def on_apply_minv(_e):
        raw = state["widgets"]["tb_minv"].text or ""
        raw = raw.strip().lower().replace("mm3","").replace("mm^3","").replace("mm³","").strip()
        try:
            v = float(raw)
        except Exception:
            print(f"[UI] min_volume_mm3 inválido: {raw}"); return
        v = float(np.clip(v, 1e3, 1e8))
        state["min_volume_mm3"] = v
        print(f"[UI] min_volume_mm3 = {v}")
        loading_win, _ = _open_loading_popup(_safe_parent_for_tk(fig), "cargando...")
        try:
            recompute_all()
        finally:
            _close_loading_popup(loading_win)

    def on_load(_event=None):
        root = tk.Tk(); root.withdraw()
        new_folder = filedialog.askdirectory(title="Seleccionar carpeta DICOM")
        root.destroy()
        if not new_folder: return

        loading_win, _ = _open_loading_popup(_safe_parent_for_tk(fig), "cargando...")
        try:
            img, files = read_best_series(new_folder)  # exige CT
            hu = to_hu(img, files[0], file_list=files)
            vol_hu = sitk.GetArrayFromImage(hu)
            state["hu"], state["vol_hu"] = hu, vol_hu
            state["vmin"], state["vmax"] = [float(x) for x in np.percentile(vol_hu, (5, 99.5))]
            state["current_z"] = vol_hu.shape[0]//2

            # Actualizar título con la carpeta cargada
            try:
                fig.suptitle(f'Segmentación de rodilla – {os.path.basename(new_folder)}', fontsize=13)
            except Exception:
                pass

            if "s_z" not in state["widgets"]:state["axes"]["img"].set_position([0.05, 0.26, 0.77, 0.66])
            # Controles laterales (derecha)
            ax_pmh_txt    = fig.add_axes([0.82, 0.60, 0.15, 0.035]); tb_pmh        = TextBox(ax_pmh_txt, "Patient HU >", initial=str(int(state.get("patient_hu_thr",150))))
            ax_pmh_apply  = fig.add_axes([0.82, 0.56, 0.15, 0.035]); btn_pmh_apply = Button(ax_pmh_apply, "Aplicar HU")

            ax_ovl = fig.add_axes([0.82, 0.44, 0.15, 0.07]); chk_ovl = CheckButtons(ax_ovl, ["Ver Bone", "Ver Patient"], [state["overlay"]["bone"], state["overlay"]["patient"]])
            ax_chk = fig.add_axes([0.82, 0.38, 0.15, 0.05]); chk_2nd  = CheckButtons(ax_chk, ["2º rango (T3–T4)"], [True])

            ax_sit = fig.add_axes([0.82, 0.34, 0.15, 0.03]); s_sit = Slider(ax_sit, "Smoothing iters", 0, 100, valinit=50, valstep=1)
            ax_pbd = fig.add_axes([0.82, 0.30, 0.15, 0.03]); s_pbd = Slider(ax_pbd, "Passband", 0.001, 0.1, valinit=0.01)
            ax_dec = fig.add_axes([0.82, 0.26, 0.15, 0.03]); s_dec = Slider(ax_dec, "Decimation %", 0.0, 90.0, valinit=0.0)
            ax_btn3d  = fig.add_axes([0.82, 0.18, 0.15, 0.05]); btn3d  = Button(ax_btn3d,  "3D (VTK)", hovercolor="0.9")
            ax_btnstl = fig.add_axes([0.82, 0.11, 0.15, 0.05]); btnstl = Button(ax_btnstl, "Guardar STL", hovercolor="0.9")
            ax_btnval = fig.add_axes([0.82, 0.04, 0.15, 0.05]); btnval = Button(ax_btnval, "Validar (sin GT)", hovercolor="0.9")

            # Sliders inferiores (izquierda)
            ax_z   = fig.add_axes([0.10, 0.16, 0.66, 0.03]); s_z   = Slider(ax_z, "Slice (Z)", 0, vol_hu.shape[0]-1, valinit=state["current_z"], valstep=1)
            # Superior: estrecho T2–T3 | Inferior: amplio T1–T4
            ax_T23 = fig.add_axes([0.10, 0.10, 0.66, 0.04]); s_T23 = RangeSlider(ax_T23, "T2–T3 (HU)", -500, 3000, valinit=(300, 1200))
            ax_T14 = fig.add_axes([0.10, 0.04, 0.66, 0.04]); s_T14 = RangeSlider(ax_T14, "T1–T4 (HU)", -500, 3000, valinit=(150, 2000))

            state["widgets"].update({
                "tb_pmh": tb_pmh, "btn_pmh_apply": btn_pmh_apply,
                "chk_2nd": chk_2nd, "chk_ovl": chk_ovl,
                "s_sit": s_sit, "s_pbd": s_pbd, "s_dec": s_dec,
                "btn3d": btn3d, "btnstl": btnstl, "btnval": btnval,
                "s_z": s_z, "s_T23": s_T23, "s_T14": s_T14
            })

            # Ajustar etiquetas de sliders y formatear valtext
            try:
                for _s in (s_sit, s_pbd, s_dec, s_z, s_T23, s_T14):
                    _s.valtext.set_fontsize(9)
            except Exception:
                pass
            _format_slider_labels()


            # Callbacks
            s_T23.on_changed(lambda _v: (_format_slider_labels(), recompute_all()))
            s_T14.on_changed(lambda _v: (_format_slider_labels(), recompute_all()))
            s_z.on_changed(lambda v: (_format_slider_labels(), on_slice_change(v)))
            chk_2nd.on_clicked(recompute_all)
            chk_ovl.on_clicked(on_overlay_clicked)
            btn3d.on_clicked(on_3d)
            btnstl.on_clicked(on_stl)
            btnval.on_clicked(on_validate)
            s_sit.on_changed(lambda _v: (_format_slider_labels(), recompute_all()))
            s_pbd.on_changed(lambda _v: (_format_slider_labels(), recompute_all()))
            s_dec.on_changed(lambda _v: (_format_slider_labels(), recompute_all()))
            state["widgets"]["btn_pmh_apply"].on_clicked(on_apply_patient_hu)
            use2 = state["widgets"]["chk_2nd"].get_status()[0] if "chk_2nd" in state["widgets"] else True
            (T1v, T4v) = state["widgets"].get("s_T14", type("obj", (), {"val": (150,2000)})()).val
            (T2v, T3v) = state["widgets"].get("s_T23", type("obj", (), {"val": (300,1200)})()).val
            T1v, T2v, T3v, T4v = int(T1v), int(T2v), int(T3v), int(T4v)

            bone_s = double_threshold_mask(hu, T1v, T2v, T3v, T4v, use2)
            patient_s = build_patient_mask(
                hu,
                hu_threshold=float(state.get("patient_hu_thr", 150.0)),
                xy_margin_mm=6.0,
                keep_components='auto',
                min_volume_mm3=1.5e5,
            )
            final_s = combine_bone_with_patient(bone_s, patient_s)

            state["mask_bone"]    = sitk.GetArrayFromImage(bone_s).astype(np.float32)
            state["mask_patient"] = sitk.GetArrayFromImage(patient_s).astype(np.float32)
            state["mask_final"]   = sitk.GetArrayFromImage(final_s).astype(np.float32)

            show_main_image(); update_hist_window()
            print(f"[LOAD] Serie DICOM cargada: {new_folder}")
        except Exception as e:
            # Cerrar el popup de carga ANTES de mostrar el info
            _close_loading_popup(loading_win)
            msg = str(e)
            if "NoSeriesFound" in msg or "No Series were found" in msg:
                _show_info_message("La carpeta seleccionada no contiene archivos DICOM válidos.", title="Aviso")
            elif ("CT válida" in msg) or ("Modality" in msg) or (" CT " in msg) or ("no son TC" in msg):
                _show_info_message("Las imágenes cargadas no son TC.", title="Aviso")
            else:
                _show_info_message(f"Error al cargar:{msg}", title="Error")
            print(f"[LOAD] Error al cargar nueva serie: {e}")
            return
        finally:
            try:
                _close_loading_popup(loading_win)
            except Exception:
                pass

    btnhist.on_clicked(on_hist)
    btnload.on_clicked(on_load)

    def on_click(event):
        if event.inaxes != state["axes"]["img"] or state["vol_hu"] is None: return
        if event.xdata is None or event.ydata is None: return
        z = state["current_z"]; x = int(round(event.xdata)); y = int(round(event.ydata))
        H, W = state["vol_hu"].shape[1:3]
        if 0 <= y < H and 0 <= x < W:
            hu_val = float(state["vol_hu"][z, y, x])
            print(f"HU en (Z={z}, Y={y}, X={x}) = {hu_val:.2f}")
            state["axes"]["img"].plot(x, y, 'ro', markersize=5); fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

if __name__ == "__main__":
    launch_viewer()