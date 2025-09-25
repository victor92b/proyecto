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


class NoSeriesFound(RuntimeError):
    def __init__(self, folder: str, reason: str, found_modalities=None, code: str = "no_series"):
        self.folder = folder
        self.reason = reason
        self.code = code
        self.found_modalities = tuple(sorted(set(found_modalities or [])))
        msg_reason = reason.strip()
        super().__init__(f"NoSeriesFound: {msg_reason} -> {folder}")


class NoCTSeriesFound(NoSeriesFound):
    def __init__(self, folder: str, found_modalities=None):
        reason = "Solo se encontraron series no-CT"
        if found_modalities:
            listed = ", ".join(sorted(set(found_modalities)))
            reason = f"Solo se encontraron series no-CT (modalidades: {listed})"
        super().__init__(folder, reason, found_modalities=found_modalities, code="no_ct")


def _get_dicom_tags(path):
    modality = None
    sop_uid = None
    if pydicom is not None:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            modality = str(getattr(ds, "Modality", "")).upper() or None
            sop_uid = str(getattr(ds, "SOPClassUID", "")) or None
        except Exception:
            modality = modality or None
            sop_uid = sop_uid or None
    if modality is None:
        try:
            r = sitk.ImageFileReader(); r.SetFileName(path); r.ReadImageInformation()
            if r.HasMetaDataKey("0008|0060"):
                modality = r.GetMetaData("0008|0060").upper() or None
        except Exception:
            modality = modality or None
    return modality, sop_uid


def read_best_series(folder: str, require_modality_ct=True, require_ct_sopclass=False):
    reader = sitk.ImageSeriesReader()
    folder = os.path.abspath(folder)
    candidates = {folder}
    for root, _dirs, files in os.walk(folder):
        if files:
            candidates.add(root)

    best_files, best_len = None, -1
    found_any_series = False
    found_modalities = set()
    for path in sorted(candidates):
        try:
            series_ids = reader.GetGDCMSeriesIDs(path) or []
            for uid in series_ids:
                files = reader.GetGDCMSeriesFileNames(path, uid) or []
                if not files:
                    continue
                found_any_series = True
                modality, sop_uid = _get_dicom_tags(files[0])
                recorded_mod = modality if modality else "Desconocida"
                found_modalities.add(recorded_mod)
                modality_ok = True
                sop_ok = True
                if require_modality_ct:
                    modality_ok = (modality == "CT")
                if require_ct_sopclass:
                    sop_ok = (sop_uid in {"1.2.840.10008.5.1.4.1.1.2", "1.2.840.10008.5.1.4.1.1.2.1"})
                if not (modality_ok and sop_ok):
                    continue
                if len(files) > best_len:
                    best_len = len(files)
                    best_files = files
        except Exception:
            continue
    if not best_files:
        if require_modality_ct and found_modalities:
            raise NoCTSeriesFound(folder, found_modalities=found_modalities)
        if not found_any_series:
            raise NoSeriesFound(folder, "No se encontró ninguna serie DICOM en la carpeta.",
                                found_modalities=found_modalities, code="no_series")
        raise NoSeriesFound(folder, "No se encontró una serie DICOM que cumpla los filtros solicitados.",
                            found_modalities=found_modalities, code="no_match")
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


    fig, ax = plt.subplots(figsize=(11.2, 7.8)); _deactivate_toolbar(fig)
    fig.patch.set_facecolor('#f4f6fb')
    plt.subplots_adjust(left=0.06, right=0.73, bottom=0.20, top=0.89)
    fig.suptitle('Segmentación de rodilla – visor interactivo', fontsize=14, fontweight='bold', y=0.98)
    ax.set_position([0.06, 0.29, 0.66, 0.57])
    state["fig"] = fig; state["axes"]["img"] = ax

    # --- Layout helpers / posiciones ---
    panel_left, panel_width = 0.74, 0.23
    slider_left, slider_width = 0.08, 0.64

    def _build_panel_layout():
        panel_top, min_panel_bottom = 0.90, 0.05
        info_height = 0.12
        info_gap = 0.015
        controls_top = panel_top - info_height - info_gap
        slots = [
            ("btnload", 0.075, 0.016),
            ("btnhist", 0.060, 0.018),
            ("section_seg", 0.032, 0.008),
            ("pmh_txt", 0.070, 0.010),
            ("pmh_apply", 0.055, 0.016),
            ("section_overlay", 0.030, 0.008),
            ("ovl", 0.088, 0.012),
            ("chk_second", 0.055, 0.014),
            ("section_post", 0.030, 0.008),
            ("s_sit", 0.060, 0.012),
            ("s_pbd", 0.060, 0.012),
            ("s_dec", 0.060, 0.020),
            ("section_export", 0.030, 0.008),
            ("btn3d", 0.060, 0.012),
            ("btnstl", 0.060, 0.012),
            ("btnval", 0.060, 0.0),
        ]

        total_consumption = sum(height + gap for _, height, gap in slots)
        max_available = controls_top - min_panel_bottom
        factor = max_available / total_consumption if total_consumption > max_available else 1.0

        cursor = controls_top
        panel_coords = {}
        for name, height, gap in slots:
            height_scaled = height * factor
            gap_scaled = gap * factor
            cursor -= height_scaled
            panel_coords[name] = [panel_left, cursor, panel_width, height_scaled]
            cursor -= gap_scaled

        panel_bottom = max(cursor - 0.02, min_panel_bottom)

        return {
            "panel_left": panel_left,
            "panel_width": panel_width,
            "panel_bottom": panel_bottom,
            "panel_top": panel_top,
            "panel_coords": panel_coords,
            "slider_coords": {
                "s_z":   [slider_left, 0.195, slider_width, 0.045],
                "s_T23": [slider_left, 0.118, slider_width, 0.050],
                "s_T14": [slider_left, 0.045, slider_width, 0.050],
            },
            "panel_bg": [panel_left - 0.015, panel_bottom - 0.02,
                          panel_width + 0.03, panel_top - panel_bottom + 0.07],
            "slider_bg": [slider_left - 0.025, 0.040, slider_width + 0.05, 0.235],
            "slider_info": [panel_left - 0.005, panel_top - info_height, panel_width + 0.01, info_height],
        }

    layout = _build_panel_layout()
    state["layout"] = layout

    # Panel lateral y zona de sliders: fondos suaves
    ax_panel_bg = fig.add_axes(layout["panel_bg"])

    ax_panel_bg.add_patch(Rectangle((0, 0), 1, 1, transform=ax_panel_bg.transAxes,
                                    facecolor='#ffffff', edgecolor='#d0d7de', linewidth=1.2))
    ax_panel_bg.axis('off')


    ax_slider_info = fig.add_axes(layout["slider_info"])
    ax_slider_info.add_patch(Rectangle((0, 0), 1, 1, transform=ax_slider_info.transAxes,
                                       facecolor='#ffffff', edgecolor='#d0d7de', linewidth=1.2))
    ax_slider_info.axis('off')
    slider_texts = (
        (0.76, 'Explorar volumen y umbrales HU', 10.5, True),
        (0.44, 'Usá el slider superior para navegar cortes axiales.', 9.0, False),
        (0.22, 'Los rangos T1–T4 y T2–T3 afinan la máscara ósea.', 9.0, False),
    )
    for y, msg, size, is_title in slider_texts:
        ax_slider_info.text(
            0.05,
            y,
            msg,
            fontsize=size,
            color='#111827' if is_title else '#4b5563',
            fontweight='bold' if is_title else 'normal',
            va='center',
        )

    ax_slider_bg = fig.add_axes(layout["slider_bg"])
    ax_slider_bg.add_patch(Rectangle((0, 0), 1, 1, transform=ax_slider_bg.transAxes,
                                     facecolor='#ffffff', edgecolor='#d0d7de', linewidth=1.2))
    ax_slider_bg.axis('off')

    # Encabezado informativo con fondo
    ax_header = fig.add_axes([0.055, 0.895, 0.66, 0.085])
    ax_header.add_patch(Rectangle((0, 0), 1, 1, transform=ax_header.transAxes,
                                  facecolor='#ffffff', edgecolor='#d0d7de', linewidth=1.1))
    ax_header.axis('off')
    ax_header.text(0.03, 0.66, '1. Cargar estudio · 2. Ajustar umbrales · 3. Validar / exportar',
                   fontsize=11, color='#1f2937', weight='bold', va='center')
    ax_header.text(0.03, 0.30,
                   'Bone → rojo   |   Paciente → verde   |   STL exportado desde la máscara final',
                   fontsize=9.6, color='#4b5563', va='center')


    # Placeholder inicial en el eje principal
    ax.set_facecolor('#111827')
    ax.axis('off')
    ax.text(0.5, 0.52, "Cargá una carpeta DICOM\ncon el panel derecho",
            ha='center', va='center', fontsize=15, color='white', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#1f2937', alpha=0.8, pad=0.8))

    # Barra de estado inferior
    ax_status = fig.add_axes([0.06, 0.03, 0.64, 0.04])
    ax_status.add_patch(Rectangle((0, 0), 1, 1, transform=ax_status.transAxes,
                                  facecolor='#eef2ff', edgecolor='#d0d7de'))
    ax_status.axis('off')
    state['axes']['status'] = ax_status
    def _update_status(msg=None):
        ax_status.clear()
        ax_status.add_patch(Rectangle((0, 0), 1, 1, transform=ax_status.transAxes,
                                      facecolor='#eef2ff', edgecolor='#d0d7de'))
        ax_status.axis('off')
        if state['vol_hu'] is not None:
            z = int(state['current_z'])
            overlay_txt = f"Bone={'ON' if state['overlay']['bone'] else 'OFF'} | Patient={'ON' if state['overlay']['patient'] else 'OFF'}"
            folder_txt = state.get('current_folder', '')
            base = f"Z={z}   |   {overlay_txt}"
            if folder_txt:
                base = f"{folder_txt}   |   {base}"
        else:
            base = 'Listo para cargar un estudio DICOM.'
        if msg:
            base += '   |   ' + str(msg)
        ax_status.text(0.02, 0.5, base, va='center', fontsize=9.5, color='#1f2937')
        fig.canvas.draw_idle()

    def _style_slider(slider, *, face='#2563eb', track='#dbeafe'):
        try:
            slider.ax.set_facecolor('#ffffff')
        except Exception:
            pass
        for attr in ("poly", "track", "hline"):
            obj = getattr(slider, attr, None)
            if obj is None:
                continue
            try:
                if attr == "poly":
                    obj.set_color(face)
                elif attr == "track":
                    obj.set_facecolor(track)
                else:
                    obj.set_color('#475569')
            except Exception:
                continue
        try:
            slider.handle.set_color(face)
        except Exception:
            pass
        try:
            for h in getattr(slider, 'handles', []):
                h.set_facecolor(face)
        except Exception:
            pass

    def _style_button(btn, *, color=None, text_color='#1f2937', size=10, weight='bold'):
        try:
            if color is not None and hasattr(btn, 'color'):
                btn.color = color
        except Exception:
            pass
        try:
            btn.label.set_fontsize(size)
            btn.label.set_fontweight(weight)
            btn.label.set_color(text_color)
        except Exception:
            pass


    def _inset_axes(pos, pad_x=0.012, pad_y=0.008):
        x, y, w, h = pos
        return [x + pad_x, y + pad_y, max(w - 2 * pad_x, 0.01), max(h - 2 * pad_y, 0.01)]

    def _style_checkbuttons(chk, *, label_colors=None, facecolors=None, label_size=8.8):
        try:
            n_rects = len(chk.rectangles)
            facecolors = facecolors or []
            label_colors = label_colors or []
            for idx, rect in enumerate(chk.rectangles):
                cy = rect.get_y() + rect.get_height() * 0.5
                rect.set_width(0.24)
                rect.set_height(0.44)
                rect.set_x(0.05)
                rect.set_y(cy - rect.get_height() * 0.5)
                rect.set_edgecolor('#94a3b8')
                rect.set_linewidth(1.1)
                if idx < len(facecolors) and facecolors[idx] is not None:
                    rect.set_facecolor(facecolors[idx])
            for idx, lbl in enumerate(chk.labels):
                lbl.set_fontsize(label_size)
                lbl.set_x(0.36)
                if idx < len(label_colors) and label_colors[idx] is not None:
                    lbl.set_color(label_colors[idx])
            for idx, line in enumerate(getattr(chk, 'lines', [])):
                rect = chk.rectangles[min(idx // 2, n_rects - 1)]
                x0, y0 = rect.get_x(), rect.get_y()
                x1, y1 = x0 + rect.get_width(), y0 + rect.get_height()
                pad_x = rect.get_width() * 0.25
                pad_y = rect.get_height() * 0.25
                if idx % 2 == 0:
                    xs = [x0 + pad_x, x1 - pad_x]
                    ys = [y0 + pad_y, y1 - pad_y]
                else:
                    xs = [x0 + pad_x, x1 - pad_x]
                    ys = [y1 - pad_y, y0 + pad_y]
                line.set_data(xs, ys)
                line.set_linewidth(1.1)
                line.set_color('#0f172a')
            try:
                chk.ax.set_xlim(0, 1.05)
                chk.ax.set_ylim(-0.1, len(chk.rectangles) - 0.1)
            except Exception:
                pass
        except Exception:
            pass


    # Botones fijos y secciones del panel
    coords = layout["panel_coords"]
    ax_btnload = fig.add_axes(coords["btnload"])
    btnload = Button(ax_btnload, "Cargar DICOM", color='#2563eb', hovercolor='#1d4ed8')
    btnload.label.set_color('white'); btnload.label.set_fontsize(11); btnload.label.set_fontweight('bold')

    ax_btnhist = fig.add_axes(coords["btnhist"])
    btnhist = Button(ax_btnhist, "Histograma", color='#e0f2fe', hovercolor='#bae6fd')
    btnhist.label.set_color('#1f2937'); btnhist.label.set_fontsize(10)

    ax_section_seg = fig.add_axes(coords["section_seg"]); ax_section_seg.axis('off')
    ax_section_seg.text(0.0, 0.5, 'Parámetros de segmentación', fontsize=10.5, color='#111827', fontweight='bold', va='center')


    ax_section_overlay = fig.add_axes(coords["section_overlay"]); ax_section_overlay.axis('off')
    ax_section_overlay.text(0.0, 0.5, 'Visibilidad de máscaras', fontsize=10, color='#1f2937', fontweight='bold', va='center')

    ax_section_post = fig.add_axes(coords["section_post"]); ax_section_post.axis('off')
    ax_section_post.text(0.0, 0.5, 'Suavizado y refinamiento', fontsize=10, color='#1f2937', fontweight='bold', va='center')

    ax_section_export = fig.add_axes(coords["section_export"]); ax_section_export.axis('off')
    ax_section_export.text(0.0, 0.5, 'Exportar / validar', fontsize=10, color='#1f2937', fontweight='bold', va='center')

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
        aximg.set_facecolor('#111827')
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
        fallback_used = False
        fallback_trigger = None
        try:
            try:
                img, files = read_best_series(new_folder)  # exige CT
            except NoCTSeriesFound as err:
                fallback_used = True
                fallback_trigger = err
                try:
                    img, files = read_best_series(new_folder, require_modality_ct=False)
                except NoSeriesFound as inner_err:
                    setattr(inner_err, "fallback_trigger", err)
                    raise
            hu = to_hu(img, files[0], file_list=files)
            vol_hu = sitk.GetArrayFromImage(hu)
            state["hu"], state["vol_hu"] = hu, vol_hu
            state["vmin"], state["vmax"] = [float(x) for x in np.percentile(vol_hu, (5, 99.5))]
            state["current_z"] = vol_hu.shape[0]//2

            # Actualizar título con la carpeta cargada
            base_name = os.path.basename(new_folder)
            state["current_folder"] = base_name
            try:
                fig.suptitle(f'Segmentación de rodilla – {base_name}', fontsize=13, fontweight='bold')
            except Exception:
                pass


            state["axes"]["img"].set_position([0.06, 0.29, 0.66, 0.57])


            coords = state["layout"]["panel_coords"]
            if not state.get("ui_initialized"):
                # ---- Controles del panel (creados una sola vez) ----
                ax_pmh_txt = fig.add_axes(coords["pmh_txt"])
                tb_pmh = TextBox(ax_pmh_txt, "Patient HU >", initial=str(int(state.get("patient_hu_thr", 150))))
                try:
                    tb_pmh.label.set_color('#1f2937'); tb_pmh.label.set_fontsize(9.5)
                    tb_pmh.text_disp.set_fontsize(10)
                except Exception:
                    pass

                ax_pmh_apply = fig.add_axes(coords["pmh_apply"])
                btn_pmh_apply = Button(ax_pmh_apply, "Aplicar HU", color='#dcfce7', hovercolor='#bbf7d0')
                _style_button(btn_pmh_apply, text_color='#166534', size=10)


                ax_ovl_bg = fig.add_axes(coords["ovl"])
                ax_ovl_bg.add_patch(Rectangle((0, 0), 1, 1, transform=ax_ovl_bg.transAxes,
                                               facecolor='#f8fafc', edgecolor='#d0d7de', linewidth=1.0))
                ax_ovl_bg.axis('off')
                ax_ovl = fig.add_axes(_inset_axes(coords["ovl"], pad_x=0.022, pad_y=0.016))
                chk_ovl = CheckButtons(ax_ovl, ["Ver Bone", "Ver Patient"],
                                      [state["overlay"]["bone"], state["overlay"]["patient"]])
                ax_ovl.set_facecolor('none')
                _style_checkbuttons(chk_ovl,
                                    label_colors=['#b91c1c', '#047857'],
                                    facecolors=['#fee2e2', '#dcfce7'])

                ax_chk_bg = fig.add_axes(coords["chk_second"])
                ax_chk_bg.add_patch(Rectangle((0, 0), 1, 1, transform=ax_chk_bg.transAxes,
                                               facecolor='#f8fafc', edgecolor='#d0d7de', linewidth=1.0))
                ax_chk_bg.axis('off')
                ax_chk = fig.add_axes(_inset_axes(coords["chk_second"], pad_x=0.022, pad_y=0.014))
                chk_2nd = CheckButtons(ax_chk, ["2º rango (T3–T4)"], [True])
                ax_chk.set_facecolor('none')
                _style_checkbuttons(chk_2nd, label_colors=['#1f2937'], facecolors=['#e2e8f0'])


                ax_sit = fig.add_axes(coords["s_sit"])
                s_sit = Slider(ax_sit, "Suavizado (iters)", 0, 100, valinit=50, valstep=1)
                _style_slider(s_sit, face='#f97316', track='#ffedd5')

                ax_pbd = fig.add_axes(coords["s_pbd"])
                s_pbd = Slider(ax_pbd, "Passband", 0.001, 0.1, valinit=0.01)
                _style_slider(s_pbd, face='#0ea5e9', track='#dbeafe')

                ax_dec = fig.add_axes(coords["s_dec"])
                s_dec = Slider(ax_dec, "Decimation %", 0.0, 90.0, valinit=0.0)
                _style_slider(s_dec, face='#22c55e', track='#dcfce7')

                ax_btn3d = fig.add_axes(coords["btn3d"])
                btn3d = Button(ax_btn3d, "Visualizar 3D", color='#ede9fe', hovercolor='#ddd6fe')
                _style_button(btn3d, text_color='#4c1d95', size=10)

                ax_btnstl = fig.add_axes(coords["btnstl"])
                btnstl = Button(ax_btnstl, "Guardar STL", color='#fef3c7', hovercolor='#fde68a')
                _style_button(btnstl, text_color='#92400e', size=10)

                ax_btnval = fig.add_axes(coords["btnval"])
                btnval = Button(ax_btnval, "Validar (sin GT)", color='#e2e8f0', hovercolor='#cbd5e1')
                _style_button(btnval, text_color='#111827', size=10)

                slider_axes = {}
                for key, pos in state["layout"]["slider_coords"].items():
                    slider_axes[key] = fig.add_axes(pos)
                state["layout"]["slider_axes"] = slider_axes

                state["widgets"].update({
                    "tb_pmh": tb_pmh, "btn_pmh_apply": btn_pmh_apply,
                    "chk_ovl": chk_ovl, "chk_2nd": chk_2nd,
                    "s_sit": s_sit, "s_pbd": s_pbd, "s_dec": s_dec,
                    "btn3d": btn3d, "btnstl": btnstl, "btnval": btnval,
                })

                btn_pmh_apply.on_clicked(on_apply_patient_hu)
                chk_ovl.on_clicked(on_overlay_clicked)
                chk_2nd.on_clicked(lambda _evt: recompute_all())
                btn3d.on_clicked(on_3d)
                btnstl.on_clicked(on_stl)
                btnval.on_clicked(on_validate)
                s_sit.on_changed(lambda _v: (_format_slider_labels(), recompute_all()))
                s_pbd.on_changed(lambda _v: (_format_slider_labels(), recompute_all()))
                s_dec.on_changed(lambda _v: (_format_slider_labels(), recompute_all()))

                state["ui_initialized"] = True
            else:
                # Actualizar controles existentes
                tb_pmh = state["widgets"].get("tb_pmh")
                if tb_pmh is not None:
                    try:
                        tb_pmh.set_val(str(int(state.get("patient_hu_thr", 150))))
                    except Exception:
                        pass
                chk_ovl = state["widgets"].get("chk_ovl")
                if chk_ovl is not None:
                    desired = [True, True]
                    for idx, want in enumerate(desired):
                        cur = chk_ovl.get_status()[idx]
                        if cur != want:
                            chk_ovl.set_active(idx)
                chk_2nd = state["widgets"].get("chk_2nd")
                if chk_2nd is not None:
                    if not chk_2nd.get_status()[0]:
                        chk_2nd.set_active(0)

            slider_axes = state["layout"].get("slider_axes", {})
            if not slider_axes:
                slider_axes = {}
                for key, pos in state["layout"]["slider_coords"].items():
                    slider_axes[key] = fig.add_axes(pos)
                state["layout"]["slider_axes"] = slider_axes

            for key in ("s_z", "s_T23", "s_T14"):
                widget = state["widgets"].pop(key, None)
                if widget is not None:
                    try:
                        widget.disconnect_events()
                    except Exception:
                        pass
                    try:
                        widget.ax.cla()
                    except Exception:
                        pass

            s_z = Slider(slider_axes["s_z"], "Slice (Z)", 0, max(vol_hu.shape[0]-1, 0),
                         valinit=float(state["current_z"]), valstep=1, color='#0ea5e9')
            s_T23 = RangeSlider(slider_axes["s_T23"], "T2–T3 (HU)", -500, 3000, valinit=(300, 1200), facecolor='#22c55e')
            s_T14 = RangeSlider(slider_axes["s_T14"], "T1–T4 (HU)", -500, 3000, valinit=(150, 2000), facecolor='#16a34a')

            for _s in (state["widgets"].get("s_sit"), state["widgets"].get("s_pbd"),
                       state["widgets"].get("s_dec"), s_z, s_T23, s_T14):
                if _s is None:
                    continue
                try:
                    _s.valtext.set_fontsize(9)
                except Exception:
                    pass


            _style_slider(s_z, face='#0ea5e9', track='#bae6fd')
            _style_slider(s_T23, face='#22c55e', track='#dcfce7')
            _style_slider(s_T14, face='#16a34a', track='#d1fae5')

            state["widgets"].update({"s_z": s_z, "s_T23": s_T23, "s_T14": s_T14})


            _format_slider_labels()

            s_T23.on_changed(lambda _v: (_format_slider_labels(), recompute_all()))
            s_T14.on_changed(lambda _v: (_format_slider_labels(), recompute_all()))
            s_z.on_changed(lambda v: (_format_slider_labels(), on_slice_change(v)))
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

            modality_loaded, _sop_uid = _get_dicom_tags(files[0])
            modality_label = modality_loaded if modality_loaded else "Desconocida"
            if fallback_used:
                print(f"[LOAD] Serie DICOM cargada en fallback (modalidad={modality_label}): {new_folder}")
            else:
                print(f"[LOAD] Serie DICOM cargada (modalidad={modality_label}): {new_folder}")
        except NoSeriesFound as e:
            _close_loading_popup(loading_win)
            fallback_trigger = getattr(e, "fallback_trigger", fallback_trigger)
            if isinstance(e, NoCTSeriesFound) or (fallback_trigger is not None and getattr(e, "code", "") != "no_series"):
                mods = getattr(fallback_trigger, "found_modalities", None) or getattr(e, "found_modalities", None)
                if mods:
                    mods_text = ", ".join(mods)
                    _show_info_message(f"Se detectaron series DICOM pero ninguna es TC (modalidades: {mods_text}).", title="Aviso")
                    print(f"[LOAD] No se encontraron series CT en {new_folder}. Modalidades: {mods_text}")
                else:
                    _show_info_message("Se detectaron series DICOM pero ninguna es TC.", title="Aviso")
                    print(f"[LOAD] No se encontraron series CT en {new_folder}. Modalidades: desconocidas")
            elif getattr(e, "code", "") == "no_series":
                _show_info_message("La carpeta seleccionada no contiene archivos DICOM válidos.", title="Aviso")
                print(f"[LOAD] No se encontraron series DICOM en {new_folder}")
            else:
                _show_info_message("No se encontró una serie DICOM compatible en la carpeta seleccionada.", title="Aviso")
                print(f"[LOAD] No se encontró serie compatible en {new_folder}: {e}")
            return
        except Exception as e:
            # Cerrar el popup de carga ANTES de mostrar el info
            _close_loading_popup(loading_win)
            msg = str(e)
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