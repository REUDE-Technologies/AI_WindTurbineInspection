"""
Wind Turbine Inspection AI Dashboard – Streamlit app.
Drone aerial video inspection → defect categorization → instant report.
"""
import io
import os
import re
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

from data.sample_data import get_sample_defects, get_default_case_data, get_pipeline_stages
from data.local_db import (
    get_blades,
    get_inspection_options,
    add_blade,
    remove_blade,
    set_blades,
    set_number_of_blades,
    get_include_nacelle,
    set_include_nacelle,
)

st.set_page_config(
    page_title="Wind Turbine Inspection AI",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dashboard look
st.markdown("""
<style>
    .kpi-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem 1.25rem;
        border-radius: 10px;
        border-left: 4px solid #0ea5e9;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .kpi-card h3 { margin: 0; font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; }
    .kpi-card .value { font-size: 1.75rem; font-weight: 700; color: #f1f5f9; }
    .pipeline-stage { padding: 0.5rem 0.75rem; border-radius: 6px; margin: 0.25rem 0; font-size: 0.9rem; }
    .pipeline-done { background: #14532d; color: #bbf7d0; }
    .severity-critical { color: #dc2626; font-weight: 700; }
    .severity-high { color: #ea580c; font-weight: 600; }
    .severity-medium { color: #ca8a04; }
    .severity-low { color: #16a34a; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

def _detect_blade_nacelle_roi_and_mask(pil_image):
    """
    Object detection + background removal: find blade/nacelle and a foreground mask.
    Returns (roi, mask). roi = (x1,y1,x2,y2). mask = (H,W) uint8, 255=foreground (blade/nacelle), 0=background.
    Mask is used to remove noise and improve defect prediction.
    """
    img_rgb = np.array(pil_image.convert("RGB"))
    h_img, w_img = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, light = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, 50, 150)
    combined = cv2.bitwise_or(light, edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (w_img * h_img) * 0.015
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    rects = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if 1.2 <= aspect <= 12:
            rects.append((x, y, x + bw, y + bh))
            cv2.drawContours(mask, [c], -1, 255, -1)
    if not rects:
        margin = 0.12
        roi = (
            int(w_img * margin), int(h_img * margin),
            int(w_img * (1 - margin)), int(h_img * (1 - margin)),
        )
        mask[:, :] = 255  # no mask, use full ROI
        return roi, mask
    x1 = max(0, min(r[0] for r in rects) - int(0.02 * w_img))
    y1 = max(0, min(r[1] for r in rects) - int(0.02 * h_img))
    x2 = min(w_img, max(r[2] for r in rects) + int(0.02 * w_img))
    y2 = min(h_img, max(r[3] for r in rects) + int(0.02 * h_img))
    roi = (x1, y1, x2, y2)
    # Clean mask: fill small holes and remove small components
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return roi, mask


def _detect_blade_nacelle_roi(pil_image):
    """Convenience: return only ROI for code that doesn't need the mask."""
    roi, _ = _detect_blade_nacelle_roi_and_mask(pil_image)
    return roi


def _detect_defects_threshold(pil_image, roi, mask=None):
    """
    Detect defects inside the blade/nacelle ROI. If mask is provided (background removal),
    only foreground pixels are used for thresholding to reduce noise and improve prediction.
    """
    x1, y1, x2, y2 = roi
    img_rgb = np.array(pil_image.convert("RGB"))
    full_h, full_w = img_rgb.shape[:2]
    crop = img_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    rw, rh = gray.shape[1], gray.shape[0]
    roi_area = rw * rh
    mask_crop = None
    if mask is not None:
        mask_crop = mask[y1:y2, x1:x2]
        if mask_crop.size == gray.size and np.any(mask_crop > 0):
            fg_mean = float(np.mean(gray[mask_crop > 0]))
            gray = gray.copy()
            gray[mask_crop == 0] = fg_mean  # set background to foreground mean to remove noise
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    defects = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # ----- 1) Dark anomaly detection (only on foreground if mask given) -----
    block = max(11, min(rw, rh) // 8)
    if block % 2 == 0:
        block += 1
    dark = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 4
    )
    if mask_crop is not None:
        dark[mask_crop == 0] = 0  # ignore background
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if mask_crop is not None:
        otsu[mask_crop == 0] = 0
    combined_dark = cv2.bitwise_or(dark, otsu)
    combined_dark = cv2.morphologyEx(combined_dark, cv2.MORPH_OPEN, kernel)
    combined_dark = cv2.morphologyEx(combined_dark, cv2.MORPH_CLOSE, kernel)
    contours_d, _ = cv2.findContours(combined_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_a = roi_area * 0.0003
    max_a = roi_area * 0.15
    for c in contours_d:
        area = cv2.contourArea(c)
        if not (min_a <= area <= max_a):
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw < 5 or bh < 5:
            continue
        if mask_crop is not None:
            roi_slice = mask_crop[by : by + bh, bx : bx + bw]
            if np.sum(roi_slice > 0) < 0.4 * (bw * bh):
                continue  # skip if mostly background
        defects.append((x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh, "Dark anomaly / defect", "#dc2626"))

    # ----- 2) High-intensity detection (thermal-style) -----
    mean_val = np.mean(gray[mask_crop > 0]) if mask_crop is not None and np.any(mask_crop > 0) else np.mean(gray)
    std_val = np.std(gray[mask_crop > 0]) if mask_crop is not None and np.any(mask_crop > 0) else np.std(gray)
    if std_val > 5:
        _, high = cv2.threshold(gray, int(mean_val + 1.2 * std_val), 255, cv2.THRESH_BINARY)
        if mask_crop is not None:
            high[mask_crop == 0] = 0
        high = cv2.morphologyEx(high, cv2.MORPH_OPEN, kernel)
        high = cv2.morphologyEx(high, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        contours_h, _ = cv2.findContours(high, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_h:
            area = cv2.contourArea(c)
            if not (min_a <= area <= max_a):
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw < 5 or bh < 5:
                continue
            if mask_crop is not None:
                roi_slice = mask_crop[by : by + bh, bx : bx + bw]
                if np.sum(roi_slice > 0) < 0.4 * (bw * bh):
                    continue
            defects.append((x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh, "High intensity (thermal)", "#ea580c"))

    # Deduplicate overlapping boxes
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a[0], a[1], a[2], a[3]
        bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-6)
    kept = []
    for d in defects:
        if any(_iou(d, k) > 0.4 for k in kept):
            continue
        kept.append(d)
    return kept


def _apply_background_removal(pil_image, mask, background_color=(40, 44, 52)):
    """Return a copy of the image with background (mask==0) set to a solid color to remove noise."""
    img = np.array(pil_image.convert("RGB"))
    out = img.copy()
    out[mask == 0] = background_color
    return Image.fromarray(out)


def _draw_defect_segmentation_overlay(pil_image, roi=None, mask=None):
    """
    Two-stage with background removal: (1) Detect blade/nacelle ROI + foreground mask (or use provided);
    (2) Detect defects inside ROI using mask to remove noise and improve prediction.
    """
    img = pil_image.convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    if roi is None or mask is None:
        roi, mask = _detect_blade_nacelle_roi_and_mask(pil_image)
    if roi:
        x1, y1, x2, y2 = roi
        draw.rectangle([x1, y1, x2, y2], outline="#0ea5e9", width=max(3, min(w, h) // 150))
        draw.text((x1, max(0, y1 - 14)), "Blade/Nacelle ROI", fill="#0ea5e9")
        rw, rh = x2 - x1, y2 - y1
        if rw > 20 and rh > 20:
            defects = _detect_defects_threshold(pil_image, roi, mask=mask)
            try:
                font = ImageFont.truetype("arial.ttf", max(12, min(w, h) // 35))
            except Exception:
                font = ImageFont.load_default()
            for dx1, dy1, dx2, dy2, label, color in defects:
                draw.rectangle([dx1, dy1, dx2, dy2], outline=color, width=max(2, min(w, h) // 250))
                draw.text((dx1, max(0, dy1 - 16)), label, fill=color, font=font)
            if not defects:
                draw.text((x1 + rw // 2 - 60, y1 + rh // 2 - 10), "No defects detected (threshold)", fill="#64748b", font=font)
    else:
        try:
            font = ImageFont.truetype("arial.ttf", max(14, min(w, h) // 40))
        except Exception:
            font = ImageFont.load_default()
        draw.text((w // 2 - 80, h // 2 - 10), "No blade/nacelle structure detected", fill="#94a3b8", font=font)
    return img


def _load_pil_image(name, source):
    """Load a PIL Image from either a file path (str/Path) or bytes (uploaded file)."""
    if isinstance(source, (str, Path)):
        return Image.open(Path(source)).convert("RGB")
    return Image.open(io.BytesIO(source)).convert("RGB")


# Load data (cached)
@st.cache_data
def load_defects():
    return get_sample_defects()

@st.cache_data
def load_case_data():
    return get_default_case_data()

def severity_order(s):
    order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    return s.map(lambda x: order.get(x, 4))

# Initialize session state for case data edits
if "case_data" not in st.session_state:
    st.session_state.case_data = load_case_data()

df = load_defects()

# ----- SIDEBAR: Filters + Case data -----
with st.sidebar:
    st.title("🌀 Inspection")
    st.caption("AI Wind Turbine Inspection Dashboard")

    # Quick filters
    st.subheader("Filters")
    run_options = ["REU_WT_IR_001_2026", "REU_WT_IR_002_2026"]
    selected_run = st.selectbox("Inspection run", run_options, index=0)
    site = st.text_input("Site / WTG", value="WTTS Kayathar | Vestas RRB 225 kW", key="site")
    date_range = st.date_input("Field work dates", value=(date(2025, 12, 29), date(2025, 12, 30)), key="dates")

    st.divider()
    st.subheader("Case data (for report)")
    cd = st.session_state.case_data
    cd["report_number"] = st.text_input("Report number", value=cd.get("report_number", ""), key="report_no")
    cd["location"] = st.text_input("Location", value=cd.get("location", ""), key="location")
    cd["wtg_type"] = st.text_input("WTG type", value=cd.get("wtg_type", ""), key="wtg_type")
    cd["blade_serials"] = st.text_input("Blade serials", value=cd.get("blade_serials", ""), key="blade_serials")
    cd["operator"] = st.text_input("UAV operator", value=cd.get("operator", ""), key="operator")
    cd["reviewer"] = st.text_input("Report reviewer", value=cd.get("reviewer", ""), key="reviewer")
    cd["verifier"] = st.text_input("Report verifier", value=cd.get("verifier", ""), key="verifier")
    cd["drone_model"] = st.text_input("Drone model", value=cd.get("drone_model", ""), key="drone")
    cd["weather"] = st.text_input("Weather", value=cd.get("weather", ""), key="weather")
    cd["wind_ms"] = st.text_input("Wind (m/s)", value=cd.get("wind_ms", ""), key="wind")
    cd["temp_c"] = st.text_input("Temperature (°C)", value=cd.get("temp_c", ""), key="temp")
    st.session_state.case_data = cd

    st.divider()
    st.subheader("Blade configuration")
    n_blades = st.number_input("Number of blades", min_value=1, max_value=20, value=len(get_blades()), key="n_blades")
    if st.button("Apply number of blades", key="apply_n_blades"):
        set_number_of_blades(n_blades)
        st.rerun()
    blades_list = get_blades()
    edited = []
    for i, b in enumerate(blades_list):
        col_a, col_b = st.columns([2, 1])
        with col_a:
            new_name = st.text_input(f"Blade {i+1} name", value=b.get("name") or f"Blade {b.get('id','')}", key=f"blade_name_{b.get('id')}")
        with col_b:
            if st.button("🗑", key=f"del_blade_{b.get('id')}"):
                remove_blade(b.get("id"))
                st.rerun()
        edited.append({"id": b.get("id", str(i)), "name": new_name})
    if st.button("Save blade names", key="save_blades"):
        set_blades(edited, include_nacelle=get_include_nacelle())
        st.rerun()
    include_nacelle = st.checkbox("Include Nacelle in inspection", value=get_include_nacelle(), key="inc_nacelle")
    if include_nacelle != get_include_nacelle():
        set_include_nacelle(include_nacelle)
        st.rerun()
    new_id = st.text_input("Add new blade (ID)", placeholder="e.g. 380", key="new_blade_id")
    new_name = st.text_input("Add new blade (display name)", placeholder="e.g. Blade 380", key="new_blade_name")
    if st.button("➕ Append blade to list", key="append_blade") and new_id.strip():
        add_blade(new_id.strip(), new_name.strip() or None)
        st.rerun()

# ----- MAIN: Title + Quick actions -----
st.title("Wind Turbine Inspection AI Dashboard")
st.caption("Drone-based inspection → defect detection & classification → instant report")

# ----- Inspection input: dropdown + upload + ML -----
st.subheader("Inspection input")
inspection_options = get_inspection_options()
if not inspection_options:
    inspection_options = ["Blade 361", "Blade 363", "Blade 379", "Nacelle"]
selected_input = st.selectbox(
    "Select component for inspection (input images from same folder)",
    options=inspection_options,
    index=0,
    key="inspection_input",
    help="Choose Blade 361, Blade 363, Blade 379, or Nacelle to load images and run ML segmentation & defect detection.",
)

# Option 1: Folder path – list images and select from folder (for local run)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
folder_path = st.text_input(
    "Inspection folder path (optional)",
    value="",
    placeholder=r"e.g. C:\Users\You\inspection_images or ./inspection_images",
    key="inspection_folder",
    help="Enter the path to the folder containing inspection images. Images from this folder can be selected below.",
)
folder_images = []
selected_from_folder = []
if folder_path and folder_path.strip():
    p = Path(folder_path.strip())
    if p.is_dir():
        folder_images = sorted([f.name for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS])
        if folder_images:
            selected_from_folder = st.multiselect(
                "Select images from folder for inspection",
                options=folder_images,
                default=folder_images,
                key="selected_folder_images",
                help="Choose which images in the folder to use for ML inspection.",
            )
        else:
            st.caption("No image files (e.g. .jpg, .png) found in this folder.")
    else:
        st.caption("Folder not found. Enter a valid path or use the file upload below.")

# Option 2: Upload files (drag & drop or browse)
uploaded_files = st.file_uploader(
    "Or upload images from folder",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
    accept_multiple_files=True,
    key="inspection_images",
    help="Alternatively, drag and drop or browse to select images from your inspection folder.",
)

if st.button("🔬 Run ML inspection (segment & identify defects)", key="run_ml", use_container_width=True):
    n_images = len(selected_from_folder) if selected_from_folder else 0
    image_sources = []  # list of (display_name, source: path or bytes)
    if selected_from_folder and folder_path and folder_path.strip():
        p = Path(folder_path.strip())
        if p.is_dir():
            for fname in selected_from_folder:
                image_sources.append((fname, p / fname))
    if not image_sources and uploaded_files:
        for uf in uploaded_files:
            image_sources.append((uf.name, uf.read()))
    if image_sources:
        st.success(f"Running segmentation and defect detection for **{selected_input}** on {len(image_sources)} image(s). Results appear below.")
        results = []
        for name, source in image_sources:
            try:
                img = _load_pil_image(name, source)
                roi, mask = _detect_blade_nacelle_roi_and_mask(img)
                img_no_bg = _apply_background_removal(img, mask)
                img_overlay = _draw_defect_segmentation_overlay(img.copy(), roi=roi, mask=mask)
                results.append((name, img, img_no_bg, img_overlay))
            except Exception as e:
                st.error(f"Could not process {name}: {e}")
        st.session_state.ml_results = results
        st.session_state.ml_component = selected_input
    else:
        st.warning("Select at least one image from the folder above or upload files, then click Run ML inspection.")
        if "ml_results" in st.session_state:
            del st.session_state.ml_results

# Show defect segmentation results (from last Run ML)
if st.session_state.get("ml_results"):
    st.subheader("Defect segmentation results")
    st.caption(
        f"Component: **{st.session_state.get('ml_component', '')}** — "
        "**Background removal** reduces noise; defect detection runs on blade/nacelle only (threshold + high-intensity). "
        "Use thermal imagery for real thermal hot-spot detection."
    )
    for item in st.session_state.ml_results:
        if len(item) == 4:
            name, img_orig, img_no_bg, img_seg = item
        else:
            name, img_orig, img_seg = item[0], item[1], item[2]
            img_no_bg = None
        with st.expander(f"📷 {name}", expanded=True):
            if img_no_bg is not None:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(img_orig, caption="Original", use_container_width=True)
                with c2:
                    st.image(img_no_bg, caption="Background removed (noise removed)", use_container_width=True)
                with c3:
                    st.image(img_seg, caption="Defect segmentation", use_container_width=True)
            else:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(img_orig, caption="Original", use_container_width=True)
                with c2:
                    st.image(img_seg, caption="Defect segmentation", use_container_width=True)

st.divider()

col_act1, col_act2, col_act3, _ = st.columns([1, 1, 1, 3])
with col_act1:
    if st.button("📤 New inspection", use_container_width=True):
        st.info("Use the inspection input dropdown above to select component and upload images.")
with col_act2:
    if st.button("📄 Generate report", use_container_width=True, key="btn_gen_report"):
        st.session_state.show_report = True
with col_act3:
    st.download_button("📥 Export data (CSV)", data=df.to_csv(index=False).encode("utf-8"), file_name="defects_export.csv", mime="text/csv", use_container_width=True, key="export_csv")

# ----- KPIs -----
st.subheader("Key metrics")
sev_counts = df["severity"].value_counts().reindex(["Critical", "High", "Medium", "Low"], fill_value=0)
c_nacelle = (df["component"] == "Nacelle").sum()
c_blades = (df["component"] == "Blade").sum()
total_views = 6 + 6  # 6 nacelle + 6 blade sides
covered = len(df["view_zone"].dropna().unique()) + df[df["component"] == "Blade"].groupby(["blade_id", "side"]).ngroups
coverage_pct = min(100, round(covered / total_views * 100)) if total_views else 0
conf_avg = round(df["confidence"].mean() * 100)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total defects", len(df))
k2.metric("Critical / High", int(sev_counts.get("Critical", 0) + sev_counts.get("High", 0)), help="Immediate action count")
k3.metric("Nacelle / Blades", f"{c_nacelle} / {c_blades}")
k4.metric("Coverage", f"{coverage_pct}%", help="Planned views/segments completed")
k5.metric("AI confidence (avg)", f"{conf_avg}%")

# ----- Pipeline status -----
st.subheader("Inspection pipeline")
stages = get_pipeline_stages()
pipe_cols = st.columns(len(stages))
for i, stage in enumerate(stages):
    with pipe_cols[i]:
        status = stage["status"]
        st.markdown(f"**{i+1}. {stage['name']}**")
        st.caption("✅ Done" if status == "done" else "⏳ " + status)
        if stage.get("eta"):
            st.caption(stage["eta"])

# ----- Tabs: Charts | Defect table | Visual explorer | Report -----
tab_charts, tab_table, tab_explorer, tab_report = st.tabs(["📊 Defect summary", "📋 Defect list", "🔍 Visual explorer", "📄 Report"])

with tab_charts:
    st.subheader("Defects by severity")
    fig_sev = px.pie(values=sev_counts.values, names=sev_counts.index, color_discrete_sequence=px.colors.qualitative.Set2)
    fig_sev.update_layout(height=320, margin=dict(t=20, b=20))
    st.plotly_chart(fig_sev, use_container_width=True)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("By defect type")
        type_counts = df["defect_type"].value_counts().head(10)
        fig_type = px.bar(x=type_counts.values, y=type_counts.index, orientation="h", labels={"x": "Count", "y": "Type"})
        fig_type.update_layout(height=360, margin=dict(t=20, b=20), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_type, use_container_width=True)
    with col_c2:
        st.subheader("By component / view")
        df["comp_view"] = df.apply(lambda r: f"Nacelle-{r['view_zone']}" if r["component"] == "Nacelle" else f"Blade {r['blade_id']} {r['side']}", axis=1)
        comp_counts = df["comp_view"].value_counts()
        fig_comp = px.bar(x=comp_counts.index, y=comp_counts.values, labels={"x": "Component", "y": "Count"})
        fig_comp.update_layout(height=360, margin=dict(t=20, b=20), xaxis_tickangle=-45)
        st.plotly_chart(fig_comp, use_container_width=True)

    st.subheader("Blade zones (blades only)")
    blade_df = df[df["component"] == "Blade"].copy()
    if not blade_df.empty:
        zone_counts = blade_df["zone"].value_counts()
        fig_zone = px.bar(x=zone_counts.index, y=zone_counts.values, labels={"x": "Zone", "y": "Count"})
        fig_zone.update_layout(height=300, margin=dict(t=20, b=20), xaxis_tickangle=-30)
        st.plotly_chart(fig_zone, use_container_width=True)
    else:
        st.info("No blade defects in current data.")

with tab_table:
    st.subheader("Defect list")
    f_sev = st.multiselect("Filter by severity", options=df["severity"].unique().tolist(), default=df["severity"].unique().tolist(), key="f_sev")
    f_comp = st.multiselect("Filter by component", options=df["component"].unique().tolist(), default=df["component"].unique().tolist(), key="f_comp")
    f_type = st.multiselect("Filter by defect type", options=df["defect_type"].unique().tolist(), default=df["defect_type"].unique().tolist(), key="f_type")

    df_f = df[df["severity"].isin(f_sev) & df["component"].isin(f_comp) & df["defect_type"].isin(f_type)].copy()
    df_f["_sev_order"] = severity_order(df_f["severity"])
    df_f = df_f.sort_values(["_sev_order", "component", "view_zone", "blade_id", "side"]).drop(columns=["_sev_order"])

    display_cols = ["defect_id", "component", "view_zone", "blade_id", "side", "zone", "condition", "defect_type", "severity", "recommendation", "confidence"]
    display_cols = [c for c in display_cols if c in df_f.columns]
    st.dataframe(df_f[display_cols], use_container_width=True, height=400)

    with st.expander("Defect detail (select by ID)"):
        sel_id = st.selectbox("Defect ID", options=df_f["defect_id"].tolist(), key="sel_id")
        row = df_f[df_f["defect_id"] == sel_id].iloc[0]
        st.json(row.to_dict())

with tab_explorer:
    st.subheader("Visual explorer")
    comp_choice = st.radio("Component", ["Nacelle", "Blade"], horizontal=True, key="expl_comp")

    if comp_choice == "Nacelle":
        view_choice = st.selectbox("View", ["Top", "Bottom", "Back", "Left", "Right", "Front"], key="nacelle_view")
        sub = df[(df["component"] == "Nacelle") & (df["view_zone"] == view_choice)]
        if sub.empty:
            st.info(f"No defects for Nacelle – {view_choice} view.")
        else:
            st.markdown(f"**Defects in Nacelle {view_choice}:** {len(sub)}")
            for _, r in sub.iterrows():
                with st.expander(f"{r['defect_id']}: {r['condition'][:50]}... — **{r['severity']}**"):
                    st.write("**Type:**", r["defect_type"])
                    st.write("**Recommendation:**", r["recommendation"])
                    st.caption(f"Confidence: {r['confidence']*100:.0f}%")
    else:
        blade_ids = [b.get("id") for b in get_blades()]
        blade_choice = st.selectbox("Blade", blade_ids if blade_ids else ["361", "363", "379"], key="blade_choice")
        side_choice = st.selectbox("Side", ["SS", "PS"], key="side_choice")
        sub = df[(df["component"] == "Blade") & (df["blade_id"] == blade_choice) & (df["side"] == side_choice)]
        if sub.empty:
            st.info(f"No defects for Blade {blade_choice} {side_choice}.")
        else:
            st.markdown(f"**Blade {blade_choice} {side_choice}:** {len(sub)} defects")
            for _, r in sub.iterrows():
                with st.expander(f"{r['defect_id']} | {r['zone']} — **{r['severity']}**"):
                    st.write("**Condition:**", r["condition"])
                    st.write("**Type:**", r["defect_type"])
                    st.write("**Recommendation:**", r["recommendation"])
                    st.caption(f"Confidence: {r['confidence']*100:.0f}%")

with tab_report:
    st.subheader("Instant report")
    cd = st.session_state.case_data
    if st.button("Generate report preview", key="gen_prev"):
        st.session_state.show_report = True
    if st.session_state.get("show_report", False):
        # Build HTML report
        severity_table = """| Severity | Description | Recommended action |
|----------|-------------|---------------------|
| Critical | Immediate risk to life or asset | Stop WTG right away |
| High | Potential risk if not remedied | Significant defect – immediate action |
| Medium | Recommend attention | Take action / monitor |
| Low | Minor non-conformance | Monitor |"""

        nacelle_views = df[df["component"] == "Nacelle"]["view_zone"].unique().tolist()
        nacelle_sections = []
        for v in nacelle_views:
            sub = df[(df["component"] == "Nacelle") & (df["view_zone"] == v)]
            rows = "\n".join([f"| {r['defect_id']} | {r['condition'][:40]}... | {r['defect_type']} | {r['severity']} | {r['recommendation']} |" for _, r in sub.iterrows()])
            nacelle_sections.append(f"**{v} view** ({len(sub)} defects)\n\n| ID | Condition | Type | Class | Recommendation |\n|----|--------|------|-------|----------------|\n{rows}")

        blades = df[df["component"] == "Blade"].groupby(["blade_id", "side"])
        blade_sections = []
        for (bid, side), grp in blades:
            rows = "\n".join([f"| {r['defect_id']} | {r['zone']} | {r['condition'][:35]}... | {r['defect_type']} | {r['severity']} | {r['recommendation']} |" for _, r in grp.iterrows()])
            blade_sections.append(f"**Blade {bid} {side}** ({len(grp)} defects)\n\n| ID | Zone | Condition | Type | Class | Recommendation |\n|----|------|--------|------|-------|----------------|\n{rows}")

        report_md = f"""
# WIND TURBINE INSPECTION REPORT

**Report number:** {cd.get('report_number', '')}  
**Report date:** {cd.get('report_date', '')}  
**Location:** {cd.get('location', '')}  
**WTG:** {cd.get('wtg_type', '')}  
**Blade serials:** {cd.get('blade_serials', '')}

---

## 1. Defect classification

{severity_table}

---

## 2. Introduction – Case data

- **Location:** {cd.get('location', '')}
- **WTG type:** {cd.get('wtg_type', '')}
- **Blade serials:** {cd.get('blade_serials', '')}
- **UAV operator:** {cd.get('operator', '')}
- **Report reviewer:** {cd.get('reviewer', '')}
- **Report verifier:** {cd.get('verifier', '')}
- **Drone:** {cd.get('drone_model', '')}
- **Weather:** {cd.get('weather', '')}, Wind {cd.get('wind_ms', '')} m/s, Temp {cd.get('temp_c', '')} °C

---

## 3. Nacelle inspection

""" + "\n\n".join(nacelle_sections) + """

---

## 4. Blade inspection

""" + "\n\n".join(blade_sections) + """

---

## 5. Conclusion

Overall condition is **serviceable** based on drone imagery. No confirmed major cracks or structural failures. Most findings are surface contamination and coating wear. Continue operation with recommended cleaning, re-imaging, and localized maintenance within the next maintenance window.

## 6. Recommendation

- Perform verification re-flight (higher zoom) for priority areas with linear marks and fastener staining.
- Apply clean-and-reinspect to confirm true defect presence.
- Conduct localized coating touch-up where peel-off or substrate exposure is suspected.
- Maintain standardized periodic drone inspection for trend tracking.
"""

        st.markdown(report_md)

        # Download as HTML (simple) or Markdown
        html_body = re.sub(r"\n## (.+)\n", r"\n<h2>\1</h2>\n", report_md)
        html_body = html_body.replace("\n---\n", "\n<hr>\n").replace("\n", "<br>\n")
        html_body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html_body)
        html_content = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Inspection Report - {cd.get('report_number', '')}</title>
<style>body{{ font-family: system-ui; max-width: 800px; margin: 2rem auto; padding: 1rem; }} table {{ border-collapse: collapse; width: 100%; }} th, td {{ border: 1px solid #ddd; padding: 8px; }} th {{ background: #1a1a2e; color: #f1f5f9; }}</style></head><body>{html_body}</body></html>"""
        st.download_button("📥 Download report (HTML)", data=html_content.encode("utf-8"), file_name=f"InspectionReport_{cd.get('report_number', 'report')}.html", mime="text/html", key="dl_html")
        st.download_button("📥 Download report (Markdown)", data=report_md.encode("utf-8"), file_name=f"InspectionReport_{cd.get('report_number', 'report')}.md", mime="text/markdown", key="dl_md")
