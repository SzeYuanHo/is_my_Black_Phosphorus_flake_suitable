# streamlit_app.py
import os
import io
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="BP Flake Classifier", page_icon="🔬", layout="centered")
st.title("BP Flake Classifier")
st.caption("Upload an image; the app will draw bounding boxes and labels using YOLOv11.")

# ---------- Weights resolution ----------
# Default local path (commit your weights as models/best.pt) or set env var LOCAL_WEIGHTS
DEFAULT_LOCAL_WEIGHTS = "models/yolov11s_best.pt"

def resolve_weights_path() -> str:
    # 1) Environment variable override (useful in containers)
    env_path = os.getenv("LOCAL_WEIGHTS")
    if env_path and os.path.exists(env_path):
        return env_path

    # 2) Default local file
    if os.path.exists(DEFAULT_LOCAL_WEIGHTS):
        return DEFAULT_LOCAL_WEIGHTS

    # 3) If neither exists, show a friendly error in the UI
    st.error(
        "Model weights not found. "
        "Place your weights at **models/best.pt** or set the **LOCAL_WEIGHTS** env var to a valid path."
    )
    st.stop()

# ---------- Load model once ----------
@st.cache_resource(show_spinner=True)
def load_model(weights_path: str):
    return YOLO(weights_path)

weights_path = resolve_weights_path()
model = load_model(weights_path)

# ---------- Sidebar ----------
st.sidebar.header("Inference Settings")
conf = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.2, 0.01)
iou = st.sidebar.slider("NMS IoU threshold", 0.1, 1.0, 0.2, 0.01)
# imgsz = st.sidebar.selectbox("Inference size (imgsz)", [384, 512, 640, 960], index=2)
show_labels = st.sidebar.checkbox("Show labels", value=True)
show_conf = st.sidebar.checkbox("Show confidences", value=True)
st.sidebar.markdown("---")
st.sidebar.write("**Weights:**", os.path.abspath(weights_path))

# ---------- Helpers ----------
def _auto_orient_rgb(pil_img: Image.Image) -> Image.Image:
    pil_img = ImageOps.exif_transpose(pil_img)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return pil_img

def _to_download_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", compress_level=6)
    buf.seek(0)
    return buf.getvalue()

# ---------- UI ----------
uploaded = st.file_uploader("Upload an image (PNG/JPG/WebP/BMP)", type=["png", "jpg", "jpeg", "webp", "bmp"])

if uploaded is not None:
    img = Image.open(uploaded)
    img = _auto_orient_rgb(img)
    st.image(img, caption="Original", use_container_width=True)

    with st.spinner("Running YOLO inference…"):
        results = model.predict(
            source=np.array(img),
            conf=conf,
            iou=iou,
            # imgsz=imgsz,
            device="cpu",
            verbose=False,
            show=False,
        )

    res = results[0]

    plotted_bgr = res.plot(boxes=True, labels=show_labels, conf=show_conf)
    plotted_rgb = Image.fromarray(plotted_bgr[:, :, ::-1])

    st.image(plotted_rgb, caption="Detections", use_container_width=True)

    # Class counts with ordered output and confidence-sorted coordinates
    if res.boxes is not None and len(res.boxes) > 0:
        names = res.names if hasattr(res, "names") else getattr(model, "names", {})
        boxes = res.boxes.xywh.cpu().numpy()          
        classes = res.boxes.cls.cpu().numpy().astype(int) 
        confs = res.boxes.conf.cpu().numpy() 

        CLASS_ORDER = ["suitable", "further_review"]
        
        class_coords = {}
        for cls, xywh, conf in zip(classes, boxes, confs):
            label = names.get(cls, str(cls))
            x, y = int(xywh[0]), int(xywh[1])
            if label not in class_coords:
                class_coords[label] = []
            class_coords[label].append((x, y, conf))

        # Sort coordinates within each class by confidence (descending)
        for label in class_coords:
            class_coords[label].sort(key=lambda item: item[2], reverse=True)
            class_coords[label] = [(x, y) for x, y, _ in class_coords[label]]

        if isinstance(img, np.ndarray):
            height, width = img.shape[:2]
        else:
            width, height = img.size

        st.subheader("Detections Summary")
        st.markdown(
                    f'<p style="color:grey; font-style:italic; margin-top:0; margin-bottom:1rem;">'
                    f'Image Dimensions: {width} × {height}. Coordinates from top-left corner.'
                    f'</p>',
                    unsafe_allow_html=True
                    )
        
        def sort_key(label):
            if label in CLASS_ORDER:
                return (0, CLASS_ORDER.index(label))
            else:
                return (1, label)

        sorted_labels = sorted(class_coords.keys(), key=sort_key)

        for label in sorted_labels:
            coords_list = class_coords[label]
            count = len(coords_list)
            coord_str = ", ".join([f"({x},{y})" for x, y in coords_list[:5]])
            if len(coords_list) > 5:
                coord_str += ", ..."
            st.write(f"- **{label}**: {count} at {coord_str}")
    else:
        st.write("No objects detected.")

    st.download_button(
        label="⬇️ Download annotated image (PNG)",
        data=_to_download_bytes(plotted_rgb),
        file_name="yolo_annotated.png",
        mime="image/png",
    )
else:
    st.info("Upload an image to get started.")
