# streamlit_app.py
import os
import io
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="BP Flake Classifier", page_icon="🔬", layout="centered")
st.title("BP Flake Classifier")
st.caption("Upload an image; the app will draw bounding boxes and labels using your YOLO model.")

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
            device="cpu",   # Codespaces typically has no CUDA
            verbose=False,
            show=False,
        )

    res = results[0]

    # Ultralytics .plot() returns BGR ndarray -> convert to RGB for PIL/Streamlit
    plotted_bgr = res.plot(boxes=True, labels=show_labels, conf=show_conf)
    plotted_rgb = Image.fromarray(plotted_bgr[:, :, ::-1])

    st.image(plotted_rgb, caption="Detections", use_container_width=True)

    # Class counts
    if res.boxes is not None and len(res.boxes) > 0:
        names = res.names if hasattr(res, "names") else getattr(model, "names", {})
        classes = res.boxes.cls.cpu().numpy().astype(int).tolist()
        unique, counts = np.unique(classes, return_counts=True)
        st.subheader("Detections Summary")
        for c, n in zip(unique, counts):
            label = names.get(c, str(c)) if isinstance(names, dict) else str(c)
            st.write(f"- **{label}**: {int(n)}")

    st.download_button(
        label="⬇️ Download annotated image (PNG)",
        data=_to_download_bytes(plotted_rgb),
        file_name="yolo_annotated.png",
        mime="image/png",
    )
else:
    st.info("Upload an image to get started.")
