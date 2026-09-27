import hashlib
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch

from pneumolens.gradcam import grad_cam
from pneumolens.imaging import overlay, png_bytes, preprocess, read_image, unpad_cam
from pneumolens.model import CLASSES, load_checkpoint

st.set_page_config(page_title="PneumoLens · X-ray Explorer", page_icon="◉", layout="wide")
st.markdown("<style>.block-container{padding-top:4rem;padding-bottom:2rem}div[data-testid='stMetricValue']{font-size:1.65rem}</style>", unsafe_allow_html=True)
torch.set_num_threads(4)
ROOT = Path(__file__).parent


@st.cache_data(show_spinner=False)
def checkpoint_digest(path, modified_ns):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

st.caption("PNEUMOLENS / APPLIED COMPUTER VISION")
st.title("Look beyond the prediction.")
st.write("Explore a chest X-ray, inspect the model’s classification, and see which regions influence its class score.")
st.info("Educational research demo. Grad-CAM shows model influence, not a diagnosis or a disease boundary.")

with st.sidebar:
    st.header("Study controls")
    checkpoint_path = ROOT / "artifacts/baseline/model.pt"
    st.caption("ResNet-18 · two dataset classes")
    target = st.selectbox("Explain class", CLASSES, index=1,
                          help="The overlay explains this class even when the model predicts the other class.")
    opacity = st.slider("Overlay opacity", 0.0, 1.0, 0.55, 0.05)
    st.divider()
    st.markdown("**How to read the overlay**\n\nAmber/red highlights stronger positive influence on the selected class score. Transparent areas have low or no positive activation.")
    st.caption("Each heatmap is normalized separately. Color strength cannot be compared as confidence across images or classes.")

explore, method, results = st.tabs(["X-ray explorer", "Method & limitations", "Evaluation"])
with explore:
    payload = None
    examples_path = ROOT / "docs/examples.json"
    examples = json.loads(examples_path.read_text()) if examples_path.exists() else []
    source_col, choice_col = st.columns([1, 2])
    source = source_col.radio("Image source", ["Dataset example", "Upload image"], horizontal=True) if examples else "Upload image"
    if source == "Dataset example":
        selected = choice_col.selectbox("Choose a held-out example", range(len(examples)),
                                format_func=lambda index: examples[index]["name"])
        example = examples[selected]
        payload = (ROOT / example["image"]).read_bytes()
        st.caption(f"Dataset label: {example['label']} · {example['original_filename']}")
    else:
        upload = st.file_uploader("Upload a de-identified chest X-ray", type=["jpg", "jpeg", "png"],
                                  help="8-bit JPEG/PNG, at most 10 MB and 20 megapixels. DICOM is not supported.")
        if upload is not None:
            payload = upload.getvalue()
        st.caption("Uploads are processed in this Python app’s memory and are not saved. A hosted deployment would send uploads to its server.")
    if payload is not None:
        if len(payload) > 10 * 1024 * 1024:
            st.error("Please choose an image smaller than 10 MB.")
        else:
            try:
                image = read_image(io.BytesIO(payload))
                left, right = st.columns(2)
                left.image(image, caption="Original X-ray", width="stretch")
                if not checkpoint_path.exists():
                    right.info("Model not trained yet. Follow the README to prepare the dataset and train the baseline.")
                else:
                    # Session-owned results: do not share a model with mutable hooks between users.
                    signature = (hashlib.sha256(payload).hexdigest(), target,
                                 checkpoint_path.stat().st_mtime_ns)
                    if st.session_state.get("signature") != signature:
                        with st.spinner("Computing class score and Grad-CAM…"):
                            model, metadata = load_checkpoint(checkpoint_path)
                            tensor, geometry = preprocess(image)
                            scores, raw_cam = grad_cam(model, model.layer4[-1], tensor[None], CLASSES.index(target))
                            cam = unpad_cam(raw_cam, geometry, image.size)
                            st.session_state["analysis"] = (scores, cam, metadata["threshold"])
                            st.session_state["signature"] = signature
                    scores, cam, threshold = st.session_state["analysis"]
                    rendered = overlay(image, cam, opacity)
                    right.image(rendered, caption=f"Grad-CAM · {target}", width="stretch")
                    a, b, c = st.columns(3)
                    a.metric("Model classification", CLASSES[int(scores[1] >= threshold)])
                    b.metric("Pneumonia model score", f"{scores[1]:.1%}")
                    c.metric("Decision threshold", f"{threshold:.2f}")
                    st.caption("The score is a softmax output, not a calibrated probability that a patient has pneumonia. The app cannot verify that an upload is a chest X-ray.")
                    if not np.any(cam > 1e-6):
                        st.warning("No positive Grad-CAM activation for this class. This is not evidence of a normal scan.")
                    st.download_button("Download overlay", png_bytes(rendered), "pneumolens-overlay.png", "image/png")
                    if source == "Dataset example":
                        with st.expander("Example selection & image attribution"):
                            st.write("These examples deliberately include successes and errors from the saved evaluation; they are not a representative performance sample. Retrained models may produce different outcomes.")
                            st.markdown("Images: Kermany, Zhang & Goldbaum · CC BY 4.0 · [Original source](https://data.mendeley.com/datasets/rscbjbr9sj/2). Overlays are modified versions of these images.")
            except (ValueError, OSError, RuntimeError) as error:
                st.error(f"Unable to analyze this image: {error}")
    else:
        st.markdown("### One image. Two views.")
        st.write("Upload an X-ray to compare the original with a class-specific heatmap. Switch the explained class to inspect how the same model supports different outputs.")

with method:
    st.subheader("From pixels to an inspectable prediction")
    st.markdown("1. Preserve the full image using aspect-ratio-preserving padding to 224 × 224.\n2. Extract features with an ImageNet-pretrained ResNet-18 and classify with a trained linear head.\n3. Differentiate the selected **class logit** with respect to the final residual block’s activations.\n4. Average gradients over space, weight the activation channels, apply ReLU, then align the heatmap to the original image.")
    st.latex(r"L^c = \mathrm{ReLU}\left(\sum_k \left[\frac{1}{Z}\sum_{i,j}\frac{\partial y^c}{\partial A^k_{ij}}\right] A^k\right)")
    st.write("The native map is only 7 × 7. Upsampling makes it easier to view but adds no localization detail. Borders, text, and acquisition artifacts can influence predictions. Image-level labels do not establish whether highlighted areas match disease locations.")
    st.markdown("[Grad-CAM paper](https://arxiv.org/abs/1610.02391) · [Dataset source](https://data.mendeley.com/datasets/rscbjbr9sj/2)")
with results:
    metrics_path = ROOT / "artifacts/evaluation/metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        if not checkpoint_path.exists() or metrics.get("checkpoint_sha256") != checkpoint_digest(str(checkpoint_path), checkpoint_path.stat().st_mtime_ns):
            st.warning("The saved evaluation does not match the current model. Rerun the evaluation command.")
            st.stop()
        st.subheader("Held-out test results")
        cols = st.columns(4)
        for col, key in zip(cols, ["sensitivity", "specificity", "roc_auc", "average_precision"]):
            value = metrics[key]
            col.metric(key.replace("_", " ").title(), f"{value:.3f}" if value is not None else "Undefined")
        st.write("Confusion matrix · rows: actual NORMAL/PNEUMONIA; columns: predicted NORMAL/PNEUMONIA")
        st.dataframe(np.array(metrics["confusion_matrix"]))
        st.caption(f"{metrics['n']} test images. One dataset and one training seed; no external clinical validation.")
        training_path = ROOT / "artifacts/baseline/training.json"
        if training_path.exists():
            training = json.loads(training_path.read_text())
            st.subheader("Checkpoint selection")
            st.caption(f"Selected epoch {training['selected_epoch']} using validation loss. The test split was not used for selection.")
            st.line_chart({"Validation cross-entropy": [row["validation_loss"] for row in training["history"]]}, x_label="Epoch index (0 = epoch 1)", y_label="Validation loss")
        st.json(metrics)
    else:
        st.info("No test results yet. The evaluation command writes measured metrics here after training.")
