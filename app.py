"""
app.py — Streamlit UI for the ResNet50-ViT model.

Run with:
    streamlit run app.py

Features:
    - Upload a single image (jpg/png/jpeg/webp/bmp).
    - Optionally upload a trained .keras / .h5 weights file.
    - See top-K predictions with probabilities as a bar chart.
    - See a "how uniform" diagnostic so you can tell untrained from broken.
    - Adjust image size, batch behaviour, and class names in the sidebar.

The model definition is imported dynamically from " ResNet50-ViT.py"
so this app works even though the file name has a leading space.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf


# --- One-time, cached resource loaders --------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
MODEL_SCRIPT = REPO_ROOT / " ResNet50-ViT.py"


@st.cache_resource(show_spinner="Loading model definition...")
def load_model_module():
    """Import the model file. Cached so we only do it once per Streamlit session."""
    if not MODEL_SCRIPT.exists():
        raise FileNotFoundError(
            f"Could not find {MODEL_SCRIPT}. Run Streamlit from the repo root."
        )
    spec = importlib.util.spec_from_file_location("ResNet50ViT", str(MODEL_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@st.cache_resource(show_spinner="Building model graph (~2.03 B params)...")
def build_random_head_model(image_size: int, num_classes: int):
    """Build a model with a randomly-initialized ViT head.

    Used when the user hasn't uploaded trained weights yet, so the UI is
    immediately responsive. Predictions from this model are essentially noise.
    """
    mod = load_model_module()
    cfg = {
        "num_layers": 12, "hidden_dim": 768, "mlp_dim": 3072, "num_heads": 12,
        "dropout_rate": 0.1, "image_size": image_size, "patch_size": 32,
        "num_patches": (image_size // 32) ** 2, "num_channels": 3,
        "num_classes": num_classes,
    }
    return mod.ResNet50ViT(cfg), cfg


@st.cache_resource(show_spinner="Loading trained weights...")
def load_trained_model(weights_bytes: bytes, image_size: int):
    """Load a user-uploaded .keras / .h5 file and return (model, model_output_dim)."""
    from tensorflow.keras.models import load_model
    # load_model needs a file path, so write to a temp file.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp.write(weights_bytes)
        tmp_path = tmp.name
    try:
        model = load_model(tmp_path, compile=False)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return model


# --- Image helpers ----------------------------------------------------------

IMG_EXTS = ("jpg", "jpeg", "png", "bmp", "webp")


def decode_to_array(uploaded_file, image_size: int) -> np.ndarray | None:
    """Read an uploaded image, force RGB, resize, return float32 in [0, 1].

    Returns None if the bytes can't be decoded as an image.
    """
    raw = uploaded_file.read()
    try:
        # tf.io handles a wide variety of formats without requiring PIL.
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    except Exception as e:
        st.error(f"Could not decode that file as an image: {e}")
        return None
    img = tf.image.resize(img, (image_size, image_size), method="bilinear")
    arr = tf.cast(img, tf.float32).numpy() / 255.0
    return arr  # (H, W, 3)


def distribution_stats(probs: np.ndarray) -> dict:
    """Return diagnostics about how 'spiky' the output distribution is."""
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(len(probs)))
    # 1.0 = perfectly uniform (random); 0.0 = one-hot (very confident)
    return {
        "max_p": float(probs.max()),
        "normalized_entropy": entropy,
        "top_class": int(probs.argmax()),
        "top_p": float(probs.max()),
    }


# --- Streamlit UI -----------------------------------------------------------

def main():
    st.set_page_config(
        page_title="ResNet50-ViT Tester",
        page_icon="🖼️",
        layout="wide",
    )

    st.title("🖼️ ResNet50-ViT Tester")
    st.caption(
        "Upload an image, optionally load trained weights, and inspect the "
        "model's predictions. The model file ` ResNet50-ViT.py` defines the "
        "ResNet50 + Vision Transformer hybrid."
    )

    # --- Sidebar: configuration ---------------------------------------------
    with st.sidebar:
        st.header("Configuration")

        image_size = st.selectbox(
            "Image size (square)",
            options=[224, 256, 384, 512],
            index=3,
            help="Must be divisible by 32. Larger = more memory.",
        )
        num_classes = st.number_input(
            "Number of classes (random-head model)",
            min_value=1, max_value=1000, value=10,
            help="Ignored if you upload trained weights — the weight file's "
                 "output dimension is used instead.",
        )
        top_k = st.slider("Top-K to display", min_value=1, max_value=20, value=5)

        st.divider()
        st.header("Trained weights")
        weights_file = st.file_uploader(
            "Upload a trained .keras or .h5 model (optional)",
            type=["keras", "h5"],
            help="Leave empty to use a randomly-initialized ViT head.",
        )

        st.divider()
        with st.expander("What is this model?"):
            st.markdown(
                "- **Backbone:** ImageNet-pretrained ResNet50 (frozen weights, "
                "loaded once on first run).\n"
                "- **Head:** 12-layer Vision Transformer + learned `[CLS]` token.\n"
                "- **Output:** softmax over `num_classes`.\n\n"
                "Until you upload trained weights, the ViT head + final "
                "softmax layer are random, so predictions are noise."
            )

    # --- Load the appropriate model ----------------------------------------
    model = None
    model_output_dim = None
    used_weights = False

    if weights_file is not None:
        try:
            model = load_trained_model(weights_file.getvalue(), image_size)
            model_output_dim = model.output_shape[-1]
            used_weights = True
        except Exception as e:
            st.error(f"Failed to load weights: {e}")
            st.info("Falling back to a random-head model.")
            model = None

    if model is None:
        model, cfg = build_random_head_model(image_size, int(num_classes))
        model_output_dim = int(num_classes)
    else:
        # Robust against models that report NoneType output_shape (Keras quirk).
        shape = getattr(model, "output_shape", None)
        if shape and isinstance(shape, tuple) and shape[-1] is not None:
            model_output_dim = int(shape[-1])
        else:
            model_output_dim = int(num_classes)
            st.warning(
                f"Could not read model output shape; assuming {model_output_dim} classes."
            )

    # Sidebar status pill
    with st.sidebar:
        if used_weights:
            st.success(f"Loaded trained model (output dim = {model_output_dim})")
        else:
            st.warning("Using random ViT head — predictions will look like noise")

    # --- Main area: image upload + prediction -------------------------------
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("1. Upload an image")
        uploaded = st.file_uploader(
            "Choose an image file",
            type=list(IMG_EXTS),
            label_visibility="collapsed",
        )
        if uploaded is None:
            st.info("👆 Upload an image to see predictions.")
            return

        arr = decode_to_array(uploaded, image_size)
        if arr is None:
            return
        st.image(arr, caption=f"{uploaded.name}  →  resized to {image_size}×{image_size}",
                 use_column_width=True)

    with col_right:
        st.subheader("2. Predictions")

        batch = np.expand_dims(arr, axis=0)        # (1, H, W, 3)
        probs = model(batch, training=False).numpy()[0]
        # If the model has more output classes than the dataset (shouldn't
        # happen with weights, but guard anyway), slice down.
        if len(probs) > model_output_dim:
            probs = probs[:model_output_dim]

        stats = distribution_stats(probs)
        c1, c2, c3 = st.columns(3)
        c1.metric("Top class", f"#{stats['top_class']}")
        c2.metric("Top-1 confidence", f"{stats['top_p']:.3f}")
        c3.metric("Output entropy", f"{stats['normalized_entropy']:.3f}",
                  help="1.0 = uniform (untrained head); 0.0 = one-hot (confident)")

        # Interpretation hint
        ent = stats["normalized_entropy"]
        if not used_weights:
            st.info(
                f"🔍 **Diagnostic:** entropy = {ent:.3f}. "
                "With a randomly-initialized head, values near 0.7–1.0 are "
                "expected. If you see entropy ≈ 0 with no weights loaded, "
                "something else (input pipeline?) is suspicious."
            )
        elif ent > 0.85:
            st.warning(
                f"🔍 Trained model output looks almost uniform "
                f"(entropy = {ent:.3f}). This usually means the model "
                "isn't well-matched to the input (wrong preprocessing, "
                "wrong class count, or under-trained)."
            )
        else:
            st.success(
                f"🔍 Output distribution looks healthy "
                f"(entropy = {ent:.3f})."
            )

        # Top-K table + bar chart
        top_k = min(top_k, len(probs))
        top_idx = probs.argsort()[::-1][:top_k]
        top_p = probs[top_idx]
        st.bar_chart({"probability": top_p}, height=300)

        with st.expander("Show raw top-K values"):
            st.dataframe(
                {
                    "rank": range(1, top_k + 1),
                    "class_index": top_idx,
                    "probability": [f"{p:.6f}" for p in top_p],
                },
                use_container_width=True,
            )

    # --- Footer / explainer -------------------------------------------------
    st.divider()
    with st.expander("How is the model being used here?"):
        st.markdown(
            f"1. The uploaded image is decoded with `tf.io.decode_image` "
            f"(no PIL required) and resized to **{image_size}×{image_size}**.\n"
            f"2. It's normalized to `[0, 1]` by dividing by 255.\n"
            f"3. The image is passed through the ResNet50 backbone "
            f"(ImageNet-pretrained) and then through the ViT head.\n"
            f"4. Softmax over `{model_output_dim}` classes gives the "
            f"probabilities shown above.\n\n"
            f"**Important:** If you trained the model with a different "
            f"preprocessing function (e.g. `tf.keras.applications.resnet50."
            f"preprocess_input`), you should apply that here too — this "
            f"demo uses simple `/255.0` normalization to match the training "
            f"template in the README."
        )


if __name__ == "__main__":
    main()