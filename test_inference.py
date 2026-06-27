"""
test_inference.py — Run the untrained ResNet50-ViT model on real images.

This script verifies the full inference pipeline (load -> resize -> normalize
-> forward -> softmax -> top-k) without requiring any training. It also
makes the "the model is untrained" limitation explicit by reporting how
uniform the output distribution is.

Usage:
    python test_inference.py --image-dir path/to/your/images
    python test_inference.py --image path/to/single.jpg
    python test_inference.py                # uses any images it finds under ./img
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

try:
    # Preferred path: PIL-backed, supports many formats.
    from tensorflow.keras.utils import load_img, img_to_array
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


def _decode_image(path: Path, image_size: int) -> np.ndarray:
    """Pure-TF fallback when PIL is unavailable."""
    raw = tf.io.read_file(str(path))
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (image_size, image_size), method="bilinear")
    return tf.cast(img, tf.float32).numpy() / 255.0


# ---- Model loading ----------------------------------------------------------

def load_model_module(script_path: str = " ResNet50-ViT.py"):
    if not os.path.exists(script_path):
        sys.exit(f"Could not find {script_path!r}. Run from the repo root.")
    spec = importlib.util.spec_from_file_location("ResNet50ViT", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def default_config():
    return {
        "num_layers": 12, "hidden_dim": 768, "mlp_dim": 3072, "num_heads": 12,
        "dropout_rate": 0.1, "image_size": 512, "patch_size": 32,
        "num_patches": (512 // 32) ** 2, "num_channels": 3, "num_classes": 10,
    }


# ---- Image helpers ----------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def find_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS])


def load_and_preprocess(path: Path, image_size: int) -> np.ndarray:
    """Load an image, resize to (image_size, image_size), return float32 batch of 1.

    Uses PIL via tf.keras.utils when available, falls back to tf.io otherwise.
    """
    if _HAS_PIL:
        try:
            img = load_img(path, target_size=(image_size, image_size))
            arr = img_to_array(img)                          # (H, W, 3), float32, 0..255
            arr = arr / 255.0                                # normalize to 0..1
            return np.expand_dims(arr, axis=0)               # (1, H, W, 3)
        except Exception:
            # PIL import succeeded but the runtime is missing (common on slim
            # TF installs). Fall through to the tf.io decoder.
            pass
    arr = _decode_image(path, image_size)
    return np.expand_dims(arr, axis=0)


# ---- Pretty-printing --------------------------------------------------------

def top_k(probs: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    idx = probs.argsort()[::-1][:k]
    return [(int(i), float(probs[i])) for i in idx]


def distribution_stats(probs: np.ndarray) -> str:
    """How uniform is the output? Random head -> ~0.1 for each of 10 classes."""
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(len(probs)))
    # 1.0 = perfectly uniform (random), 0.0 = one-hot (confident)
    max_p = float(probs.max())
    return f"max_p={max_p:.3f}  normalized_entropy={entropy:.3f}  (1.0=uniform/random)"


# ---- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test ResNet50-ViT inference.")
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a single image.",
    )
    parser.add_argument(
        "--image-dir", type=str, default=None,
        help="Path to a directory of images (recursive).",
    )
    parser.add_argument(
        "--k", type=int, default=3,
        help="Show top-k predictions per image (default: 3).",
    )
    parser.add_argument(
        "--script", type=str, default=" ResNet50-ViT.py",
        help="Path to the model script.",
    )
    args = parser.parse_args()

    # Collect images
    if args.image:
        images = find_images(Path(args.image))
    elif args.image_dir:
        images = find_images(Path(args.image_dir))
    else:
        # Fall back to ./img if it has anything usable
        images = find_images(Path("img"))
        if not images:
            sys.exit("No images found. Pass --image <file> or --image-dir <dir>.")

    if not images:
        sys.exit("No images found at that path (jpg/png/jpeg/bmp/webp/tiff).")

    print(f"Found {len(images)} image(s). Building model...\n")

    # Build model
    mod = load_model_module(args.script)
    cfg = default_config()
    model = mod.ResNet50ViT(cfg)
    print(f"Model built: input={model.input_shape}  output={model.output_shape}\n")
    print("Note: the ViT head + classifier are RANDOMLY INITIALIZED.")
    print("      Output will be near-uniform unless you have trained the model.\n")

    image_size = cfg["image_size"]

    # Inference loop
    for img_path in images:
        try:
            batch = load_and_preprocess(img_path, image_size)
            probs = model(batch, training=False).numpy()[0]
        except Exception as e:
            print(f"[SKIP] {img_path}: {e}")
            continue

        print(f"── {img_path}")
        print(f"   {distribution_stats(probs)}")
        for rank, (cls, p) in enumerate(top_k(probs, args.k), start=1):
            bar = "█" * int(round(p * 40))
            print(f"   #{rank}  class {cls:>2d}  p={p:.4f}  {bar}")
        print()


if __name__ == "__main__":
    main()