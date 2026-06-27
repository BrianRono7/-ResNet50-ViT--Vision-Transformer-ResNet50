"""
evaluate.py — Compute accuracy / top-k accuracy on a labelled test directory.

Expects the same layout that tf.keras.utils.image_dataset_from_directory uses:
    test_dir/
        class_0/  *.jpg ...
        class_1/  *.jpg ...
        ...
        class_N/  *.jpg ...

If the dataset was created via image_dataset_from_directory, the alphabetical
class-name -> index mapping is reproducible (it sorts entries). That mapping
must match what you used during training.

Usage:
    python evaluate.py --test-dir data/test --weights path/to/model.keras
    python evaluate.py --test-dir data/test --weights path/to/model.keras --batch-size 4
    python evaluate.py --test-dir data/test --weights path/to/model.keras --top-k 5

Without --weights, the model is built with random ViT head weights and accuracy
will be ~1/N. Useful only to confirm the evaluation pipeline works.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


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


def build_or_load_model(weights: str | None, cfg: dict, num_dataset_classes: int | None = None):
    if weights:
        if not os.path.exists(weights):
            sys.exit(f"--weights file does not exist: {weights}")
        print(f"Loading model from {weights}...")
        return load_model(weights, compile=False)

    # No weights: build a fresh model whose output matches the dataset.
    if num_dataset_classes is not None and num_dataset_classes != cfg["num_classes"]:
        print(f"Dataset has {num_dataset_classes} classes; overriding num_classes from {cfg['num_classes']}.")
        cfg = {**cfg, "num_classes": num_dataset_classes}
    print("No --weights given. Building model with RANDOM ViT head.")
    print("Expected accuracy is ~{:.0f}% (random for {} classes).".format(
        100.0 / cfg["num_classes"], cfg["num_classes"]))
    mod = load_model_module()
    return mod.ResNet50ViT(cfg)


def evaluate(model, test_dir: str, image_size: int, batch_size: int, top_k_choices: list[int]):
    if not os.path.isdir(test_dir):
        sys.exit(f"--test-dir does not exist or is not a directory: {test_dir}")

    ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,           # keep order so we can map preds back to files if needed
    )
    class_names = ds.class_names
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    # Sanity: warn if model output dim doesn't match dataset class count
    out_dim = model.output_shape[-1]
    if out_dim != num_classes:
        print(f"WARNING: model has {out_dim} output classes but dataset has {num_classes}.")
        print(f"         Slicing predictions to the first {num_classes} columns.")
        out_dim = min(out_dim, num_classes)

    # Normalize to 0..1 to match the training pipeline we documented.
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

    n = 0
    correct = {k: 0 for k in top_k_choices}
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for batch_x, batch_y in ds:
        probs = model(batch_x, training=False).numpy()[:, :out_dim]
        labels = batch_y.numpy()
        # Clip labels that are out-of-range for the model output
        labels = np.clip(labels, 0, num_classes - 1)
        n += len(labels)
        for k in top_k_choices:
            kk = min(k, out_dim)
            topk = np.argsort(probs, axis=1)[:, -kk:]
            correct[k] += int(np.sum([labels[i] in topk[i] for i in range(len(labels))]))
        preds = probs.argmax(axis=1)
        for t, p in zip(labels, preds):
            confusion[t, p] += 1

    print()
    print(f"Evaluated {n} images.")
    for k in sorted(correct):
        acc = correct[k] / max(n, 1)
        print(f"  top-{k} accuracy: {acc:.4f}  ({correct[k]}/{n})")

    # Per-class accuracy (top-1)
    print("\nPer-class top-1 accuracy:")
    for i, name in enumerate(class_names):
        total_i = confusion[i].sum()
        if total_i == 0:
            print(f"  {name:>20s}: (no samples)")
            continue
        correct_i = confusion[i, i]
        print(f"  {name:>20s}: {correct_i/total_i:.3f}  ({correct_i}/{total_i})")

    # Most confused pairs
    print("\nTop confused pairs (true -> predicted):")
    flat = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and confusion[i, j] > 0:
                flat.append((confusion[i, j], class_names[i], class_names[j]))
    flat.sort(reverse=True)
    for count, t, p in flat[:5]:
        print(f"  {t} -> {p}: {count}")

    return float(correct[sorted(correct)[0]] / max(n, 1))


def main():
    parser = argparse.ArgumentParser(description="Evaluate ResNet50-ViT on a labelled test directory.")
    parser.add_argument("--test-dir", required=True, help="Path to test directory (one subdir per class).")
    parser.add_argument("--weights", default=None, help="Path to a saved .keras / .h5 model file.")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size. Default 2 keeps memory low; raise on bigger GPUs.")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5],
                        help="Which top-k accuracies to compute. Default: 1 3 5")
    parser.add_argument("--image-size", type=int, default=512,
                        help="Square image side. Must match training config.")
    parser.add_argument("--script", default=" ResNet50-ViT.py",
                        help="Path to the model definition script (only used when --weights is omitted).")
    args = parser.parse_args()

    cfg = default_config()
    if args.image_size != cfg["image_size"]:
        cfg["image_size"] = args.image_size
        cfg["num_patches"] = (args.image_size // cfg["patch_size"]) ** 2
        print(f"Overriding image_size -> {args.image_size}, num_patches -> {cfg['num_patches']}")

    # Peek at the dataset class count BEFORE building the model so the random-head
    # case picks a matching output dim.
    import glob, os as _os
    subdirs = sorted([d for d in glob.glob(_os.path.join(args.test_dir, "*")) if _os.path.isdir(d)])
    num_dataset_classes = len(subdirs) if subdirs else None

    model = build_or_load_model(args.weights, cfg, num_dataset_classes=num_dataset_classes)
    evaluate(model, args.test_dir, cfg["image_size"], args.batch_size, args.top_k)


if __name__ == "__main__":
    main()