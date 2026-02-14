#!/usr/bin/env python3
"""
Zero-shot ImageNet-1K accuracy evaluation for SigLIP2 models.

Supports:
  - HuggingFace (PyTorch) original model
  - CoreML FP32 models (Vision + Text .mlpackage)
  - CoreML INT8 models (Vision + Text .mlpackage)

Usage examples:
  # Evaluate all backends (HF + CoreML FP32 + CoreML INT8) on 1% of validation data
  python evaluate_siglip2_imagenet.py

  # Evaluate only HF model on 10% of data
  python evaluate_siglip2_imagenet.py --backend hf --sample-ratio 0.10

  # Evaluate only CoreML INT8 on full validation set
  python evaluate_siglip2_imagenet.py --backend coreml_int8 --sample-ratio 1.0

  # Use a specific batch size
  python evaluate_siglip2_imagenet.py --batch-size 64
"""

import argparse
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths (defaults)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = str(BASE_DIR / "google" / "siglip2-base-patch16-224")
DEFAULT_DATASET_DIR = str(BASE_DIR / "datasets" / "data")
DEFAULT_CLASSES_PY = str(BASE_DIR / "datasets" / "classes.py")

COREML_VISION_FP32 = str(BASE_DIR / "SigLIP2_Vision.mlpackage")
COREML_TEXT_FP32 = str(BASE_DIR / "SigLIP2_Text.mlpackage")
COREML_VISION_INT8 = str(BASE_DIR / "SigLIP2_Vision_INT8.mlpackage")
COREML_TEXT_INT8 = str(BASE_DIR / "SigLIP2_Text_INT8.mlpackage")

VALID_BACKENDS = ["hf", "coreml_fp32", "coreml_int8"]

# Zero-shot prompt template (matches SigLIP2 pipeline default)
# See: https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/siglip2.md
# "To get the same results as the Pipeline, a prompt template of 'This is a photo of {label}.' should be passed."
# NOTE: model was trained with lowercased text, so class names are lowercased below.
PROMPT_TEMPLATE = "This is a photo of {}."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_imagenet_classes(classes_py_path: str) -> list[str]:
    """Load the 1000 ImageNet class names from datasets/classes.py."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("classes", classes_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ordered = mod.IMAGENET2012_CLASSES  # OrderedDict synset -> label
    # Take the first name before the comma for a cleaner prompt
    labels = []
    for _synset, raw_label in ordered.items():
        # e.g. "tench, Tinca tinca" -> "tench"
        first_name = raw_label.split(",")[0].strip()
        labels.append(first_name)
    return labels


def load_dataset_parquet(dataset_dir: str, sample_ratio: float, seed: int):
    """Load ImageNet validation set from parquet shards.

    Each row has columns: ``image`` (dict with ``bytes``), ``label`` (int 0-999).
    Returns a list of (PIL.Image, label_int) tuples.
    """
    import pyarrow.parquet as pq
    from PIL import Image

    parquet_files = sorted(Path(dataset_dir).glob("validation-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No validation parquet files found in {dataset_dir}")

    print(f"Found {len(parquet_files)} parquet shards in {dataset_dir}")

    # Read all shards into a single PyArrow table
    tables = [pq.read_table(str(f)) for f in parquet_files]
    import pyarrow as pa

    table = pa.concat_tables(tables)
    total = len(table)
    print(f"Total validation samples: {total}")

    # Sub-sample
    rng = np.random.default_rng(seed)
    n_samples = max(1, int(total * sample_ratio))
    indices = rng.choice(total, size=n_samples, replace=False)
    indices.sort()
    print(f"Sampled {n_samples} images ({sample_ratio * 100:.1f}%)")

    samples = []
    for idx in indices:
        row = table.slice(int(idx), 1).to_pydict()
        img_bytes = row["image"][0]["bytes"]
        label = int(row["label"][0])
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        samples.append((img, label))

    return samples


# ---------------------------------------------------------------------------
# Text embedding pre-computation
# ---------------------------------------------------------------------------
def precompute_text_embeddings_hf(model, processor, class_names: list[str], device: str):
    """Compute L2-normalised text embeddings for all 1000 classes using HF model."""
    import torch

    prompts = [PROMPT_TEMPLATE.format(name.lower()).lower() for name in class_names]
    seq_len = model.config.text_config.max_position_embeddings

    tokenized = processor.tokenizer(
        prompts,
        padding="max_length",
        max_length=seq_len,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"]

    all_embeds = []
    batch_size = 128
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            ids = input_ids[start:end].to(device)
            embeds = model.get_text_features(input_ids=ids)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            all_embeds.append(embeds.cpu().numpy())

    return np.concatenate(all_embeds, axis=0)  # (1000, dim)


def precompute_text_embeddings_coreml(
    text_model_path: str,
    processor,
    class_names: list[str],
    seq_len: int,
):
    """Compute L2-normalised text embeddings for all 1000 classes using CoreML text model."""
    import coremltools as ct

    text_mlmodel = ct.models.MLModel(text_model_path)
    prompts = [PROMPT_TEMPLATE.format(name.lower()).lower() for name in class_names]

    tokenized = processor.tokenizer(
        prompts,
        padding="max_length",
        max_length=seq_len,
        truncation=True,
        return_tensors="np",
    )
    input_ids = tokenized["input_ids"].astype(np.int32)

    all_embeds = []
    for i in range(len(prompts)):
        ids = input_ids[i : i + 1]
        out = text_mlmodel.predict({"input_ids": ids})
        emb = np.array(out["embedding"], dtype=np.float32).reshape(1, -1)
        all_embeds.append(emb)

    text_embeds = np.concatenate(all_embeds, axis=0)  # (1000, dim)
    norms = np.linalg.norm(text_embeds, axis=-1, keepdims=True)
    text_embeds = text_embeds / norms
    return text_embeds


# ---------------------------------------------------------------------------
# Image embedding computation
# ---------------------------------------------------------------------------
def compute_image_embeddings_hf(model, processor, images, device: str, batch_size: int):
    """Compute L2-normalised image embeddings using HF model. Returns (N, dim)."""
    import torch

    all_embeds = []
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        batch_imgs = images[start:end]
        inputs = processor.image_processor(images=batch_imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            embeds = model.get_image_features(pixel_values=pixel_values)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        all_embeds.append(embeds.cpu().numpy())

    return np.concatenate(all_embeds, axis=0)


def compute_image_embeddings_coreml(vision_model_path: str, images, image_size: int):
    """Compute L2-normalised image embeddings using CoreML vision model. Returns (N, dim)."""
    import coremltools as ct
    from PIL import Image

    vision_mlmodel = ct.models.MLModel(vision_model_path)

    all_embeds = []
    for img in images:
        img_resized = img.resize((image_size, image_size), resample=Image.BICUBIC)
        out = vision_mlmodel.predict({"image": img_resized})
        emb = np.array(out["embedding"], dtype=np.float32).reshape(1, -1)
        all_embeds.append(emb)

    image_embeds = np.concatenate(all_embeds, axis=0)
    norms = np.linalg.norm(image_embeds, axis=-1, keepdims=True)
    image_embeds = image_embeds / norms
    return image_embeds


# ---------------------------------------------------------------------------
# Top-k accuracy
# ---------------------------------------------------------------------------
def topk_accuracy(
    image_embeds: np.ndarray,
    text_embeds: np.ndarray,
    labels: np.ndarray,
    k_values: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Compute top-k zero-shot classification accuracy.

    image_embeds: (N, dim)  – L2-normalised
    text_embeds:  (C, dim)  – L2-normalised  (C=1000)
    labels:       (N,)      – ground-truth class indices
    """
    # Similarity matrix (N, C)
    logits = image_embeds @ text_embeds.T
    results = {}
    for k in k_values:
        topk_preds = np.argsort(logits, axis=1)[:, -k:]  # (N, k) – highest scores
        correct = np.any(topk_preds == labels[:, None], axis=1)
        results[f"top{k}"] = float(correct.mean())
    return results


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------
def evaluate_hf(
    model_id: str,
    samples: list,
    class_names: list[str],
    batch_size: int,
    device: str,
) -> dict[str, float]:
    """Run zero-shot eval with the original HuggingFace model."""
    import torch
    from transformers import AutoModel, AutoProcessor

    print(f"\n{'='*60}")
    print("Evaluating: HuggingFace (PyTorch) model")
    print(f"{'='*60}")

    print("Loading model and processor...")
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    images = [s[0] for s in samples]
    labels = np.array([s[1] for s in samples])

    print("Pre-computing text embeddings for 1000 classes...")
    t0 = time.time()
    text_embeds = precompute_text_embeddings_hf(model, processor, class_names, device)
    print(f"  Text embeddings done in {time.time() - t0:.1f}s  shape={text_embeds.shape}")

    print(f"Computing image embeddings for {len(images)} images (batch_size={batch_size})...")
    t0 = time.time()
    image_embeds = compute_image_embeddings_hf(model, processor, images, device, batch_size)
    print(f"  Image embeddings done in {time.time() - t0:.1f}s  shape={image_embeds.shape}")

    results = topk_accuracy(image_embeds, text_embeds, labels)
    print(f"  Top-1 accuracy: {results['top1'] * 100:.2f}%")
    print(f"  Top-5 accuracy: {results['top5'] * 100:.2f}%")

    # Free memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return results


def evaluate_coreml(
    label: str,
    vision_model_path: str,
    text_model_path: str,
    model_id: str,
    samples: list,
    class_names: list[str],
    image_size: int,
    seq_len: int,
) -> dict[str, float]:
    """Run zero-shot eval with CoreML Vision + Text models."""
    from transformers import AutoProcessor

    print(f"\n{'='*60}")
    print(f"Evaluating: CoreML {label}")
    print(f"  Vision: {vision_model_path}")
    print(f"  Text:   {text_model_path}")
    print(f"{'='*60}")

    processor = AutoProcessor.from_pretrained(model_id)

    images = [s[0] for s in samples]
    labels = np.array([s[1] for s in samples])

    print("Pre-computing text embeddings for 1000 classes...")
    t0 = time.time()
    text_embeds = precompute_text_embeddings_coreml(
        text_model_path, processor, class_names, seq_len
    )
    print(f"  Text embeddings done in {time.time() - t0:.1f}s  shape={text_embeds.shape}")

    print(f"Computing image embeddings for {len(images)} images...")
    t0 = time.time()
    image_embeds = compute_image_embeddings_coreml(vision_model_path, images, image_size)
    print(f"  Image embeddings done in {time.time() - t0:.1f}s  shape={image_embeds.shape}")

    results = topk_accuracy(image_embeds, text_embeds, labels)
    print(f"  Top-1 accuracy: {results['top1'] * 100:.2f}%")
    print(f"  Top-5 accuracy: {results['top5'] * 100:.2f}%")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot ImageNet-1K evaluation for SigLIP2 (HF & CoreML)."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Local path or HuggingFace model id. Default: %(default)s",
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="Directory containing validation parquet shards. Default: %(default)s",
    )
    parser.add_argument(
        "--classes-py",
        default=DEFAULT_CLASSES_PY,
        help="Path to classes.py with IMAGENET2012_CLASSES. Default: %(default)s",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.01,
        help="Fraction of validation set to use (0.0-1.0). Default: 0.01 (1%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling. Default: 42",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for HF image embedding computation. Default: 32",
    )
    parser.add_argument(
        "--backend",
        nargs="+",
        choices=VALID_BACKENDS,
        default=VALID_BACKENDS,
        help="Which backend(s) to evaluate. Default: all three.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for HF inference. Default: cpu",
    )
    # CoreML model paths (overridable)
    parser.add_argument("--coreml-vision-fp32", default=COREML_VISION_FP32)
    parser.add_argument("--coreml-text-fp32", default=COREML_TEXT_FP32)
    parser.add_argument("--coreml-vision-int8", default=COREML_VISION_INT8)
    parser.add_argument("--coreml-text-int8", default=COREML_TEXT_INT8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ---- Load class names ----
    print("Loading ImageNet class names...")
    class_names = load_imagenet_classes(args.classes_py)
    assert len(class_names) == 1000, f"Expected 1000 classes, got {len(class_names)}"
    print(f"  {len(class_names)} classes loaded. Examples: {class_names[:3]}")

    # ---- Load dataset ----
    print(f"\nLoading dataset (sample_ratio={args.sample_ratio}, seed={args.seed})...")
    samples = load_dataset_parquet(args.dataset_dir, args.sample_ratio, args.seed)

    # ---- Resolve model config values needed by CoreML ----
    image_size = 224
    seq_len = 64
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(args.model_id)
        size = getattr(processor.image_processor, "size", None)
        if isinstance(size, dict):
            image_size = size.get("height", 224)

        # Try to read seq_len from config
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(args.model_id)
        seq_len = getattr(config.text_config, "max_position_embeddings", 64)
        del processor
    except Exception as e:
        print(f"  Warning: could not auto-detect model params: {e}. Using defaults.")

    print(f"  image_size={image_size}, seq_len={seq_len}")

    # ---- Run evaluations ----
    all_results = {}

    if "hf" in args.backend:
        results = evaluate_hf(
            model_id=args.model_id,
            samples=samples,
            class_names=class_names,
            batch_size=args.batch_size,
            device=args.device,
        )
        all_results["HuggingFace (PyTorch)"] = results

    if "coreml_fp32" in args.backend:
        results = evaluate_coreml(
            label="FP32",
            vision_model_path=args.coreml_vision_fp32,
            text_model_path=args.coreml_text_fp32,
            model_id=args.model_id,
            samples=samples,
            class_names=class_names,
            image_size=image_size,
            seq_len=seq_len,
        )
        all_results["CoreML FP32"] = results

    if "coreml_int8" in args.backend:
        results = evaluate_coreml(
            label="INT8",
            vision_model_path=args.coreml_vision_int8,
            text_model_path=args.coreml_text_int8,
            model_id=args.model_id,
            samples=samples,
            class_names=class_names,
            image_size=image_size,
            seq_len=seq_len,
        )
        all_results["CoreML INT8"] = results

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset: ImageNet-1K validation ({len(samples)} samples, {args.sample_ratio*100:.1f}%)")
    print(f"Prompt template: \"{PROMPT_TEMPLATE}\"")
    print(f"{'─'*60}")
    print(f"{'Backend':<25} {'Top-1':>10} {'Top-5':>10}")
    print(f"{'─'*60}")
    for name, res in all_results.items():
        print(f"{name:<25} {res['top1']*100:>9.2f}% {res['top5']*100:>9.2f}%")
    print(f"{'─'*60}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
