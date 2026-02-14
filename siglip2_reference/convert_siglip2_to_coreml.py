import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import coremltools as ct
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


DEFAULT_MODEL_ID = "google/siglip2-base-patch16-224"
DEFAULT_OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
VISION_OUTPUT_NAME = "SigLIP2_Vision.mlpackage"
TEXT_OUTPUT_NAME = "SigLIP2_Text.mlpackage"
MAX_ABS_DIFF_THRESHOLD = 3e-2
COSINE_SIM_THRESHOLD = 0.999


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SigLIP2 HuggingFace weights to CoreML Vision/Text models (FP16)."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Local path or HuggingFace model id. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for exported mlpackage files. Default: %(default)s",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip PyTorch vs CoreML numeric validation.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override image size. Default: from processor.image_processor.size",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Override text sequence length. Default: from model.config.text_config.max_position_embeddings",
    )
    return parser.parse_args()


def resolve_image_size(processor, override: int | None) -> int:
    if override is not None:
        if override <= 0:
            raise ValueError(f"image_size must be > 0, got {override}")
        return int(override)

    size = getattr(processor.image_processor, "size", None)
    if isinstance(size, int):
        return int(size)
    if isinstance(size, dict):
        h = size.get("height")
        w = size.get("width")
        if h is not None and w is not None:
            h = int(h)
            w = int(w)
            if h != w:
                raise ValueError(
                    f"Only square image size is supported for this exporter, got height={h}, width={w}"
                )
            return h

    raise ValueError("Could not resolve image size from processor.image_processor.size")


def resolve_seq_len(config, override: int | None) -> int:
    if override is not None:
        if override <= 0:
            raise ValueError(f"seq_len must be > 0, got {override}")
        return int(override)

    seq_len = getattr(config.text_config, "max_position_embeddings", None)
    if seq_len is None:
        raise ValueError("Could not resolve max_position_embeddings from model config")
    return int(seq_len)


def resolve_pad_token_id(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)

    special_map = getattr(tokenizer, "special_tokens_map", {}) or {}
    pad_token = special_map.get("pad_token")
    if pad_token is not None:
        pad_id = tokenizer.convert_tokens_to_ids(pad_token)
        if pad_id is not None and int(pad_id) >= 0:
            return int(pad_id)

    raise ValueError(
        "tokenizer.pad_token_id is None and no valid pad token could be resolved from special_tokens_map."
    )


def report_missing_local_files(model_id: str) -> None:
    if not os.path.isdir(model_id):
        return
    expected = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "model.safetensors",
    ]
    missing = [name for name in expected if not os.path.exists(os.path.join(model_id, name))]
    if missing:
        print(
            f"[ERROR] Missing files under local model path '{model_id}': {', '.join(missing)}",
            file=sys.stderr,
        )


class VisionModelWrapper(torch.nn.Module):
    def __init__(self, vision_model: torch.nn.Module, mean, std):
        super().__init__()
        self.vision_model = vision_model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(dtype=self.mean.dtype)
        x = (image - self.mean) / self.std
        outputs = self.vision_model(pixel_values=x)
        return outputs.pooler_output


class TextModelWrapper(torch.nn.Module):
    def __init__(self, text_model: torch.nn.Module):
        super().__init__()
        self.text_model = text_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.long()
        outputs = self.text_model(input_ids=input_ids)
        return outputs.pooler_output


def build_vision_coreml_model(
    wrapper: torch.nn.Module,
    image_size: int,
    rescale_factor: float,
) -> ct.models.MLModel:
    dummy_image = torch.rand(1, 3, image_size, image_size, dtype=torch.float32)
    traced = torch.jit.trace(wrapper, dummy_image)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, image_size, image_size),
                scale=rescale_factor,
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[ct.TensorType(name="embedding")],
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
    )
    mlmodel.short_description = "SigLIP2 Vision tower — FP16 mlprogram (pooler output embedding)."
    mlmodel.input_description["image"] = (
        f"RGB image ({image_size}x{image_size}). Input is scaled by {rescale_factor} and normalized in-model."
    )
    mlmodel.output_description["embedding"] = "Image embedding tensor."
    return mlmodel


def build_text_coreml_model(wrapper: torch.nn.Module, seq_len: int) -> ct.models.MLModel:
    dummy_ids = torch.ones((1, seq_len), dtype=torch.long)
    traced = torch.jit.trace(wrapper, dummy_ids)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=(1, seq_len), dtype=np.int32)],
        outputs=[ct.TensorType(name="embedding")],
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
    )
    mlmodel.short_description = "SigLIP2 Text tower — FP16 mlprogram (pooler output embedding)."
    mlmodel.input_description["input_ids"] = (
        f"Token IDs Int32 shape (1, {seq_len}). No attention_mask — model uses all-ones mask internally."
    )
    mlmodel.output_description["embedding"] = "Text embedding tensor."
    return mlmodel


def pick_validation_image(model_id: str, image_size: int) -> Tuple[Image.Image, str]:
    candidates = [
        os.path.join(model_id, "figure", "fig1.jpg"),
        os.path.join(model_id, "fig1.jpg"),
        os.path.join(model_id, "sample.jpg"),
        "fig1.jpg",
        "sample.jpg",
        "test.jpg",
        "test.png",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return Image.open(path).convert("RGB"), path

    rng = np.random.default_rng(12345)
    arr = rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB"), "generated_random_image"


def embedding_metrics(reference: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    ref = np.asarray(reference, dtype=np.float32).reshape(-1)
    pred = np.asarray(predicted, dtype=np.float32).reshape(-1)

    diff = np.abs(ref - pred)
    max_abs_diff = float(diff.max())
    mean_abs_diff = float(diff.mean())
    denom = float(np.linalg.norm(ref) * np.linalg.norm(pred))
    cosine = float(np.dot(ref, pred) / denom) if denom > 0 else 0.0
    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "cosine_similarity": cosine,
    }


def metrics_pass(
    metrics: Dict[str, float],
    max_abs_diff_threshold: float,
    cosine_sim_threshold: float,
) -> bool:
    return (
        metrics["max_abs_diff"] <= max_abs_diff_threshold
        and metrics["cosine_similarity"] >= cosine_sim_threshold
    )


def format_metrics(prefix: str, metrics: Dict[str, float]) -> str:
    return (
        f"{prefix} max_abs_diff={metrics['max_abs_diff']:.6f}, "
        f"mean_abs_diff={metrics['mean_abs_diff']:.6f}, "
        f"cosine_similarity={metrics['cosine_similarity']:.6f}"
    )


def validate_export(
    model,
    processor,
    model_id: str,
    vision_model_path: str,
    text_model_path: str,
    image_size: int,
    seq_len: int,
    pad_token_id: int,
) -> None:
    print("\nRunning HuggingFace reference inference...")

    image, image_source = pick_validation_image(model_id, image_size)
    image_resized = image.resize((image_size, image_size), resample=Image.BICUBIC)
    print(f"Validation image source: {image_source}")

    with torch.no_grad():
        try:
            vision_inputs = processor.image_processor(
                images=image_resized,
                return_tensors="pt",
                do_resize=False,
            )
        except Exception:
            vision_inputs = processor(images=image_resized, return_tensors="pt")
        pt_image_embedding = model.get_image_features(
            pixel_values=vision_inputs["pixel_values"]
        ).cpu().numpy()

    texts = ["a photo of a cat", "a photo of a dog"]
    tokenized = processor.tokenizer(
        texts,
        padding="max_length",
        max_length=seq_len,
        truncation=True,
        return_tensors="np",
    )
    input_ids = tokenized["input_ids"].astype(np.int32)

    pt_text_embeddings = []
    for index, text in enumerate(texts):
        sample_ids = input_ids[index : index + 1]
        ids_torch = torch.from_numpy(sample_ids.astype(np.int64))

        with torch.no_grad():
            pt_text_embedding = model.get_text_features(
                input_ids=ids_torch,
            ).cpu().numpy()
        pt_text_embeddings.append(pt_text_embedding)

    vision_mlmodel = ct.models.MLModel(vision_model_path)
    text_mlmodel = ct.models.MLModel(text_model_path)

    print("\nComparing FP16 CoreML vs HuggingFace (FP32 PyTorch)...")
    overall_ok = True

    cm_image_embedding = vision_mlmodel.predict({"image": image_resized})["embedding"]
    vision_metrics = embedding_metrics(pt_image_embedding, cm_image_embedding)
    vision_ok = metrics_pass(vision_metrics, MAX_ABS_DIFF_THRESHOLD, COSINE_SIM_THRESHOLD)
    overall_ok = overall_ok and vision_ok
    print(format_metrics("FP16 Vision:", vision_metrics))
    print(f"FP16 Vision status: {'PASS' if vision_ok else 'FAIL'}")

    for index, text in enumerate(texts):
        sample_ids = input_ids[index : index + 1]
        cm_text_embedding = text_mlmodel.predict({"input_ids": sample_ids})["embedding"]
        text_metrics = embedding_metrics(pt_text_embeddings[index], cm_text_embedding)
        text_ok = metrics_pass(text_metrics, MAX_ABS_DIFF_THRESHOLD, COSINE_SIM_THRESHOLD)
        overall_ok = overall_ok and text_ok
        print(f"FP16 text sample: {text}")
        print(format_metrics("FP16 Text:  ", text_metrics))
        print(f"FP16 Text status: {'PASS' if text_ok else 'FAIL'}")

    if not overall_ok:
        raise RuntimeError(
            "FP16 validation failed. Thresholds: "
            f"cosine_similarity >= {COSINE_SIM_THRESHOLD}, "
            f"max_abs_diff <= {MAX_ABS_DIFF_THRESHOLD}."
        )

    print("Validation passed.")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model and processor: {args.model_id}")
    try:
        model = AutoModel.from_pretrained(args.model_id)
        processor = AutoProcessor.from_pretrained(args.model_id)
    except Exception as exc:
        report_missing_local_files(args.model_id)
        raise RuntimeError(
            "Failed to load model/processor. Ensure local path contains model.safetensors, "
            "tokenizer.model, config.json, preprocessor_config.json and tokenizer_config.json."
        ) from exc

    model.eval()

    image_mean = getattr(processor.image_processor, "image_mean", None)
    image_std = getattr(processor.image_processor, "image_std", None)
    rescale_factor = getattr(processor.image_processor, "rescale_factor", None)
    if image_mean is None or image_std is None or rescale_factor is None:
        raise ValueError("Could not resolve image_mean/image_std/rescale_factor from processor")

    image_size = resolve_image_size(processor, args.image_size)
    seq_len = resolve_seq_len(model.config, args.seq_len)
    pad_token_id = resolve_pad_token_id(processor.tokenizer)

    print("Resolved parameters:")
    print(f"  image_size={image_size}")
    print(f"  image_mean={image_mean}")
    print(f"  image_std={image_std}")
    print(f"  rescale_factor={rescale_factor}")
    print(f"  seq_len={seq_len}")
    print(f"  pad_token_id={pad_token_id}")

    vision_wrapper = VisionModelWrapper(model.vision_model, image_mean, image_std).eval()
    text_wrapper = TextModelWrapper(model.text_model).eval()

    print("Converting vision model to CoreML (FP16 mlprogram)...")
    vision_mlmodel = build_vision_coreml_model(vision_wrapper, image_size, float(rescale_factor))
    vision_path = str(output_dir / VISION_OUTPUT_NAME)
    vision_mlmodel.save(vision_path)
    print(f"Saved vision model: {vision_path}")

    print("Converting text model to CoreML (FP16 mlprogram)...")
    text_mlmodel = build_text_coreml_model(text_wrapper, seq_len)
    text_path = str(output_dir / TEXT_OUTPUT_NAME)
    text_mlmodel.save(text_path)
    print(f"Saved text model: {text_path}")

    if args.skip_validate:
        print("Validation skipped (--skip-validate).")
    else:
        validate_export(
            model=model,
            processor=processor,
            model_id=args.model_id,
            vision_model_path=vision_path,
            text_model_path=text_path,
            image_size=image_size,
            seq_len=seq_len,
            pad_token_id=pad_token_id,
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
