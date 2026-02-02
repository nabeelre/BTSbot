"""
Convert a BTSbot .pth checkpoint to ONNX.
Usage: python -m btsbot.to_onnx <model_dir> [--output path.onnx] [--verify]
"""
import argparse
import json
import os.path as path
import sys

import numpy as np
import pandas as pd
import torch

# Allow running from repo root or btsbot/
sys.path.insert(0, path.join(path.dirname(__file__), ".."))
from btsbot import architectures

EXAMPLE_DATA_DIR = path.join(path.dirname(__file__), "example_data")


def load_config(model_dir: str) -> dict:
    report_path = path.join(model_dir, "report.json")
    with open(report_path) as f:
        return json.load(f)["train_config"]


def load_model(model_dir: str, config: dict) -> torch.nn.Module:
    model_cls = getattr(architectures, config["model_name"])
    model = model_cls(config)
    state = torch.load(path.join(model_dir, "best_model.pth"), map_location="cpu")
    # Handle checkpoint saved under DataParallel
    if next(iter(state.keys())).startswith("module."):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _load_example_inputs(config: dict):
    """Load one sample from example_data/. Returns (img_tensor, meta_tensor)."""
    triplets = np.load(path.join(EXAMPLE_DATA_DIR, "usage_triplets.npy")).astype(np.float32)
    triplets = np.transpose(triplets, (0, 3, 1, 2))
    img = torch.from_numpy(triplets[0:1].copy())

    cand = pd.read_csv(path.join(EXAMPLE_DATA_DIR, "usage_candidates.csv"))
    meta_cols = config.get("metadata_cols", [])
    if meta_cols:
        cand = cand.reindex(columns=meta_cols, fill_value=0.0)
        meta = torch.from_numpy(cand.iloc[0:1].values.astype(np.float32))
    else:
        meta = None
    return img, meta


def _example_input_tuple(model: torch.nn.Module, config: dict):
    """Return (input_tuple, is_multimodal) using example_data/."""
    img, meta = _load_example_inputs(config)
    model_name = config["model_name"]
    metadata_only = model_name == "um_nn"
    image_only = model_name in ("MaxViT", "ConvNeXt", "um_cnn")
    if metadata_only:
        return (meta,), False
    if image_only:
        return (img,), False
    return (img, meta), True


def export_onnx(model: torch.nn.Module, config: dict, output_path: str):
    model_name = config["model_name"]
    image_only = model_name in ("MaxViT", "ConvNeXt", "um_cnn")
    metadata_only = model_name == "um_nn"

    dummy_tuple, _ = _example_input_tuple(model, config)
    if metadata_only:
        (dummy_meta,) = dummy_tuple
        torch.onnx.export(
            model,
            dummy_meta,
            output_path,
            input_names=["metadata"],
            output_names=["logits"],
            dynamic_axes={"metadata": {0: "batch"}, "logits": {0: "batch"}},
        )
    elif image_only:
        (dummy_img,) = dummy_tuple
        torch.onnx.export(
            model,
            dummy_img,
            output_path,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        )
    else:
        dummy_img, dummy_meta = dummy_tuple
        torch.onnx.export(
            model,
            (dummy_img, dummy_meta),
            output_path,
            input_names=["image", "metadata"],
            output_names=["logits"],
            dynamic_axes={
                "image": {0: "batch"},
                "metadata": {0: "batch"},
                "logits": {0: "batch"},
            },
        )


def verify_pth_vs_onnx(model_dir: str, config: dict, onnx_path: str) -> bool:
    """Run same inputs through .pth and .onnx; return True if outputs match closely."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("Verify skipped: install onnxruntime to compare .pth vs .onnx")
        return False

    model = load_model(model_dir, config)
    dummy_tuple, _ = _example_input_tuple(model, config)

    with torch.no_grad():
        out_pth = model(*dummy_tuple)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    model_name = config["model_name"]
    if model_name == "um_nn":
        feed = {"metadata": dummy_tuple[0].numpy()}
    elif model_name in ("MaxViT", "ConvNeXt", "um_cnn"):
        feed = {"image": dummy_tuple[0].numpy()}
    else:
        feed = {"image": dummy_tuple[0].numpy(), "metadata": dummy_tuple[1].numpy()}
    out_onnx = session.run(None, feed)[0]

    out_pth_np = out_pth.numpy()
    close = torch.allclose(
        torch.tensor(out_onnx), torch.tensor(out_pth_np), rtol=1e-4, atol=1e-5
    )
    max_diff = float(torch.max(torch.abs(torch.tensor(out_onnx) - out_pth_np)))
    if close:
        print(f"Verify OK: .pth and .onnx outputs match (max diff {max_diff:.2e})")
    else:
        print(f"Verify FAIL: max diff {max_diff:.2e}")
    return close


def main():
    parser = argparse.ArgumentParser(description="Convert BTSbot .pth to ONNX")
    parser.add_argument("model_dir", help="Directory containing report.json and best_model.pth")
    parser.add_argument(
        "--output", "-o", help="Output .onnx path (default: <model_dir>/model.onnx)"
    )
    parser.add_argument("--verify", action="store_true", help="Compare .pth vs .onnx outputs")
    args = parser.parse_args()

    config = load_config(args.model_dir)
    model = load_model(args.model_dir, config)
    output_path = args.output or path.join(args.model_dir, "model.onnx")
    if path.isdir(output_path):
        output_path = path.join(output_path, "model.onnx")

    print(f"Loading model from {args.model_dir}")
    print(f"Exporting model to {output_path}")

    export_onnx(model, config, output_path)
    print(f"Exported to {output_path}")
    if args.verify:
        verify_pth_vs_onnx(args.model_dir, config, output_path)


if __name__ == "__main__":
    main()
