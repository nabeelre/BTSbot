"""Score every alert in a split with a trained BTSbot model and write per-alert predictions and embeddings.

The outputs are feature sources for train_tfm.py, two CSVs per split:
    preds/<run_name>_<split>.csv      candid, raw_preds
    preds/<run_name>_emb_<split>.csv  candid, emb_0 ... emb_<k-1>: the penultimate layer (the input to the final
                                      Linear(., 1), e.g. 16-d after fc2 + GELU for ConvNeXt)

Usage (from btsbot/, like train.py):
    python predict.py models/ConvNeXt_v11_N100_cuda/divine-sweep-47/
    python predict.py models/ConvNeXt_v11_N100_cuda/divine-sweep-47/ --splits val test
    python predict.py models/ConvNeXt_v11_N100_cuda/divine-sweep-47/ --splits val --data-dir /Volumes/NR-SSD/BTSbot/data/v11

run_name is the model dir's name. Use them in a TFM config as
    "feature_sources": [{"path": "preds/divine-sweep-47_{split}.csv", "columns": ["raw_preds"], "prefix": "img_"}]
    "feature_sources": [{"path": "preds/divine-sweep-47_emb_{split}.csv", "columns": "emb_*", "prefix": "img_"}]

Works with any model dir holding report.json (with "train_config") and best_model.pth: image-only (ConvNeXt,
MaxViT, um_cnn), metadata-only (um_nn) or multi-modal (mm_*, frozen_fusion).
"""
import argparse
import inspect
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import architectures

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")


def load_model(model_dir, checkpoint):
    with open(os.path.join(model_dir, "report.json")) as f:
        config = json.load(f)["train_config"]
    # The checkpoint overwrites every weight, so skip downloading pretrained backbones and, for frozen_fusion,
    # loading the branch checkpoints from their original dirs.
    model = getattr(architectures, config["model_name"])({**config, "pretrained": 0, "skip_load_state": 1})
    model.load_state_dict(torch.load(os.path.join(model_dir, checkpoint), map_location="cpu"))
    return model.to(device).eval(), config


def inputs_needed(model, config):
    """(use_images, use_metadata) for this model."""
    if "image_input" in inspect.signature(model.forward).parameters:  # mm_*, frozen_fusion
        return True, True
    metadata_only = config["model_name"] == "um_nn"
    return not metadata_only, metadata_only


def predict(model, config, cand, triplets, batch_size):
    """Returns (scores, penultimate-layer embeddings), the latter being the input to the final Linear(., 1)."""
    use_images, use_meta = inputs_needed(model, config)
    multi_modal = use_images and use_meta
    meta = cand[config["metadata_cols"]].to_numpy(np.float32) if use_meta else None

    output_layer = [m for m in model.modules() if isinstance(m, torch.nn.Linear) and m.out_features == 1][-1]
    embs = []
    hook = output_layer.register_forward_hook(lambda mod, inp, out: embs.append(inp[0].cpu().numpy()))

    preds, t0 = [], time.time()
    n_batches = -(-len(cand) // batch_size)
    report_every = max(1, n_batches // 50)  # ~2% steps
    with torch.no_grad():
        for start in range(0, len(cand), batch_size):
            sl = slice(start, start + batch_size)
            if use_images:  # (N, H, W, C) on disk -> (N, C, H, W) float32, one batch at a time
                img = torch.from_numpy(np.ascontiguousarray(
                    np.transpose(triplets[sl], (0, 3, 1, 2)), dtype=np.float32)).to(device)
            if use_meta:
                md = torch.from_numpy(meta[sl]).to(device)
            if multi_modal:
                logits = model(image_input=img, metadata_input=md)
            else:
                logits = model(input_data=img if use_images else md)
            preds.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())
            if len(preds) % report_every == 0 or len(preds) == n_batches:
                done, elapsed = min(start + batch_size, len(cand)), time.time() - t0
                print(f"  {done:,}/{len(cand):,} ({100 * done / len(cand):.0f}%), {done / elapsed:,.0f} alerts/s, "
                      f"ETA {(len(cand) - done) * elapsed / done / 60:.1f} min", flush=True)
    hook.remove()
    return np.concatenate(preds), np.concatenate(embs)


def main():
    parser = argparse.ArgumentParser(description="Write per-alert BTSbot predictions for train_tfm.py")
    parser.add_argument("model_dir", help="dir with report.json and the checkpoint")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default="best_model.pth")
    parser.add_argument("--data-dir", default=None,
                        help="dir holding <split>_cand_*.csv and <split>_triplets_*.npy; default data/ "
                             "(or the /scratch dir on the cluster if present, as train.py)")
    parser.add_argument("--out-dir", default="preds")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    model, config = load_model(args.model_dir, args.checkpoint)
    run_name = os.path.basename(os.path.normpath(args.model_dir))
    version, N_str = config["train_data_version"], f"_N{config.get('N_max', 100)}"
    data_dir = args.data_dir
    if data_dir is None:
        cluster_dir = f"/scratch/nrc5378/BTSbot_training_{version}/"
        data_dir = (cluster_dir if sys.platform != "darwin" and os.path.exists(cluster_dir) else "") + "data/"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{run_name} ({config['model_name']}, {version}) on {device}")

    use_images, _ = inputs_needed(model, config)
    for split in args.splits:
        cand = pd.read_csv(os.path.join(data_dir, f"{split}_cand_{version}{N_str}.csv"))
        triplets = None
        if use_images:
            trips_path = os.path.join(data_dir, f"{split}_triplets_{version}{N_str}.npy")
            if not os.path.exists(trips_path):
                raise FileNotFoundError(f"{trips_path} not found (set --data-dir?)")
            triplets = np.load(trips_path, mmap_mode="r")  # memory-mapped: train triplets needn't fit in RAM
            if len(triplets) != len(cand):
                raise ValueError(f"{trips_path} has {len(triplets):,} rows but the cand file has {len(cand):,}")

        print(f"{split}: {len(cand):,} alerts")
        raw_preds, emb = predict(model, config, cand, triplets, args.batch_size)

        out, emb_out = f"{args.out_dir}/{run_name}_{split}.csv", f"{args.out_dir}/{run_name}_emb_{split}.csv"
        pd.DataFrame({"candid": cand.candid, "raw_preds": raw_preds}).to_csv(out, index=False)
        pd.concat([cand[["candid"]], pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])],
                  axis=1).to_csv(emb_out, index=False)
        print(f"  ROC AUC {roc_auc_score(cand.label, raw_preds):.4f} -> {out}; {emb.shape[1]}-d embeddings -> {emb_out}")
        if split == "train":
            print("  note: train-split scores are in-sample (the model was trained on these alerts)")


if __name__ == "__main__":
    main()
