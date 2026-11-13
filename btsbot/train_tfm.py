"""Score BTSbot alerts with a tabular foundation model (TabPFN via lazy-tfm) from metadata only.

The TFM counterpart of train.py: there's no training loop (TabPFN is in-context: `fit` only stores the context rows),
but it's configured, run and reported the same way. The context is drawn from the train split by *source* (all of a
selected source's alerts together), either as one context or as bags, each bag a separate TabPFN fit whose scores
are averaged. Evaluation uses val.diagnostic_fig, so the figure and metrics (alert-level and policy-based) are the
same as for trained models.

Usage (from btsbot/, like train.py):
    python train_tfm.py train_configs/tfm_config.json
    python train_tfm.py train_configs/tfm_config.json bag_sources=800 random_seed=3 model_version=v3.5
Trailing key=value pairs override config entries; values are parsed as JSON where possible (numbers, true/false,
null, lists) and otherwise kept as strings.

Writes to models/tfm_<model_version>_<train_data_version>_N<N_max>_<device>/<run_name>/:
    report.json                 config, per-bag history and metrics (utils.make_report, as train.py)
    <run_name>.pdf              diagnostic figure for val (and <run_name>_test.pdf with use_test_split)
    val_scores.csv              per-alert scores (and test_scores.csv)

Config keys:
    model_name          lazy backend, e.g. "tabpfn"
    model_version       e.g. "v3.5-fast" (CPU-friendly) or "v3.5"
    train_data_version  e.g. "v11"; files are data/<split>_cand_<version>_N<N_max>.csv
    N_max               e.g. 100
    metadata_cols       feature columns from the candidate CSVs, as in train.py configs (may be [])
    feature_sources     optional extra per-alert features joined on by candid, e.g. embeddings or another model's
                        scores. A list of {"path", "columns", "prefix", "key", "allow_missing"}:
                          path           CSV or parquet; "{split}" is replaced by train / val / test
                          columns        list of column names, or one wildcard pattern such as "emb_*"
                          prefix         optional, prepended to the added column names (avoids clashes)
                          key            join column, default "candid"
                          allow_missing  default false: every alert must have a row with no missing values
                        e.g. [{"path": "embeddings/mm_{split}.csv", "columns": "emb_*", "prefix": "mm_"},
                              {"path": "preds/img_{split}.csv", "columns": ["raw_preds"], "prefix": "img_"}]
                        Caution: train-split values from a model trained on that same train split are in-sample
                        (overconfident), so TabPFN would learn to trust them more than they deserve on val/test.
                        Prefer out-of-fold values for the train split.
    context_sources     null = all train sources; int = a random subset of this many sources
    bag_sources         null = one context; int = each bag is every alert of this many random sources
    n_bags              number of bags (only with bag_sources)
    n_estimators        TabPFN ensemble members per fit (per bag when bagging)
    random_seed         picks the context/bag sources; bag i is fit with random_state random_seed + i
    device              "auto" (cuda > mps > cpu), "cuda", "mps" or "cpu"
    use_test_split      also score the test split
    run_name            optional; default is built from the settings, e.g. "bag800x8_s2"
    data_base_dir       optional; default as train.py ("" on macOS, the /scratch dir on the cluster if present)
"""
import fnmatch
import gc
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

import lazy
from utils import make_report
import val

warnings.filterwarnings("ignore", category=DeprecationWarning, module="tabpfn")

BOLD, END = '\033[1m', '\033[0m'


def load_config(path, overrides):
    with open(path) as f:
        config = json.load(f)
    for item in overrides:
        key, _, value = item.partition("=")
        try:
            config[key] = json.loads(value)
        except json.JSONDecodeError:
            config[key] = value
    return config


def add_feature_sources(df, split, sources):
    """Join extra per-alert features onto `df` (keeping its row order). Returns (df, names of the added columns)."""
    added = []
    for src in sources:
        path, key, prefix = src["path"].format(split=split), src.get("key", "candid"), src.get("prefix", "")
        table = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        cols = src["columns"]
        if isinstance(cols, str):
            cols = [c for c in table.columns if fnmatch.fnmatch(c, cols) and c != key]
        absent = set(cols + [key]) - set(table.columns)
        if absent or not cols:
            raise ValueError(f"{path}: columns not found: {sorted(absent) or src['columns']}")
        if table[key].duplicated().any():
            raise ValueError(f"{path}: {table[key].duplicated().sum():,} duplicate {key} values")
        new = [prefix + c for c in cols]
        clash = set(new) & set(df.columns)
        if clash:
            raise ValueError(f"{path}: columns {sorted(clash)[:5]} already exist; set a 'prefix' for this source")
        df = df.merge(table[[key] + cols].rename(columns=dict(zip(cols, new))), on=key, how="left")
        n_missing = int(df[new].isna().any(axis=1).sum())
        if n_missing and not src.get("allow_missing", False):
            raise ValueError(f"{path}: {n_missing:,} of {len(df):,} {split} alerts have no row or missing values "
                             f"(set 'allow_missing': true to let TabPFN treat them as missing)")
        print(f"  {split}: +{len(new)} feature(s) from {path}" + (f" ({n_missing:,} alerts missing)" if n_missing else ""))
        added += new
    return df, added


def get_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def patch_lazy_for_mps():
    """lazy 0.6.0 casts TabPFN's logits to float64 before moving them to the CPU, and MPS has no float64.
    Move them to the CPU first; nothing else changes (MPS scores match CPU to ~1e-5)."""
    import lazy.models.tabpfn as lazy_tabpfn
    if getattr(lazy_tabpfn.bucket_masses, "_mps_safe", False):
        return
    original = lazy_tabpfn.bucket_masses

    def bucket_masses_mps_safe(output, *args, **kwargs):
        return original({**output, "logits": output["logits"].detach().cpu()}, *args, **kwargs)

    bucket_masses_mps_safe._mps_safe = True
    lazy_tabpfn.bucket_masses = bucket_masses_mps_safe


def free_device_memory(device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def choose_fits(train, config, rng):
    """Row labels of `train` for each TabPFN fit: one context, or one entry per bag. Selection is by source."""
    sources = train.objectId.unique()
    context = train
    if config.get("context_sources") is not None:
        if config["context_sources"] > len(sources):
            raise ValueError(f"context_sources={config['context_sources']} but train has {len(sources)} sources")
        context = train[train.objectId.isin(rng.choice(sources, config["context_sources"], replace=False))]
    if config.get("bag_sources") is None:
        return context, [context.index]
    context_sources = context.objectId.unique()
    if config["bag_sources"] > len(context_sources):
        raise ValueError(f"bag_sources={config['bag_sources']} but the context has {len(context_sources)} sources")
    bags = [context.index[context.objectId.isin(rng.choice(context_sources, config["bag_sources"], replace=False))]
            for _ in range(config["n_bags"])]
    return context, bags


def default_run_name(config):
    if config.get("bag_sources") is not None:
        base = f"bag{config['bag_sources']}x{config['n_bags']}"
    elif config.get("context_sources") is not None:
        base = f"ctx{config['context_sources']}"
    else:
        base = "ctxall"
    if config.get("n_estimators", 8) != 1:
        base += f"_e{config['n_estimators']}"
    return f"{base}_s{config['random_seed']}"


def run(config):
    device = get_device(config.get("device", "auto"))
    if device == "mps":
        patch_lazy_for_mps()

    dataset_version, N_max = config["train_data_version"], config["N_max"]
    N_str = f"_N{N_max}"
    data_base_dir = config.get("data_base_dir")
    if data_base_dir is None:  # same default as train.py
        cluster_dir = f"/scratch/nrc5378/BTSbot_training_{dataset_version}/"
        data_base_dir = cluster_dir if sys.platform != "darwin" and os.path.exists(cluster_dir) else ""
    split_path = {s: f"{data_base_dir}data/{s}_cand_{dataset_version}{N_str}.csv" for s in ("train", "val", "test")}
    splits = ["val"] + (["test"] if config.get("use_test_split", False) else [])
    metadata_cols = config["metadata_cols"]

    run_name = config.get("run_name") or default_run_name(config)
    model_dir = f"models/tfm_{config['model_version']}_{dataset_version}{N_str}_{device}/{run_name}/"
    os.makedirs(model_dir, exist_ok=True)

    train = pd.read_csv(split_path["train"])
    evals = {s: pd.read_csv(split_path[s]) for s in splits}
    for name, d in [("train", train), *evals.items()]:
        missing = set(metadata_cols + ["label", "objectId"]) - set(d.columns)
        if missing:
            raise ValueError(f"{name} split is missing columns: {sorted(missing)}")

    # Extra features (embeddings, other models' scores) joined on by candid
    sources = config.get("feature_sources") or []
    train, extra_cols = add_feature_sources(train, "train", sources)
    for s in splits:
        evals[s], _ = add_feature_sources(evals[s], s, sources)
    feature_cols = metadata_cols + extra_cols
    if not feature_cols:
        raise ValueError("no features: metadata_cols is empty and there are no feature_sources")
    print(f"{len(feature_cols)} features ({len(metadata_cols)} metadata + {len(extra_cols)} from feature_sources)")

    rng = np.random.default_rng(config["random_seed"])
    context, fits = choose_fits(train, config, rng)
    # TabPFN v3+ limits on alerts per fit: 5,000 on CPU (upstream's default), 1,000,000 on GPU
    max_fit_rows = 5_000 if device == "cpu" else 1_000_000
    largest = max(len(f) for f in fits)
    if largest > max_fit_rows:
        raise ValueError(f"a fit has {largest:,} alerts but TabPFN takes at most {max_fit_rows:,} on "
                         f"{device.upper()}; lower context_sources / bag_sources"
                         + (" or use a GPU" if device == "cpu" else ""))

    used = context.loc[np.unique(np.concatenate(fits))]
    print(f"*** {config['model_name']} {config['model_version']} on {device}: {len(fits)} fit(s) of "
          f"{min(len(f) for f in fits):,}-{largest:,} alerts, {config['n_estimators']} member(s) each; "
          f"{used.objectId.nunique():,} of {train.objectId.nunique():,} train sources used ***")
    print(f"Run: {model_dir}")

    X_ctx, y_ctx = context[feature_cols], context.label.astype(float)
    scores = {s: np.zeros(len(d)) for s, d in evals.items()}
    fit_seconds, start = [], time.time()
    for i, rows in enumerate(fits):
        t = time.time()
        model = lazy.LazyModel(config["model_name"], version=config["model_version"], device=device,
                               random_state=config["random_seed"] + i, n_estimators=config["n_estimators"])
        model.fit(X_ctx.loc[rows], y_ctx.loc[rows])
        for s, d in evals.items():
            scores[s] += model.predict(d[feature_cols], method="mean") / len(fits)
        del model
        free_device_memory(device)
        fit_seconds.append(time.time() - t)
        print(f"  fit {i + 1}/{len(fits)}: {len(rows):,} alerts, {fit_seconds[-1]:.1f}s", flush=True)

    summaries = {}
    for s, d in evals.items():
        raw_preds = np.clip(scores[s], 0, 1)  # predictive mean ~ P(BTS | metadata)
        d.assign(tfm_score=raw_preds)[["objectId", "candid", "label", "tfm_score"]].to_csv(
            f"{model_dir}/{s}_scores.csv", index=False)
        summaries[s] = val.diagnostic_fig(
            run_data={"type": config["model_name"], "raw_preds": raw_preds, "labels": d.label.to_numpy(),
                      "run_name": run_name + ("" if s == "val" else f"_{s}")},
            cand_dir=split_path[s], run_descriptor=model_dir)
        summaries[s].pop("fig", None)

    run_data = {"run_name": run_name, "fit_alerts": [len(f) for f in fits], "fit_seconds": fit_seconds,
                "feature_cols": feature_cols}
    make_report(config, f"{model_dir}/report.json", run_data, summaries["val"])
    if "test" in summaries:
        with open(f"{model_dir}/report.json") as f:
            report = json.load(f)
        report["test_summary"] = summaries["test"]
        with open(f"{model_dir}/report.json", "w") as f:
            json.dump(report, f, indent=4, default=float)

    print(BOLD + '============ Summary =============' + END)
    for s, summ in summaries.items():
        pol = summ["policy_performance"]
        print(f"{s}: AUC {summ['roc_auc']:.4f}, alert P/R {summ['alert_precision']:.4f}/{summ['alert_recall']:.4f}, "
              + ", ".join(f"{p} P/R {v['policy_precision']:.3f}/{v['policy_recall']:.3f}" for p, v in pol.items()))
    print(f"Total {(time.time() - start) / 60:.1f} min. Diagnostics at {model_dir}\n")


if __name__ == "__main__":
    run(load_config(sys.argv[1], sys.argv[2:]))
