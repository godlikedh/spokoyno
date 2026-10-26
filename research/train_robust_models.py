#!/usr/bin/env python3
"""Fixed candidate budget, source-balanced augmentation and nested real-source CV.

Evaluation-only codecs cannot enter training or inner selection. The primary
candidate and all hyperparameters are fixed before inspecting this run's scores.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from extract_audio_context import sha256
from improve_events import logistic, training_key
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from train_events import metrics
from train_models import atomic_json

PRIMARY = "consistent-hybrid"
CONFIGS = {
    PRIMARY: {"audio": True, "augment": True, "power": 8, "consistency": 1.0},
    "augmented-hybrid": {
        "audio": True,
        "augment": True,
        "power": 8,
        "consistency": 0.0,
    },
    "consistent-physical": {
        "audio": False,
        "augment": True,
        "power": 8,
        "consistency": 1.0,
    },
    "augmented-physical": {
        "audio": False,
        "augment": True,
        "power": 8,
        "consistency": 0.0,
    },
    "unaugmented-hybrid": {
        "audio": True,
        "augment": False,
        "power": 8,
        "consistency": 0.0,
    },
    "augmented-softmax": {
        "audio": True,
        "augment": True,
        "power": 2,
        "consistency": 0.0,
    },
}


def load_data(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"training matrix changed: {name}")
    arrays = {
        name: np.load(directory / (name + ".npy"), mmap_mode="r", allow_pickle=False)
        for name in ("physical", "audio", "targets", "owners")
    }
    views = {"worst-training-view": {}}
    for index, record in enumerate(manifest["records"]):
        if record["kind"] != "real":
            continue
        views.setdefault(record["variant"], {}).setdefault(record["source"], []).append(
            index
        )
        if record["split"] in ("base", "train"):
            views["worst-training-view"].setdefault(record["source"], []).append(index)
    return manifest | arrays | {"view_indices": views}


def eligible_records(records, training, augmentation=True):
    allowed = set(training)
    return np.asarray(
        [
            r["split"] != "evaluation"
            and (augmentation or r["split"] == "base")
            and r["label"] in ("positive", "negative")
            and set(r["parents"]) <= allowed
            for r in records
        ]
    )


def group_weights(records):
    """Each real source has equal within-class mass, regardless of copy count."""
    result = np.zeros(len(records))
    for kind, total in (("real", 1.0), ("synthetic", 0.2)):
        for label in ("positive", "negative"):
            selected = np.asarray(
                [
                    i
                    for i, r in enumerate(records)
                    if r["kind"] == kind and r["label"] == label
                ],
                dtype=int,
            )
            sources = np.asarray([records[i]["source"] for i in selected])
            unique, inverse, count = np.unique(
                sources, return_inverse=True, return_counts=True
            )
            if len(unique):
                result[selected] = total / 2 / len(unique) / count[inverse]
    return result


def pool(probabilities, owners, count, power=8):
    p = np.maximum(probabilities, 1e-15)
    denominator = np.bincount(owners, weights=p ** (power - 1), minlength=count)
    numerator = np.bincount(owners, weights=p**power, minlength=count)
    return numerator / np.maximum(denominator, 1e-300), denominator


def objective(
    theta,
    x,
    owners,
    labels,
    weights,
    pairs,
    strong,
    targets,
    strong_weights,
    power,
    consistency,
    penalty,
):
    z = x @ theta[:-1] + theta[-1]
    p = expit(z)
    bag, denominator = pool(p, owners, len(labels), power)
    bag = np.clip(bag, 1e-12, 1 - 1e-12)
    loss = -np.sum(weights * (labels * np.log(bag) + (1 - labels) * np.log1p(-bag)))
    dbag = weights * (bag - labels) / (bag * (1 - bag))
    if consistency:
        total = np.bincount(pairs, weights=weights)
        mean = np.bincount(pairs, weights=weights * bag) / np.maximum(total, 1e-20)
        difference = bag - mean[pairs]
        loss += consistency * np.sum(weights * difference**2)
        dbag += 2 * consistency * weights * difference
    derivative = (
        np.maximum(p, 1e-15) ** (power - 2)
        * (power * p - (power - 1) * bag[owners])
        / np.maximum(denominator[owners], 1e-300)
    )
    dz = dbag[owners] * derivative * p * (1 - p)
    loss += np.sum(strong_weights * (np.logaddexp(0, z[strong]) - targets * z[strong]))
    dz[strong] += strong_weights * (p[strong] - targets)
    loss += 0.5 * penalty * np.dot(theta, theta)
    gradient = np.r_[x.T @ dz, dz.sum()] + penalty * theta
    return float(loss), gradient


def prepare(model, physical, audio):
    spec = model["physical"]
    filled = np.where(np.isnan(physical), spec["median"], physical)
    result = (filled - spec["mean"]) / spec["scale"]
    if audio is not None:
        spec = model["audio"]
        projected = (audio - spec["pca_mean"]) @ np.asarray(spec["components"]).T
        result = np.c_[result, (projected - spec["mean"]) / spec["scale"]]
    return np.clip(result, -8.0, 8.0)


def fitted_transform(data, training):
    records = data["records"]
    eligible = eligible_records(records, training, False)
    indices = np.flatnonzero(eligible[data["owners"]])
    selected = np.flatnonzero(eligible)
    by_record = np.zeros(len(records))
    by_record[selected] = group_weights([records[i] for i in selected])
    counts = np.bincount(data["owners"][indices], minlength=len(records))
    weights = by_record[data["owners"][indices]] / counts[data["owners"][indices]]
    median = SimpleImputer(strategy="median", keep_empty_features=True).fit(
        data["physical"][indices]
    )
    values = median.transform(data["physical"][indices])
    scaler = StandardScaler().fit(values, sample_weight=weights)
    # Bounded, deterministic PCA sample of original TRAINING windows only.
    sample = np.sort(
        np.random.default_rng(20261026).choice(
            indices, min(8000, len(indices)), replace=False
        )
    )
    pca = PCA(n_components=8, svd_solver="randomized", random_state=42).fit(
        np.asarray(data["audio"][sample], dtype=float)
    )
    projection = np.empty((len(data["owners"]), 8))
    for lo in range(0, len(projection), 4096):
        projection[lo : lo + 4096] = pca.transform(
            np.asarray(data["audio"][lo : lo + 4096], dtype=float)
        )
    audio_scaler = StandardScaler().fit(projection[indices], sample_weight=weights)
    model = {
        "physical": {
            "median": median.statistics_.tolist(),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "audio": {
            "pca_mean": pca.mean_.tolist(),
            "components": pca.components_.tolist(),
            "mean": audio_scaler.mean_.tolist(),
            "scale": audio_scaler.scale_.tolist(),
        },
        "names": data["names"],
        "clip_standardized": 8.0,
    }
    x = np.c_[
        scaler.transform(median.transform(data["physical"])),
        audio_scaler.transform(projection),
    ]
    return model, np.clip(x, -8.0, 8.0)


def fit_one(data, x, training, config):
    eligible = eligible_records(data["records"], training, config["augment"])
    selected = np.flatnonzero(eligible)
    records = [data["records"][i] for i in selected]
    window_ids = np.flatnonzero(eligible[data["owners"]])
    mapping = np.full(len(data["records"]), -1, dtype=int)
    mapping[selected] = np.arange(len(selected))
    owners = mapping[data["owners"][window_ids]]
    dimensions = x.shape[1] if config["audio"] else len(data["names"])
    values = x[window_ids, :dimensions]
    labels = np.asarray([r["label"] == "positive" for r in records], dtype=np.int8)
    weights = group_weights(records)
    pairs = np.unique([r["pair"] for r in records], return_inverse=True)[1]
    strong = np.flatnonzero(data["targets"][window_ids] >= 0)
    targets = data["targets"][window_ids][strong]
    counts = np.bincount(owners[strong], minlength=len(records))
    strong_weights = weights[owners[strong]] / counts[owners[strong]]
    for label in (0, 1):
        mask = targets == label
        strong_weights[mask] *= 0.05 / max(sum(strong_weights[mask]), 1e-20)
    initial = logistic(values[strong], targets, strong_weights * len(training) / 0.1)
    theta = np.r_[initial.coef_[0], initial.intercept_[0]]
    optimized = minimize(
        objective,
        theta,
        args=(
            values,
            owners,
            labels,
            weights,
            pairs,
            strong,
            targets,
            strong_weights,
            config["power"],
            config["consistency"],
            1 / (0.05 * len(training)),
        ),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 400, "ftol": 1e-9, "gtol": 1e-6},
    )
    if not optimized.success:
        raise ValueError(f"optimization failed: {optimized.message}")
    model = config | {
        "coefficient": optimized.x[:-1].tolist(),
        "intercept": float(optimized.x[-1]),
        "optimizer_iterations": optimized.nit,
        "training_records": selected.tolist(),
        "training_windows": len(window_ids),
    }
    scores = pool(
        expit(x[:, :dimensions] @ optimized.x[:-1] + optimized.x[-1]),
        data["owners"],
        len(data["records"]),
        config["power"],
    )[0]
    if not np.all(np.isfinite(scores)):
        raise ValueError("nonfinite prediction")
    return model, scores


def run_fit(data, training, output):
    key = training_key(training)
    directory = output / "fits" / key
    if (directory / "manifest.json").exists():
        meta = json.loads((directory / "manifest.json").read_text())
        for file, expected in meta["files"].items():
            if sha256(directory / file) != expected:
                raise ValueError("resumed fit checksum mismatch")
        if meta["training_indices"] != training:
            raise ValueError("resumed fit belongs to another partition")
        with np.load(directory / "scores.npz", allow_pickle=False) as saved:
            return key, {n: saved[n] for n in CONFIGS}
    directory.mkdir(parents=True, exist_ok=True)
    transform, x = fitted_transform(data, training)
    models, scores = {}, {}
    for name, config in CONFIGS.items():
        models[name], scores[name] = fit_one(data, x, training, config)
    atomic_json(directory / "transform.json", transform)
    atomic_json(directory / "models.json", models)
    np.savez_compressed(directory / "scores.npz", **scores)
    atomic_json(
        directory / "manifest.json",
        {
            "training_indices": training,
            "training_hashes": [data["clips"][i]["audio_sha256"] for i in training],
            "files": {
                f: sha256(directory / f)
                for f in ("transform.json", "models.json", "scores.npz")
            },
        },
    )
    print(f"trained {key}: {len(training)} original groups", flush=True)
    return key, scores


def source_views(data, scores, sources, variant="original"):
    rows = []
    for source in sources:
        indices = data["view_indices"][variant].get(source, [])
        if not indices:
            raise ValueError("source lacks requested evaluation view")
        clip = data["clips"][source]
        values = scores[indices]
        score = (
            (min(values) if clip["label"] == "positive" else max(values))
            if variant == "worst-training-view"
            else values[0]
        )
        rows.append(
            {
                "path": clip["path"],
                "audio_sha256": clip["audio_sha256"],
                "label": clip["label"],
                "fold_thread": clip["fold_thread"],
                "score": float(score),
            }
        )
    return rows


def select_model(data, scores, sources):
    # Fixed thresholds only. Never sacrifice a primary-caught positive to claim
    # a lower false-alert count. Evaluation-only codecs are inaccessible here.
    base = {
        v: source_views(data, scores[PRIMARY], sources, v)
        for v in ("original", "worst-training-view")
    }
    choices = []
    for order, name in enumerate(CONFIGS):
        current = {v: source_views(data, scores[name], sources, v) for v in base}
        lost = any(
            a["label"] == "positive"
            and a["score"] >= threshold
            and b["score"] < threshold
            for v in base
            for threshold in (0.6, 0.8)
            for a, b in zip(base[v], current[v], strict=True)
        )
        if lost:
            continue
        false = sum(
            r["label"] == "negative" and r["score"] >= 0.8
            for rows in current.values()
            for r in rows
        )
        yellow = sum(
            r["label"] == "negative" and r["score"] >= 0.6
            for rows in current.values()
            for r in rows
        )
        choices.append((false, yellow, order, name))
    return min(choices)[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("research/artifacts/improve-events-20261025/plan.json"),
    )
    parser.add_argument("--jobs", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    data = load_data(args.data_dir)
    folds = json.loads(args.splits.read_text())["folds"]
    known = sorted(i for f in folds for i in f["test"])
    sets = {
        training_key(f["train"]): f["train"]
        for outer in folds
        for f in [outer, *outer["inner"]]
    }
    sets[training_key(known)] = known
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_sha256": sha256(Path(__file__)),
        "dataset_sha256": sha256(args.data_dir / "manifest.json"),
        "split_plan_sha256": sha256(args.splits),
        "candidates": CONFIGS,
        "primary": PRIMARY,
        "folds": folds,
        "selection": "At fixed .6/.8, preserve every primary-caught inner positive in original and worst training-codec view; minimize total red negative counts across those views, then any-warning counts, then declaration order. Never use evaluation-only codec profiles.",
        "limitations": "No derivative is a new independent example. Development corpus informed earlier research. Known families only. This run cannot establish absence of overfitting without new original recordings.",
    }
    if (args.output_dir / "plan.json").exists():
        previous = json.loads((args.output_dir / "plan.json").read_text())
        if any(
            previous[k] != plan[k]
            for k in (
                "source_sha256",
                "dataset_sha256",
                "split_plan_sha256",
                "candidates",
            )
        ):
            raise ValueError("cannot resume changed experiment")
    else:
        atomic_json(args.output_dir / "plan.json", plan)
    print(
        f"declared {len(CONFIGS)} configurations on {len(sets)} partitions", flush=True
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(run_fit, data, train, args.output_dir)
            for train in sets.values()
        ]
        fitted = dict(f.result() for f in concurrent.futures.as_completed(futures))
    fixed = {n: np.full(len(data["records"]), np.nan) for n in CONFIGS}
    nested = np.full(len(data["records"]), np.nan)
    decisions = []
    for outer in folds:
        inner_scores = {n: np.full(len(data["records"]), np.nan) for n in CONFIGS}
        for inner in outer["inner"]:
            testing = set(inner["test"])
            mask = np.asarray(
                [
                    r["kind"] == "real" and r["source"] in testing
                    for r in data["records"]
                ]
            )
            for name in CONFIGS:
                inner_scores[name][mask] = fitted[training_key(inner["train"])][name][
                    mask
                ]
        chosen = select_model(data, inner_scores, outer["train"])
        scores = fitted[training_key(outer["train"])]
        testing = set(outer["test"])
        mask = np.asarray(
            [r["kind"] == "real" and r["source"] in testing for r in data["records"]]
        )
        for name in CONFIGS:
            fixed[name][mask] = scores[name][mask]
        nested[mask] = scores[chosen][mask]
        decisions.append({"thread": outer["thread"], "chosen": chosen})
    outputs = fixed | {"nested-selection": nested}
    variants = ("original", "worst-training-view", "aac80", "opus48", "mp3_128")
    predictions = {
        n: {v: source_views(data, score, known, v) for v in variants}
        for n, score in outputs.items()
    }
    results = {
        "plan_sha256": sha256(args.output_dir / "plan.json"),
        "metrics": {
            n: {v: metrics(rows) for v, rows in views.items()}
            for n, views in predictions.items()
        },
        "predictions": predictions,
        "selection_decisions": decisions,
        "final_model": select_model(data, fixed, known),
        "final_fit_key": training_key(known),
    }
    atomic_json(args.output_dir / "results.json", results)
    np.savez_compressed(args.output_dir / "heldout-scores.npz", **outputs)
    print(json.dumps(results["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
