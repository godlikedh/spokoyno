"""Portable numerical inference for bounded-depth histogram boosting heads."""

import numpy as np
from scipy.special import expit


def window_probabilities(model, x):
    if model.get("kind", "linear") == "linear":
        return expit(x @ model["coefficient"] + model["intercept"])
    if model["kind"] != "histogram-boosting":
        raise ValueError("unsupported model head")
    raw = np.full(len(x), model["intercept"], dtype=float)
    for tree in model["trees"]:
        stack = [(0, np.arange(len(x)))]
        while stack:
            index, rows = stack.pop()
            if not len(rows):
                continue
            node = tree[index]
            if node["leaf"]:
                raw[rows] += node["value"]
            else:
                left = x[rows, node["feature"]] <= node["threshold"]
                stack.append((node["left"], rows[left]))
                stack.append((node["right"], rows[~left]))
    return expit(raw)


def export(estimator):
    trees = []
    for stage in estimator._predictors:
        nodes = stage[0].nodes
        if np.any(nodes["is_categorical"]):
            raise ValueError("categorical tree export is unsupported")
        trees.append(
            [
                {
                    "leaf": bool(n["is_leaf"]),
                    "value": float(n["value"]),
                    "feature": int(n["feature_idx"]),
                    "threshold": float(n["num_threshold"]),
                    "left": int(n["left"]),
                    "right": int(n["right"]),
                }
                for n in nodes
            ]
        )
    return {
        "kind": "histogram-boosting",
        "intercept": float(estimator._baseline_prediction.ravel()[0]),
        "trees": trees,
    }
