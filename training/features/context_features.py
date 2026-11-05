"""Fixed temporal/semantic measurements from frozen before/event/after embeddings.

No corpus labels, source identity, absolute audio position, or fitted PCA enter
these measurements. Category groups are fixed from the pinned AudioSet ontology.
"""

import numpy as np
from scipy.special import expit

GROUPS = {
    "speech": [0],
    "music": [132],
    "vocal_alarm": [6, 7, 8, 9, 10, 11, 19, 21, 22, 33, 34, 39, 40],
    "singing": [24],
    "laughter": [13],
    "noise": [507, 511],
    "silence": [494],
}


def cosine_distance(a, b):
    an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    denominator = an * bn
    similarity = np.sum(a * b, axis=1) / np.maximum(denominator, 1e-12)
    return np.where((an < 1e-12) & (bn < 1e-12), 0.0, 1 - np.clip(similarity, -1, 1))


def extract(embeddings, content_positions, head):
    values = np.asarray(embeddings, dtype=float).reshape(-1, 3, 1024)
    before, event, after = values[:, 0], values[:, 1], values[:, 2]
    norms = np.linalg.norm(values, axis=2)
    result = {
        "ctx_boundary_opening": (
            np.rint(np.asarray(content_positions) / 0.05) <= 20
        ).astype(float),
        "ctx_embedding_before_event_distance": cosine_distance(before, event),
        "ctx_embedding_event_after_distance": cosine_distance(event, after),
        "ctx_embedding_before_after_distance": cosine_distance(before, after),
        "ctx_embedding_context_event_distance": cosine_distance(
            (before + after) / 2, event
        ),
        "ctx_embedding_event_before_log_norm": np.log(
            (norms[:, 1] + 1e-6) / (norms[:, 0] + 1e-6)
        ),
        "ctx_embedding_event_after_log_norm": np.log(
            (norms[:, 1] + 1e-6) / (norms[:, 2] + 1e-6)
        ),
    }
    scores = expit(values @ np.asarray(head["kernel"]) + np.asarray(head["bias"]))
    positions = {index: i for i, index in enumerate(head["indices"])}
    groups = {}
    for name, indices in GROUPS.items():
        grouped = np.max(scores[:, :, [positions[i] for i in indices]], axis=2)
        groups[name] = grouped
        result[f"ctx_semantic_{name}_event"] = grouped[:, 1]
        result[f"ctx_semantic_{name}_rise"] = grouped[:, 1] - grouped[:, 0]
        result[f"ctx_semantic_{name}_fall"] = grouped[:, 1] - grouped[:, 2]
    for name in ("vocal_alarm", "noise"):
        result[f"ctx_semantic_{name}_introduced"] = groups[name][:, 1] * (
            1 - groups[name][:, 0]
        )
    for name in ("music", "singing"):
        result[f"ctx_semantic_{name}_persistent"] = np.minimum(
            groups[name][:, 0], groups[name][:, 1]
        )
    names = sorted(result)
    return np.column_stack([result[name] for name in names]), names


def relative_positions(data):
    start = np.asarray([w["start_s"] for w in data["windows"]])
    first = np.asarray(
        [r["canonical"]["trimmed_samples"] / 16000 for r in data["records"]]
    )
    return start - first[data["owners"]]
