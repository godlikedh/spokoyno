"""Extend extracted acoustic windows with the frozen context-feature schema."""

import numpy as np
from context_features import extract


def physical_features(extracted, embeddings, transform):
    specification = transform.get("context_features")
    if specification is None:
        return extracted["physical"], extracted["names"]
    if extracted["names"] != specification["base_names"]:
        raise ValueError("base acoustic schema differs from context model")
    first = extracted["metadata"]["trimmed_samples"] / 16000
    positions = [w["start_s"] - first for w in extracted["windows"]]
    extra, names = extract(embeddings, positions, specification["head"])
    if names != specification["names"]:
        raise ValueError("context schema differs from model")
    return np.c_[extracted["physical"], extra], extracted["names"] + names
