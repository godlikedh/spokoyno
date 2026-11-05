"""Waveform inference for frozen acoustic models."""

import numpy as np

from training.features.acoustic_features import extract
from training.features.extract_audio_context import context_audio
from training.models.improve_acoustics import predict_model


def encode(samples, features, windows, encoder):
    patches, _ = context_audio(samples, features)
    embedded = encoder(patches)
    return np.asarray(
        [
            np.concatenate([np.zeros(1024), embedded[9], np.zeros(1024)])
            if point["slot"] == 3
            else embedded[point["slot"] * 3 : point["slot"] * 3 + 3].ravel()
            for point in windows
        ]
    )


def score(models, samples, original, embeddings):
    vectors, points = extract(samples, original)
    names = sorted(vectors[0])
    physical = np.asarray([[v[n] for n in names] for v in vectors])
    result = {}
    for name, model in models.items():
        size = len(points) if model.get("expanded_proposals") else len(original)
        audio = np.pad(embeddings, ((0, size - len(original)), (0, 0)))
        scores, windows = predict_model(
            model, physical[:size], audio, names, np.zeros(size, dtype=int), 1
        )
        chosen = int(np.argmax(windows))
        result[name] = {
            "score": float(scores[0]),
            "event_at_s": points[chosen]["start_s"],
            "top_windows": [
                {"start_s": points[i]["start_s"], "score": float(windows[i])}
                for i in np.argsort(-windows)[:3]
            ],
        }
    return result
