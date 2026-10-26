#!/usr/bin/env python3
"""Declared follow-up: remove near-clipping dependence and strengthen consistency.

This adaptive follow-up was motivated by the first run's AAC diagnostics. It is
not a new blind test. It retains the same complete nested source exclusions and
does not train on any evaluation-only codec profile.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import train_robust_models as trainer
from extract_audio_context import sha256
from train_models import atomic_json

PRIMARY = "codec-stable-hybrid"
CONFIGS = {
    PRIMARY: {"audio": True, "augment": True, "power": 8, "consistency": 10.0},
    "codec-stable-physical": {
        "audio": False,
        "augment": True,
        "power": 8,
        "consistency": 10.0,
    },
    "codec-stable-softmax": {
        "audio": True,
        "augment": True,
        "power": 2,
        "consistency": 10.0,
    },
    "noclip-hybrid": {"audio": True, "augment": True, "power": 8, "consistency": 1.0},
}


def main():
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    output.mkdir(parents=True, exist_ok=True)
    declaration = {
        "created_at": datetime.now(UTC).isoformat(),
        "entry_sha256": sha256(Path(__file__)),
        "trainer_sha256": sha256(Path(trainer.__file__)),
        "primary": PRIMARY,
        "candidates": CONFIGS,
        "disabled_feature": "near_clip_fraction",
        "reason": "AAC changes near-clipping count despite high waveform correlation; first-round models assigned it large positive weight. Zero this input during training so its coefficient is exactly zero, and increase source-variant consistency weight.",
        "limitation": "Adaptive follow-up on the same development corpus, motivated by first-round held-out failure analysis; not prospective validation. Candidate list and thresholds fixed before follow-up fits.",
    }
    file = output / "followup-declaration.json"
    if file.exists():
        saved = json.loads(file.read_text())
        if (
            saved["entry_sha256"] != declaration["entry_sha256"]
            or saved["candidates"] != CONFIGS
        ):
            raise ValueError("cannot resume a changed follow-up")
    else:
        atomic_json(file, declaration)
    original_fit = trainer.fit_one

    def fit(data, x, training, config):
        values = x.copy()
        index = data["names"].index("near_clip_fraction")
        values[:, index] = 0.0
        model, scores = original_fit(data, values, training, config)
        if abs(model["coefficient"][index]) > 1e-12:
            raise ValueError("disabled feature acquired a nonzero coefficient")
        model["disabled_features"] = ["near_clip_fraction"]
        # Inference may retain the published 68-column schema: its zero
        # coefficient makes clipping counts irrelevant in every runtime.
        assert np.all(np.isfinite(scores))
        return model, scores

    trainer.fit_one = fit
    trainer.CONFIGS = CONFIGS
    trainer.PRIMARY = PRIMARY
    trainer.main()


if __name__ == "__main__":
    main()
