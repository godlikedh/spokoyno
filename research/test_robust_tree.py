"""Check portable tree export against sklearn and the JavaScript scorer."""

import json
import subprocess
import unittest
from pathlib import Path

import numpy as np
from robust_tree import export, window_probabilities
from sklearn.ensemble import HistGradientBoostingClassifier


class PortableTreeTests(unittest.TestCase):
    def test_python_and_javascript_preserve_fitted_predictions(self):
        rng = np.random.default_rng(24)
        x = rng.uniform(-2, 2, (400, 3))
        y = ((x[:, 0] > 0.3) & (x[:, 1] < -0.1)) | (x[:, 2] > 1.5)
        estimator = HistGradientBoostingClassifier(
            max_iter=15,
            max_leaf_nodes=7,
            min_samples_leaf=8,
            early_stopping=False,
            random_state=42,
        ).fit(x, y)
        model = export(estimator) | {"audio": False, "power": 8}
        expected = estimator.predict_proba(x)[:, 1]
        np.testing.assert_allclose(
            window_probabilities(model, x), expected, atol=1e-12, rtol=0
        )
        transform = {
            "physical": {"mean": [0, 0, 0], "scale": [1, 1, 1], "median": [0, 0, 0]}
        }
        payload = json.dumps({"model": model, "transform": transform, "x": x.tolist()})
        script = "const fs=require('node:fs'),{predict}=require('./research/robust_model.cjs');const d=JSON.parse(fs.readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(d.x.map(row=>predict(d.transform,d.model,[row],null).score)));"
        result = subprocess.run(
            ["node", "-e", script],
            input=payload,
            text=True,
            capture_output=True,
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
            timeout=30,
        )
        np.testing.assert_allclose(
            json.loads(result.stdout), expected, atol=1e-12, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
