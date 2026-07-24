from __future__ import annotations

import numpy as np
import pandas as pd

from src.econometrics_eda_v2.econometric_feature_ablation import AblationSpec, compute_feature_ablation


def test_ablation_reports_with_without_fit_change():
    rng = np.random.default_rng(7)
    rows = 240
    signal = rng.normal(size=rows)
    noise = rng.normal(size=rows)
    cited = (signal + rng.normal(scale=0.35, size=rows) > 0).astype(int)
    frame = pd.DataFrame(
        {
            "cited": cited,
            "prompt_id": [f"p{i % 12}" for i in range(rows)],
            "signal": signal,
            "noise": noise,
        }
    )
    spec = AblationSpec(
        model_family="test",
        model_label="Test model",
        full_formula="cited ~ signal + noise + C(prompt_id)",
        features=(("signal", "Signal", "signal"), ("noise", "Noise", "noise")),
    )

    result = compute_feature_ablation(frame, (spec,)).set_index("feature")

    assert set(result.index) == {"signal", "noise"}
    assert result.loc["signal", "r_squared_gain"] > result.loc["noise", "r_squared_gain"]
    assert result.loc["signal", "rmse_reduction"] > 0
    assert result["leakage_guardrail_passed"].all()
    assert result["n_obs"].eq(rows).all()
