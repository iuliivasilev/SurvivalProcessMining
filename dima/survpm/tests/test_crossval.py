import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("pm4py")
from survpm import datasets  # noqa: E402
from survpm.metrics import crossval  # noqa: E402


def median_days(model, fold):
    times = model.completion_times(500, fold.seed)
    return float(np.median(times[np.isfinite(times)]))


def test_set_metrics_adds_replaces_removes():
    raw, censoring = datasets.generate_ad(200, seed=3)
    cv = crossval.CrossValidator("A", "D", k=2, n_sim=200)
    cv.set_metrics(level1=None, level2=None, level3=None, median=crossval.per_model(median_days),
                   share=lambda fold: dict(censored_share=1 - fold.test_cases["event"].mean()))
    assert list(cv.metrics) == ["level4", "median", "share"]
    results = cv.run(raw, censoring["end"], censoring["reason"], progress=None)
    assert set(results["median"]["model"]) == {"naive", "censor_aware"}
    assert len(results["median"]) == 4 and results["median"]["median_days"].gt(0).all()
    assert len(results["share"]) == 2  # одна строка на фолд, без модели
    assert crossval.summarize(results["share"]).shape == (2, 1)
    with pytest.raises(TypeError):
        cv.set_metrics(bad=42)
