import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("pm4py")
from survpm import datasets  # noqa: E402
from survpm.metrics import crossval  # noqa: E402


def median_days(model, fold):
    times = model.completion_times(500, fold.seed)
    return float(np.median(times[np.isfinite(times)]))


def test_metrics_are_added_explicitly():
    cv = crossval.CrossValidator("A", "D")
    assert cv.metrics == {}
    cv.add_metrics("km_sup", "level4")
    assert list(cv.metrics) == ["km_sup", "unseen_share", "nll_per_interval", "nll_per_trace"]
    cv.add_metrics("all").remove_metrics("level2", "ibs")
    assert set(cv.metrics) == set(crossval.METRICS) - set(crossval.LEVELS["level2"]) - {"ibs"}
    with pytest.raises(ValueError):
        cv.add_metrics("no_such_metric")
    with pytest.raises(ValueError):
        cv.add_metrics(level3=lambda fold: {})
    with pytest.raises(TypeError):
        cv.add_metrics(bad=42)
    with pytest.raises(ValueError):
        crossval.CrossValidator("A", "D").run(None)


def test_selected_and_custom_metrics():
    raw, censoring = datasets.generate_ad(200, seed=3)
    cv = crossval.CrossValidator("A", "D", k=2, n_sim=200)
    cv.add_metrics("nll_per_interval", "km_sup", median=crossval.per_model(median_days),
                   share=lambda fold: dict(censored_share=1 - fold.test_cases["event"].mean()))
    results = cv.run(raw, censoring["end"], censoring["reason"], progress=None)
    assert list(results) == ["level3", "level4", "median", "share"]
    assert list(results["level3"].columns) == ["fold", "model", "km_sup"]
    assert list(results["level4"].columns) == ["fold", "model", "nll_per_interval"]
    assert set(results["median"]["model"]) == {"naive", "censor_aware"}
    assert len(results["median"]) == 4 and results["median"]["median_days"].gt(0).all()
    assert len(results["share"]) == 2  # одна строка на фолд, без модели
    assert crossval.summarize(results["share"]).shape == (2, 1)
