import math

import numpy as np
import pandas as pd
import pytest

from survpm import CensoredStateNet, EventLog


def journal(rows):
    return pd.DataFrame([
        dict(case_id=case, activity=activity,
             timestamp=pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(day, unit="D"))
        for case, activity, day in rows
    ])


@pytest.fixture
def model():
    # Два завершения за 1 и 3 дня: MLE среднего экспоненты равна 2 дням.
    train = journal([("a", "A", 0), ("a", "D", 1), ("b", "A", 0), ("b", "D", 3)])
    return CensoredStateNet("A", "D", candidates=("exponential",)).fit(train)


def test_evaluate_scores_held_out_data_without_refitting(model):
    test = journal([("x", "A", 0), ("x", "D", 4), ("y", "A", 0)])
    end = pd.Series({"y": pd.Timestamp("2025-01-03", tz="UTC")})
    result = model.evaluate(test, observation_end=end)
    row = result["level4"].iloc[0]
    # Один завершённый интервал: log(2)+4/2; одна цензура: 2/2.
    assert row["nll_per_interval"] == pytest.approx((math.log(2) + 3) / 2)
    assert row["nll_per_trace"] == pytest.approx((math.log(2) + 3) / 2)
    assert row["unseen_share"] == 0
    assert model.edges_[0]["dist"][1]["mean"] == pytest.approx(2)
    assert set(model.intervals_.case_id) == {"a", "b"}


def test_evaluate_selects_metrics_and_works_after_interval_fit(model):
    fitted = CensoredStateNet("A", "D").fit(intervals=model.intervals_)
    test = journal([("x", "A", 0), ("x", "D", 4)])
    result = fitted.evaluate(test, metrics=["nll_per_trace"])
    assert list(result) == ["level4"]
    assert set(result["level4"].columns) == {"model", "nll_per_trace"}
    assert result["level4"].iloc[0]["nll_per_trace"] == pytest.approx(math.log(2) + 2)


def test_evaluate_eventlog_uses_measurement_followup(model):
    events = journal([("x", "A", 0)])
    measurements = pd.DataFrame([dict(case_id="x", timestamp=pd.Timestamp("2025-01-05", tz="UTC"),
                                    feature="CRP", value=10)])
    log = EventLog(events, pd.DataFrame(), measurements)
    result = model.evaluate(log, metrics="nll_per_interval")
    assert result["level4"].iloc[0]["nll_per_interval"] == pytest.approx(2)


def test_evaluate_reports_unseen_only_test_set(model):
    test = journal([("x", "A", 0), ("x", "X", 1)])
    row = model.evaluate(test)["level4"].iloc[0]
    assert row["unseen_share"] == 1
    assert np.isnan(row["nll_per_interval"]) and np.isnan(row["nll_per_trace"])


def test_evaluate_validates_fit_and_metric_selection(model):
    test = journal([("x", "A", 0), ("x", "D", 1)])
    with pytest.raises(ValueError, match="fit"):
        CensoredStateNet("A", "D").evaluate(test)
    with pytest.raises(ValueError, match="no_such_metric"):
        model.evaluate(test, metrics="no_such_metric")
    with pytest.raises(ValueError, match="[Мм]етрик"):
        model.evaluate(test, metrics=[])


def test_evaluate_requires_training_cases_for_ipcw_after_interval_fit(model):
    fitted = CensoredStateNet("A", "D").fit(intervals=model.intervals_)
    test = journal([("x", "A", 0), ("x", "D", 1)])
    with pytest.raises(ValueError, match="fit\\(raw\\)"):
        fitted.evaluate(test, metrics="ibs")


def test_level4_crossvalidation_uses_the_shared_registry():
    pytest.importorskip("sklearn")
    from survpm.metrics.crossval import CrossValidator

    raw = journal([(str(i), activity, day) for i in range(8)
                   for activity, day in [("A", 0), ("D", i + 1)]])
    result = CrossValidator("A", "D", k=2).add_metrics("level4").run(raw, progress=None)
    assert len(result["level4"]) == 4
    assert set(result["level4"]["model"]) == {"naive", "censor_aware"}
    assert result["level4"]["unseen_share"].eq(0).all()
    assert np.isfinite(result["level4"]["nll_per_interval"]).all()


def test_evaluate_temporal_metrics(model):
    pytest.importorskip("pm4py")
    pytest.importorskip("log_distance_measures")
    pytest.importorskip("sksurv")
    pytest.importorskip("lifelines")
    test = journal([("x", "A", 0), ("x", "D", 1), ("y", "A", 0), ("y", "D", 3)])
    result = model.evaluate(test, metrics=["edge_ks", "km_sup"], n_sim=1000, seed=7)
    row = result["level3"].iloc[0]
    assert row["edge_ks"] == pytest.approx(1 - math.exp(-.5))
    assert 0 <= row["km_sup"] <= 1
    pd.testing.assert_frame_equal(result["level3"], model.evaluate(
        test, metrics=["edge_ks", "km_sup"], n_sim=1000, seed=7)["level3"])


@pytest.mark.parametrize("censored", [False, True])
def test_evaluate_process_metrics_score_only_this_model(model, censored):
    pytest.importorskip("pm4py")
    rows = [("x", "A", 0)] if censored else [("x", "A", 0), ("x", "D", 2)]
    test = journal(rows)
    result = model.evaluate(test, metrics="fitness_token")
    assert len(result["level1"]) == 1
    assert result["level1"].iloc[0]["fitness_token"] == pytest.approx(1)


@pytest.mark.parametrize("censored", [False, True])
def test_evaluate_ibs_without_eligible_completions_is_nan(model, censored):
    pytest.importorskip("pm4py")
    pytest.importorskip("log_distance_measures")
    pytest.importorskip("sksurv")
    pytest.importorskip("lifelines")
    rows = [("x", "A", 0)] if censored else [("x", "A", 0), ("x", "D", 4)]
    test = journal(rows)
    end = pd.Series({"x": pd.Timestamp("2025-01-03", tz="UTC")})
    result = model.evaluate(test, metrics="ibs", observation_end=end)
    assert np.isnan(result["level3"].iloc[0]["ibs"])


def test_evaluate_stochastic_metrics_match_deterministic_language(model):
    pytest.importorskip("pm4py")
    pytest.importorskip("ebi")
    pytest.importorskip("ot")
    pytest.importorskip("jellyfish")
    test = journal([("x", "A", 0), ("x", "D", 2)])
    row = model.evaluate(test, metrics="level2", n_sim=20)["level2"].iloc[0]
    assert row["emd"] == pytest.approx(0)
    assert row["uemsc"] == pytest.approx(1)
    assert row["entropic_relevance"] == pytest.approx(0)
