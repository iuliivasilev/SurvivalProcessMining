import numpy as np
import pandas as pd
import pytest

from survpm import CensoredStateNet, EventLog


def sample(n=700, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.choice([-1., 1.], n)
    destination = np.where(rng.random(n) < 1 / (1 + np.exp(-1.5 * x)), "D", "B")
    duration = rng.exponential(2 * np.exp(.7 * x))
    censor = rng.exponential(6, n)
    observed = duration <= censor
    iv = pd.DataFrame(dict(case_id=[str(i) for i in range(n)], src="A",
                           dst=np.where(observed, destination, None), duration=np.minimum(duration, censor),
                           observed=observed, reason=np.where(observed, "", "lost")))
    return iv, pd.DataFrame({"x": x})


def test_joint_fit_recovers_choice_and_time_with_censoring():
    iv, X = sample()
    model = CensoredStateNet("A", "D", feature_columns=["x"], candidates=("exponential",)).fit(intervals=iv, X=X)
    low = model.for_features({"x": -1}).edges_table().set_index("dst")
    high = model.for_features({"x": 1}).edges_table().set_index("dst")
    assert high.loc["D", "p"] > .7 and low.loc["D", "p"] < .3
    assert high.loc["D", "mean_days"] / low.loc["D", "mean_days"] == pytest.approx(np.exp(1.4), rel=.35)
    assert model.diagnostics_.success.all()
    with pytest.raises(ValueError, match="for_features"):
        model.survival([1, 7])
    snapshot = model.for_features({"x": 1})
    assert np.all(np.diff(snapshot.survival([1, 3, 7], method="grid")) <= 0)


def test_conditional_likelihood_improves_on_held_out_data():
    iv, X = sample()
    test, Xt = sample(300, seed=4)
    plain = CensoredStateNet("A", "D", candidates=("exponential",)).fit(intervals=iv)
    model = CensoredStateNet("A", "D", feature_columns=["x"], candidates=("exponential",)).fit(intervals=iv, X=X)
    assert model.nll(test, X=Xt)["nll_per_interval"] < plain.nll(test)["nll_per_interval"] - .1


def event_log(n=100):
    iv, X = sample(n)
    rows = []
    ends = {}
    origin = pd.Timestamp("2025-01-01", tz="UTC")
    for row in iv.itertuples():
        rows.append((row.case_id, "A", origin))
        ends[row.case_id] = origin + pd.Timedelta(row.duration, unit="D")
        if row.observed:
            rows.append((row.case_id, "D", ends[row.case_id]))
    cases = X.set_axis(iv.case_id)
    return EventLog(pd.DataFrame(rows, columns=["case_id", "activity", "timestamp"]), cases,
                    pd.DataFrame(columns=["case_id", "timestamp", "feature", "value"]), pd.Series(ends))


def test_eventlog_evaluate_and_no_training_leakage():
    log = event_log()
    model = CensoredStateNet("A", "D", feature_columns=["x"], candidates=("exponential",)).fit(log)
    means = model.encoder_.mean_.copy()
    test_cases = log.cases.copy()
    test_cases["x"] = 10.
    test_log = EventLog(log.events, test_cases, log.measurements, log.last_seen)
    result = model.evaluate(test_log)
    assert np.isfinite(result["level4"].iloc[0].nll_per_interval)
    assert result["level4"].iloc[0].nll_per_interval != pytest.approx(model.evaluate(log)["level4"].iloc[0].nll_per_interval)
    pd.testing.assert_series_equal(model.encoder_.mean_, means)
    with pytest.raises(ValueError, match="EventLog|X"):
        model.evaluate(log.events)


@pytest.mark.parametrize("dtype", ["object", "category"])
def test_encoder_handles_missing_and_unseen_categories(dtype):
    iv, X = sample(100)
    X["group"] = np.where(X.x > 0, "a", "b")
    X["group"] = X["group"].astype(dtype)
    X.loc[0, "x"] = np.nan
    model = CensoredStateNet("A", "D", feature_columns=["x", "group"], candidates=("exponential",)).fit(intervals=iv, X=X)
    snapshot = model.for_features({"x": None, "group": "new"})
    assert snapshot.edges_table().p.sum() == pytest.approx(1)


def test_feature_crossvalidation():
    pytest.importorskip("sklearn")
    from survpm.metrics.crossval import CrossValidator
    result = CrossValidator("A", "D", k=2, feature_columns=("x",)).add_metrics("level4").run(event_log(80), progress=None)
    assert len(result["level4"]) == 4
    assert np.isfinite(result["level4"].nll_per_interval).all()


def test_zero_coefficients_match_original_censored_likelihood():
    from survpm.covariates import conditional_nll
    from survpm.estimation import source_nll
    iv, X = sample(40)
    edges = [dict(src="A", dst=d, p=p, dist=("exponential", {"mean": m}),
                  choice_coef=np.zeros(1), time_coef=np.zeros(1)) for d, p, m in [("B", .4, 2.), ("D", .6, 3.)]]
    expected = source_nll(iv, [("A", "B"), ("A", "D")], [.4, .6], [e["dist"] for e in edges])
    assert conditional_nll(iv, edges, X.to_numpy()) == pytest.approx(expected)


@pytest.mark.parametrize("dist", [
    ("exponential", {"mean": 2.}), ("lognormal", {"median": 2., "sigma": .5}),
    ("gamma", {"shape": 2., "scale": 1.}), ("weibull", {"shape": 1.5, "scale": 2.}),
])
def test_aft_acceleration_for_each_family(dist):
    from survpm.covariates import conditional_nll, condition_edges
    from survpm.distributions import mean_time
    from survpm.estimation import source_nll
    edges = [dict(src="A", dst="D", p=1., dist=dist, choice_coef=np.zeros(1), time_coef=np.array([np.log(2)]))]
    conditioned = condition_edges(edges, np.ones((1, 1)))
    assert mean_time(conditioned[0]["dist"]) == pytest.approx(2 * mean_time(dist))
    iv = pd.DataFrame(dict(src=["A", "A"], dst=["D", None], duration=[1., 3.], observed=[True, False]))
    expected = source_nll(iv, [("A", "D")], [1.], [conditioned[0]["dist"]])
    assert conditional_nll(iv, edges, np.ones((2, 1))) == pytest.approx(expected)


def test_late_measurements_are_not_used_at_entry():
    log = event_log(40)
    measurements = pd.DataFrame(dict(case_id=log.cases.index,
                                     timestamp=pd.Timestamp("2025-02-01", tz="UTC"), feature="CRP", value=99.))
    log = EventLog(log.events, log.cases, measurements, log.last_seen)
    model = CensoredStateNet("A", "D", feature_columns=["CRP"], candidates=("exponential",)).fit(log)
    assert model.intervals_.CRP.isna().all()
    assert model.encoder_.names_ == []


def test_history_is_computed_at_entry():
    events = pd.DataFrame(dict(case_id=["a", "a", "a", "b", "b"], activity=["A", "A", "D", "A", "D"],
                               timestamp=pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta([0, 1, 2, 0, 3], unit="D")))
    model = CensoredStateNet("A", "D", feature_columns=["history_visits", "history_elapsed_days"],
                             candidates=("exponential",)).fit(events)
    assert model.intervals_.history_visits.tolist() == [1, 2, 1]
    assert model.intervals_.history_elapsed_days.tolist() == [0, 1, 0]


def test_feature_names_cannot_overwrite_interval_fields():
    with pytest.raises(ValueError, match="полями интервалов"):
        CensoredStateNet("A", "D", feature_columns=["duration"])
