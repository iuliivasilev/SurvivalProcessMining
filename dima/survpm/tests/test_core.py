import json
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

import survpm
from survpm import datasets


def ts(day):
    return pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(day, unit="D")


@pytest.fixture
def small_log():
    rows = [("1", "A", 0), ("1", "A", 1), ("1", "B", 2), ("1", "D", 5), ("1", "B", 6),
            ("2", "X", 0), ("2", "A", 1), ("2", "B", 3),
            ("3", "B", 0), ("3", "D", 1)]
    return pd.DataFrame([dict(case_id=c, activity=a, timestamp=ts(d)) for c, a, d in rows])


def test_prepare_cuts_collapses_and_censors(small_log):
    observation_end = pd.Series({"2": ts(10)})
    log, cases = survpm.prepare(small_log, "A", "D", observation_end)
    assert set(cases.case_id) == {"1", "2"}  # у трассы 3 нет A
    assert log.loc[log.case_id == "1", "activity"].tolist() == ["A", "B", "D"]  # повтор A схлопнут, после D обрезано
    assert log.loc[log.case_id == "2", "activity"].tolist() == ["A", "B"]  # X до A отброшен
    case2 = cases.set_index("case_id").loc["2"]
    assert case2.event == 0 and case2.duration == pytest.approx(9)

    intervals = survpm.intervals(log, cases)
    one = intervals[intervals.case_id == "1"]
    assert one[["src", "dst"]].values.tolist() == [["A", "B"], ["B", "D"]]
    assert one.duration.tolist() == pytest.approx([2, 3])  # вход в A — первое событие серии
    censored = intervals[~intervals.observed]
    assert censored.src.tolist() == ["B"] and censored.duration.tolist() == pytest.approx([7])


def test_censoring_lengthens_exponential_mean():
    # Завершения 1, 2, 3 дня и две цензуры через 10 дней: MLE экспоненты с цензурой = (6 + 20) / 3.
    intervals = pd.DataFrame(dict(case_id=list("abcde"), src="A", dst=["B", "B", "B", None, None],
                                  duration=[1., 2., 3., 10., 10.], observed=[True] * 3 + [False] * 2,
                                  reason=[""] * 3 + ["lost"] * 2))
    families = {("A", "B"): "exponential"}
    naive, _ = survpm.estimate_network(intervals, families, use_censoring=False)
    aware, _ = survpm.estimate_network(intervals, families, use_censoring=True)
    assert naive[0]["dist"][1]["mean"] == pytest.approx(2, rel=1e-3)
    assert aware[0]["dist"][1]["mean"] == pytest.approx(26 / 3, rel=1e-3)


def test_recovers_synthetic_parameters():
    raw, censoring = datasets.generate_ad(3000, seed=1)
    families = {(t["src"], t["dst"]): t["dist"][0] for t in datasets.AD_TRANSITIONS}
    model = survpm.CensoredStateNet("A", "D", families=families).fit(raw, censoring["end"], censoring["reason"])
    fitted = {(e["src"], e["dst"]): e for e in model.edges_}
    for truth in datasets.AD_TRANSITIONS:
        edge = fitted[(truth["src"], truth["dst"])]
        assert edge["p"] == pytest.approx(truth["p"], abs=.05)
        assert survpm.mean_time(edge["dist"]) == pytest.approx(survpm.mean_time(truth["dist"]), rel=.2)


@pytest.fixture(scope="module")
def ad_model():
    raw, censoring = datasets.generate_ad(500, seed=2)
    return survpm.CensoredStateNet("A", "D").fit(raw, censoring["end"], censoring["reason"])


def test_survival_is_monotone(ad_model):
    survival = ad_model.survival(np.linspace(0, 100, 50), n=2000)
    assert survival[0] == pytest.approx(1) and np.all(np.diff(survival) <= 0) and 0 <= survival[-1] < .1


def test_score_matches_nll(ad_model):
    result = ad_model.nll(ad_model.intervals_)
    assert result["unseen_share"] == 0
    assert ad_model.score(ad_model.intervals_) == pytest.approx(-result["nll_per_interval"])


@pytest.mark.parametrize("censoring", [False, True])
def test_sdfa_probabilities_sum_exactly(ad_model, censoring):
    automaton = json.loads(ad_model.to_sdfa(censoring=censoring))
    totals = {}
    for t in automaton["transitions"]:
        totals[t["from"]] = totals.get(t["from"], 0) + Fraction(t["prob"])
    assert all(total <= 1 for total in totals.values())
    if not censoring:
        assert all(total == 1 for total in totals.values())


def test_pm4py_replay_puts_token_in_cens(ad_model):
    pytest.importorskip("pm4py")
    from survpm import adapters
    net, im, fm = ad_model.to_pm4py()
    marking = adapters.replay_censored(net, im, ["A", "B"], reason="lost")
    assert [p.name for p, n in marking.items() if n] == [adapters.CENSORED]
