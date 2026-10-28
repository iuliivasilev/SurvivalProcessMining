import numpy as np
import pandas as pd
import pytest

import survpm
from survpm import features


def ts(day):
    return pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(day, unit="D")


@pytest.fixture
def wide():
    """Широкий журнал: возраст в регистрации, лаборатория — отдельные события Lab."""
    rows = [("1", "A", 0, "70", "true", "", "x"), ("1", "Lab", 0.5, "", "", "10", "x"),
            ("1", "B", 1, "", "", "", "y"), ("1", "Lab", 1.5, "", "", "30", "y"),
            ("1", "Lab", 2.5, "", "", "20", "y"), ("1", "D", 3, "", "", "", "y"),
            ("2", "A", 0, "50", "false", "", "x"), ("2", "B", 2, "", "", "", "x"),
            ("2", "Lab", 2.0, "", "", "5", "x")]
    return pd.DataFrame(rows, columns=["case", "act", "day", "Age", "Flag", "CRP", "Ward"]).assign(
        time=lambda d: d["day"].map(ts)).drop(columns="day")


def test_from_frame_classifies_and_hides_measurements(wide):
    log = features.from_frame(wide, "case", "act", "time", measurement_activities=["Lab"])
    assert log.static == ["Age", "Flag"]
    assert log.dynamic == ["CRP", "Ward"]
    assert log.cases.loc["1", "Age"] == 70 and log.cases.loc["2", "Flag"] == 0
    assert log.events["activity"].tolist() == ["A", "B", "D", "A", "B"]
    crp = log.measurements[log.measurements.feature == "CRP"]
    assert crp["value"].tolist() == [10, 30, 20, 5]


def test_at_entry_takes_last_known_value(wide):
    log = features.from_frame(wide, "case", "act", "time", dynamic=["CRP"], static=["Age"],
                              measurement_activities=["Lab"])
    model_log, cases = survpm.prepare(log.events, "A", "D")
    iv = survpm.intervals(model_log, cases)
    x = features.at_entry(iv, log, window="1D")
    one = x[iv.case_id == "1"]
    assert one["Age"].tolist() == [70, 70]
    assert np.isnan(one["CRP"].iloc[0])                 # в A замеров ещё нет
    assert one["CRP"].iloc[1] == 10 and one["CRP_age"].iloc[1] == pytest.approx(.5)
    assert one["CRP_n"].tolist() == [0, 1]
    two = x[iv.case_id == "2"]                          # замер в момент входа в B уже известен
    assert two["CRP"].tolist()[-1] == 5 and two["CRP_n_1D"].tolist()[-1] == 1


def test_split_cuts_at_measurements(wide):
    log = features.from_frame(wide, "case", "act", "time", dynamic=["CRP"], measurement_activities=["Lab"])
    model_log, cases = survpm.prepare(log.events, "A", "D")
    iv = survpm.intervals(model_log, cases)
    pieces = features.split(iv, log)
    b = pieces[(pieces.case_id == "1") & (pieces.src == "B")]
    assert b["t0"].tolist() == pytest.approx([0, .5, 1.5]) and b["t1"].tolist() == pytest.approx([.5, 1.5, 2])
    assert b["observed"].tolist() == [False, False, True] and b["dst"].tolist()[-1] == "D"
    assert b["CRP"].tolist() == [10, 30, 20]
    assert pieces.groupby("interval")["t1"].max().tolist() == pytest.approx(iv["duration"].tolist())


def test_coerce():
    assert features.coerce(["1", "", "2.5"]).tolist()[::2] == [1, 2.5]
    assert features.coerce(["true", "false", ""]).tolist()[:2] == [1, 0]
    assert features.coerce(["a", "b"]).dtype == object


def test_fit_uses_last_seen_for_censoring(wide):
    log = features.from_frame(wide, "case", "act", "time", dynamic=["CRP"], measurement_activities=["Lab"])
    model = survpm.CensoredStateNet("A", "D").fit(log)
    censored = model.intervals_.loc[~model.intervals_.observed]
    assert censored.case_id.tolist() == ["2"] and censored.duration.tolist() == pytest.approx([1 / 1440])
    late = wide.copy()
    late.loc[len(late)] = ["2", "Lab", "", "", "7", "x", ts(4)]  # анализ через 2 дня после входа в B
    model = survpm.CensoredStateNet("A", "D").fit(
        features.from_frame(late, "case", "act", "time", dynamic=["CRP"], measurement_activities=["Lab"]))
    censored = model.intervals_.loc[~model.intervals_.observed]
    assert censored.duration.tolist() == pytest.approx([2])
