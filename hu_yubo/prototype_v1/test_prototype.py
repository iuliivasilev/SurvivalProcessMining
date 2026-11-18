import json
from pathlib import Path
import subprocess
import sys

import pytest

from prototype_v1 import agent as a, demo
from prototype_v1.environment import Case, TeacherNet


def test_answers_to_three_questions_use_the_same_prediction_point():
    env = TeacherNet(0.3)
    assert env.risk("A", "P") == pytest.approx(0.3)
    assert env.risk("A", "B") == 0.0
    assert env.risk("A", "C1") == pytest.approx(0.15)
    assert env.compare("A", "P", "B") == "left"
    assert env.compare("A", "B", "P") == "right"
    assert env.compare("A", "C1", "C2") == "tie"
    assert env.compare("A", "absent", "B") == "unknown"
    assert env.counterexample("A", "P") == ("S", "C1", "C2", "P", "A", "OK")
    assert env.context_support("A", "P", "C1") == (True, False)
    assert TeacherNet(1).counterexample("A", "P") is None


@pytest.mark.parametrize("answer,preferred", [("left", "P"), ("right", "B")])
def test_preference_changes_the_shared_score_of_an_unqueried_event(answer, preferred):
    p = a.Candidate("P", 0.2, 0.6, True)
    b = a.Candidate("B", 0.0, -0.4, False)
    x = a.Candidate("new-name", 0.2, 0.6, True)
    def reply(left, right):
        # A fixed answer about P versus B, independent of presentation order.
        if left == "P":
            return answer
        return "right" if answer == "left" else "left"
    fit = a.learn([p, b], 1, reply)
    initial = a.Weights()
    sign = 1 if preferred == "P" else -1
    assert sign * (fit.weights.score(x) - initial.score(x)) > 0
    assert fit.weights.score(x) == fit.weights.score(p)
    assert [c.name for c in a.ranking([p, b], fit.weights)] == ["P"]


def test_question_budget_charges_ties_and_unknowns_and_never_repeats_a_pair():
    cs = [a.Candidate("P", 0.8, 0.2, True), a.Candidate("C", 0.1, 0.0, True),
          a.Candidate("B", 0.0, -0.3, False), a.Candidate("Z", 0.0, -0.2, False)]
    for policy in ("active", "random"):
        fit = a.learn(cs, 20, lambda left, right: "unknown" if left == "B" else "tie",
                      policy=policy, seed=19)
        pairs = {frozenset((q.left, q.right)) for q in fit.history}
        assert len(pairs) == len(fit.history) == 5
        assert frozenset(("B", "Z")) not in pairs
        assert fit.weights == a.Weights()
        first = a.learn(cs, 1, lambda *_: "unknown", policy=policy, seed=19).history[0]
        assert (first.left, first.right) == (fit.history[0].left, fit.history[0].right)
    first_active = a.learn(cs, 1, lambda *_: "unknown").history[0]
    assert {first_active.left, first_active.right} == {"B", "C"}


def test_portable_miner_supplies_final_candidates_and_logged_b_is_only_a_reference():
    env = TeacherNet(0.3)
    cases = env.sample(80, 7, "source")
    pool = demo.prepare_pool(cases)
    assert pool.anchor == "A"
    assert {c.name for c in pool.candidates if c.ranked} == {"P", "C1", "C2", "S"}
    assert {c.name for c in pool.candidates if not c.ranked} == {"B"}
    feature = {c.name: c.contrast for c in pool.candidates}
    assert feature["P"] > feature["C1"] == 0 > feature["B"]
    no_failures = demo.prepare_pool(TeacherNet(0).sample(80, 7, "source"))
    assert no_failures.status == "no_target" and not no_failures.candidates


def test_evaluation_uses_the_next_event_and_reports_unsupported_and_equal_risks():
    def case(name, labels):
        return Case(name, tuple(labels), tuple(range(len(labels))))
    cases = [case("1", ["C", "P", "A", "OK", "F"]), case("2", ["C", "B", "A", "F"])]
    risks = demo.observed_risks(cases, "A", ["P", "B", "C", "X"])
    assert risks == {"P": 0.0, "B": 1.0, "C": 0.5, "X": None}
    cs = [a.Candidate(name, score, 0.0, True)
          for name, score in [("P", 1.0), ("B", 0.0), ("C", 0.0), ("X", 0.0)]]
    result = demo.evaluate(cs, a.Weights(), risks)
    assert result.strict == 3 and result.credit == 0.5 and result.unsupported == 3
    risks["P"] = risks["B"]
    assert demo.evaluate(cs, a.Weights(), risks).equal == 1


def test_saved_weights_reproduce_new_id_scores_in_a_fresh_process(tmp_path):
    path = tmp_path / "weights.json"
    cmd = [sys.executable, "-m", "prototype_v1.demo"]
    root = Path(__file__).resolve().parents[1]
    train = subprocess.run([*cmd, "--save-model", str(path)], cwd=root, capture_output=True, text=True)
    assert train.returncode == 0, train.stderr
    weights = json.loads(path.read_text())["weights"]
    assert weights["contrast"] > 0
    frozen = subprocess.run([*cmd, "--load-model", str(path)], cwd=root, capture_output=True, text=True)
    assert frozen.returncode == 0, frozen.stderr
    def new_id_rows(text):
        return [line for line in text.splitlines() if line.strip().startswith("X")]
    assert len(new_id_rows(frozen.stdout)) == 4
    assert new_id_rows(frozen.stdout) == new_id_rows(train.stdout)
    # Deliberately change the on-disk model to distinguish loading from silent retraining.
    data = json.loads(path.read_text())
    data["weights"] = {"baseline": 0.0, "contrast": 0.0}
    path.write_text(json.dumps(data))
    changed = subprocess.run([*cmd, "--load-model", str(path)], cwd=root,
                             capture_output=True, text=True, check=True)
    assert [float(row.split()[2]) for row in new_id_rows(changed.stdout)] == [0.0] * 4


def test_run_without_candidates_signals_that_no_new_model_was_saved(tmp_path):
    path = tmp_path / "weights.json"
    path.write_text("previous model")
    with pytest.raises(ValueError, match="no_target"):
        demo.demo(TeacherNet(0), 7, 4, path)
    assert path.read_text() == "previous model"
