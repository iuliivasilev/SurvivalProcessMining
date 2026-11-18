"""From the repository's hu_yubo directory, run python -m prototype_v1.demo."""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from .agent import Candidate, Weights, learn, ranking
from .environment import Case, TeacherNet, at_anchor

# Call the original miner through its public entry point and its shipped configuration.
BASELINE = Path(__file__).resolve().parents[1] / "precursor-mining-simple"
sys.path.insert(0, str(BASELINE))
from precursor_mining import Config, analyze


@dataclass(frozen=True)
class Pool:
    anchor: str | None
    candidates: tuple[Candidate, ...]
    status: str


def counts_at_anchor(cases, anchor):
    failed, normal = Counter(), Counter()
    n_failed = n_normal = 0
    for case in cases:
        observation = at_anchor(case.labels, anchor)
        if observation is None:
            continue
        prefix, following = observation
        if following == "F":
            n_failed += 1
            failed.update(set(prefix))
        else:
            n_normal += 1
            normal.update(set(prefix))
    return failed, normal, n_failed, n_normal


def prepare_pool(cases):
    """Fit on completed historical cases only, once per catalogue of observable labels."""
    config = json.loads((BASELINE / "config/example.json").read_text())
    config.pop("input")
    rows = ({"case": case.case_id, "t": t, "event": label}
            for case in cases for t, label in zip(case.times, case.labels, strict=True))
    result = analyze(rows, Config(**config))
    summary = result["summary"]
    anchor = summary["anchor"][0] if summary["anchor"] else None
    if summary["status"] != "ok":
        return Pool(anchor, (), summary["status"])
    base = {node["node"][0]: node["normalized_pagerank"] for node in result["nodes"]}
    failed, normal, n_failed, n_normal = counts_at_anchor(cases, anchor)
    if not n_failed or not n_normal:
        return Pool(anchor, (), "insufficient_next_event_outcomes")
    candidates = []
    for name in sorted(set(base) | set(failed) | set(normal)):
        contrast = failed[name] / n_failed - normal[name] / n_normal
        candidates.append(Candidate(name, base.get(name, 0.0), contrast, name in base))
    return Pool(anchor, tuple(candidates), "ok")


def observed_risks(cases, anchor, names):
    """Evaluation only. The denominator is cases with the event in the eligible prefix."""
    failed, normal, _, _ = counts_at_anchor(cases, anchor)
    risks = {}
    for name in names:
        total = failed[name] + normal[name]
        risks[name] = failed[name] / total if total else None
    return risks


@dataclass(frozen=True)
class Quality:
    credit: float
    strict: int
    equal: int
    unsupported: int


def evaluate(candidates, weights, risks):
    credit = 0.0
    strict = equal = unsupported = 0
    for a, b in combinations(ranking(candidates, weights), 2):
        left, right = risks[a.name], risks[b.name]
        if left is None or right is None:
            unsupported += 1
        elif math.isclose(left, right):
            equal += 1
        else:
            strict += 1
            if math.isclose(weights.score(a), weights.score(b), abs_tol=1e-12):
                credit += 0.5
            elif left > right:  # ranking() puts the higher model score first.
                credit += 1
    return Quality(credit, strict, equal, unsupported)


def relabel(cases):
    # These new upstream names are used only to construct the evaluation fixture.
    names = {"P": "X7", "B": "X3", "C1": "X8", "C2": "X1", "S": "X5"}
    return [Case(c.case_id, tuple(names.get(label, label) for label in c.labels), c.times)
            for c in cases]


def save_model(path, weights):
    payload = {"model": "precursor-shared-score-v1", "weights": asdict(weights)}
    contents = json.dumps(payload, allow_nan=False, indent=2) + "\n"
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / "weights.json"
        temporary.write_text(contents, encoding="utf-8")
        temporary.replace(path)


def load_model(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("model") != "precursor-shared-score-v1":
        raise ValueError("Unrecognized model format")
    raw = data.get("weights")
    if not isinstance(raw, dict) or set(raw) != {"baseline", "contrast"}:
        raise ValueError("The model needs baseline and contrast weights")
    if not all(type(x) in (int, float) and math.isfinite(x) for x in raw.values()):
        raise ValueError("Weights must be finite numbers")
    return Weights(**raw)


def demo(env, seed, budget, save_path=None):
    source_cases = env.sample(80, seed, "source")
    source = prepare_pool(source_cases)
    response_names = {"left": "выше у первой группы", "right": "выше у второй группы",
                      "tie": "риски равны", "unknown": "недостаточно данных"}
    print("Примеры ответов среды. Прогнозируем отказ F на следующем шаге после A.")
    print(f"  Риск в группе P равен {env.risk('A', 'P'):.1%}, в группе B — {env.risk('A', 'B'):.1%}.")
    witness = env.counterexample("A", "P")
    print("  Траектория с P и нормальным исходом после A —",
          " → ".join(witness) if witness else "в этой сети отсутствует")
    with_c1, without_c1 = env.context_support("A", "P", "C1")
    print(f"  Циклы с C1 перед P — {'есть' if with_c1 else 'отсутствуют'}. "
          f"Циклы без C1 перед P — {'есть' if without_c1 else 'отсутствуют'}.")
    failed, normal, _, _ = counts_at_anchor(source_cases, "A")
    print(f"\nИсходный журнал, {len(source_cases)} циклов. Исход сразу после A")
    print("Ветвь | Отказ F | Нормальный исход OK")
    for branch in ("P", "B"):
        print(f"  {branch} | {failed[branch]} | {normal[branch]}")
    if not source.candidates:
        if save_path:
            raise ValueError(f"Нет кандидатов для обучения новой модели ({source.status})")
        print(f"Базовый алгоритм не выделил кандидатов в исходном журнале ({source.status}).")
        return

    def answer(left, right):
        return env.compare(source.anchor, left, right)

    active = learn(source.candidates, budget, answer, policy="active", seed=seed)
    random_fit = learn(source.candidates, budget, answer, policy="random", seed=seed)
    references = [c.name for c in source.candidates if not c.ranked]
    print(f"\nЯкорь базового алгоритма — {source.anchor}. "
          f"События только для сравнения — {', '.join(references) or 'отсутствуют'}.")
    for title, fit in [("Активный выбор", active), ("Случайный выбор", random_fit)]:
        print(f"{title}. Каждая строка соответствует одному запросу.")
        for i, q in enumerate(fit.history, 1):
            print(f"  {i}. {q.left} / {q.right}   {response_names[q.answer]}   "
                  f"веса ({q.weights.baseline:.3f}, {q.weights.contrast:.3f})")
    asked = {name for q in active.history for name in (q.left, q.right)}
    print("\nКандидаты после активного обучения на исходном журнале")
    print("Событие  Оценка до  Разность долей  Оценка после  Был запрос")
    for c in ranking(source.candidates, active.weights):
        print(f"  {c.name:>3}  {c.baseline:7.3f}  {c.contrast:9.3f}  "
              f"{active.weights.score(c):9.3f}   {'да' if c.name in asked else 'нет'}")
    if save_path:
        save_model(save_path, active.weights)
        print(f"Веса модели сохранены в {save_path}")
    report_new_ids(env, seed, [("Базовый алгоритм", Weights(), 0),
                               ("Случайный выбор", random_fit.weights, len(random_fit.history)),
                               ("Активный выбор", active.weights, len(active.history))])


def report_new_ids(env, seed, methods):
    target = prepare_pool(relabel(env.sample(80, seed, "new-history")))
    if not target.candidates:
        print(f"В журнале с новыми обозначениями нет кандидатов для оценки ({target.status}).")
        return
    test = relabel(env.sample(200, seed, "evaluation"))
    risks = observed_risks(test, target.anchor, [c.name for c in target.candidates])
    failed, normal, _, _ = counts_at_anchor(test, target.anchor)
    print("\nТа же сеть, новые обозначения событий до A. Веса зафиксированы.")
    print("80 новых циклов для вычисления признаков, ещё 200 для проверки.")
    print("Метод | Запросы в этом запуске | Баллы/пары с разным риском | Пары с равным риском | Пары без наблюдений")
    for label, weights, questions in methods:
        q = evaluate(target.candidates, weights, risks)
        pair_result = f"{q.credit:g}/{q.strict}" if q.strict else "нет пар с разным риском"
        print(f"  {label} | {questions} | {pair_result} | {q.equal} | {q.unsupported}")
    print("Новые кандидаты. Запросов о них не было.")
    print("Событие  Оценка до  Оценка после  Доля отказов  Отказы/циклы в проверочной выборке")
    displayed_weights = methods[-1][1]
    for c in ranking(target.candidates, displayed_weights):
        rate = risks[c.name]
        rate_text = f"{rate:.1%}" if rate is not None else "нет наблюдений"
        count = failed[c.name] + normal[c.name]
        print(f"  {c.name:>3}   {c.baseline:.3f}  {displayed_weights.score(c):.3f}   "
              f"{rate_text}   {failed[c.name]}/{count}")


def main():
    parser = argparse.ArgumentParser(description="Эксперимент 1. Активное обучение при поиске предвестников отказа")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--p", type=float, default=0.30, help="вероятность отказа на ветви P после A")
    model_flags = parser.add_mutually_exclusive_group()
    model_flags.add_argument("--save-model", type=Path, help="сохранить обученные веса в JSON")
    model_flags.add_argument("--load-model", type=Path, help="применить сохранённые веса к новым обозначениям")
    args = parser.parse_args()
    if args.budget < 0 or not 0 <= args.p <= 1:
        parser.error("бюджет должен быть неотрицательным, p должно лежать в [0, 1]")
    env = TeacherNet(args.p)
    try:
        if args.load_model:
            weights = load_model(args.load_model)
            report_new_ids(env, args.seed, [("Базовый алгоритм", Weights(), 0), ("Загруженная модель", weights, 0)])
        else:
            demo(env, args.seed, args.budget, args.save_model)
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
