"""Метрики качества восстановленной сети с цензурой: 5-fold по трассам, четыре уровня.

Вход — журнал ``case_id, activity, timestamp`` и, если известен, момент окончания наблюдения
каждой незавершённой трассы (``observation_end``). Без него цензура наступает в момент
последней записи трассы. В каждом фолде на обучающих трассах оцениваются две модели
из :mod:`censored_petri` — наивная MLE (``naive``) и MLE с цензурой (``censor_aware``), —
а метрики считаются на тестовых трассах.

* Уровень 1 — process mining: fitness и precision (token replay, alignments), generalization,
  simplicity для сети состояний и Inductive Miner. Вариант ``A`` — только полные тестовые трассы;
  вариант ``Б`` — все, цензурированная трасса считается префиксом (без фишки в конце).
* Уровень 2 — вероятности путей: entropic relevance и uEMSC (Ebi), EMD языков (метрика PM4Py, решение через POT).
* Уровень 3 — время: temporal profile, расстояния журналов (log-distance-measures),
  KS по рёбрам, расстояние до Каплана–Майера, log-rank, integrated Brier score.
* Уровень 4 — −log L тестовых интервалов с цензурой.

Тестовые трассы с переходом или состоянием, которых нет в обучающих, на уровне 4 исключаются,
их доля — отдельная метрика ``unseen_share``.

Пример::

    import censored_metrics as cm
    results = cm.cross_validate(raw, start="Registration", terminal="Discharge")
    cm.summarize(results["level1"])
"""
import datetime
import json

import ebi
import jellyfish
import ot
import numpy as np
import pandas as pd
import pm4py
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance
from log_distance_measures.config import EventLogIDs
from log_distance_measures.control_flow_log_distance import control_flow_log_distance
from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.objects.petri_net.obj import Marking
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv

import censored_petri as cp

CASE, ACTIVITY, TIME = "case:concept:name", "concept:name", "time:timestamp"
LOG_IDS = EventLogIDs(case=CASE, activity=ACTIVITY, start_time=TIME, end_time=TIME)
DAY = pd.Timedelta(1, unit="D")
MODELS = ("naive", "censor_aware")


# --- Данные ---

def state_log(raw, start, terminal, observation_end=None):
    """Журнал входов в состояния и таблица трасс.

    Трасса начинается с первого ``start`` и обрезается после первого ``terminal``;
    подряд идущие повторы состояния схлопываются. ``cases``: ``case_id, entry, end, event,
    duration`` (в днях от ``start`` до исхода или цензуры). Конец незавершённой трассы —
    ``observation_end[case_id]``, если задан, иначе её последняя запись.
    """
    raw = raw.sort_values(["case_id", "timestamp"], kind="stable")
    parts, case_rows = [], []
    for case_id, group in raw.groupby("case_id", sort=False):
        activities = group["activity"].to_numpy()
        starts = np.flatnonzero(activities == start)
        if not len(starts):
            continue
        group, activities = group.iloc[starts[0]:], activities[starts[0]:]
        ends = np.flatnonzero(activities == terminal)
        if len(ends):
            group, activities = group.iloc[:ends[0] + 1], activities[:ends[0] + 1]
        event = len(ends) > 0
        end = group["timestamp"].iloc[-1]
        if not event and observation_end is not None and case_id in observation_end.index:
            end = max(end, observation_end[case_id])
        parts.append(group.iloc[np.flatnonzero(np.r_[True, activities[1:] != activities[:-1]])])
        case_rows.append(dict(case_id=case_id, entry=group["timestamp"].iloc[0], end=end, event=int(event)))
    log = pd.concat(parts)[["case_id", "activity", "timestamp"]].set_axis([CASE, ACTIVITY, TIME], axis=1)
    cases = pd.DataFrame(case_rows)
    cases["duration"] = (cases["end"] - cases["entry"]) / DAY
    return log.reset_index(drop=True), cases


def make_intervals(log, cases, terminal, min_duration=1 / 1440):
    """Интервалы :func:`censored_petri.intervals_from_log`; цензурированное ожидание длится
    от входа в последнее состояние до ``cases.end``."""
    intervals = cp.intervals_from_log(log, terminal, min_duration=min_duration)
    last_entry = log.groupby(CASE)[TIME].max()
    censored = ~intervals["observed"]
    ids = intervals.loc[censored, "case_id"]
    end = cases.set_index("case_id")["end"]
    intervals.loc[censored, "duration"] = ((end[ids].to_numpy() - last_entry[ids].to_numpy()) / DAY).clip(min_duration)
    return intervals


def trace_folds(cases, k=5, seed=2026):
    """``k`` пар ``(train_ids, test_ids)``; доля цензурированных трасс в фолдах одинакова."""
    ids = cases["case_id"].to_numpy()
    folds = StratifiedKFold(k, shuffle=True, random_state=seed).split(ids, cases["event"])
    return [(ids[train], ids[test]) for train, test in folds]


def fit_models(intervals):
    """Семейства по AIC и оценки двух моделей: ``(families, {"naive": ..., "censor_aware": ...})``."""
    families, _ = cp.select_families(intervals)
    return families, {model: cp.estimate_network(intervals, families, model == "censor_aware")[0]
                      for model in MODELS}


def stop_probabilities(intervals):
    """Наблюдаемая вероятность цензуры за одно ожидание в состоянии: ``{state: q}``."""
    return cp.censoring_summary(intervals).set_index("src")["q_censored"].to_dict()


# --- Симуляция восстановленной модели ---

def simulate(estimates, start, terminal, n, seed, stop=None, horizon=None, max_steps=1000):
    """Трассы модели ``[(states, times), ...]``, время в днях от входа в ``start``.

    ``stop`` — вероятность цензуры за одно ожидание (обрыв без события);
    ``horizon`` — момент цензуры каждой трассы (массив длины ``n``).
    Трасса из состояния без выходов обрывается.
    """
    rng = np.random.default_rng(seed)
    outgoing = {}
    for t in estimates:
        rv, args, kwargs = cp.scipy_args(t["dist"])
        outgoing.setdefault(t["src"], []).append((t["dst"], t["p"], rv(*args, **kwargs)))
    choice = {src: ([o[0] for o in out], np.array([o[1] for o in out]) / sum(o[1] for o in out),
                    [o[2] for o in out]) for src, out in outgoing.items()}
    traces = []
    for i in range(n):
        state, now, states, times = start, 0., [start], [0.]
        for _ in range(max_steps):
            if state == terminal or state not in choice:
                break
            if stop and rng.random() < stop.get(state, 0.):
                break
            destinations, p, frozen = choice[state]
            j = rng.choice(len(p), p=p)
            now += float(frozen[j].rvs(random_state=rng))
            if horizon is not None and now > horizon[i]:
                break
            state = destinations[j]
            states.append(state)
            times.append(now)
        traces.append((states, times))
    return traces


def completion_times(traces, terminal):
    """Время до ``terminal`` по трассам симуляции; ``inf`` — не дошла."""
    return np.array([times[-1] if states[-1] == terminal else np.inf for states, times in traces])


def traces_to_log(traces, entries, prefix="sim"):
    """Трассы симуляции → журнал PM4Py; ``entries`` — момент входа каждой трассы."""
    rows = [{CASE: f"{prefix}_{i}", ACTIVITY: state, TIME: entry + pd.to_timedelta(t, unit="D")}
            for i, ((states, times), entry) in enumerate(zip(traces, entries))
            for state, t in zip(states, times)]
    return pd.DataFrame(rows)


# --- Уровень 1: process mining ---

def _replay_sums(log, net, im, fm):
    """Суммы ``missing, consumed, remaining, produced`` token replay по журналу."""
    replay = token_replay.apply(pm4py.convert_to_event_log(log), net, im, fm)
    return np.array([[r["missing_tokens"], r["consumed_tokens"], r["remaining_tokens"], r["produced_tokens"]]
                     for r in replay], dtype=float).sum(axis=0)


def prefix_fitness(complete, censored, net, im, fm):
    """Token-based fitness, где цензурированная трасса — префикс: она проигрывается без финальной
    маркировки, оставшиеся фишки не штрафуются."""
    missing, consumed, remaining, produced = _replay_sums(complete, net, im, fm)
    if len(censored):
        m, c, _, p = _replay_sums(censored, net, im, Marking())
        missing, consumed, produced = missing + m, consumed + c, produced + p
    return .5 * (1 - missing / consumed) + .5 * (1 - remaining / produced)


def level1(train_log, test_log, complete_ids, train_intervals, start, terminal):
    """Сеть состояний (без места CENS) и Inductive Miner, обученные на ``train_log``."""
    observed = train_intervals.loc[train_intervals["observed"]]
    nets = {"state_net": cp.state_petri_net(observed, start, terminal),
            "inductive": pm4py.discover_petri_net_inductive(train_log)}
    complete = test_log.loc[test_log[CASE].isin(complete_ids)]
    censored = test_log.loc[~test_log[CASE].isin(complete_ids)]
    rows = []
    for name, (net, im, fm) in nets.items():
        simplicity = pm4py.simplicity_petri_net(net, im, fm)
        rows.append(dict(model=name, variant="A",
                         fitness_token=pm4py.fitness_token_based_replay(complete, net, im, fm)["log_fitness"],
                         fitness_alignments=pm4py.fitness_alignments(complete, net, im, fm)["log_fitness"],
                         precision_token=pm4py.precision_token_based_replay(complete, net, im, fm),
                         precision_alignments=pm4py.precision_alignments(complete, net, im, fm),
                         generalization=pm4py.generalization_tbr(complete, net, im, fm),
                         simplicity=simplicity))
        rows.append(dict(model=name, variant="Б",
                         fitness_token=prefix_fitness(complete, censored, net, im, fm),
                         precision_token=pm4py.precision_token_based_replay(test_log, net, im, fm),
                         generalization=pm4py.generalization_tbr(test_log, net, im, fm),
                         simplicity=simplicity))
    return rows


# --- Уровень 2: вероятности путей ---

def _decimals(values, total, digits):
    """Вероятности с ``digits`` знаками после запятой, сумма ровно ``total``: Ebi считает точно,
    и уже при знаменателе 10⁴ entropic relevance на Sepsis не вычисляется («cannot compute ER»)."""
    unit = 10 ** digits
    counts = [max(1, round(v * unit)) for v in values]
    counts[int(np.argmax(counts))] += round(total * unit) - sum(counts)
    return [f"{c}/{unit}" for c in counts]


def sdfa(estimates, start, stop=None, digits=3):
    """Стохастический детерминированный автомат для Ebi (JSON).

    Состояние 0 — до ``start``; метка перехода — состояние, в которое он ведёт.
    ``stop`` — вероятность закончить трассу в состоянии (цензура, вариант Б);
    вероятности округляются до ``digits`` знаков.
    """
    states = sorted({start} | {t["src"] for t in estimates} | {t["dst"] for t in estimates})
    index = {state: i + 1 for i, state in enumerate(states)}
    transitions = [{"from": 0, "to": index[start], "label": start, "prob": "1"}]
    for src in states:
        out = [t for t in estimates if t["src"] == src]
        if not out:
            continue
        keep = 1 - (stop or {}).get(src, 0.)
        probabilities = _decimals([t["p"] * keep for t in out], round(keep, digits), digits)
        transitions += [{"from": index[src], "to": index[t["dst"]], "label": t["dst"], "prob": p}
                        for t, p in zip(out, probabilities)]
    return json.dumps({"initialState": 0, "transitions": transitions})


def _ebi_number(result):
    """Ebi отдаёт точное выражение строкой (``"... Approximately 0.93"``) или список."""
    if isinstance(result, list):
        return float(result[0])
    return float(str(result).split("Approximately")[-1])


def entropic_relevance(event_log, estimates, start, stop=None):
    """Entropic relevance (Ebi), бит на трассу; при «cannot compute ER» округление грубее."""
    for digits in (3, 2):
        try:
            return _ebi_number(ebi.conformance_entropic_relevance(event_log, sdfa(estimates, start, stop, digits)))
        except Exception as error:
            if "cannot compute ER" not in str(error):
                raise
    return np.nan


def _language(traces):
    counts = pd.Series([tuple(states) for states, _ in traces]).value_counts(normalize=True)
    return counts.to_dict()


def language_emd(log_language, model_language):
    """EMD между стохастическими языками с нормированным Левенштейном, как в PM4Py.

    PM4Py строит матрицу по объединению вариантов и без пакета POT решает плотную задачу ЛП:
    на Sepsis это >14 ГБ памяти; здесь матрица только «варианты журнала × варианты модели»."""
    activities = sorted({a for trace in [*log_language, *model_language] for a in trace})
    code = {a: chr(0x4E00 + i) for i, a in enumerate(activities)}
    left = ["".join(code[a] for a in trace) for trace in log_language]
    right = ["".join(code[a] for a in trace) for trace in model_language]
    distance = np.array([[jellyfish.levenshtein_distance(x, y) / max(len(x), len(y)) for y in right] for x in left])
    weights = [np.array(list(language.values()), dtype=float) for language in (log_language, model_language)]
    return float(ot.emd2(weights[0] / weights[0].sum(), weights[1] / weights[1].sum(), distance))


def level2(test_log, complete_ids, models, train_intervals, start, terminal, n_sim, seed):
    """Ebi: entropic relevance (бит на трассу, меньше — лучше), uEMSC (больше — лучше);
    EMD между языками журнала и симуляции модели (меньше — лучше)."""
    stop = stop_probabilities(train_intervals)
    logs = {"A": test_log.loc[test_log[CASE].isin(complete_ids)], "Б": test_log}
    rows = []
    for model, estimates in models.items():
        for variant, log in logs.items():
            variant_stop = stop if variant == "Б" else None
            event_log = pm4py.convert_to_event_log(log)
            traces = simulate(estimates, start, terminal, n_sim, seed, stop=variant_stop)
            log_language = {tuple(trace): p for trace, p in pm4py.get_stochastic_language(log).items()}
            rows.append(dict(model=model, variant=variant,
                             entropic_relevance=entropic_relevance(event_log, estimates, start, variant_stop),
                             uemsc=_ebi_number(ebi.conformance_unit_earth_movers(event_log, sdfa(estimates, start, variant_stop))),
                             emd=language_emd(log_language, _language(traces))))
    return rows


# --- Уровень 3: время ---

def _temporal_deviation_share(log, profile):
    deviations = pm4py.conformance_temporal_profile(log, profile)
    return float(np.mean([len(d) > 0 for d in deviations]))


def survival_metrics(test_cases, train_cases, model_times):
    """Сравнение кривой выживания модели (симуляция без цензуры) с Каплан–Майером теста."""
    km = KaplanMeierFitter().fit(test_cases["duration"], test_cases["event"])
    grid = np.linspace(0, test_cases["duration"].max(), 400)
    model_times = np.sort(model_times)
    model_survival = 1 - np.searchsorted(model_times, grid, side="right") / len(model_times)
    difference = np.abs(model_survival - km.survival_function_at_times(grid).to_numpy())
    finite = np.isfinite(model_times)
    logrank = logrank_test(test_cases["duration"], np.where(finite, model_times, grid[-1] * 10),
                           test_cases["event"], finite.astype(int))
    # IPCW-веса считаются по цензуре обучения: тестовые времена должны быть меньше последнего в обучении.
    scored = test_cases.loc[test_cases["duration"] < train_cases["duration"].max()]
    event_times = scored.loc[scored["event"] == 1, "duration"]
    times = np.unique(np.quantile(event_times, np.linspace(.1, .8, 15)))
    times = times[(times > scored["duration"].min()) & (times < scored["duration"].max())]
    estimate = 1 - np.searchsorted(model_times, times, side="right") / len(model_times)
    ibs = integrated_brier_score(Surv.from_arrays(train_cases["event"].astype(bool), train_cases["duration"]),
                                 Surv.from_arrays(scored["event"].astype(bool), scored["duration"]),
                                 np.tile(estimate, (len(scored), 1)), times) if len(times) > 1 else np.nan
    return dict(km_sup=difference.max(), km_l1_days=np.trapezoid(difference, grid),
                logrank_statistic=logrank.test_statistic, logrank_p=logrank.p_value, ibs=ibs)


def edge_ks(test_intervals, estimates):
    """Средний по тестовым интервалам KS завершённых времён против распределения ребра."""
    completed = test_intervals.loc[test_intervals["observed"]]
    values, weights = [], []
    for t in estimates:
        x = completed.loc[(completed["src"] == t["src"]) & (completed["dst"] == t["dst"]), "duration"]
        if len(x):
            rv, args, kwargs = cp.scipy_args(t["dist"])
            values.append(stats.kstest(x, lambda v: rv.cdf(v, *args, **kwargs)).statistic)
            weights.append(len(x))
    return float(np.average(values, weights=weights))


def level3(train_log, test_log, train_cases, test_cases, test_intervals, models, start, terminal, n_sim, seed):
    """Temporal profile: доля тестовых трасс с отклонениями от профиля журнала обучения
    (``model=train_log``, базовая линия) или симуляции модели.
    Расстояния log-distance-measures: тест против симуляции того же размера с теми же
    моментами входа; в варианте Б симулированная трасса обрывается в момент цензуры
    соответствующей тестовой трассы. ``cycle_time_emd`` и ``absolute_event_emd`` — в часах."""
    test_cases = test_cases.set_index("case_id").loc[test_log[CASE].unique()].reset_index()
    complete_ids = test_cases.loc[test_cases["event"] == 1, "case_id"]
    logs = {"A": test_log.loc[test_log[CASE].isin(complete_ids)], "Б": test_log}
    horizons = {"A": test_cases.loc[test_cases["event"] == 1],
                "Б": test_cases}
    train_profile = pm4py.discover_temporal_profile(train_log)
    rows = [dict(model="train_log", variant=v, temporal_deviation_share=_temporal_deviation_share(log, train_profile))
            for v, log in logs.items()]
    for model, estimates in models.items():
        free = simulate(estimates, start, terminal, n_sim, seed)
        profile = pm4py.discover_temporal_profile(traces_to_log(free, [train_cases["entry"].min()] * n_sim))
        time_rows = survival_metrics(test_cases, train_cases, completion_times(free, terminal))
        for variant, log in logs.items():
            cases = horizons[variant]
            horizon = np.where(cases["event"] == 1, np.inf, cases["duration"])
            traces = simulate(estimates, start, terminal, len(cases), seed + 1, horizon=horizon)
            simulated = traces_to_log(traces, cases["entry"])
            rows.append(dict(model=model, variant=variant,
                             temporal_deviation_share=_temporal_deviation_share(log, profile),
                             cfld=control_flow_log_distance(log, LOG_IDS, simulated, LOG_IDS, parallel=False),
                             ngram_3=n_gram_distribution_distance(log, LOG_IDS, simulated, LOG_IDS, n=3, normalize=True),
                             cycle_time_emd=cycle_time_distribution_distance(log, LOG_IDS, simulated, LOG_IDS,
                                                                             bin_size=datetime.timedelta(hours=1)),
                             absolute_event_emd=absolute_event_distribution_distance(log, LOG_IDS, simulated, LOG_IDS),
                             edge_ks=edge_ks(test_intervals, estimates),
                             **(time_rows if variant == "Б" else {})))
    return rows


# --- Уровень 4: правдоподобие на тесте ---

def heldout_nll(test_intervals, estimates):
    """−log L тестовых интервалов (завершённый: p·f(t), цензурированный: Σ p·S(c)).
    Трассы с невиданным в обучении переходом или состоянием исключаются."""
    edges = {(t["src"], t["dst"]): t for t in estimates}
    sources = {t["src"] for t in estimates}
    observed = test_intervals["observed"]
    known_edge = pd.Series([(s, d) in edges for s, d in zip(test_intervals["src"], test_intervals["dst"])],
                           index=test_intervals.index)
    unseen = (observed & ~known_edge) | (~observed & ~test_intervals["src"].isin(sources))
    unseen_cases = set(test_intervals.loc[unseen, "case_id"])
    kept = test_intervals.loc[~test_intervals["case_id"].isin(unseen_cases)]
    nll = 0.
    for src, group in kept.groupby("src"):
        out = [edge for edge in edges if edge[0] == src]
        nll += cp.source_nll(group, out, [edges[e]["p"] for e in out], [edges[e]["dist"] for e in out])
    n_cases = test_intervals["case_id"].nunique()
    return dict(unseen_share=len(unseen_cases) / n_cases, nll_per_interval=nll / len(kept),
                nll_per_trace=nll / kept["case_id"].nunique())


# --- Кросс-валидация ---

def cross_validate(raw, start, terminal, observation_end=None, k=5, seed=2026, min_duration=1 / 1440,
                   n_sim=5000, progress=print):
    """Все уровни на ``k`` фолдах. Возвращает словарь таблиц ``level1 … level4`` с колонкой ``fold``."""
    log, cases = state_log(raw, start, terminal, observation_end)
    intervals = make_intervals(log, cases, terminal, min_duration)
    tables = {f"level{i}": [] for i in range(1, 5)}
    for fold, (train_ids, test_ids) in enumerate(trace_folds(cases, k, seed)):
        train_log, test_log = log.loc[log[CASE].isin(train_ids)], log.loc[log[CASE].isin(test_ids)]
        train_cases, test_cases = cases.loc[cases["case_id"].isin(train_ids)], cases.loc[cases["case_id"].isin(test_ids)]
        train_intervals = intervals.loc[intervals["case_id"].isin(train_ids)]
        test_intervals = intervals.loc[intervals["case_id"].isin(test_ids)]
        complete_ids = test_cases.loc[test_cases["event"] == 1, "case_id"]
        _, models = fit_models(train_intervals)
        fold_seed = seed + 1000 * fold
        parts = {
            "level1": level1(train_log, test_log, complete_ids, train_intervals, start, terminal),
            "level2": level2(test_log, complete_ids, models, train_intervals, start, terminal, n_sim, fold_seed),
            "level3": level3(train_log, test_log, train_cases, test_cases, test_intervals, models, start, terminal,
                             n_sim, fold_seed),
            "level4": [dict(model=m, **heldout_nll(test_intervals, e)) for m, e in models.items()],
        }
        for name, rows in parts.items():
            tables[name].extend(dict(fold=fold, **row) for row in rows)
        if progress:
            progress(f"фолд {fold + 1}/{k} готов")
    return {name: pd.DataFrame(rows) for name, rows in tables.items()}


def summarize(table):
    """Среднее и стандартное отклонение по фолдам."""
    keys = [c for c in ("model", "variant") if c in table.columns]
    values = table.drop(columns="fold").groupby(keys, sort=False)
    return values.agg(["mean", "std"])
