"""survpm: процессы с правой цензурой — сеть состояний, время переходов, генерация и прогноз.

Ядро зависит только от numpy, pandas и scipy. PM4Py и graphviz нужны для сети Петри и рисунков
(``pip install survpm[pm4py]``), метрики уровней 1–3 и кросс-валидация — ``survpm[metrics]``.

Основной сценарий (v2)::

    import survpm
    log = survpm.datasets.load_sepsis_log(path).prepare({"starts": ["Registration"], "terminals": ["Discharge"]})
    train, test = log.split(seed=1)
    model = survpm.ProcessModel().fit(train)              # сеть — DFG по train
    synthetic = survpm.Simulator(model, seed=1).run(survpm.ObjectStream.from_log(train), horizon=60)
    survpm.evaluate(model, test, "all")                   # все метрики, одна таблица
    model.plot("petri")                                   # сеть Петри с местом CENS

Прежний API (``CensoredStateNet``, семейства распределений по AIC, функции подготовки
журнала) по-прежнему импортируется отсюда и собран в :mod:`survpm.legacy`.
"""
__version__ = "0.2.0"

# --- v2: данные, структура, модель, движки, метрики, рисунки ---------------------------------
from . import datasets, features
from .eventlog import EventLog, process_bounds
from .features import read_log
from .network import Network
from .discovery import PetriBackend, DirectlyFollowsBackend, discover
from .adapters import Pm4pyBackend
from .models import (TransitionModel, TransitionPicker, MinTimePicker, ExponentialTransitionModel,
                     register_transition_model, create_transition_model,
                     DefaultHistoryFeatures, FeatureUpdater, LatentModel)
from .encoding import FeatureEncoder
from .process import ProcessModel
from .engine import ObjectStream, Simulator, Router, Forecaster, Evaluator
from .engine import HistoryFeatures  # прежнее имя DefaultHistoryFeatures; протокол — survpm.models.HistoryFeatures
from .metrics import (Metric, register_metric, evaluate, compare, compare_networks, available_metrics,
                      long_format)
from .viz import Visualizer, MatplotlibVisualizer, GraphvizVisualizer

# --- прежний API (совместимость; см. survpm.legacy) ------------------------------------------
from .curves import completion_cdf, curves_table, residual_cdf
from .distributions import CANDIDATE_FAMILIES, dist_str, mean_time, scipy_args, select_families
from .estimation import estimate_network, source_nll
from .log import censoring_summary, intervals, prepare, read_csv
from .model import CensoredStateNet
from .simulation import completion_times, simulate, traces_to_log

__all__ = [
    # v2
    "EventLog", "process_bounds", "read_log", "datasets", "features",
    "Network", "PetriBackend", "DirectlyFollowsBackend", "Pm4pyBackend", "discover",
    "TransitionModel", "TransitionPicker", "MinTimePicker", "ExponentialTransitionModel",
    "register_transition_model", "create_transition_model",
    "HistoryFeatures", "DefaultHistoryFeatures", "FeatureUpdater", "LatentModel", "FeatureEncoder",
    "ProcessModel", "ObjectStream", "Simulator", "Router", "Forecaster", "Evaluator",
    "Metric", "register_metric", "evaluate", "compare", "compare_networks", "available_metrics", "long_format",
    "Visualizer", "MatplotlibVisualizer", "GraphvizVisualizer",
    # прежний API
    "CensoredStateNet", "prepare", "intervals", "read_csv", "censoring_summary",
    "simulate", "completion_times", "traces_to_log", "completion_cdf", "curves_table", "residual_cdf",
    "CANDIDATE_FAMILIES", "dist_str", "mean_time", "scipy_args", "select_families",
    "estimate_network", "source_nll",
]
