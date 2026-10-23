"""survpm: стохастические сети состояний с правым цензурированием по журналам событий.

Ядро зависит только от numpy, pandas и scipy. PM4Py и graphviz нужны для адаптеров
и схем (``pip install survpm[pm4py]``), метрики и кросс-валидация — в :mod:`survpm.metrics`
(``pip install survpm[metrics]``).

Пример::

    import survpm
    raw = survpm.datasets.load_sepsis("Sepsis Cases - Event Log_grouped.csv")
    model = survpm.CensoredStateNet("Registration", "Discharge").fit(raw)
    model.edges_table()
"""
from . import datasets
from .distributions import CANDIDATE_FAMILIES, dist_str, mean_time, scipy_args, select_families
from .estimation import estimate_network, source_nll
from .log import censoring_summary, intervals, prepare, read_csv
from .model import CensoredStateNet
from .simulation import completion_times, simulate, traces_to_log

__version__ = "0.1.0"
