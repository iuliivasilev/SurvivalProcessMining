"""Прежний API (до v2): ``CensoredStateNet`` с режимами ``min_time``/``choice_then_time``,
семейства распределений по AIC, функции подготовки журнала, сеточные кривые и симуляция.

Новые сценарии стоит строить на :class:`survpm.ProcessModel`; мост — :meth:`ProcessModel.legacy`.
"""
from .curves import completion_cdf, curves_table, residual_cdf  # noqa: F401
from .distributions import CANDIDATE_FAMILIES, dist_str, mean_time, scipy_args, select_families  # noqa: F401
from .estimation import estimate_network, source_nll  # noqa: F401
from .log import censoring_summary, intervals, prepare, read_csv  # noqa: F401
from .model import CensoredStateNet  # noqa: F401
from .simulation import completion_times, simulate, traces_to_log  # noqa: F401
