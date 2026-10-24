"""Метрики качества и кросс-валидация (``pip install survpm[metrics]``).

* :mod:`.process` — уровень 1, process mining (PM4Py);
* :mod:`.stochastic` — уровень 2, вероятности путей (Ebi, POT);
* :mod:`.temporal` — уровень 3, время (PM4Py, log-distance-measures, lifelines, scikit-survival);
* уровень 4 — :meth:`survpm.CensoredStateNet.nll`;
* :mod:`.crossval` — всё вместе на k фолдах; свои метрики — ``CrossValidator.set_metrics``.
"""
