"""Метрики качества и кросс-валидация (``pip install survpm[metrics]``).

* :mod:`.process` — уровень 1, process mining (PM4Py);
* :mod:`.stochastic` — уровень 2, вероятности путей (Ebi, POT);
* :mod:`.temporal` — уровень 3, время (PM4Py, log-distance-measures, lifelines, scikit-survival);
* :mod:`.likelihood` — уровень 4, −log L с цензурой;
* :mod:`.crossval` — выбранные метрики на k фолдах: ``CrossValidator.add_metrics``.
"""
