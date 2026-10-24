# Статус 24.10: метрики качества

Метрики считаются на 5-fold кросс-валидации по целым трассам (фолды равны по доле
цензурированных трасс) для двух моделей: `naive` и `censor_aware`. Корпус: синтетика A–D
и Sepsis. Код: `survpm.metrics` (`process`, `stochastic`, `temporal`, `crossval`),
результаты: раздел 7 ноутбуков `petri_net_time_estimation(_sepsis).ipynb`,
формулы: `metrics.pdf`. Цензурированная трасса везде считается префиксом.

survpm - начало библиотеки, пока сырая версия, нужна еще неделя, чтобы доработать метрики и валидацию. Пока не думал насчет способом расчета функций выживания с учетом времени delta_t перехода между состояниями. Попробую на этой неделе.

Обзор: `survey/Probabilistic Transition Petri Net/Краткий обзор методов оценки сетей Петри
и генерируемых ими журналов событий.pdf', откуда брал метрики. +добавил свои и PM4py

## Что добавлено

| Уровень | Метрика | Чем считается | В обзоре |
|---|---|---|---|
| 1 | fitness (token replay, для префиксов) | PM4Py + своя обёртка | да, F_tok |
| 1 | precision (ETC, token) | PM4Py | да, P_EE |
| 1 | generalization | PM4Py | нет |
| 1 | simplicity | PM4Py | частично (вектор сложности) |
| 1 | сравнение сети состояний с Inductive Miner | PM4Py | нет |
| 2 | entropic relevance | Ebi, SDFA с остановкой q | да, ER |
| 2 | uEMSC = 1 − TV по трассам | Ebi | да |
| 2 | EMD языков, нормированный Левенштейн | POT + jellyfish, своё | да, EMSC |
| 3 | доля отклонений temporal profile (ζ = 6) | PM4Py | нет |
| 3 | CFLD, `ngram_3`, `cycle_time_emd`, `absolute_event_emd` | log-distance-measures | нет (только ссылка [10]) |
| 3 | `edge_ks`, KS по времени на рёбрах | scipy, своё | нет |
| 3 | `km_sup`, `km_l1_days` (до Каплана–Майера) | lifelines, своё | нет |
| 3 | log-rank | lifelines | нет |
| 3 | IBS с IPCW | scikit-survival | близко к CRPS |
| 4 | `nll_per_interval`, `nll_per_trace` с цензурой | своё (`CensoredStateNet.nll`) | близко к LL времени |
| 4 | `unseen_share` | своё | нет |
| — | `median_error_days` (пример через `set_metrics`) | своё, lifelines | нет |

Новое в библиотеке: `CrossValidator(...).set_metrics(имя=функция | None)` добавляет,
заменяет или убирает метрику; `per_model` превращает `metric(model, fold)` в метрику по всем моделям.

## Из обзора

**Взято:** token replay, escaping-edge precision, uEMSC (1 − TV по трассам), EMSC/EMD
с нормированным редакционным расстоянием, entropic relevance.

**Взято с отличиями:**
* сложность: только `simplicity` из PM4Py, без вектора |P|, |T|, |F|, d_max;
* LL времени: параметрические плотности модели и цензура вместо ядерной оценки (KDE);
* CRPS: отдельно не считается, его аналог с учётом цензуры — IBS;
* alignment fitness была, удалена вместе с вариантом A (только полные трассы).

**Не взято:** soundness / корректность завершения, F_strict, охранные условия и duEMSC
(в модели нет данных), TV по действиям, JS-расстояние, условное сравнение ветвей D_z,
раздельные длительность действия и ожидание, весь раздел 4 (сравнение двух сетей:
сходство узлов, GED, эквивалентность языков, бисимуляция, causal footprint).

## Сверх обзора

generalization и сравнение с Inductive Miner; temporal profile; метрики Chapela-Campa
(CFLD, `ngram_3`, `cycle_time_emd`, `absolute_event_emd`); `edge_ks`; метрики выживания
(`km_sup`, `km_l1_days`, log-rank, IBS); `unseen_share`; `median_error_days`.

## Открытые вопросы

* На Sepsis время цензуры почти всегда ≈ 0 (момент окончания наблюдения неизвестен),
  поэтому `naive` и `censor_aware` там почти не различаются; разницу показывает синтетика A–D.
