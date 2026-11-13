"""Кодирование признаков по обучающей выборке: пропуски, категории, стандартизация, обрезка."""
import numpy as np
import pandas as pd


class FeatureEncoder:
    """Медиана + индикатор пропуска, one-hot, стандартизация по обучению.

    Неизвестная категория кодируется как базовая; постоянные колонки удаляются.

    ``clip=True`` (по умолчанию) обрезает стандартизованные значения до диапазона, который
    встречался при обучении: значение вне диапазона train (выброс анализа) кодируется как
    ближайшее крайнее, и линейная по z модель не экстраполирует ``exp(β·z)`` за пределы данных.
    ``log_features`` — числовые признаки, к которым до стандартизации применяется ``log1p``
    (скошенные положительные величины: анализы, суммы, длительности); отрицательные значения
    в них — ошибка.
    """

    def __init__(self, clip=True, log_features=()):
        self.clip = clip
        self.log_features = list(log_features)

    def _numbers(self, column, values):
        numbers = pd.to_numeric(values, errors="coerce")
        if column in getattr(self, "log_features", ()):
            if (numbers < 0).any():
                raise ValueError(f"log_features: отрицательные значения признака {column}")
            numbers = np.log1p(numbers)
        return numbers

    def fit(self, X):
        unknown = set(self.log_features) - set(X.columns)
        if unknown:
            raise ValueError(f"log_features нет среди признаков: {sorted(unknown)}")
        self.columns_ = list(X.columns)
        self.numeric_, self.categories_, self.medians_ = [], {}, {}
        for column in self.columns_:
            values = X[column]
            numbers = self._numbers(column, values)
            if numbers.notna().sum() == values.notna().sum():
                self.numeric_.append(column)
                self.medians_[column] = numbers.median() if numbers.notna().any() else 0.
            elif column in self.log_features:
                raise ValueError(f"log_features: признак {column} не числовой")
            else:
                self.categories_[column] = sorted(values.astype(object).fillna("<missing>").astype(str).unique())
        frame = self._frame(X)
        self.mean_ = frame.mean()
        std = frame.std(ddof=0)
        self.names_ = list(std.index[std > 1e-12])
        self.std_ = std[self.names_]
        z = (frame[self.names_] - self.mean_[self.names_]) / self.std_
        self.low_, self.high_ = z.min().to_numpy(), z.max().to_numpy()
        return self

    def _frame(self, X):
        missing = set(self.columns_) - set(X.columns)
        if missing:
            raise ValueError(f"Нет признаков X: {sorted(missing)}")
        out = {}
        for column in self.numeric_:
            values = self._numbers(column, X[column])
            if np.isinf(values).any():
                raise ValueError(f"Бесконечные значения признака {column}")
            out[column] = values.fillna(self.medians_[column])
            out[f"{column}__missing"] = values.isna().astype(float)
        for column, categories in self.categories_.items():
            values = X[column].astype(object).fillna("<missing>").astype(str)
            for category in categories[1:]:
                out[f"{column}={category}"] = (values == category).astype(float)
        return pd.DataFrame(out, index=X.index, dtype=float)

    def transform(self, X):
        frame = self._frame(X)
        z = ((frame[self.names_] - self.mean_[self.names_]) / self.std_).to_numpy()
        if getattr(self, "clip", False) and hasattr(self, "low_"):
            z = np.clip(z, self.low_, self.high_)
        return z
