import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.special import erf, erfinv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_array, check_is_fitted


class GaussRankScaler(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        epsilon=1e-4,
        copy=True,
        n_jobs=None,
        interp_kind="linear",
        interp_copy=False,
    ):
        self.epsilon = epsilon
        self.copy = copy
        self.interp_kind = interp_kind
        self.interp_copy = interp_copy
        self.fill_value = "extrapolate"
        self.n_jobs = n_jobs
        self.bound = 1.0 - self.epsilon

    def fit(self, X, y=None):
        X = check_array(
            X, copy=self.copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite=True
        )

        self.interp_func_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit)(x) for x in X.T
        )
        return self

    def _fit(self, x):
        x = self.drop_duplicates(x)
        rank = np.argsort(np.argsort(x))
        factor = np.max(rank) / 2.0 * self.bound
        scaled_rank = np.clip(rank / factor - self.bound, -self.bound, self.bound)
        return interp1d(
            x,
            scaled_rank,
            kind=self.interp_kind,
            copy=self.interp_copy,
            fill_value=self.fill_value,
        )

    def transform(self, X, copy=None):
        check_is_fitted(self, "interp_func_")

        copy = copy if copy is not None else self.copy
        X = check_array(
            X, copy=copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite=True
        )

        X = np.array(
            Parallel(n_jobs=self.n_jobs)(
                delayed(self._transform)(i, x) for i, x in enumerate(X.T)
            )
        ).T
        return X

    def _transform(self, i, x):
        clipped = np.clip(self.interp_func_[i](x), -self.bound, self.bound)
        return erfinv(clipped)

    def inverse_transform(self, X, copy=None):
        check_is_fitted(self, "interp_func_")

        copy = copy if copy is not None else self.copy
        X = check_array(
            X, copy=copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite=True
        )

        X = np.array(
            Parallel(n_jobs=self.n_jobs)(
                delayed(self._inverse_transform)(i, x) for i, x in enumerate(X.T)
            )
        ).T
        return X

    def _inverse_transform(self, i, x):
        inv_interp_func = interp1d(
            self.interp_func_[i].y,
            self.interp_func_[i].x,
            kind=self.interp_kind,
            copy=self.interp_copy,
            fill_value=self.fill_value,
        )
        return inv_interp_func(erf(x))

    @staticmethod
    def drop_duplicates(x):
        is_unique = np.zeros_like(x, dtype=bool)
        is_unique[np.unique(x, return_index=True)[1]] = True
        return x[is_unique]
