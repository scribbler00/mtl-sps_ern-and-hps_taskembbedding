import torch
import random
import numpy as np
from sklearn.base import BaseEstimator

__author__ = "Jens Schreiber"
__copyright__ = "Copyright 2017, Intelligent Embedded Systems, Universität Kassel"
__status__ = "Prototype"


class TrAdaBoostR2(BaseEstimator):
    """ TrAdaBoost.R2 following details of [Pardoe2010]."""

    def __init__(
        self, estimator, training_function, get_preds, n_iters=10,
    ):
        self.estimator = estimator
        self.training_function = training_function
        self.get_preds = get_preds
        self.n_iters = n_iters

    def _normalize_weights(self, weights):
        return weights / np.sum(weights)

    def fit(self, X_source, y_source, X_target, y_target, init_weights=None):
        X, y = (
            np.concatenate([X_source, X_target], axis=1),
            np.concatenate([y_source, y_target], axis=1),
        )

        n = X_source.shape[0]
        m = X_target.shape[0]
        n_samples = n + m

        if init_weights is None:
            w = np.ones(n_samples)

        for t in range(self.n_iters):
            w = self._normalize_weights(w)
            self.estimator = self.training_function(self.estimator, w)

            y_source_preds = self.get_preds(self.estimator, X_source)
            e = y_source - y_source_preds
            Dt = np.max(e)
            e = e / Dt
            e_t = np.sum(e) * w

            if e_t > 0.5:
                break

            beta_t = e_t / (1 - e_t)
            w = self._normalize_weights(w * beta_t)
