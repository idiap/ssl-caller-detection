# Copyright (c) 2023, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Enno Hermann, Eklavya Sarkar <eklavya.sarkar@idiap.ch>


import numpy as np


class Gaussian:
    """Gaussian with diagonal covariance matrix."""

    def __init__(
        self,
        mean: np.ndarray = None,
        cov: np.ndarray = None,
        mean_inv_cov: np.ndarray = None,
        inv_cov: np.ndarray = None,
    ):
        """
        Two ways to initialise:
        1. Means and covariances
        2. Means * inverse covariances and inverse covariances (Kaldi-style)
        """
        if mean is not None and cov is not None:
            assert mean.shape == cov.shape

            self.mean = mean.reshape(-1, 1)
            self.cov = cov.reshape(-1, 1)
            self.inv_cov = 1 / self.cov
        elif mean_inv_cov is not None and inv_cov is not None:
            assert mean_inv_cov.shape == inv_cov.shape

            self.inv_cov = inv_cov.reshape(-1, 1)
            self.cov = 1 / self.inv_cov
            self.mean = mean_inv_cov.reshape(-1, 1) * self.cov
        else:
            raise ValueError

        self.d = self.mean.shape[0]
        self.det = np.prod(self.cov)
        self.log_det = np.sum(np.log(self.cov))

    def __key(self):
        return (self.mean.data.tobytes(), self.cov.data.tobytes())

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return np.all(self.mean == other.mean) and np.all(self.cov == other.cov)

    @staticmethod
    @np.vectorize
    def D_KL(g1: "Gaussian", g2: "Gaussian", symmetric=False) -> float:
        """KL divergence."""
        assert g1.d == g2.d
        D_KL1 = 0.5 * (
            np.log(g2.det / g1.det)
            + np.sum(g2.inv_cov * g1.cov)
            + (g1.mean - g2.mean).T.dot(g2.inv_cov * (g1.mean - g2.mean))
            - g1.d
        )
        if symmetric:
            D_KL2 = 0.5 * (
                np.log(g2.det / g1.det)
                + np.sum(g2.inv_cov * g1.cov)
                + (g1.mean - g2.mean).T.dot(g2.inv_cov * (g1.mean - g2.mean))
                - g1.d
            )
            return 0.5 * (D_KL1 + D_KL2)
        else:
            return D_KL1

    @staticmethod
    @np.vectorize
    def D_KL_log(g1: "Gaussian", g2: "Gaussian", symmetric=False) -> float:
        """KL divergence using log."""
        assert g1.d == g2.d
        D_KL1 = 0.5 * (
            g2.log_det
            - g1.log_det
            + np.sum(g2.inv_cov * g1.cov)
            + (g1.mean - g2.mean).T.dot(g2.inv_cov * (g1.mean - g2.mean))
            - g1.d
        )

        if symmetric:
            D_KL2 = 0.5 * (
                g1.log_det
                - g2.log_det
                + np.sum(g1.inv_cov * g2.cov)
                + (g2.mean - g1.mean).T.dot(g1.inv_cov * (g2.mean - g1.mean))
                - g2.d
            )
            return 0.5 * (D_KL1 + D_KL2)
        else:
            return D_KL1

    @staticmethod
    @np.vectorize
    def D_Bhatt(g1: "Gaussian", g2: "Gaussian") -> float:
        """Bhattacharyya distance."""
        mean_cov = 0.5 * (g1.cov + g2.cov)
        mean_inv_cov = 1 / mean_cov
        term_1 = 0.125 * (g1.mean - g2.mean).T.dot(
            mean_inv_cov * (g1.mean - g2.mean)
        )
        term_2 = 0.5 * np.log(np.prod(mean_cov) / np.sqrt(g1.det * g2.det))
        return term_1 + term_2

    @staticmethod
    @np.vectorize
    def D_Bhatt_log(g1: "Gaussian", g2: "Gaussian") -> float:
        """Bhattacharyya distance using log."""
        mean_cov = 0.5 * (g1.cov + g2.cov)
        mean_inv_cov = 1 / mean_cov
        term_1 = 0.125 * (g1.mean - g2.mean).T.dot(
            mean_inv_cov * (g1.mean - g2.mean)
        )
        term_2 = 0.5 * (
            np.sum(np.log(mean_cov))
            - (
                np.sum(np.log(np.sqrt(g1.cov)))
                + np.sum(np.log(np.sqrt(g2.cov)))
            )
        )
        return term_1 + term_2
