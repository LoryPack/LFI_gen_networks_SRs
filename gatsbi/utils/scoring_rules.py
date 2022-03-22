import logging
from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np
import torch
from torch import Tensor


# define an abstract class for the SRs
class ScoringRule(metaclass=ABCMeta):
    """This is the abstract class for the ScoringRules"""

    @abstractmethod
    def estimate_score_batch(self, forecast,
                             verification):
        """
        Add docstring
        """

        raise NotImplementedError


class EnergyScore(ScoringRule):
    """ Estimates the EnergyScore. Here, I assume the observations and simulations are lists of
    length respectively n_obs and n_sim. Then, for each fixed observation the n_sim simulations are used to estimate the
    scoring rule. Subsequently, the values are summed over each of the n_obs observations.

    Note this scoring rule is connected to the energy distance between probability distributions.
    """

    def __init__(self, beta=1, mean=True):
        """default value is beta=1"""
        self.beta = beta
        self.logger = logging.getLogger("Energy Score")

        self.mean = mean

        if not 0 < beta < 2:
            self.logger.warning("Beta should be in (0,2) for the Energy Score to be strictly proper. Computations "
                                "will still proceed but the results may be incongrous.")

    def score(self, observations, simulations):
        """observations is an array of size (n_obs, p) (p being the dimensionality), while simulations is an array
        of size (n_sim, p).  This works on numpy in the framework of the genBayes with SR paper.
        We estimate this by building an empirical unbiased estimate of Eq. (2) in Ziel and Berk 2019"""

        return self.estimate_energy_score_numpy(observations, simulations)

    def estimate_energy_score_numpy(self, observations, simulations):
        """observations is an array of size (n_obs, p) (p being the dimensionality), while simulations is an array
        of size (n_sim, p). This works on numpy in the framework of the genBayes with SR paper.
        We estimate this by building an empirical unbiased estimate of Eq. (2) in Ziel and Berk 2019"""
        n_obs = observations.shape[0]
        n_sim, p = simulations.shape
        diff_X_y = observations.reshape(n_obs, 1, -1) - simulations.reshape(1, n_sim, p)
        # check (specifically in case n_sim==p):
        # diff_X_y2 = np.zeros((observations.shape[0], *simulations.shape))
        # for i in range(observations.shape[0]):
        #     for j in range(n_sim):
        #         diff_X_y2[i, j] = observations[i] - simulations[j]
        # assert np.allclose(diff_X_y2, diff_X_y)
        diff_X_y = np.einsum('ijk, ijk -> ij', diff_X_y, diff_X_y)

        diff_X_tildeX = simulations.reshape(1, n_sim, p) - simulations.reshape(n_sim, 1, p)
        # check (specifically in case n_sim==p):
        # diff_X_tildeX2 = np.zeros((n_sim, n_sim, p))
        # for i in range(n_sim):
        #     for j in range(n_sim):
        #         diff_X_tildeX2[i, j] = simulations[j] - simulations[i]
        # assert np.allclose(diff_X_tildeX2, diff_X_tildeX)
        diff_X_tildeX = np.einsum('ijk, ijk -> ij', diff_X_tildeX, diff_X_tildeX)

        if self.beta != 2:
            diff_X_y **= (self.beta / 2.0)
            diff_X_tildeX **= (self.beta / 2.0)

        result = 2 * np.sum(np.mean(diff_X_y, axis=1)) - n_obs * np.sum(diff_X_tildeX) / (n_sim * (n_sim - 1))

        if self.mean:
            result /= observations.shape[0]

        return result

    def estimate_score_batch(self, forecast,
                             verification):
        """The previous implementation considered a set of simulations and a set of observations, and estimated the
        score separately for each observation with the provided simulations. Here instead we have a batch
        of (simulations, observation); then it corresponds to the one above when batch_size=1 and the observation size
        is =1. We want therefore an implementation which works parallely over batches."""

        batch_size, ensemble_size, data_size = forecast.shape

        # old version: the gradient computation when using this failed, when taking the power of diff_X_tildeX, due to
        # that matrix containing 0 entries; if self.beta_over_2 < 1, the gradient had a 0 term in the denominator, which
        # lead to nan values. The new version uses a torch command which computes the pairwise distances and does not
        # lead to nan gradients. It is also slightly faster.
        # diff_X_y = verification.reshape(batch_size, 1, data_size) - forecast
        # diff_X_y = torch.einsum('bep, bep -> be', diff_X_y, diff_X_y)
        #
        # diff_X_tildeX = forecast.reshape(batch_size, 1, ensemble_size, data_size) - (forecast.reshape(
        #     batch_size, ensemble_size, 1,
        #     data_size))  # idea could be adding an epsilon for numerical stability, but does not seem to work.
        # diff_X_tildeX = torch.einsum('befp, befp -> bef', diff_X_tildeX, diff_X_tildeX)
        #
        # if self.beta_over_2 != 1:
        #     diff_X_y = torch.pow(diff_X_y, self.beta_over_2)
        #     diff_X_tildeX = torch.pow(diff_X_tildeX, self.beta_over_2)

        # the following should have shape  ["batch", "ensemble_size", "data_size"], contains all differences of each
        # verification from its own forecasts
        diff_X_y = torch.cdist(verification.reshape(batch_size, 1, data_size), forecast, p=2)
        diff_X_y = torch.squeeze(diff_X_y, dim=1)

        # the following should have shape  ["batch", "ensemble_size", "ensemble_size", "data_size"], contains all
        # differences of each verification from each other verification for each batch element
        diff_X_tildeX = torch.cdist(forecast, forecast, p=2)

        if self.beta != 1:
            diff_X_tildeX = torch.pow(diff_X_tildeX, self.beta)
            diff_X_y = torch.pow(diff_X_y, self.beta)

        result = 2 * torch.sum(torch.mean(diff_X_y, dim=1)) - torch.sum(diff_X_tildeX) / (
                ensemble_size * (ensemble_size - 1))

        if self.mean:
            result /= verification.shape[0]

        return result


class KernelScore(ScoringRule):

    def __init__(self, kernel="gaussian", biased_estimator=False, torch=True, mean=True, **kernel_kwargs):
        """
        Parameters
        ----------
        kernel : str or callable, optional
            Can be a string denoting the kernel, or a function. If a string, only gaussian is implemented for now; in
            that case, you can also provide an additional keyword parameter 'sigma' which is used as the sigma in the
            kernel.
        """

        self.mean = mean

        self.kernel_vectorized = False
        if not isinstance(kernel, str) and not callable(kernel):
            raise RuntimeError("'kernel' must be either a string or a function of two variables returning a scalar.")
        if isinstance(kernel, str):
            if kernel == "gaussian":
                if torch:
                    self.kernel = self.def_gaussian_kernel_torch(**kernel_kwargs)
                else:
                    self.kernel = self.def_gaussian_kernel(**kernel_kwargs)
                self.kernel_vectorized = True  # the gaussian kernel is vectorized
            elif kernel == "rational_quadratic":
                if torch:
                    self.kernel = self.def_rational_quadratic_kernel_torch(**kernel_kwargs)
                else:
                    self.kernel = self.def_rational_quadratic_kernel(**kernel_kwargs)
                self.kernel_vectorized = True  # the rational_quadratic kernel is vectorized
            else:
                raise NotImplementedError("The required kernel is not implemented.")
        else:
            self.kernel = kernel  # if kernel is a callable already

        self.biased_estimator = biased_estimator

    def score(self, observations, simulations):
        """observations is an array of size (n_obs, p) (p being the dimensionality), while simulations is an array
        of size (n_sim, p). This works on numpy in the framework of the genBayes with SR paper.
        We estimate this by building an empirical unbiased estimate of Eq. (2) in Ziel and Berk 2019"""

        # compute the Gram matrix
        K_sim_sim, K_obs_sim = self.compute_Gram_matrix(observations, simulations)

        # Estimate MMD
        if self.biased_estimator:
            result = self.MMD_V_estimator(K_sim_sim, K_obs_sim)
        else:
            result = self.MMD_unbiased(K_sim_sim, K_obs_sim)

        if self.mean:
            result /= observations.shape[0]
        return result

    def estimate_score_batch(self, forecast,
                             verification):
        """The previous implementation considered a set of simulations and a set of observations, and estimated the
        score separately for each observation with the provided simulations. Here instead we have a batch
        of (simulations, observation); then it corresponds to the one above when batch_size=1 and the observation size
        is =1. We want therefore an implementation which works parallely over batches."""

        # compute the Gram matrix
        K_sim_sim, K_obs_sim = self.compute_Gram_matrix_batch(forecast, verification)

        # Estimate MMD
        if self.biased_estimator:
            result = self.MMD_V_estimator_batch(K_sim_sim, K_obs_sim)
        else:
            result = self.MMD_unbiased_batch(K_sim_sim, K_obs_sim)

        if self.mean:
            result /= verification.shape[0]

        return result

    @staticmethod
    def def_gaussian_kernel(sigma=1):
        sigma_2 = 2 * sigma ** 2

        def Gaussian_kernel_vectorized(X, Y):
            """Here X and Y have shape (n_samples_x, n_features) and (n_samples_y, n_features);
            this directly computes the kernel for all pairwise components"""
            XY = X.reshape(X.shape[0], 1, -1) - Y.reshape(1, Y.shape[0], -1)  # pairwise differences
            return np.exp(- np.einsum('xyi,xyi->xy', XY, XY) / sigma_2)

        return Gaussian_kernel_vectorized

    @staticmethod
    def def_rational_quadratic_kernel(alpha=1):
        # this follows definition in Bińkowski, M., Sutherland, D. J., Arbel, M., & Gretton, A. (2018).
        # Demystifying MMD GANs. arXiv preprint arXiv:1801.01401.
        alpha_2 = 2 * alpha

        def rational_quadratic_vectorized(X, Y):
            """Here X and Y have shape (n_samples_x, n_features) and (n_samples_y, n_features);
            this directly computes the kernel for all pairwise components"""
            XY = X.reshape(X.shape[0], 1, -1) - Y.reshape(1, Y.shape[0], -1)  # pairwise differences
            return (1 + np.einsum('xyi,xyi->xy', XY, XY) / alpha_2) ** (-alpha)

        return rational_quadratic_vectorized

    def compute_Gram_matrix(self, observations, simulations):

        if self.kernel_vectorized:
            K_sim_sim = self.kernel(simulations, simulations)
            K_obs_sim = self.kernel(observations, simulations)
        else:
            n_obs = observations.shape[0]
            n_sim = simulations.shape[0]

            K_sim_sim = np.zeros((n_sim, n_sim))
            K_obs_sim = np.zeros((n_obs, n_sim))

            for i in range(n_sim):
                # we assume the function to be symmetric; this saves some steps:
                for j in range(i, n_sim):
                    K_sim_sim[j, i] = K_sim_sim[i, j] = self.kernel(simulations[i], simulations[j])

            for i in range(n_obs):
                for j in range(n_sim):
                    K_obs_sim[i, j] = self.kernel(observations[i], simulations[j])

        return K_sim_sim, K_obs_sim

    @staticmethod
    def MMD_unbiased(K_sim_sim, K_obs_sim):
        # Adapted from https://github.com/eugenium/MMD/blob/2fe67cbc7378f10f3b273cfd8d8bbd2135db5798/mmd.py
        # The estimate when distribution of x is not equal to y
        n_obs, n_sim = K_obs_sim.shape

        t_obs_sim = (2. / n_sim) * np.sum(K_obs_sim)
        t_sim_sim = (1. / (n_sim * (n_sim - 1))) * np.sum(K_sim_sim - np.diag(np.diagonal(K_sim_sim)))

        return n_obs * t_sim_sim - t_obs_sim

    @staticmethod
    def MMD_V_estimator(K_sim_sim, K_obs_sim):
        # The estimate when distribution of x is not equal to y
        n_obs, n_sim = K_obs_sim.shape

        t_obs_sim = (2. / n_sim) * np.sum(K_obs_sim)
        t_sim_sim = (1. / (n_sim * n_sim)) * np.sum(K_sim_sim)

        return n_obs * t_sim_sim - t_obs_sim

    @staticmethod
    def def_gaussian_kernel_torch(sigma=1):
        sigma_2 = 2 * sigma ** 2

        def Gaussian_kernel_vectorized(X,
                                       Y):
            """Here X and Y have shape (n_samples_x, n_features) and (n_samples_y, n_features);
            this directly computes the kernel for all pairwise components"""
            XY = torch.cdist(X, Y)
            return torch.exp(- torch.pow(XY, 2) / sigma_2)

        return Gaussian_kernel_vectorized

    @staticmethod
    def def_rational_quadratic_kernel_torch(alpha=1):
        alpha_2 = 2 * alpha

        def rational_quadratic_kernel_vectorized(X, Y):
            """Here X and Y have shape (n_samples_x, n_features) and (n_samples_y, n_features);
            this directly computes the kernel for all pairwise components"""
            XY = torch.cdist(X, Y)
            return torch.pow(1 + torch.pow(XY, 2) / alpha_2, -alpha)

        return rational_quadratic_kernel_vectorized

    def compute_Gram_matrix_batch(self, forecast, verification):

        batch_size, ensemble_size, data_size = forecast.shape

        if self.kernel_vectorized:
            verification = verification.reshape(batch_size, 1, data_size)
            K_sim_sim = self.kernel(forecast, forecast)
            K_obs_sim = self.kernel(verification, forecast)
        else:

            K_sim_sim = torch.zeros((batch_size, ensemble_size, ensemble_size))
            K_obs_sim = torch.zeros((batch_size, 1, ensemble_size))

            for b in range(batch_size):
                for i in range(ensemble_size):
                    # we assume the function to be symmetric; this saves some steps:
                    for j in range(i, ensemble_size):
                        K_sim_sim[b, j, i] = K_sim_sim[b, i, j] = self.kernel(forecast[b, i], forecast[b, j])

                for j in range(ensemble_size):
                    K_obs_sim[b, 0, j] = self.kernel(verification[b], forecast[b, j])

        return K_sim_sim, K_obs_sim

    @staticmethod
    def MMD_unbiased_batch(K_sim_sim, K_obs_sim):
        # Adapted from https://github.com/eugenium/MMD/blob/2fe67cbc7378f10f3b273cfd8d8bbd2135db5798/mmd.py
        # The estimate when distribution of x is not equal to y
        batch_size, ensemble_size, _ = K_sim_sim.shape

        t_obs_sim = (2. / ensemble_size) * torch.sum(K_obs_sim)

        # sum only the off-diagonal elements of K_sim_sim: first set them to 0:
        # this does not work inside automatic differentiation!
        # K_sim_sim[:, range(ensemble_size), range(ensemble_size)] = 0
        # t_sim_sim = (1. / (ensemble_size * (ensemble_size - 1))) * torch.sum(K_sim_sim)

        # alternatively, sum only the off-diagonal elements.
        off_diagonal_sum = torch.sum(
            K_sim_sim.masked_select(
                torch.stack([~torch.eye(ensemble_size, dtype=bool, device=K_sim_sim.device)] * batch_size)))
        t_sim_sim = (1. / (ensemble_size * (ensemble_size - 1))) * off_diagonal_sum

        return t_sim_sim - t_obs_sim

    @staticmethod
    def MMD_V_estimator_batch(K_sim_sim, K_obs_sim):
        # Adapted from https://github.com/eugenium/MMD/blob/2fe67cbc7378f10f3b273cfd8d8bbd2135db5798/mmd.py
        # The estimate when distribution of x is not equal to y
        batch_size, ensemble_size, _ = K_sim_sim.shape

        t_obs_sim = (2. / ensemble_size) * torch.sum(K_obs_sim)

        t_sim_sim = (1. / (ensemble_size * ensemble_size)) * torch.sum(K_sim_sim)

        return t_sim_sim - t_obs_sim

    # todo speed up by reciclying previous computations of 2. / ensemble_size and similar?


def energy_score(forecast, verification):
    return EnergyScore().estimate_score_batch(forecast, verification)


def kernel_score(forecast, verification):
    return KernelScore().estimate_score_batch(forecast, verification)


sr_dict = {"energy_score": energy_score, "kernel_score": kernel_score}