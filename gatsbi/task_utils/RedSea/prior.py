from typing import Optional

import numpy as np
from torch import Tensor, FloatTensor


class SpatialConditionalExtremesPrior:

    def __init__(
        self,
        kappa: Optional[tuple] = (1,2),
        _lambda: Optional[tuple] = (1,5),
        beta: Optional[tuple] = (0.05,1),
        rho: Optional[tuple] = (0.2,5),
        nu: Optional[tuple] = (0.4,2),
        mu: Optional[tuple] = (-0.5,0.5),
        tau: Optional[tuple] = (0.2,1),
        delta1: Optional[tuple] = (1,3)

    ):
        """
        Set up prior.

        Args:
            kappa, lambda, beta: parameters associated with a(.) and b(.)
            rho, nu: Covariance parameters associated with the Gaussian process
            mu, tau: Location and scale parameters for the Subbotin distribution
            delta1: Parameters used to construct the shape parameter delta

        """
        self.kappa = kappa
        self._lambda = _lambda
        self.beta = beta
        self.rho = rho
        self.nu = nu
        self.mu = mu
        self.tau = tau
        self.delta1 = delta1


    def __call__(self, num_samples: int = 1, seed: int = 42) -> Tensor:
        """Return random batch of parameters from Uniform."""
        return self.sample(num_samples, seed).unsqueeze(1)

    def sample(self, num_samples: int, seed: Optional[int] = 42) -> Tensor:
        """
        Forward pass.

        num_samples: number of depth profile samples from prior
        seed: random-sampling seed. Default is 42.
        """
        param_sample = np.ndarray((num_samples,8))
        
        for idx, param in enumerate([self.kappa,
                                     self._lambda,
                                     self.beta,
                                     self.rho,
                                     self.nu,
                                     self.mu,
                                     self.tau,
                                     self.delta1]):
            param_sample[:,idx] = np.random.uniform(param[0],param[1],num_samples)
            
        return FloatTensor(param_sample).unsqueeze(1)

