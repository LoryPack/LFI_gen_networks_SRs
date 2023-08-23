from scipy import stats
import torch
import numpy as np
from torch import Tensor
from mpmath import gammainc
from sklearn.gaussian_process.kernels import Matern
#from prior import SpatialConditionalExtremesPrior as Prior
#import matplotlib.pyplot as plt

# ---- Helper functions for Conditional Extremes model ----

def incgammalowerunregularised(a,z):
    return np.asarray([gammainc(a, 0, zi, regularized=False)
                       for zi in z]).astype(float)


def subbotin_density(x, mu, tau, d):   
    return d * np.exp(-(abs((x - mu)/tau)) ** d) / (2* tau * stats.gamma(1/d))


def subbotin_distr(q, mu, tau, d):
    return 0.5 + 0.5 * np.sign(q - mu) * (1 / stats.gamma(1/d)) * incgammalowerunregularised(1/d, abs((q - mu)/tau) ** d)


def subbotin_quantile(p, mu, tau, d):
    return mu + np.sign(p - 0.5) * (tau ** d * stats.gamma.ppf(2 * abs(p - 0.5), (1/d))) ** (1/d)


def a(h, z, lam, kappa):
    return z * np.exp(-(np.apply_along_axis(np.linalg.norm, 1, h) / lam) ** kappa)


def b(h, z, beta, lam, kappa):
    return 1 + a(h, z, lam = lam, kappa = kappa) ** beta


def delta(h, d1):
    return 1 + np.exp(-(h / d1) ** 2)


def cov_matrix(h, rho, nu):
    kernel = Matern(length_scale=rho, nu=nu)
    return kernel(h)


def std_dev_proc(h, rho, nu):
    return np.sqrt(2 - 2 * cov_matrix(h, rho, nu))


def t(y, mu, tau, d):
    return  subbotin_quantile(stats.norm.cdf(y), mu, tau, d)


def array_to_grid_with_padding(data, rows, cols, data_idx):
    grid = np.zeros((rows,cols))

    for num,idx in enumerate(data_idx):
        row = idx % rows
        col = idx // rows
        grid[row,col] = data[num]

    return np.array(grid)

def simulate(
    theta: np.ndarray,
    h: np.ndarray,
    s0_idx: int,
    u: float,
    data_idx: np.ndarray
) -> Tensor:
    """
    Return realisation of a field modeled using the Conditional Extremes model.

    Args:
        theta: input parameters.
        h: s - s0, where s0 is the conditioning site, and s is the location (from data)
        s0_ind: the index of the conditioning site (from data)
        u: the threshold over which the temp is considered extreme (empirical from data)
        data_idx: the indices at which we have observations in the padded grid
    """

    # Parameters associated with a(.) and b(.):
    kappa = theta[0]
    lam = theta[1]
    beta = theta[2]
    # Covariance parameters associated with the Gaussian process
    rho = theta[3]
    nu = theta[4]
    # Location and scale parameters for the residual process
    mu = theta[5]
    tau = theta[6]
    d1 = theta[7]

    L = np.linalg.cholesky(cov_matrix(h, rho, nu))

    # Construct the parameter δ used in the Subbotin distribution:
    d = delta(np.apply_along_axis(np.linalg.norm, 1, h), d1)

    # Observed datum at the conditioning site, Z₀:    
    Z_0 = u + np.random.exponential()

	# Simulate a mean-zero Gaussian random field with unit marginal variance,
    # independently of Z₀. Note that Ỹ inherits the order of L. Therefore, we
	# can use s₀_idx to access s₀ in all subsequent vectors.
    Y_hat  = np.dot(L, np.random.normal(size=(678)))


	# Adjust the Gaussian process so that it is 0 at s₀
    Y_hat_0 = Y_hat - Y_hat[s0_idx]

	# Transform to unit variance:
	# σ̃₀ = sqrt.(2 .- 2 *  matern.(h, ρ, ν))
    Y_hat_0_1 = Y_hat_0 / std_dev_proc(h, rho, nu)[s0_idx]
    Y_hat_0_1[s0_idx] = 0 # avoid pathology by setting Ỹ₀₁(s₀) = 0.

	# Probability integral transform from the standard Gaussian scale to the
	# standard uniform scale, and then inverse probability integral transform
	# from the standard uniform scale to the Subbotin scale:
    
    Y = t(Y_hat_0_1, mu, tau, d)
    
	# Apply the functions a(⋅) and b(⋅) to simulate data throughout the domain:
    Z = a(h, Z_0, lam = lam, kappa = kappa) + b(h, Z_0, beta = beta, lam = lam, kappa = kappa) * Y


    #Variance stabiliser in the original paper
    #Z = np.cbrt(Z)

    full_grid = array_to_grid_with_padding(Z.flatten(), 29,37,data_idx)


    return torch.FloatTensor(full_grid)




class SpatialConditionalExtremesSim:
    """Spatial Conditional Extremes Simulator."""

    def __init__(self,):
        """Initialise Spatial Conditional Extremes simulator for the RedSea dataset."""
        

        S = np.genfromtxt('data_RedSea/S.csv', delimiter=',')[1:,1:]
        s0 = np.genfromtxt('data_RedSea/s0.csv', delimiter=',')[1:,1:]
        u = np.genfromtxt('data_RedSea/u.csv', delimiter=',')[1:,1:]
        h = S-s0
        s0_idx = next(idx for idx, x in enumerate(h.tolist()) if x == [0,0])
        data_idx = np.genfromtxt('data_RedSea/data_idx.csv', delimiter=',', dtype=int)[1:,1:].flatten() - 1
        
        self.h = h
        self.s0_idx = s0_idx
        self.u = u
        self.data_idx = data_idx

    def __call__(self, theta: Tensor) -> Tensor:
        """Call to simulator."""
        return self.simulate_full_field(theta).unsqueeze(1)

    def simulate_full_field(self, theta: Tensor,) -> Tensor:
        """Forward pass."""

        return torch.stack([simulate(th.squeeze().numpy(), self.h, self.s0_idx, self.u, self.data_idx) for th in theta])
    


'''
prior = Prior()
sim = SpatialConditionalExtremesSim()

print(prior(10).shape)
print(sim(prior(10)).shape)

plt.imshow(sim(prior(1))[0][0])
plt.show()
'''