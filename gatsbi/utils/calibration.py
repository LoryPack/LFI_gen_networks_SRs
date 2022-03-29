import numpy as np
from scipy.stats import binom
from sklearn.metrics import r2_score
import torch
from gatsbi.networks.base import WrapGenMultipleSimulations
from gatsbi.optimize.utils import _sample


def sbc(theta_samples, theta_test, param_names, bins=20,
        figsize=(15, 5), interval=0.99, show=False, filename=None, font_size=12):
    """
    Plots the simulation-based posterior checking histograms as advocated by Talts et al. (2018).
    """
    theta_samples = theta_samples.transpose((1, 0, 2))
    # Plot settings
    # plt.rcParams['font.size'] = font_size
    N = int(theta_test.shape[0])

    # # Prepare figure
    # if len(param_names) >= 6:
    #     n_col = int(np.ceil(len(param_names) / 2))
    #     n_row = 2
    # else:
    #     n_col = int(len(param_names))
    #     n_row = 1
    # # Initialize figure
    # f, axarr = plt.subplots(n_row, n_col, figsize=figsize)
    # if n_row == n_col == 1:
    #     axarr = [axarr]
    # if n_row > 1:
    #     axarr = axarr.flat

    # Compute ranks (using broadcasting)
    ranks = np.sum(theta_samples < theta_test[:, np.newaxis, :], axis=1)

    # Compute interval
    endpoints = binom.interval(interval, N, 1 / (bins + 1))

    # Plot histograms
    for j in range(len(param_names)):

        # Add interval
        axarr[j].axhspan(endpoints[0], endpoints[1], facecolor='gray', alpha=0.3)
        axarr[j].axhline(np.mean(endpoints), color='gray', zorder=0, alpha=0.5)

        sns.distplot(ranks[:, j], kde=False, ax=axarr[j], color='#a34f4f',
                     hist_kws=dict(edgecolor="k", linewidth=1, alpha=1.), bins=bins)

        axarr[j].set_title(param_names[j])
        axarr[j].spines['right'].set_visible(False)
        axarr[j].spines['top'].set_visible(False)
        if j == 0:
            axarr[j].set_xlabel('Rank statistic')
        axarr[j].get_yaxis().set_ticks([])

        f.tight_layout(pad=0.6)

    # Show, if specified
    if show:
        plt.show()
    # Save if specified
    if filename is not None:
        f.savefig(filename, dpi=200)
    plt.close()


def generate_test_set_for_calibration(task, generator, n_test_samples, n_generator_simulations, sample_seed):
    simulator = task.get_simulator()
    prior = task.get_prior()

    test_theta, test_obs = _sample(
        prior=prior,
        simulator=simulator,
        sample_seed=sample_seed + 1,
        num_samples=n_test_samples,  # number of samples to use for calibration
    )

    # need to generate synthetic thetas, some of them for each of the observation
    gen_wrapped = WrapGenMultipleSimulations(
        generator, n_simulations=n_generator_simulations)  # number of generated theta per simulation

    gen_wrapped.eval()

    test_theta_fake_all_obs = []
    for obs in test_obs:
        obs = obs.unsqueeze(0)

        sample_size = n_generator_simulations
        test_theta_fake_obs = []
        while sample_size > 0:
            test_theta_fake = gen_wrapped(obs, n_simulations=sample_size)
            rej_thresh = task.prior_params["high"]
            rej_thresh = torch.tensor([1.0, 1e-03])
            # print('rej_thresh', rej_thresh)
            # print(test_theta_fake)
            # print(torch.abs(test_theta_fake) < rej_thresh)
            inds = torch.all(torch.abs(test_theta_fake) < rej_thresh, -1).reshape(-1)
            ok_elements = inds.sum()
            if ok_elements > 0:
                sample_size -= ok_elements
                test_theta_fake_obs.append(test_theta_fake[0][inds])

        test_theta_fake_all_obs.append(torch.concat(test_theta_fake_obs, 0))

    test_theta_fake_all_obs = torch.stack(test_theta_fake_all_obs, 0)

    return test_theta_fake_all_obs, test_theta


def compute_calibration_metrics(theta_samples, theta_test):
    test_theta_fake_numpy = theta_samples.transpose(1, 0).cpu().detach().numpy()
    test_theta_numpy = theta_test.cpu().numpy()

    cal_err_val = calibration_error(test_theta_fake_numpy, test_theta_numpy, alpha_resolution=100)
    r2_val = R2(test_theta_fake_numpy, test_theta_numpy)
    rmse_val = rmse(test_theta_fake_numpy, test_theta_numpy)

    return {
        "cal_err_val_mean": cal_err_val.mean(),
        "r2_val_mean": r2_val.mean(),
        "rmse_val_mean": rmse_val.mean(),
        "cal_err_val_std": cal_err_val.std(),
        "r2_val_std": r2_val.std(),
        "rmse_val_std": rmse_val.std(),
    }


def calibration_error(theta_samples, theta_test, alpha_resolution=100):
    """
    Computes the calibration error of an approximate posterior per parameters.
    The calibration error is given as the median of the absolute deviation
    between alpha (0 - 1) (credibility level) and the relative number of inliers from
    theta test.

    ----------

    Arguments:
    theta_samples       : np.ndarray of shape (n_samples, n_test, n_params) -- the samples from
                          the approximate posterior
    theta_test          : np.ndarray of shape (n_test, n_params) -- the 'true' test values
    alpha_resolution    : int -- the number of intervals to consider

    ----------

    Returns:

    cal_errs  : np.ndarray of shape (n_params, ) -- the calibration errors per parameter
    """

    n_params = theta_test.shape[1]
    n_test = theta_test.shape[0]
    alphas = np.linspace(0.01, 1.0, alpha_resolution)
    cal_errs = np.zeros(n_params)

    # Loop for each parameter
    for k in range(n_params):
        alphas_in = np.zeros(len(alphas))
        # Loop for each alpha
        for i, alpha in enumerate(alphas):
            # Find lower and upper bounds of posterior distribution
            region = 1 - alpha
            lower = np.round(region / 2, 3)
            upper = np.round(1 - (region / 2), 3)

            # Compute quantiles for given alpha using the entire sample
            quantiles = np.quantile(theta_samples[:, :, k], [lower, upper], axis=0).T

            # Compute the relative number of inliers
            inlier_id = (theta_test[:, k] > quantiles[:, 0]) & (theta_test[:, k] < quantiles[:, 1])
            inliers_alpha = np.sum(inlier_id) / n_test
            alphas_in[i] = inliers_alpha

        # Compute calibration error for k-th parameter
        diff_alphas = np.abs(alphas - alphas_in)
        cal_err = np.round(np.median(diff_alphas), 3)
        cal_errs[k] = cal_err

    return cal_errs


def rmse(theta_samples, theta_test, normalized=True):
    """
    Computes the RMSE or normalized RMSE (NRMSE) between posterior means
    and true parameter values for each parameter

    ----------

    Arguments:
    theta_samples   : np.ndarray of shape (n_samples, n_test, n_params) -- the samples from
                          the approximate posterior
    theta_test      : np.ndarray of shape (n_test, n_params) -- the 'true' test values
    normalized      : boolean -- whether to compute nrmse or rmse (default True)

    ----------

    Returns:

    (n)rmse  : np.ndarray of shape (n_params, ) -- the (n)rmse per parameter
    """

    # Convert tf.Tensors to numpy, if passed
    if type(theta_samples) is not np.ndarray:
        theta_samples = theta_samples.numpy()
    if type(theta_test) is not np.ndarray:
        theta_test = theta_test.numpy()

    theta_approx_means = theta_samples.mean(0)
    rmse = np.sqrt(np.mean((theta_approx_means - theta_test) ** 2, axis=0))

    if normalized:
        rmse = rmse / (theta_test.max(axis=0) - theta_test.min(axis=0))
    return rmse


def R2(theta_samples, theta_test):
    """
    Computes the R^2 score as a measure of reconstruction (percentage of variance
    in true parameters captured by estimated parameters)

    ----------
    Arguments:
    theta_samples   : np.ndarray of shape (n_samples, n_test, n_params) -- the samples from
                          the approximate posterior
    theta_test      : np.ndarray of shape (n_test, n_params) -- the 'true' test values

    ----------
    Returns:

    r2s  : np.ndarray of shape (n_params, ) -- the r2s per parameter
    """

    # Convert tf.Tensors to numpy, if passed
    if type(theta_samples) is not np.ndarray:
        theta_samples = theta_samples.numpy()
    if type(theta_test) is not np.ndarray:
        theta_test = theta_test.numpy()

    theta_approx_means = theta_samples.mean(0)
    return r2_score(theta_test, theta_approx_means, multioutput='raw_values')
