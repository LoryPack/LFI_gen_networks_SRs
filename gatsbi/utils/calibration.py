import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from matplotlib.lines import Line2D
from scipy.stats import binom
from sklearn.metrics import r2_score

from gatsbi.networks import WrapGenMultipleSimulations
from gatsbi.optimize.utils import _sample


def generate_test_set_for_calibration(prior, simulator, generator, n_test_samples, n_generator_simulations,
                                      sample_seed, rej_thresh=None, data_is_image=False):
    test_theta, test_obs = _sample(
        prior=prior,
        simulator=simulator,
        sample_seed=sample_seed + 1,
        num_samples=n_test_samples,  # number of samples to use for calibration
    )

    return generate_test_set_for_calibration_from_obs(test_theta, test_obs, generator, n_test_samples,
                                                      n_generator_simulations, rej_thresh, data_is_image)


def generate_test_set_for_calibration_from_obs(test_theta, test_obs, generator, n_test_samples, n_generator_simulations,
                                               rej_thresh=None, data_is_image=False):
    test_theta = test_theta[0:n_test_samples]
    test_obs = test_obs[0:n_test_samples]

    # need to generate synthetic thetas, some of them for each of the observation
    gen_wrapped = WrapGenMultipleSimulations(
        generator, n_simulations=n_generator_simulations)  # number of generated theta per simulation

    gen_wrapped.eval()
    device = list(gen_wrapped.parameters())[0].device

    if rej_thresh is None:
        test_obs = test_obs.to(device)
        test_theta_fake_all_obs = gen_wrapped(test_obs, n_simulations=n_generator_simulations)
    else:
        test_theta_fake_all_obs = []
        for obs in test_obs:
            obs = obs.unsqueeze(0).to(device)

            test_theta_fake_obs = []
            sample_size = n_generator_simulations
            while sample_size > 0:
                test_theta_fake = gen_wrapped(obs, n_simulations=sample_size)
                # print('rej_thresh', rej_thresh)
                # print(test_theta_fake)
                # print(torch.abs(test_theta_fake) < rej_thresh)
                inds = torch.all(torch.abs(test_theta_fake) < rej_thresh, -1).reshape(-1)
                ok_elements = inds.sum()
                if ok_elements > 0:
                    sample_size -= ok_elements
                    test_theta_fake_obs.append(test_theta_fake[0][inds])
            test_theta_fake_obs = torch.concat(test_theta_fake_obs, 0)

            test_theta_fake_all_obs.append(test_theta_fake_obs)

        test_theta_fake_all_obs = torch.stack(test_theta_fake_all_obs, 0)

    # flatten the test set if it is image:
    if data_is_image:
        test_theta_fake_all_obs = test_theta_fake_all_obs.squeeze(1)
        test_theta = test_theta.flatten(1, -1)
        test_theta_fake_all_obs = test_theta_fake_all_obs.flatten(2, -1)

    return test_theta_fake_all_obs, test_theta


def compute_calibration_metrics(theta_samples, theta_test, sbc_hist=False, sbc_lines=False, sbc_hist_kwargs={},
                                sbc_lines_kwargs={}):
    test_theta_fake_numpy = theta_samples.transpose(1, 0).cpu().detach().numpy()
    test_theta_numpy = theta_test.cpu().numpy()

    # print("Compute metrics...")
    cal_err_val = calibration_error(test_theta_fake_numpy, test_theta_numpy, alpha_resolution=100)
    r2_val = R2(test_theta_fake_numpy, test_theta_numpy)
    rmse_val = rmse(test_theta_fake_numpy, test_theta_numpy)
    # print("Done")

    return_dict = {
        "cal_err_val_mean": cal_err_val.mean(),
        "r2_val_mean": r2_val.mean(),
        "rmse_val_mean": rmse_val.mean(),
        "cal_err_val_std": cal_err_val.std(),
        "r2_val_std": r2_val.std(),
        "rmse_val_std": rmse_val.std(),
    }

    if sbc_hist or sbc_lines:
        # print("Computing SBC")
        ranks = sbc(test_theta_fake_numpy, test_theta_numpy)
    if sbc_hist:
        # print("Plotting SBC histogram")
        fig, ax = make_sbc_plot_histogram(ranks, **sbc_hist_kwargs)
        return_dict["sbc_hist"] = wandb.Image(fig)
    if sbc_lines:
        # print("Plotting SBC lines")
        fig, ax = make_sbc_plot_lines(ranks, **sbc_lines_kwargs)
        return_dict["sbc_lines"] = wandb.Image(fig)

    return return_dict


def sbc(theta_samples, theta_test):
    """
    Plots the simulation-based posterior checking histograms as advocated by Talts et al. (2018).

    ----------

    Arguments:
    theta_samples       : np.ndarray of shape (n_samples, n_test, n_params) -- the samples from
                          the approximate posterior
    theta_test          : np.ndarray of shape (n_test, n_params) -- the 'true' test values
    alpha_resolution    : int -- the number of intervals to consider

    """
    theta_samples = theta_samples.transpose((1, 0, 2))

    # N = int(theta_test.shape[0])

    # Compute ranks (using broadcasting)
    ranks = np.sum(theta_samples < theta_test[:, np.newaxis, :], axis=1)

    return ranks


def make_sbc_plot_histogram(ranks, param_names=None, bins=20,
                            figsize=(15, 5), interval=0.99, show=False, filename=None, font_size=12):
    # Plot settings
    plt.rcParams['font.size'] = font_size

    N, ndim = ranks.shape

    if param_names is None:
        param_names = [r"$\theta_{}$".format(i) for i in range(ndim)]

    # Prepare figure
    if len(param_names) >= 6:
        n_col = int(np.ceil(len(param_names) / 2))
        n_row = 2
    else:
        n_col = int(len(param_names))
        n_row = 1
    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row == n_col == 1:
        axarr = [axarr]
    if n_row > 1:
        axarr = axarr.flat

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
    # plt.close()

    return f, axarr


def make_sbc_plot_lines(ranks, fig=None, ax=None, name="", color="r", show=False, filename=None):
    ranks = ranks.transpose((1, 0))
    ndim, N = ranks.shape
    nbins = int(N / 20)
    repeats = 1

    hb = binom(N, p=1 / nbins).ppf(0.5) * np.ones(nbins)
    hbb = hb.cumsum() / hb.sum()

    lower = [binom(N, p=p).ppf(0.005) for p in hbb]
    upper = [binom(N, p=p).ppf(0.995) for p in hbb]

    # Plot CDF
    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        # fig.tight_layout(pad=3.0)
        spec = fig.add_gridspec(ncols=1,
                                nrows=1)
        ax = fig.add_subplot(spec[0, 0])
    for i in range(ndim):
        hist, *_ = np.histogram(ranks[i], bins=nbins, density=False)
        histcs = hist.cumsum()
        ax.plot(np.linspace(0, nbins, repeats * nbins),
                np.repeat(histcs / histcs.max(), repeats),
                color=color,
                alpha=.1
                )
    ax.plot(np.linspace(0, nbins, repeats * nbins),
            np.repeat(hbb, repeats),
            color="k", lw=2,
            alpha=.8,
            label="uniform CDF")

    ax.fill_between(x=np.linspace(0, nbins, repeats * nbins),
                    y1=np.repeat(lower / np.max(lower), repeats),
                    y2=np.repeat(upper / np.max(lower), repeats),
                    color='k',
                    alpha=.5)

    # Ticks and axes
    ax.set_xticks([0, 25, 50])
    ax.set_xlim([0, 50])
    ax.set_xlabel("Rank")
    ax.set_yticks([0, .5, 1.])
    ax.set_ylim([0., 1.])
    ax.set_ylabel("CDF")

    # Legend
    custom_lines = [Line2D([0], [0], color="k", lw=1.5, linestyle="-"),
                    Line2D([0], [0], color=color, lw=1.5, linestyle="-")]
    ax.legend(custom_lines, ['Uniform CDF', name])

    # Show, if specified
    if show:
        plt.show()
    # Save if specified
    if filename is not None:
        fig.savefig(filename, dpi=200)
    # plt.close()

    return fig, ax


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
