import functools
import jax.numpy as jnp
import jax
from jax import grad
import numpy as np


def extreme_value_cdf_family(x, mu=6.0, sigma=1.0, e=-0.75):
    tail_prob = 1e-3
    tail_length = 10

    upper_limit = mu - (sigma / e)
    mask = x >= upper_limit
    z = (x[~mask] - mu) / sigma
    tx = (1 + e * z) ** (-1 / e)
    valid_cdf = jnp.exp(-tx)

    # Extend CDF to above upper limit of CDF to avoid numerical issues
    extension_cdf = 1 + (tail_prob / tail_length) * jnp.clip(
        x[mask] - upper_limit, 0, tail_length)
    extended_cdf = jnp.zeros_like(x)
    extended_cdf = jax.ops.index_update(extended_cdf, ~mask, valid_cdf)
    extended_cdf = jax.ops.index_update(extended_cdf, mask, extension_cdf)
    return extended_cdf / (1 + tail_prob)


def make_extreme_value_cdf(mu, sigma, e):
    return functools.partial(extreme_value_cdf_family, mu=mu, sigma=sigma, e=e)


def make_normal_cdf(mu, sigma):
    return lambda x: jax.scipy.stats.norm.cdf((x - mu) / sigma)


def make_pdf_from_cdf(cdf_fn):
    return grad(lambda x: jnp.sum(cdf_fn(x)))


def make_left_truncated_cdf(cdf_fn):
    def _truncated_cdf(x, t):
        cdf_x = cdf_fn(x)
        cdf_t = cdf_fn(t)
        truncated_cdf = (cdf_x - cdf_t) / jnp.clip(1 - cdf_t, 1e-6, 1.0)
        mask = x < t
        return jax.ops.index_update(truncated_cdf, mask, 0)
    return _truncated_cdf


def make_left_truncated_pdf_from_cdf(cdf_fn):
    left_truncated_cdf_fn = make_left_truncated_cdf(cdf_fn)
    unsafe_pdf = grad(lambda x, t: jnp.sum(left_truncated_cdf_fn(x, t)))
    return unsafe_pdf


def make_hierarchical_logbeta_prior_log_pdf(beta_0, tau, sigma):
    def _log_pdf(beta, mouse_indicators):
        n = jnp.sum(mouse_indicators, axis=0)
        mean_beta = beta.dot(mouse_indicators)
        mean_beta = mean_beta / n
        first_term = -(beta ** 2) / (2 * sigma ** 2)
        C = (
            jnp.log(sigma)
            - n * jnp.log(jnp.sqrt(2 * np.pi) * sigma)
            - (0.5 * jnp.log(n * tau ** 2 + sigma ** 2)))
        second_term = C
        second_term += (
            (tau * n * mean_beta) ** 2 / sigma ** 2
            + (sigma * beta_0) ** 2 / tau ** 2
            + 2 * n * mean_beta * beta_0
        ) / (2 * (n * tau ** 2 + sigma ** 2))
        second_term -= 0.5 * (beta_0 / tau) ** 2
        return jnp.sum(first_term) + jnp.sum(second_term)
    return _log_pdf


# Just for testing purposes, get approximate prior using Riemann sum
# and test that the analytic expression matches
def make_approximate_logbeta_prior_log_pdf(beta_0, tau, sigma):
    def _approximate_log_pdf(beta, mouse_indicators):
        delta = 1e-1
        min, max = -50, 50
        overall_loglikelihood = 0
        for row in mouse_indicators.T:
            mouse_betas = beta[row].flatten()
            beta_0s = jnp.arange(min, max, delta)
            cond_log_likelihood = jnp.sum(
                jax.scipy.stats.norm.logpdf(
                    (mouse_betas[:, None] - beta_0s[None, :]), scale=sigma),
                axis=0,)
            prior_log_likelihood = jax.scipy.stats.norm.logpdf(
                (beta_0s - beta_0), scale=tau)
            joint_log_likelihood = cond_log_likelihood + prior_log_likelihood
            data_log_likelihood = jax.scipy.special.logsumexp(
                joint_log_likelihood + jnp.log(delta))
            overall_loglikelihood += data_log_likelihood
        return overall_loglikelihood
    return _approximate_log_pdf
