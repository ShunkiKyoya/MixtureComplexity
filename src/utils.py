"""Utilities.
"""

import numpy as np
from numpy.random import RandomState
from scipy.stats import multivariate_normal
from scipy.special import logsumexp


def _log_prob(*, X, means, covariances):
    """Create log probabilities.

    Args:
        X (np.ndarray, shape=(N, D)): data.
        means (np.ndarray, shape=(K, D)): mean vectors.
        covariances (np.ndarray, shape=(K, D, D)): covariance matrices.
    Returns:
        np.ndarray (shape=(N, K)): log probability matrix.
            Prob[n, k] = log N(X[n]; means[k], covariances[k])
    """
    N = len(X)
    K = len(means)

    log_prob = np.zeros([N, K])
    for k in range(K):
        log_prob[:, k] = multivariate_normal.logpdf(X, mean=means[k], cov=covariances[k])

    return log_prob


def posterior_g(*, X, pi, means, covariances):
    """Posterior probabilities for g1, ..., gK.

    Args:
        X (np.ndarray, shape=(N, D)): data.
        pi (np.ndarray, shape=(K,)): mixture weights.
        means (np.ndarray, shape=(K, D)): mean vectors.
        covariances (np.ndarray, shape=(K, D, D)): covariance matrices.
    Returns:
        np.ndarray, shape=(N, K): posterior probability matrix for g1, ..., gK.
            Prob[n, k] = Pr(Z = k|X[n])
    """
    log_prob = _log_prob(X=X, means=means, covariances=covariances)
    log_pi_prob = np.log(pi + 1e-50) + log_prob
    log_pi_prob_norm = logsumexp(log_pi_prob, axis=1).reshape([-1, 1])
    log_post_prob = log_pi_prob - log_pi_prob_norm
    return np.exp(log_post_prob)


def posterior_h(*, X, pi, means, covariances, partition):
    """Posterior probabilities for h1, ..., hL.

    Args:
        X (np.ndarray, shape=(N, D)): data.
        pi (np.ndarray, shape=(K,)): mixture weights.
        means (np.ndarray, shape=(K, D)): mean vectors.
        covariances (np.ndarray, shape=(K, D, D)): covariance matrices.
        partition (np.ndarray, shape=(K, L)): partition matrix.
    Returns:
        np.ndarray, shape=(N, L): posterior probability matrix for h1, ..., hL.
            Prob[n, l] = Pr(Z = l|X[n])
    """
    log_prob = _log_prob(X=X, means=means, covariances=covariances)
    N = len(X)
    L = np.shape(partition)[1]

    rho = np.dot(pi, partition)  # shape=(L,)
    phi = np.reshape(pi, [-1, 1]) * partition / rho  # shape=(K, L)

    log_prob_h = np.zeros([N, L])
    for l in range(L):
        log_prob_h[:, l] = logsumexp(log_prob, b=phi[:, l], axis=1)

    log_rho_prob_h = np.log(rho) + log_prob_h
    log_rho_prob_h_norm = logsumexp(log_rho_prob_h, axis=1).reshape([-1, 1])
    log_w = log_rho_prob_h - log_rho_prob_h_norm

    return np.exp(log_w)
