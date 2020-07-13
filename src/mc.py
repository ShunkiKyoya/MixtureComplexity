"""Mixture Complexity.
"""

import numpy as np
from scipy.special import logsumexp

from .utils import _log_prob


def _mc(*, pi, log_prob, weights):
    """Calculate mixture complexity from log_prob.

    Args:
        log_prob (np.ndarray, shape=(N, K)): log probability matrix.
        pi (np.ndarray, shape=(K,)): mixture proportions.
        weights (np.ndarray): data weights.
    Returns:
        float: mixture complexity.
    """
    log_pi_prob = np.log(pi + 1e-50) + log_prob
    log_pi_prob_norm = logsumexp(log_pi_prob, axis=1).reshape([-1, 1])

    term1 = log_pi_prob - log_pi_prob_norm
    term2 = log_prob - log_pi_prob_norm

    return sum(np.dot(weights, np.exp(term1) * term2)) / sum(weights)


def mc(*, X, pi, means, covariances, weights=[]):
    """Calculate mixture complexity.

    Args:
        X (np.ndarray, shape=(N, D)): data.
        pi (np.ndarray, shape=(K,)): mixture proportions.
        means (np.ndarray, shape=(K, D)): mean vectors.
        covariances (np.ndarray, shape=(K, D, D)): covariance matrices.
        weights (np.ndarray, shape=(N,)): data weights.
    Returns:
        float: mixture complexity.
    """
    N = len(X)
    if len(weights) == 0:
        weights = np.ones(N)
    log_prob = _log_prob(X=X, means=means, covariances=covariances)

    return _mc(pi=pi, log_prob=log_prob, weights=weights)


def _mc_decomp(log_prob, pi, partition):
    """Calculate the decomposition of MC from log_prob.

    Args:
        log_prob (np.ndarray, shape=(N, K)): log probability.
        pi (np.ndarray, shape=(K,)): mixture proportions.
        partition (np.ndarray, shape=(K, L)): partition matrix.
    Returns:
        float, (np.ndarray, shape=(L,)), (np.ndarray, shape=(L,)): MC_interaction, W, MC_local.
    """
    N = len(log_prob)
    L = np.shape(partition)[1]

    rho = np.dot(pi, partition)  # shape=(L,)
    phi = np.reshape(pi, [-1, 1]) * partition / rho  # shape=(K, L)

    log_prob_h = np.zeros([N, L])
    for l in range(L):
        log_prob_h[:, l] = logsumexp(log_prob, b=phi[:, l], axis=1)

    log_rho_prob_h = np.log(rho) + log_prob_h
    log_rho_prob_h_norm = logsumexp(log_rho_prob_h, axis=1).reshape([-1, 1])
    log_w = log_rho_prob_h - log_rho_prob_h_norm
    w = np.exp(log_w)

    MC_interaction = _mc(log_prob=log_prob_h, pi=rho, weights=np.ones(N))

    W = np.zeros(L)

    MC_local = np.zeros(L)
    for l in range(L):
        W[l] = sum(w[:, l]) / N
        MC_local[l] = _mc(log_prob=log_prob, pi=phi[:, l], weights=w[:, l])

    return MC_interaction, W, MC_local


def mc_decomp(*, X, pi, means, covariances, partition):
    """Calculate the decomposition of MC.

    Args:
        X (np.ndarray, shape=(N, D)): data.
        pi (np.ndarray, shape=(K,)): mixture weights.
        means (np.ndarray, shape=(K, D)): mean vectors.
        covariances (np.ndarray, shape=(K, D, D)): covariance matrices.
        weights (np.ndarray, shape=(N,)): data weights.
        partition (np.ndarray, shape=(K, L)): partition matrix.
    Returns:
        float, (np.ndarray, shape=(L,)), (np.ndarray, shape=(L,)): MC_interaction, W, MC_local.
    """
    log_prob = _log_prob(X=X, means=means, covariances=covariances)

    return _mc_decomp(log_prob=log_prob, pi=pi, partition=partition)
