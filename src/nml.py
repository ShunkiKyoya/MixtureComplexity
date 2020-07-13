"""Functions to calculate the parametric complexity for NML.

.. References:
    * Petri Kontkanen and Petri Myllymäki,
        A linear-time algorithm for computing the multinomial stochastic complexity,
        Information Process Letters, 103, 227--233, 2007.
    * So Hirai and Kenji Yamanishi,
        Efficient computation of normalized maximum likelihood codes for Gaussian
        mixture models with its applications to clustering,
        IEEE Transactions on Information Theory, 59(11), 7718--7728, 2013
    * So Hirai and Kenji Yamanishi,
        Correction to efficient computation of normalized maximum likelihood codes
        for Gaussian mixture models with its applications to clustering,
        IEEE Transactions on Information Theory, 65(10), 6827--6828, 2019.
"""

import math
import numpy as np
from scipy.special import logsumexp, loggamma


def _pc_multinomial(N, K):
    """parametric complexity for multinomial distributions.

    Args:
        N (int): number of data.
        K (int): number of clusters.

    Returns:
        float: parametric complexity for multinomial distributions.
    """
    PC_list = [0]

    # K = 1
    if K >= 1:
        PC_list.append(1)

    # K = 2
    if K >= 2:
        r1 = np.arange(N + 1)
        r2 = N - r1
        logpc_2 = logsumexp(sum([
            loggamma(N + 1),
            (-1) * loggamma(r1 + 1),
            (-1) * loggamma(r2 + 1),
            r1 * np.log(r1 / N + 1e-50),
            r2 * np.log(r2 / N + 1e-50)
        ]))
        PC_list.append(np.exp(logpc_2))

    # K >= 3
    for k in range(3, K + 1):
        PC_list.append(PC_list[k - 1] + N * PC_list[k - 2] / (k - 2))

    return PC_list[-1]


def _log_pc_gaussian(N_list, D, R, lmd_min):
    """log parametric complexity for Gaussian distributions.

    Args:
        N_list (np.ndarray): list of the number of data.
        D (int): dimension of data.
        R (float): upper bound of ||mean||^2.
        lmd_min (float): lower bound of the eigenvalues of the covariance matrix.

    Returns:
        np.ndarray: list of the parametric complexity.
    """
    N_list = np.array(N_list)

    log_PC_list = sum([
        D * N_list * np.log(N_list / 2 / math.e) / 2,
        (-1) * D * (D - 1) * np.log(math.pi) / 4,
        (-1) * np.sum(loggamma((N_list.reshape(-1, 1) - np.arange(1, D + 1)) / 2), axis=1),
        (D + 1) * np.log(2 / D),
        (-1) * loggamma(D / 2),
        D * np.log(R) / 2,
        (-1) * D**2 * np.log(lmd_min) / 2
    ])

    return log_PC_list


def log_pc_gmm(K_max, N_max, D, *, R=1e+3, lmd_min=1e-3):
    """log PC of GMM.

    Calculate (log) parametric complexity of Gaussian mixture model.

    Args:
        K_max (int): max number of clusters.
        N_max (int): max number of data.
        D (int): dimension of data.
        R (float): upper bound of ||mean||^2.
        lmd_min (float): lower bound of the eigenvalues of the covariance matrix.

    Returns:
        np.ndarray: array of (log) parametric complexity.
            returns[K, N] = log C(K, N)
    """
    log_PC_array = np.zeros([K_max + 1, N_max + 1])
    r1_min = D + 1

    # N = 0
    log_PC_array[:, 0] = -np.inf

    # K = 0
    log_PC_array[0, :] = -np.inf

    # K = 1
    # N <= r1_min
    log_PC_array[1, :r1_min] = -np.inf
    # N > r1_min
    N_list = np.arange(r1_min, N_max + 1)
    log_PC_array[1, r1_min:] = _log_pc_gaussian(N_list, D=D, R=R, lmd_min=lmd_min)

    # K > 1
    for k in range(2, K_max + 1):
        for n in range(1, N_max + 1):
            r1 = np.arange(n + 1)
            r2 = n - r1
            log_PC_array[k, n] = logsumexp(sum([
                loggamma(n + 1),
                (-1) * loggamma(r1 + 1),
                (-1) * loggamma(r2 + 1),
                r1 * np.log(r1 / n + 1e-100),
                r2 * np.log(r2 / n + 1e-100),
                log_PC_array[1, r1],
                log_PC_array[k - 1, r2]
            ]))

    return log_PC_array
