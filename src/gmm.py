import math

import numpy as np
from numpy.linalg import det, inv
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

from .nml import _pc_multinomial
from .utils import _log_prob


def _loglike(*, X, pi, means, covariances):
    """log-likelihood

    Args:
        X (np.ndarray, shape=(N, D)): data.
        pi (np.ndarray, shape=(K,)): mixture weights.
        means (np.ndarray, shape=(K, D)): means.
        covarianvces (np.ndarray, shape=(K, D, D)): covariance matrices.
    Returns:
        float: log-likelihood.
    """
    log_prob = _log_prob(X=X, means=means, covariances=covariances)
    log_pi_prob = np.log(pi + 1e-50) + log_prob

    return sum(logsumexp(log_pi_prob, axis=1))


def _comp_loglike(*, X, Z, pi, means, covariances):
    """complete log-likelihood

    Args:
        X (np.ndarray, shape=(N, D)): data.
        Z (np.ndarray, shape=(N,)): latent variables.
        means (np.ndarray, shape=(K, D)): means.
        covarianvces (np.ndarray, shape=(K, D, D)): covariance matrices.
    Returns:
        float: complete log-likelihood.
    """
    _, D = X.shape
    K = len(means)
    nk = np.bincount(Z, minlength=K)

    if min(nk) <= 0:
        return np.nan
    else:
        c_loglike = 0
        for k in range(K):
            c_loglike += nk[k] * np.log(pi[k])
            c_loglike -= 0.5 * nk[k] * D * np.log(2 * math.pi * math.e)
            c_loglike -= 0.5 * nk[k] * np.log(det(covariances[k]))
        return c_loglike


class GMM():
    """Gaussian Mixture Model.
    """

    def __init__(self, *, K, reg_covar=1e-6, random_state=None):
        """
        Args:
            K (int): mixture size.
            reg_covar (float): regularization for covariance matrices.
            random_state (Optional[Union[int, np.random.RandomState]): random_state.
        """
        self.K = K
        self.model = GaussianMixture(
            n_components=K,
            reg_covar=reg_covar,
            random_state=random_state
        )

    def fit(self, X):
        """fit GMM

        Args:
            X (np.ndarray, shape=(N, D)): data
        """
        self.model.fit(X)
    
    def get_params(self):
        """get model parameters

        Returns:
            Dict[str, Any]: gmm_parameters.
        """
        gmm_parameters = {
            'pi': self.model.weights_,
            'means': self.model.means_,
            'covariances': self.model.covariances_
        }
        return gmm_parameters

    def model_score(self, X, *, criterion='NML', log_pc_array=[]):
        """model criterion score

        Args:
            X (np.ndarray, shape=(N, D)): data.
            criterion (str): criterion name.
            log_PC_array (np.ndarray, optional): log PC matrix.
        Returns:
            float: model criterion score.
        """
        # check parameters
        N, D = np.shape(X)
        assert criterion in ['AIC', 'AICcomp', 'BIC', 'BICcomp', 'NML', 'DNML']
        if criterion in ['NML', 'DNML']:
            assert len(log_pc_array) > 0

        # log_like
        gmm_parameters = self.get_params()
        if criterion in ['AIC', 'BIC']:
            log_like = _loglike(X=X, **gmm_parameters)
        elif criterion in ['AICcomp', 'BICcomp', 'NML', 'DNML']:
            Z = self.model.predict(X)
            log_like = _comp_loglike(X=X, Z=Z, **gmm_parameters)

        # complexity
        params = (self.K - 1) + 0.5 * self.K * D * (D + 3)
        if criterion in ['AIC', 'AICcomp']:
            complexity = params
        elif criterion in ['BIC', 'BICcomp']:
            complexity = 0.5 * params * np.log(N)
        elif criterion == 'NML':
            complexity = log_pc_array[self.K, N]
        elif criterion == 'DNML':
            complexity = np.log(_pc_multinomial(N, self.K))
            for k in range(self.K):
                Z_k = sum(Z == k)
                if log_pc_array[1, Z_k] == - np.inf:
                    return np.nan
                else:
                    complexity += log_pc_array[1, Z_k]

        return - log_like + complexity
