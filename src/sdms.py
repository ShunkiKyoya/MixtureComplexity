"""Sequential dynamic model selection (SDMS)

References:
    * So Hirai and Kenji Yamanishi:
        Detecting Changes of Clustering Structures Using Normalized Maximum Likelihood Coding,
        in Proceedings of the 18th ACM SIGKDD International Conference
        on Knowledge Discovery and Data Mining, pages 343–351, Beijing, China, 2012.
"""

from copy import deepcopy
import numpy as np
from numpy.random import RandomState
from .gmm import GMM


class SDMS():

    def __init__(
        self, *, K_max=15, criterion='NML',
        log_pc_array=[], beta=0.01, random_state=0, num_em=10,
        reg_covar=1e-3
    ):
        """
        Args:
            K_max (int): max mixture / cluster size.
            criterion (str): criterion name.
            log_PC_array (np.ndarray): log PC matrix.
            beta (float): change rate.
            random_state (Optional[int]): random_state.
            num_em (int): number of EM trials.
            reg_covar (float): regularization for covariance matrices.
        """
        self.K_max = K_max
        self.criterion = criterion
        self.log_pc_array = log_pc_array
        self.beta = beta
        self.num_em = num_em
        self.random_state = random_state
        self.reg_covar = reg_covar

        # random_state
        self.random = RandomState(seed=self.random_state)

        # for next
        self.K_next = np.arange(1, K_max + 1).tolist()
        self.penalty_next = (-np.log(np.ones(K_max) / K_max)).tolist()

    def update(self, X):
        """update

        Args:
            X (np.ndarray): data
        Returns:
            GMM: gmm object
        """

        # create model
        model_list = []
        score_list = []
        for k in range(len(self.K_next)):
            for _ in range(self.num_em):
                model_tmp = GMM(K=self.K_next[k], random_state=self.random, reg_covar=self.reg_covar)
                model_tmp.fit(X)
                model_list.append(deepcopy(model_tmp))
                score_list.append(
                    model_tmp.model_score(X=X, criterion=self.criterion, log_pc_array=self.log_pc_array)
                    + self.penalty_next[k]
                )
        score_list = np.array(score_list)

        # choose the best model
        idx_best = np.nanargmin(score_list)
        model_best = model_list[idx_best]

        # for next update
        K_now = model_best.K
        if K_now == 1:
            self.K_next = np.array([1, 2])
            self.penalty_next = - np.log([1 - 0.5 * self.beta, 0.5 * self.beta])
        elif K_now == self.K_max:
            self.K_next = np.array([self.K_max - 1, self.K_max])
            self.penalty_next = - np.log([0.5 * self.beta, 1 - 0.5 * self.beta])
        else:
            self.K_next = np.array([K_now - 1, K_now, K_now + 1])
            self.penalty_next = - np.log([0.5 * self.beta, 1 - self.beta, 0.5 * self.beta])

        return model_best
