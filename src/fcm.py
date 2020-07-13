"""Fuzzy c-means

.. References:
    * James C. Bezdec, Robert Ehrlich, and Willan Full,
        FCM: The fuzzy c-means clustering algorithm.
        Computers and Geosciences. 10(2-3), 191--203, 1984.
"""

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans


class FCM():

    def __init__(self, *, K=20, m=2, max_iter=100, random_state=None):
        """
        Args:
            K (int): number of clusters.
            m (float): fuzzy parameter.
            max_iter (int): max number of iterations,
            random_state (Optional[Union[int, np.random.RandomState]): random_state.
        """
        self.K = K
        self.m = m
        self.max_iter = max_iter
        self.random_state = random_state

    def init_U(self, X):
        """initialization of U.

        Initialization of the responsibility parameter U.

        Args:
            X (np.ndarray): data.
        Returns:
            U: responsibility matrix.
        """
        N = len(X)

        km = KMeans(n_clusters=self.K, max_iter=10, random_state=self.random_state).fit(X)
        U = np.zeros([N, self.K])
        U[np.arange(N), km.labels_] = 1

        return U

    def create_dist(self, X, means):
        """create distance matrix.

        Args:
            X (np.ndarray): data.
            means (np.ndarray): cluster means.
        Returns:
            dist: distance matrix.
        """
        N = len(X)
        dist = np.zeros([N, self.K])

        for k in range(self.K):
            X_mean = X - means[k]
            dist[:, k] = np.sum(X_mean**2, axis=1) + 1e-100

        return dist

    def objective(self, X, U, means, weight):
        """objective function for FCM.

        Args:
            X (np.ndarray): data.
            U (np.ndarray): responsibility matrix.
            means (np.ndarray): cluster means.
            weight (np.ndarray): weight of each data.
        Returns:
            float: objective function for FCM.
        """
        return np.sum(weight.reshape((-1, 1)) * U**self.m * self.create_dist(X, means))

    def update_log_U(self, X, means, weight):
        """update log U.

        Update log U.

        Args:
            X (np.ndarray): data.
            means (np.ndarray): cluster means.
            weight (np.ndarray): weight of each data.
        Returns:
            np.ndarray: log U.
        """
        dist = weight.reshape((-1, 1)) * self.create_dist(X, means)
        log_U = - 1 / (self.m - 1) * np.log(dist)
        log_U -= logsumexp(log_U, axis=1).reshape((-1, 1))
        return log_U

    def update_means(self, X, U, weight):
        """update means.

        Update means.

        Args:
            X (np.ndarray): data.
            U (np.ndarray): responsibility matrix.
            weight (np.ndarray): weight of each data.
        Returns:
            means (np.ndarray): cluster means.
        """
        Um = U.T**self.m * weight
        nk = np.sum(Um, axis=1)
        return np.dot(Um, X) / np.reshape(nk, (-1, 1))

    def fit(self, X, weight=[]):
        """fit.

        Fit FCM to data.

        Args:
            X (np.ndarray): data.
            verbose (bool): if True, print objectives.
            weight (np.ndarray): weight of each data.
        Returns:
            None
        """
        weight = np.array(weight)
        N = len(X)
        if len(weight) == 0:
            weight = np.ones(N)

        U = self.init_U(X)
        self.means_ = self.update_means(X, U, weight)

        for _ in range(self.max_iter):
            self.log_U = self.update_log_U(X, self.means_, weight)
            self.means_ = self.update_means(X, np.exp(self.log_U), weight)
