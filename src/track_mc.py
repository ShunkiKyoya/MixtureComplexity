"""tracking MC
"""

import numpy as np
from tqdm import tqdm

from .fcm import FCM
from .nml import log_pc_gmm
from .mc import mc, mc_decomp
from .sdms import SDMS


class TrackMC():
    """track MC

    Attributes:
        gmm_list_ (List[GMM]): sequence of GMM objects.
    """
    def __init__(
        self, *, K_max=20, criterion='NML',
        beta=0.01, random_state=None, num_em=10, reg_covar=1e-3,
        R=1e+3, lmd_min=1e-3, verbose=True
    ):
        """
        Args:
            K_max (int): max mixture / cluster size.
            criterion (str): criterion name.
            beta (float): change rate.
            random_state (Optional[int]): random_state.
            num_em (int): number of EM trials.
            reg_covar (float): regularization for covariance matrices.
            R (float): upper bound of ||mean||^2.
            lmd_min (float): lower bound of the eigenvalues of the covariance matrix.
            verbose (bool): verbosity.
        """
        self.K_max = K_max
        self.criterion = criterion
        self.beta = beta
        self.random_state = random_state
        self.num_em = num_em
        self.reg_covar = reg_covar
        self.R = R
        self.lmd_min = lmd_min
        self.verbose = verbose

    def fit(self, XX):
        """
        fit to dataset XX.

        Args:
            XX (np.ndarray, shape=(T, N, D)): dataset
        """
        self.gmm_list_ = []

        # create log_pc_array
        D = XX[0].shape[1]
        N_list = []
        for X in XX:
            N_list.append(len(X))
        N_max = max(N_list)
        log_pc_array = log_pc_gmm(K_max=self.K_max, N_max=N_max, D=D, R=self.R, lmd_min=self.lmd_min)

        # SDMS
        sdms = SDMS(
            K_max=self.K_max, criterion=self.criterion, log_pc_array=log_pc_array, beta=self.beta,
            random_state=self.random_state, num_em=self.num_em, reg_covar=self.reg_covar
        )

        if self.verbose:
            XX_iter = tqdm(XX)
        else:
            XX_iter = XX

        for X in XX_iter:
            self.gmm_list_.append(sdms.update(X))
    
    @property
    def K_list(self):
        """
        get K_list. fit is needed before.

        Returns:
            np.ndarray, shape=(T,): sequence of K.
        """
        K_list = []
        for gmm in self.gmm_list_:
            K_list.append(gmm.K)
        K_list = np.array(K_list)
        return K_list

    def mc_list(self, XX):
        """
        get mc_list. fit is needed before.

        Args:
            XX (np.ndarray, shape=(T, N, D)): dataset
        Returns:
            np.ndarray, shape=(T,): sequence of MC.
        """
        mc_list = []
        for gmm, X in zip(self.gmm_list_, XX):
            gmm_parameters = gmm.get_params()
            mc_list.append(mc(X=X, **gmm_parameters))
        mc_list = np.array(mc_list)
        return mc_list
    
    @property
    def means_list(self):
        """
        get means_list. fit is needed before.

        Returns:
            np.ndarray, shape=(D, -1): sequence of means.
        """
        means_list = []
        for gmm in self.gmm_list_:
            means_list.extend(gmm.model.means_)
        means_list = np.array(means_list)
        return means_list
    
    @property
    def pi_list(self):
        """
        get pi_list. fit is needed before.

        Returns:
            np.ndarray, shape=(-1,): sequence of pi.
        """
        pi_list = []
        for gmm in self.gmm_list_:
            pi_list.extend(gmm.model.weights_)
        pi_list = np.array(pi_list)
        return pi_list


class TrackMCDecomp():
    """track the decomposition of MC

    Attributes:
        mc_total_ (np.ndarray, shape=(T,)): MC (total).
        mc_interaction_ (np.ndarray, shape=(T,)): MC (interaction).
        W_ (np.ndarray, shape=(T, L)): W (cluster l).
        mc_local_ (np.ndarray, shape=(T, L)): MC (cluster l).
    """

    def __init__(
        self,
        *,
        K_max=20, L=3, m=2, criterion='NML',
        beta=0.01, random_state=0, num_em=10, reg_covar=1e-6,
        R=1e+3, lmd_min=1e-3, verbose=True
    ):
        """
        Args:
            K_max (int): max mixture / cluster size.
            L (int): number of upper clusters.
            m (float): fuzzy parameter.
            criterion (str): criterion name.
            beta (float): change rate.
            random_state (int): random_state.
            num_em (int): number of EM trials.
            reg_covar (float): regularization for covariance matrices.
            R (float): upper bound of ||mean||^2.
            lmd_min (float): lower bound of the eigenvalues of the covariance matrix.
            verbose (bool): verbosity.
        """
        self.K_max = K_max
        self.L = L
        self.m = m
        self.criterion = criterion
        self.beta = beta
        self.random_state = random_state
        self.num_em = num_em
        self.reg_covar = reg_covar
        self.R = R
        self.lmd_min = lmd_min
        self.verbose = verbose

    def fit(self, XX):
        """fit

        Args:
            XX (np.ndarray, shape=(T, N, D)): dataset
        """
        T = len(XX)

        # step 1: estimate lower clusters
        track_mc = TrackMC(
            K_max=self.K_max, criterion=self.criterion,
            beta=self.beta, random_state=self.random_state,
            num_em=self.num_em, reg_covar=self.reg_covar,
            R=self.R, lmd_min=self.lmd_min, verbose=self.verbose
        )
        track_mc.fit(XX)
        self.mc_total_ = track_mc.mc_list(XX)

        # step 2: estimate upper clusters
        means_list = track_mc.means_list
        pi_list = track_mc.pi_list
        fcm = FCM(K=self.L, m=self.m, max_iter=1000, random_state=self.random_state)
        fcm.fit(means_list, weight=pi_list)
        U = np.exp(fcm.log_U)
        self.centers_ = fcm.means_

        # step 3: decomposition of MC
        start = 0
        self.mc_interaction_ = np.zeros(T)
        self.W_ = np.zeros([T, self.L])
        self.mc_local_ = np.zeros([T, self.L])

        for t, X in enumerate(XX):

            model_tmp = track_mc.gmm_list_[t]
            gmm_parameters = model_tmp.get_params()
            partition = U[start: start + model_tmp.K]
            start += model_tmp.K
            
            self.mc_interaction_[t], self.W_[t], self.mc_local_[t] = mc_decomp(
                X=X, partition=partition, **gmm_parameters
            )
