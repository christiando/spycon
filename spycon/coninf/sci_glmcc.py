"""
Credits to Taehoon Kim
"""

from spycon.spycon_inference import SpikeConnectivityInference
import numpy
from scipy.stats import norm
import multiprocessing
from itertools import repeat

import scipy.special as special
import numpy as np
import math
import csv
from copy import copy
from tqdm import tqdm


class GLMCC(SpikeConnectivityInference):
    """
    Adaptation of the original Python script from:

    Kobayashi, R., Kurita, S., Kurth, A., Kitano, K., Mizuseki, K., Diesmann, M., Richmond, B. J., & Shinomoto, S. (2019). Reconstructing neuronal circuitry from parallel spike trains. Nature Communications, 10(1). https://doi.org/10.1038/s41467-019-12225-2

    Args:
        params (dict): Parameter dictionary with the following keys:
            - 'binsize' (float): Time step in seconds used for time discretization. Default is 1e-3.
            - 'ccg_tau' (float): The maximum lag for which the cross-correlogram (CCG) is calculated, in seconds. Default is 50e-3.
            - 'syn_delay' (float): Assumed synaptic delay in seconds. Default is 3e-3.
            - 'tau' (list of float): Time constants tau for exponential decay in seconds. Default is [1e-3, 1e-3].
            - 'beta' (float): Corresponds to the penalty term (gamma in the paper) for slow trend fitting. A larger beta results in a higher penalty. Default is 4000.
            - 'alpha' (float): Threshold. Default is 1e-2.

    Returns:
        None
    """

    def __init__(self, params: dict = {}):

        super().__init__(params)
        self.method = "glmcc"
        self.default_params = {
            "binsize": 1e-3,  # bin size (s)
            "ccg_tau": 50e-3,  # window size for ccg gives 100ms
            "syn_delay": 3e-3,  # synaptic delay assumed for 'sim' mode
            # "NPAR": 102,  # length of parameter variable (2*len(Win) +2 )
            "tau": [1e-3, 1e-3],  # time constant tau for exponential decay (spike rate)
            "beta": 4000,  # corresponds to penalty term gamma in the paper (for slow trend fitting), bigger the beta, higher the penalty
            "alpha": 0.001,  # alpha value to be compared in end
        }

    def _infer_connectivity(
        self, times: numpy.ndarray, ids: numpy.ndarray, pairs: numpy.ndarray
    ) -> tuple:
        """
        GLMCC connectivity inference.

        Args:
            times (numpy.ndarray): Spike times in milliseconds (*1000) converted from seconds.
            ids (numpy.ndarray): Unit IDs corresponding to spike times.

        Returns:
            tuple: A tuple containing the following arrays:
                1) nodes (numpy.ndarray): An array of node labels with shape [number_of_nodes].
                2) edges (numpy.ndarray): An array with shape [number_of_edges, 2], where the first column represents the outgoing node and the second column represents the incoming node.
                3) weights (numpy.ndarray): An array of graded strengths of connections with shape [number_of_edges].
                4) stats (numpy.ndarray): A 2D array representing a fully connected graph with shape [number_of_edges, 3].
                The columns are as follows:
                    - The first column represents outgoing nodes.
                    - The second column represents incoming nodes.
                    - The third column contains the statistic used to decide whether it is an edge or not.
                        A higher value indicates that an edge is more probable.

        """
        alpha = self.params.get("alpha", self.default_params["alpha"])
        # converting times var into miliseconds
        times_ms = times * 1000  # converting seconds into ms

        nodes = numpy.unique(ids)

        weights = []
        stats = []
        num_connections_to_test = len(nodes) * (len(nodes) - 1)

        conn_count = 0
        print_step = numpy.amin([1000, numpy.round(num_connections_to_test / 10.0)])
        pairs_already_computed = numpy.empty((0, 2))

        for pair in tqdm(pairs):
            id1, id2 = pair
            if not any(numpy.prod(pairs_already_computed == [id2, id1], axis=1)):

                zscore1, weight1, zscore2, weight2, pair = self._test_connection_pair(
                    times_ms, ids, pair
                )
                id1, id2 = pair
                weights.append(weight1)
                stats.append(numpy.array([id1, id2, zscore1]))

                pairs_already_computed = numpy.vstack(
                    [pairs_already_computed, numpy.array([id1, id2])]
                )
                if any(numpy.prod(pairs == [id2, id1], axis=1)):
                    weights.append(weight2)
                    stats.append(numpy.array([id2, id1, zscore2]))
                    conn_count += 2
                    pairs_already_computed = numpy.vstack(
                        [pairs_already_computed, numpy.array([id2, id1])]
                    )
                else:
                    conn_count += 1

        print(
            "Test connection %d of %d (%d %%)"
            % (
                conn_count,
                num_connections_to_test,
                100 * conn_count / num_connections_to_test,
            )
        )
        weights = numpy.vstack(weights)
        stats = numpy.vstack(stats)
        threshold = norm.ppf(1.0 - 0.5 * alpha)
        return nodes, weights, stats, threshold

    def _infer_connectivity_parallel(
        self,
        times: numpy.ndarray,
        ids: numpy.ndarray,
        pairs: numpy.ndarray,
        num_cores: int,
    ) -> tuple:
        """
        CCG connectivity inference. Parallel version.

        Args:
            times (numpy.ndarray): Spike times in seconds.
            ids (numpy.ndarray): Unit IDs corresponding to spike times.
            pairs (numpy.ndarray): Array of [pre, post] pair node IDs for which inference will be done.
            num_cores (int): Number of CPU cores to be used for multiprocessing.

        Returns:
            tuple: A tuple containing the following elements:
                1) nodes (numpy.ndarray): An array of node labels with shape [number_of_nodes].
                2) weights (numpy.ndarray): An array of graded strengths of connections with shape [number_of_edges].
                3) stats (numpy.ndarray): A 2D array representing a fully connected graph with shape [number_of_edges, 3].
                The columns are as follows:
                    - The first column represents outgoing nodes.
                    - The second column represents incoming nodes.
                    - The third column contains the statistic used to decide whether it is an edge or not.
                        A higher value indicates that an edge is more probable.
                4) threshold (float): A float value that considers an edge to be a connection if stats > threshold.

        """

        alpha = self.params.get("alpha", self.default_params["alpha"])
        # converting times var into miliseconds
        times_ms = times * 1000  # converting seconds into ms
        nodes = numpy.unique(ids)
        weights = []
        stats = []
        pairs_already_computed = numpy.empty((0, 2))
        pairs_to_compute = []

        for pair in pairs:
            id1, id2 = pair
            if not any(numpy.prod(pairs_already_computed == [id2, id1], axis=1)):
                pairs_to_compute.append(pair)
                pairs_already_computed = numpy.vstack(
                    [pairs_already_computed, numpy.array([id1, id2])]
                )
                if any(numpy.prod(pairs == [id2, id1], axis=1)):
                    pairs_already_computed = numpy.vstack(
                        [pairs_already_computed, numpy.array([id2, id1])]
                    )

        job_arguments = zip(repeat(times_ms), repeat(ids), pairs_to_compute)
        pool = multiprocessing.Pool(processes=num_cores)
        results = pool.starmap(self._test_connection_pair, job_arguments)
        pool.close()

        for result in results:
            zscore1, weight1, zscore2, weight2, pair = result
            id1, id2 = pair
            weights.append(weight1)
            stats.append(numpy.array([id1, id2, zscore1]))
            if any(numpy.prod(pairs == [id2, id1], axis=1)):
                weights.append(weight2)
                stats.append(numpy.array([id2, id1, zscore2]))

        weights = numpy.vstack(weights)
        stats = numpy.vstack(stats)
        threshold = norm.ppf(1.0 - 0.5 * alpha)
        return nodes, weights, stats, threshold

    def _test_connection_pair(
        self, times_ms: numpy.ndarray, ids: numpy.ndarray, pair: tuple
    ) -> tuple:
        """
        Test connections in both directions.

        Args:
            times (numpy.ndarray): Spike times in seconds.
            ids (numpy.ndarray): Unit IDs corresponding to spike times.
            id1 (int): ID of the first node.
            id2 (int): ID of the second node.

        Returns:
            tuple: A tuple containing the following elements:
                - pval_id1_to_id2 (float): p-value for the edge from id1 to id2.
                - weight_id1_to_id2 (float): Weight for the edge from id1 to id2.
                - pval_id2_to_id1 (float): p-value for the edge from id2 to id1.
                - weight_id2_to_id1 (float): Weight for the edge from id2 to id1.

        """

        T = np.amax(times_ms) - np.amin(times_ms)
        id1, id2 = pair
        times1 = times_ms[ids == id1]
        times2 = times_ms[ids == id2]
        glmcc_ccg_result = self._linear_crossCorrelogram(times1, times2, T)
        zscore1, weight1, zscore2, weight2 = self._test_connection(glmcc_ccg_result)

        return zscore1, weight1, zscore2, weight2, pair

    def _test_connection(self, glmcc_ccg_result: list) -> tuple:
        """
        Compute connection statistics.

        Args:
            glmcc_ccg_result (list): Output from the GLMCC original code.

        Returns:
            tuple: A tuple containing the following elements:
                - zscore_1_to_2 (float): Z-score for the connection from 1 to 2.
                - weight_1_to_2 (float): Weight for the connection from 1 to 2.
                - zscore_2_to_1 (float): Z-score for the connection from 2 to 1.
                - weight_2_to_1 (float): Weight for the connection from 2 to 1.

        """

        tau = copy(self.params.get("tau", self.default_params["tau"]))
        tau[0] *= 1e3
        tau[1] *= 1e3
        beta = self.params.get("beta", self.default_params["beta"])
        mode = "sim"

        # Fitting a GLM
        if mode == "sim":
            delay_synapse = (
                self.params.get("syn_delay", self.default_params["syn_delay"]) * 1e3
            )
            par, log_pos = self._GLMCC(
                glmcc_ccg_result[1],
                glmcc_ccg_result[0],
                tau,
                beta,
                glmcc_ccg_result[2],
                glmcc_ccg_result[3],
                delay_synapse,
            )
        elif mode == "exp":
            log_pos = 0
            for m in range(2, 5):
                tmp_par, tmp_log_pos = self._GLMCC(
                    glmcc_ccg_result[1],
                    glmcc_ccg_result[0],
                    tau,
                    beta,
                    glmcc_ccg_result[2],
                    glmcc_ccg_result[3],
                    m,
                )
                if m == 2 or tmp_log_pos > log_pos:
                    log_pos = tmp_log_pos
                    par = tmp_par
                    delay_synapse = m
        else:
            print("Input error: You must write sim or exp in mode")
            exit(0)

        WIN = self.params.get("ccg_tau", self.default_params["ccg_tau"]) * 1e3
        DELTA = self.params.get("binsize", self.default_params["binsize"]) * 1e3

        # Connection parameters
        nb = int(WIN / DELTA)
        NPAR = 2 * (nb + 1)
        cc_0 = [0 for l in range(2)]
        max_list = [0 for l in range(2)]
        Jmin = [0 for l in range(2)]
        for l in range(2):
            cc_0[l] = 0
            max_list[l] = int(tau[l] + 0.1)

            if l == 0:
                for m in range(max_list[l]):
                    cc_0[l] += np.exp(par[nb + int(delay_synapse) + m])
            if l == 1:
                for m in range(max_list[l]):
                    cc_0[l] += np.exp(par[nb - int(delay_synapse) - m])

            cc_0[l] = cc_0[l] / max_list[l]

            # Jmin[l] = 1.57*abs(norm.ppf(alpha))*math.sqrt(1/ tau[l]/ cc_0[l])
            n12 = tau[l] * cc_0[l]
            # if n12 <= 10: # chekcing for sensible amount of spikes
            #    par[NPAR-2+l] = 0

        zscore1 = (
            numpy.abs(par[NPAR - 1]) * numpy.sqrt(tau[1] * cc_0[1]) / 1.57
        )  # numpy.amin([abs(norm(Jmin[1],1).logcdf(par[NPAR-1])), abs(norm(Jmin[1],1).logcdf(-par[NPAR-1]))])
        zscore2 = (
            numpy.abs(par[NPAR - 2]) * numpy.sqrt(tau[0] * cc_0[0]) / 1.57
        )  # numpy.amin([abs(norm(Jmin[0],1).logcdf(par[NPAR-2])), norm(Jmin[0],1).logcdf(-par[NPAR-2])])

        # below is the cutoff part (instead of 1.266 value ~ 10% percentile in the code we use alpha value again)
        weight1 = self._calc_PSP(
            par[NPAR - 1]
        )  # Here just applying PSP transformation of vanilla GLMACC implementation without arbitrary 1.266 scaling factor (*it assumes fitting of the EPSP, IPSP based on the simulation. check calc_PSP() function below for details)
        weight2 = self._calc_PSP(par[NPAR - 2])  # same as above

        return zscore1, weight1, zscore2, weight2

    ############################################# original glmcc ##########################################

    def _index_linear_search(self, array, target, index):
        """
        search for index with the smallest value which is bigger than target.
        ---------------------------------------------------------------------
        targetよりも大きくて、その中でも一番小さい値のindexを返す関数
        """

        result = 0
        if index == -1:
            while len(array) > result and array[result] <= target:
                result += 1
            return result
        else:
            result = index
            while len(array) > result and array[result] <= target:
                result += 1
            return result

    def _linear_crossCorrelogram(self, times1, times2, T):
        """
        make Cross correlogram.

        Input:
        spike times for unit 1, spike times for unit 2, T(s)(float or int)

        Output:
        list of spike time (list)
        list of histogram (list)
        the number of cell1's spike time (list)
        the number of cell2's spike time (list)

        -------------------------------------------
        Cross correlogramの図を作成する。

        入力:
        ファイルの名前1, ファイルの名前2, T(s)

        出力:
        スパイク時間のリスト
        ヒストグラムのリスト
        cell1のスパイク時間の数
        cell2のスパイク時間の数

        """

        WIN = self.params.get("ccg_tau", self.default_params["ccg_tau"]) * 1e3
        DELTA = self.params.get("binsize", self.default_params["binsize"]) * 1e3

        cell1 = times1
        cell2 = times2

        cell1 = cell1[np.logical_and(cell1 > 0, cell1 < T)]
        cell2 = cell2[np.logical_and(cell2 > 0, cell2 < T)]

        # print('n_pre: '+str(len(cell1)))
        # print('n_post: '+str(len(cell2)))

        # make c_ij(spike time)
        w = int(WIN)
        c = []
        min_index = -1
        max_index = -1

        for i in range(len(cell2)):
            min = cell2[i] - w
            max = cell2[i] + w

            min_j = self._index_linear_search(cell1, min, min_index)
            min_index = min_j
            max_j = self._index_linear_search(cell1, max, max_index)
            max_index = max_j

            c_i = []
            for j in range(max_j - min_j):
                if (cell1[min_j + j] - cell2[i]) < WIN:
                    c_i.append(cell1[min_j + j] - cell2[i])

            c.extend(c_i)

        # make histogram
        bin_width = DELTA  # bin width
        bin_num = int(2 * w / bin_width)  # the number of bin

        hist_array = np.histogram(np.array(c), bins=bin_num, range=(-1 * w, w))
        result = [0, 0, 0, 0]
        result[0] = c
        result[1] = hist_array[0].tolist()
        result[2] = len(cell1)
        result[3] = len(cell2)

        return result

    def _init_par(self, rate, NPAR):
        """
        initialize parameter

        ---------------------
        パラメータを初期化する
        """
        par = np.ones((NPAR, 1))
        par = math.log(rate) * par
        par[NPAR - 2][0] = 0.1
        par[NPAR - 1][0] = 0.1
        return par

    def _calc_hessian(self, par, beta, tau, c, n_sp, t_sp, delay_synapse, Gk):
        """
        calculate hessian of log posterior probability

        Output:
        numpy matrix

        -------------------------------------------------
        対数事後確率のヘシアンを計算する

        出力:
        numpyの行列
        """

        WIN = self.params.get("ccg_tau", self.default_params["ccg_tau"]) * 1e3
        DELTA = self.params.get("binsize", self.default_params["binsize"]) * 1e3
        nb = int(WIN / DELTA)
        NPAR = 2 * (nb + 1)

        hessian = np.zeros((NPAR, NPAR))

        # d^2P/da_kda_l, d^2P/da_kdJ
        for i in range(0, NPAR - 2):
            for j in range(0, NPAR):
                # d^2P/da_kdJ
                if j == NPAR - 2:
                    x_k = (i + 1) * DELTA - WIN
                    if x_k > delay_synapse:
                        # if abs(J) < 1.0e-3, approximate J=0
                        if abs(par[NPAR - 2][0]) < 1.0e-3:
                            hessian[i][j] = (
                                tau[0]
                                * np.exp(par[i][0])
                                * self._func_f(x_k - DELTA, delay_synapse, tau[0])
                                * (1 - np.exp(-DELTA / tau[0]))
                            )

                        else:
                            hessian[i][j] = (
                                (-1)
                                * (tau[0] * np.exp(par[i][0]) / par[NPAR - 2][0])
                                * (
                                    np.exp(
                                        par[NPAR - 2][0]
                                        * self._func_f(
                                            x_k - DELTA, delay_synapse, tau[0]
                                        )
                                    )
                                    - np.exp(
                                        par[NPAR - 2][0]
                                        * self._func_f(x_k, delay_synapse, tau[0])
                                    )
                                )
                            )

                elif j == NPAR - 1:
                    x_k = (i + 1) * DELTA - WIN
                    if x_k <= (-1) * delay_synapse:
                        # if abs(J) < 1.0e-3, approximate J=0
                        if abs(par[NPAR - 1][0]) < 1.0e-3:
                            hessian[i][j] = (
                                tau[1]
                                * np.exp(par[i][0])
                                * self._func_f(-x_k, delay_synapse, tau[1])
                                * (1 - np.exp(-DELTA / tau[1]))
                            )

                        else:
                            hessian[i][j] = (
                                (-1)
                                * (tau[1] * np.exp(par[i][0]) / par[NPAR - 1][0])
                                * (
                                    np.exp(
                                        par[NPAR - 1][0]
                                        * self._func_f(-x_k, delay_synapse, tau[1])
                                    )
                                    - np.exp(
                                        par[NPAR - 1][0]
                                        * self._func_f(
                                            -x_k + DELTA, delay_synapse, tau[1]
                                        )
                                    )
                                )
                            )

                # d^2p/da_kda_l
                else:
                    if i == j:
                        hessian[i][j] = (-1) * Gk[i] + (beta / DELTA) * (
                            self._K_delta(i, 0) + self._K_delta(i, NPAR - 3) - 2
                        )
                    else:
                        hessian[i][j] = (beta / DELTA) * (
                            self._K_delta(i - 1, j) + self._K_delta(i + 1, j)
                        )

        # d^2P/dJ^2
        for i in range(NPAR - 2, NPAR):
            for j in range(0, NPAR):
                if j >= NPAR - 2:
                    if i == j == NPAR - 2:
                        tmp = 0
                        for k in range(0, NPAR - 2):
                            x_k = (k + 1) * DELTA - WIN
                            if x_k > delay_synapse:
                                # if abs(J) < 1.0e-3, approximate J=0
                                if abs(par[NPAR - 2][0]) < 1.0e-3:
                                    tmp = (
                                        (tau[0] / 2)
                                        * (
                                            self._func_f(
                                                x_k - DELTA, delay_synapse, tau[0]
                                            )
                                            ** 2
                                        )
                                        * (1 - np.exp(-2 * DELTA / tau[0]))
                                    )
                                    hessian[i][j] -= tmp
                                else:
                                    tmp = (
                                        par[NPAR - 2][0]
                                        * self._func_f(
                                            x_k - DELTA, delay_synapse, tau[0]
                                        )
                                        - 1
                                    ) * np.exp(
                                        par[NPAR - 2][0]
                                        * self._func_f(
                                            x_k - DELTA, delay_synapse, tau[0]
                                        )
                                    )
                                    tmp -= (
                                        par[NPAR - 2][0]
                                        * self._func_f(x_k, delay_synapse, tau[0])
                                        - 1
                                    ) * np.exp(
                                        par[NPAR - 2][0]
                                        * self._func_f(x_k, delay_synapse, tau[0])
                                    )
                                    hessian[i][j] -= (
                                        (tau[0] * np.exp(par[k][0]))
                                        / (par[NPAR - 2][0] ** 2)
                                    ) * tmp

                    elif i == j == NPAR - 1:
                        tmp = 0
                        for k in range(0, NPAR - 2):
                            x_k = (k + 1) * DELTA - WIN
                            if x_k <= -delay_synapse:
                                # if abs(J) < 1.0e-3, approximate J=0
                                if abs(par[NPAR - 1][0]) < 1.0e-3:
                                    tmp = (
                                        (tau[1] / 2)
                                        * (
                                            self._func_f(-x_k, delay_synapse, tau[1])
                                            ** 2
                                        )
                                        * (1 - np.exp(-2 * DELTA / tau[1]))
                                    )
                                    hessian[i][j] -= tmp
                                else:
                                    tmp = (
                                        par[NPAR - 1][0]
                                        * self._func_f(-x_k, delay_synapse, tau[1])
                                        - 1
                                    ) * np.exp(
                                        par[NPAR - 1][0]
                                        * self._func_f(-x_k, delay_synapse, tau[1])
                                    )
                                    tmp -= (
                                        par[NPAR - 1][0]
                                        * self._func_f(
                                            -x_k + DELTA, delay_synapse, tau[1]
                                        )
                                        - 1
                                    ) * np.exp(
                                        par[NPAR - 1][0]
                                        * self._func_f(
                                            -x_k + DELTA, delay_synapse, tau[1]
                                        )
                                    )
                                    hessian[i][j] -= (
                                        (tau[1] * np.exp(par[k][0]))
                                        / (par[NPAR - 1][0] ** 2)
                                    ) * tmp

                else:
                    hessian[i][j] = hessian[j][i]

        return hessian

    def _calc_grad_log_p(self, par, beta, tau, c, n_sp, t_sp, delay_synapse, Gk):
        """
        calculate gradient of log posterior probability

        Output:
        numpy column vector

        -----------------------------------------------------
        対数事後確率の勾配を計算する

        出力:
        numpyの列ベクトル
        """
        WIN = self.params.get("ccg_tau", self.default_params["ccg_tau"]) * 1e3
        DELTA = self.params.get("binsize", self.default_params["binsize"]) * 1e3
        nb = int(WIN / DELTA)
        NPAR = 2 * (nb + 1)

        g_log_p = np.zeros((NPAR, 1))

        # dP/da_k

        for i in range(0, NPAR - 2):
            tmp = 0
            g_log_p[i][0] = -Gk[i]

            if i == 0:
                tmp = (-1) * (par[i][0] - par[i + 1][0])
            elif i == NPAR - 3:
                tmp = (-1) * (par[i][0] - par[i - 1][0])
            else:
                tmp = (-1) * (par[i][0] - par[i - 1][0]) + (-1) * (
                    par[i][0] - par[i + 1][0]
                )
            g_log_p[i][0] += (beta / DELTA) * tmp + c[i]

        # dP/dJ_ij, dp/dJ_ji
        tmp_ij = 0
        tmp_ji = 0

        for i in range(0, n_sp):
            """
            #実験データの時使用する
            if 1 < abs(t_sp[i]):

                if t_sp[i] > delay_synapse:
                    tmp_ij += func_f(t_sp[i], delay_synapse, tau[0])

                elif t_sp[i] < -delay_synapse:
                    tmp_ji += func_f(-t_sp[i], delay_synapse, tau[1])
            """
            if t_sp[i] > delay_synapse:
                tmp_ij += self._func_f(t_sp[i], delay_synapse, tau[0])

            elif t_sp[i] < -delay_synapse:
                tmp_ji += self._func_f(-t_sp[i], delay_synapse, tau[1])

        for i in range(0, NPAR - 2):
            x_k = (i + 1) * DELTA - WIN
            # if abs(J) < 1.0e-3, approximate J=0
            if x_k > delay_synapse:
                if abs(par[NPAR - 2][0]) < 1.0e-3:
                    tmp_ij -= (
                        tau[0]
                        * np.exp(par[i][0])
                        * self._func_f(x_k - DELTA, delay_synapse, tau[0])
                        * (1 - np.exp(-DELTA / tau[0]))
                    )
                else:
                    tmp_ij -= (tau[0] * np.exp(par[i][0]) / par[NPAR - 2][0]) * (
                        np.exp(
                            par[NPAR - 2][0]
                            * self._func_f(x_k - DELTA, delay_synapse, tau[0])
                        )
                        - np.exp(
                            par[NPAR - 2][0] * self._func_f(x_k, delay_synapse, tau[0])
                        )
                    )
            elif x_k <= (-1) * delay_synapse:
                if abs(par[NPAR - 1][0]) < 1.0e-3:
                    tmp_ji -= (
                        tau[1]
                        * np.exp(par[i][0])
                        * self._func_f(-x_k, delay_synapse, tau[1])
                        * (1 - np.exp(-DELTA / tau[1]))
                    )
                else:
                    tmp_ji -= (tau[1] * np.exp(par[i][0]) / par[NPAR - 1][0]) * (
                        np.exp(
                            par[NPAR - 1][0] * self._func_f(-x_k, delay_synapse, tau[1])
                        )
                        - np.exp(
                            par[NPAR - 1][0]
                            * self._func_f(-x_k + DELTA, delay_synapse, tau[1])
                        )
                    )

        g_log_p[NPAR - 2][0] = tmp_ij
        g_log_p[NPAR - 1][0] = tmp_ji

        return g_log_p

    def _calc_Gk(self, par, beta, tau, c, n_sp, t_sp, delay_synapse):
        """
        calculate Gk using scipy.special.expi()

        Output: Gk (list)

        -------------------------------------------
        scipyのモジュールを使ってGkを計算する

        出力: Gk (list)
        """
        WIN = self.params.get("ccg_tau", self.default_params["ccg_tau"]) * 1e3
        DELTA = self.params.get("binsize", self.default_params["binsize"]) * 1e3
        nb = int(WIN / DELTA)
        NPAR = 2 * (nb + 1)

        Gk = [0 for i in range(0, NPAR - 2)]
        for i in range(0, NPAR - 2):
            x_k = (i + 1) * DELTA - WIN
            tmp = 0

            """
            #実験データの時に使用する
            if i == int(WIN-1) or i == int(WIN):
                continue
            """

            if (
                x_k <= -delay_synapse
                and abs(par[NPAR - 1][0] * self._func_f(-x_k, delay_synapse, tau[1]))
                > 1.0e-6
            ):
                tmp = special.expi(
                    par[NPAR - 1][0] * self._func_f(-x_k, delay_synapse, tau[1])
                )
                tmp -= special.expi(
                    par[NPAR - 1][0] * self._func_f(-x_k + DELTA, delay_synapse, tau[1])
                )

                Gk[i] = tmp * np.exp(par[i][0]) * tau[1]
            elif (
                x_k > delay_synapse
                and abs(par[NPAR - 2][0] * self._func_f(x_k, delay_synapse, tau[0]))
                > 1.0e-6
            ):
                tmp = special.expi(
                    par[NPAR - 2][0] * self._func_f(x_k - DELTA, delay_synapse, tau[0])
                )
                tmp -= special.expi(
                    par[NPAR - 2][0] * self._func_f(x_k, delay_synapse, tau[0])
                )

                Gk[i] = tmp * np.exp(par[i][0]) * tau[0]
            else:
                Gk[i] = DELTA * np.exp(par[i][0])

        return Gk

    def _K_delta(self, i, j):
        """
        Kronecker delta

        ---------------------
        クロネッカーのデルタ
        """
        if i == j:
            return 1
        else:
            return 0

    def _func_f(self, sec, delay, tau):
        """
        The time profile of the synaptic interaction

        ----------------------------------------------
        シナプス電流の効果を表す関数

        """
        if sec >= delay:
            return np.exp(-(sec - delay) / tau)
        else:
            return 0

    def _calc_log_posterior(self, par, beta, tau, c, n_sp, t_sp, delay_synapse, Gk):
        """
        calculate log posterior probability
        Output: log_post (float)

        --------------------------------------
        対数事後確率を求める関数
        出力: log_post(float)
        """

        WIN = self.params.get("ccg_tau", self.default_params["ccg_tau"]) * 1e3
        DELTA = self.params.get("binsize", self.default_params["binsize"]) * 1e3
        nb = int(WIN / DELTA)
        NPAR = 2 * (nb + 1)

        log_post = 0

        for i in range(0, NPAR):
            log_post += par[i][0] * c[i]

        for i in range(0, NPAR - 2):
            log_post = log_post - Gk[i]

        tmp = 0
        for i in range(0, NPAR - 3):
            tmp += (par[i + 1][0] - par[i][0]) ** 2
        tmp = (beta / (2 * DELTA)) * tmp

        log_post -= tmp

        return log_post

    def _LM(self, par, beta, tau, c, n_sp, t_sp, delay_synapse):
        """
        calculate the best parameter whose log posterior probability is biggest by LM method
        This function does not end until the termination condition is satisfied.

        Output:
        parameter (list), log_post (float) (if LM method's convergence condition is satisfied.)
        or
        false (if loop count is 1000.)

        ----------------------------------------------------------------------------------------
        対数事後確率が一番大きいパラメータをLM法で計算する関数。
        この関数は収束条件を満たさない限り終了しない。(繰り返し回数を1000回超えたら強制的に終了する。)

        """

        WIN = self.params.get("ccg_tau", self.default_params["ccg_tau"]) * 1e3
        DELTA = self.params.get("binsize", self.default_params["binsize"]) * 1e3
        nb = int(WIN / DELTA)
        NPAR = 2 * (nb + 1)

        C_lm = 0.01
        eta = 0.1
        l_c = 0

        # for_reporting=[]
        while True:
            # for_reporting.append(par)

            l_c += 1
            # Update parameters
            # print("現在のパラメータ")
            # print(par)
            Gk = self._calc_Gk(par, beta, tau, c, n_sp, t_sp, delay_synapse)
            log_pos = self._calc_log_posterior(
                par, beta, tau, c, n_sp, t_sp, delay_synapse, Gk
            )
            grad = self._calc_grad_log_p(
                par, beta, tau, c, n_sp, t_sp, delay_synapse, Gk
            )
            hessian = self._calc_hessian(
                par, beta, tau, c, n_sp, t_sp, delay_synapse, Gk
            )
            h_diag = np.diag(hessian)
            tmp = np.eye(NPAR, NPAR)
            for i in range(0, NPAR):
                tmp[i][i] = h_diag[i]
            new_par = par - np.dot(np.linalg.inv(hessian + C_lm * tmp), grad)

            # Adjust J
            p_min = -3
            p_max = 5
            for i in range(2):
                if new_par[NPAR - 2 + i] < p_min:
                    new_par[NPAR - 2 + i] = p_min
                if p_max < new_par[NPAR - 2 + i]:
                    new_par[NPAR - 2 + i] = p_max

            # Whether log posterior probability is increasing
            Gk = self._calc_Gk(new_par, beta, tau, c, n_sp, t_sp, delay_synapse)
            new_log_pos = self._calc_log_posterior(
                new_par, beta, tau, c, n_sp, t_sp, delay_synapse, Gk
            )
            if new_log_pos >= log_pos:
                par = new_par
                C_lm = C_lm * eta
            else:
                C_lm = C_lm * (1 / eta)
                continue

            # Whether the convergence condition is satisfied
            if abs(new_log_pos - log_pos) < 1.0e-4:
                # print('done')
                return (par.T).tolist()[0], new_log_pos

            if l_c > 1000:
                # print('lapse done')
                return (par.T).tolist()[
                    0
                ], new_log_pos  # for now just ignore convergence

    def _GLMCC(self, c, t_sp, tau, beta, pre, post, delay_synapse):
        """
        fit a GLM to the Cross correlogram

        Input:
        Cross correlogram(list), spike time(list), tau(list), beta(float), the number of cell1's spike time(int), the number of cell2's spike time(int), delay_synapse

        Output:
        parameter (list), log_post (float) (if LM method's convergence condition is satisfied.)
        or
        false (if func LM returns false)

        ----------------------------------------------------------------------------------------
        Cross correlogramをGLMでフィッティングする関数。

        """

        WIN = self.params.get("ccg_tau", self.default_params["ccg_tau"]) * 1e3
        DELTA = self.params.get("binsize", self.default_params["binsize"]) * 1e3
        nb = int(WIN / DELTA)
        NPAR = 2 * (nb + 1)

        new_c = [0 for i in range(NPAR)]

        for i in range(len(t_sp)):
            """
            if 1 < abs(t_sp[i]):
                k = (t_sp[i]+WIN)/DELTA
                tmp = math.floor(k)
                if k-tmp == 0:
                    new_c[tmp] = new_c[tmp] + 0.5
                    new_c[tmp-1] = new_c[tmp-1] + 0.5
                elif 0 <= tmp and tmp < NPAR-2:
                    new_c[tmp] += 1
                else:
                    print('Error: '+str(t_sp[i]))

                if delay_synapse < t_sp[i]:
                    new_c[NPAR-2] += np.exp((-1)*(t_sp[i]-delay_synapse)/tau[0])
                if t_sp[i] < -delay_synapse:
                    new_c[NPAR-1] += np.exp((t_sp[i]+delay_synapse)/tau[1])
            """
            k = (t_sp[i] + WIN) / DELTA
            tmp = math.floor(k)
            if k - tmp == 0:
                new_c[tmp] = new_c[tmp] + 0.5
                new_c[tmp - 1] = new_c[tmp - 1] + 0.5
            elif 0 <= tmp and tmp < NPAR - 2:
                new_c[tmp] += 1
            else:
                print("Error: " + str(t_sp[i]))

            if delay_synapse < t_sp[i]:
                new_c[NPAR - 2] += np.exp((-1) * (t_sp[i] - delay_synapse) / tau[0])
            if t_sp[i] < -delay_synapse:
                new_c[NPAR - 1] += np.exp((t_sp[i] + delay_synapse) / tau[1])

        # print(new_c)
        # rate = len(t_sp)/(2*WIN)

        rate = len(t_sp) / (2 * WIN)  # work around for low number of spikes
        if rate == 0:
            rate = 0.0000001

        n_sp = len(t_sp)
        n_pre = pre
        n_post = post
        # Make a prior distribution
        par = self._init_par(rate, NPAR)
        # print("rate: "+str(rate))
        # print("par: ", par)
        # print(t_sp)

        return self._LM(par, beta, tau, new_c, n_sp, t_sp, delay_synapse)

    def _calc_PSP(self, J, c_E=2.532, c_I=0.612):
        """
        calculate PSP

        Output: PSP (float)

        --------------------
        PSPを計算する関数

        出力: PSP (float)
        """
        PSP = 0
        if J > 0:
            PSP = J * c_E
        if J < 0:
            PSP = J * c_I

        return PSP

    def _divide_into_E_I(W_file, n, cell_dir):
        """
        sort cell data from least to most firing rate
        and
        divide W into excitatory and inhibitory connections.

        Input:
        W file name, n(the number of cell), cell directory name

        ----------------------------------------------------------
        セルデータを発火率の低い順に並べ替え、興奮性と抑制性の二つに分ける関数。

        入力:
        Wのファイル名, n(セルの数), セルデータの入っているディレクトリ名

        """
        W = [[0 for i in range(n)] for j in range(n)]
        # read W file
        with open(W_file, "r") as f:
            reader = csv.reader(f)

            i = 0
            for row in reader:
                for j in range(0, n):
                    W[i][j] = float(row[j])
                i += 1

        firing_rate = []
        for i in range(0, n):
            cell_file = open(cell_dir + "/cell" + str(i) + ".txt", "r")
            cell = cell_file.readlines()

            firing_rate.append(len(cell))

        sorted_firing_rate = sorted(firing_rate)

        for i in range(0, n):
            if sorted_firing_rate[i] != firing_rate[i]:
                for j in range(i, n):
                    if sorted_firing_rate[i] == firing_rate[j]:
                        tmp = j
                        break

                tmp_i = []
                for j in range(0, n):
                    tmp_i.append(W[i][j])
                for j in range(0, n):
                    W[i][j] = W[tmp][j]
                for j in range(0, n):
                    W[tmp][j] = tmp_i[j]

                for j in range(0, n):
                    tmp_i[j] = W[j][i]
                for j in range(0, n):
                    W[j][i] = W[j][tmp]
                for j in range(0, n):
                    W[j][tmp] = tmp_i[j]

        # transport W list
        W_t = [list(x) for x in zip(*W)]

        W_e_t = []
        W_i_t = []
        e_cell_list = []
        i_cell_list = []
        cell_list = []

        e_i_rate = [0 for i in range(0, n)]
        for i in range(0, n):
            i_rate = 0
            e_rate = 0
            for j in range(0, n):
                if W_t[i][j] > 0:
                    e_rate += 1
                elif W_t[i][j] < 0:
                    i_rate += 1

            if (e_rate - i_rate) < 0:
                W_i_t.append(W_t[i])
                i_cell_list.append(i)
            else:
                W_e_t.append(W_t[i])
                e_cell_list.append(i)

        W_d_t = W_e_t + W_i_t
        cell_list = e_cell_list + i_cell_list

        W = [list(x) for x in zip(*W_d_t)]

        for i in range(0, n):
            if cell_list[i] != i:
                for j in range(i, n):
                    if cell_list[j] == i:
                        tmp = j
                        break

                tmp_i = []
                for j in range(0, n):
                    tmp_i.append(W[i][j])

                for j in range(0, n):
                    W[i][j] = W[tmp][j]
                for j in range(0, n):
                    W[tmp][j] = tmp_i[j]

        W_d_f = open("sorted_W.csv", "w")

        for i in range(0, n):
            for j in range(0, n):
                if W[i][j] == 0:
                    W[i][j] = int(W[i][j])
                W_d_f.write(str(W[i][j]))
                if j == n - 1:
                    W_d_f.write("\n")
                else:
                    W_d_f.write(", ")

        W_d_f.close()
