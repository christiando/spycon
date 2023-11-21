import numpy
from scipy.special import digamma, loggamma
from scipy.stats import norm
from scipy import linalg as sc_linalg

from spycon.spycon_inference import SpikeConnectivityInference
import multiprocessing
from itertools import repeat
import time
from tqdm import tqdm


class UnitModel_VI:
    def __init__(
        self,
        unit_idx,
        T,
        times,
        history,
        coupled_neurons,
        r0=True,
        couplings=True,
        coupling_sparsity=False,
        max_rate=None,
        coupling_tau=5e-3,
        coupling_tau_pop=[5e-3],
        model_name=None,
        unit_id=None,
        conv_crit=1e-1,
        max_iter=50,
    ):
        self.unit_idx = unit_idx
        if unit_id is None:
            self.unit_id = self.unit_idx
        else:
            self.unit_id = unit_id
        if model_name is None:
            self.model_name = "%03d" % self.unit_id
        else:
            self.model_name = model_name
        self.num_iters = 0
        self.conv_crit = conv_crit
        self.max_iter = max_iter
        self.T = T
        self.dt = 2e-3
        self.trange = numpy.arange(0, self.T + self.dt, self.dt)
        self.times = times
        self.num_times = len(times)
        self.time_indices = numpy.searchsorted(self.trange, self.times) - 2
        self.time_indices = self.time_indices[
            numpy.logical_and(
                self.time_indices >= 0, self.time_indices < len(self.trange)
            )
        ]
        self.history = history
        self.coupled_neurons = coupled_neurons
        self.N_coupled = len(self.coupled_neurons)
        self.N_tot = len(set(self.history[1]))
        self.r0 = r0
        self.couplings = couplings
        self.coupling_sparsity = coupling_sparsity
        self.noise = 1e-4
        self.coupling_tau = coupling_tau
        self.coupling_tau_pop = coupling_tau_pop
        self.D_pop = len(coupling_tau_pop)
        if max_rate is None:
            max_rate = 2.0 * len(times) / self.T

        # Priors
        self.alpha0 = max_rate
        self.beta0 = 1.0
        self.mu0_r0 = -numpy.log(1.0 / (len(times) / self.T / max_rate) - 1.0)
        self.sigma0_r0 = 5.0 / 1.68
        self.mu0_couplings = 0.0
        self.log_sigma_0_couplings = 0.0
        self.sigma0_couplings = numpy.exp(self.log_sigma_0_couplings)
        self.create_alpha_prior()
        self.construct_phi_matrix()
        self.initialize_q1()
        self.lb_list = []

    def create_distance_dict(self, coupled_pos):
        self.distance = {}
        for unit_id, pos in coupled_pos.items():
            self.distance[unit_id] = numpy.sqrt(numpy.sum((self.unit_pos - pos) ** 2))

    def run(self, update_hyperparams=True):
        converged = False
        while not converged:
            if self.num_iters >= self.max_iter:
                break
            t1 = time.perf_counter()
            self.optimal_q2()
            t2 = time.perf_counter()
            # print("Update q2: %.1f sec" % (t2 - t1))
            self.optimal_q1()
            # print("Update q1: %.1f sec" % (time.perf_counter() - t2))
            if self.num_iters > 1:
                cur_conv_crit = (self.lb_list[-1] - self.lb_list[-2]) / numpy.amax(
                    [1, numpy.abs(self.lb_list[-1]), numpy.abs(self.lb_list[-2])]
                )
                converged = numpy.abs(cur_conv_crit) < self.conv_crit
            self.lb_list.append(self.lower_bound()[0, 0])
            self.num_iters += 1
            if self.num_iters % 10 == 0 and self.num_iters > 2:
                print(
                    """+----------------+\n|Iterations   %03d|\n|Conv-crit %.4f|\n+----------------+"""
                    % (self.num_iters, cur_conv_crit)
                )

    def initialize_q1(self):
        self.Sigma1 = self.Sigma0
        self.Sigma1_inv = self.Sigma0_inv
        self.mu1 = self.mu0
        self.log_det_Sigma1 = self.log_det_Sigma0
        self.alpha1 = self.alpha0
        self.beta1 = self.beta0
        self.mu_lmbda_bar = self.alpha1 / self.beta1
        self.mu_ln_lmbda_bar = digamma(self.alpha1) - numpy.log(self.beta1)
        self.lmbda_bar_1 = numpy.exp(self.mu_ln_lmbda_bar)

    def create_alpha_prior(self):
        mu0_list = []
        Sigma0_list, Sigma0_inv_list, self.log_det_Sigma0 = (
            [],
            [],
            numpy.zeros((1,)),
        )
        if self.r0:
            diag_r0 = self.sigma0_r0**2 * numpy.ones((1,))
            Sigma0_r0 = numpy.diag(diag_r0)
            Sigma0_inv_r0 = numpy.diag(1.0 / diag_r0)
            Sigma0_list.append(Sigma0_r0)
            Sigma0_inv_list.append(Sigma0_inv_r0)
            self.log_det_Sigma0 += numpy.sum(numpy.log(diag_r0))
            mu0_list.append(
                self.mu0_r0
                * numpy.ones(
                    (1, 1),
                )
            )
        if self.couplings:
            self.sigma0_couplings = numpy.exp(self.log_sigma_0_couplings)
            if self.coupling_sparsity:
                diag_couplings = 1.0 / self.mu_tau
                self.log_det_Sigma0 += -numpy.reduce_sum(self.mu_ln_tau)
            else:
                diag_couplings = self.sigma0_couplings**2 * numpy.ones(
                    (self.N_coupled,),
                )
                self.log_det_Sigma0 += numpy.sum(numpy.log(diag_couplings))
            Sigma0_couplings = numpy.diag(diag_couplings)
            Sigma0_inv_couplings = numpy.diag(1.0 / diag_couplings)
            Sigma0_list.append(Sigma0_couplings)
            Sigma0_inv_list.append(Sigma0_inv_couplings)
            mu0_list.append(self.mu0_couplings * numpy.ones((self.N_coupled, 1)))
        self.Sigma0 = sc_linalg.block_diag(*Sigma0_list)
        self.Sigma0_inv = sc_linalg.block_diag(*Sigma0_inv_list)
        self.mu0 = numpy.concatenate(mu0_list, axis=0)

    def construct_phi_matrix(self, convolve=True):
        phi_list = []
        if self.r0:
            phi_r0 = numpy.ones((len(self.trange), 1))
            phi_list.append(phi_r0)
        if self.couplings:
            if convolve:
                self.compute_coupling_convolution()
            phi_list.append(self.h_couplings)
        self.phi = numpy.concatenate(phi_list, axis=1)

    def optimal_q2(self):
        mu_phi = numpy.dot(self.phi, self.mu1)
        EQ_g2 = self.Sigma1 + numpy.dot(self.mu1, self.mu1.T)
        phi_E2_phi = numpy.sum(
            numpy.dot(self.phi, EQ_g2) * self.phi, axis=1, keepdims=True
        )
        self.c_t = numpy.sqrt(phi_E2_phi)
        self.c_t = numpy.clip(self.c_t, 1e-10, numpy.inf)  # numpy.max(self.c_t, 1e-10)
        self.mu_omega_t = numpy.exp(
            numpy.log(numpy.tanh(0.5 * self.c_t)) - numpy.log(2.0 * self.c_t)
        )
        self.lmbda_2 = 0.5 * numpy.exp(
            self.mu_ln_lmbda_bar - 0.5 * mu_phi - numpy.log(numpy.cosh(0.5 * self.c_t))
        )

        """
        if self.coupling_sparsity:
            self.alpha_tau = 3.0 / 2.0 * numpy.ones(self.N_coupled, dtype=default_dtype)
            E_alpha_coupling2 = tf.constant(
                tf.gather(tf.linalg.diag_part(self.Sigma1), self.coupling_indices)
                + tf.gather(self.mu1[:, 0], self.coupling_indices) ** 2
            )
            self.beta_tau = tf.constant(
                0.5 * E_alpha_coupling2 + 0.5 * self.sigma0_couplings**2
            )
            self.mu_tau = self.alpha_tau / self.beta_tau
            self.mu_ln_tau = tf.math.digamma(self.alpha_tau) - tf.math.log(
                self.beta_tau
            )
            self.create_alpha_prior(init_hyperparams=False)
        """

    def optimal_q1(self):
        phi_events = self.phi[self.time_indices]
        mu_omega_events = numpy.reshape(
            self.mu_omega_t[self.time_indices], [len(self.time_indices), 1]
        )
        weighted_phi_events = mu_omega_events * phi_events
        A = 0.5 * numpy.dot(weighted_phi_events.T, phi_events)
        mu_omega_lmbda2 = numpy.reshape(
            self.lmbda_2 * self.mu_omega_t, [len(self.trange), 1]
        )
        weighted_phi = mu_omega_lmbda2 * self.phi
        B = 0.5 * numpy.dot(weighted_phi.T, self.phi) * self.T / len(self.trange)
        self.Sigma1_inv = self.Sigma0_inv + 2.0 * (A + B)
        self.L1_inv = numpy.linalg.cholesky(self.Sigma1_inv)
        L1 = sc_linalg.solve_triangular(
            self.L1_inv, numpy.eye(self.L1_inv.shape[0]), lower=True, check_finite=False
        )
        self.Sigma1 = L1.T.dot(L1)
        self.log_det_Sigma1 = -2.0 * numpy.sum(numpy.log(numpy.diagonal(self.L1_inv)))
        phi_events = self.phi[self.time_indices]
        a = (0.5 * numpy.sum(phi_events, axis=0, keepdims=True)).T
        weighted_phi = numpy.reshape(self.lmbda_2, [len(self.trange), 1]) * self.phi
        b = (-0.5 * numpy.mean(weighted_phi, axis=0, keepdims=True) * self.T).T
        self.mu1 = numpy.dot(self.Sigma1, a + b + self.Sigma0_inv.dot(self.mu0))
        self.alpha1 = self.num_times + numpy.mean(self.lmbda_2) * self.T + self.alpha0
        self.beta1 = self.T + self.beta0
        self.mu_lmbda_bar = self.alpha1 / self.beta1
        self.mu_ln_lmbda_bar = digamma(self.alpha1) - numpy.log(self.beta1)
        self.lmbda_bar_1 = numpy.exp(self.mu_ln_lmbda_bar)

    def lower_bound(self):
        self.create_alpha_prior()
        phi_events = self.phi[self.time_indices]
        a = (0.5 * numpy.sum(phi_events, axis=0, keepdims=True)).T
        weighted_phi = numpy.reshape(self.lmbda_2, [len(self.trange), 1]) * self.phi
        b = (-0.5 * numpy.mean(weighted_phi, axis=0, keepdims=True) * self.T).T
        mu_omega_events = numpy.reshape(
            self.mu_omega_t[self.time_indices], [len(self.time_indices), 1]
        )
        weighted_phi_events = mu_omega_events * phi_events
        A = 0.5 * numpy.dot(weighted_phi_events.T, phi_events)
        mu_omega_lmbda2 = numpy.reshape(
            self.lmbda_2 * self.mu_omega_t, [len(self.trange), 1]
        )
        weighted_phi = mu_omega_lmbda2 * self.phi
        B = 0.5 * numpy.dot(weighted_phi.T, self.phi) * self.T / len(self.trange)
        EQ_g2 = self.Sigma1 + numpy.dot(self.mu1, self.mu1.T)
        EQ_ln_L = (
            numpy.sum(self.mu1 * (a + b), keepdims=True)
            - numpy.trace(numpy.dot(A + B, EQ_g2))
            + self.num_times * (self.mu_ln_lmbda_bar - numpy.log(2))
        )
        EQ_pq_alpha = 0.5 * (
            self.log_det_Sigma1
            - self.log_det_Sigma0
            - numpy.trace(self.Sigma0_inv.dot(self.Sigma1))
            - numpy.dot(
                (self.Sigma0_inv.dot(self.mu1 - self.mu0)).T,
                self.mu1 - self.mu0,
            )
            + self.Sigma1.shape[0]
        )
        EQ_pq_lmbda_bar = (
            self.alpha0 * numpy.log(self.beta0)
            - self.alpha1 * numpy.log(self.beta1)
            - loggamma(self.alpha0)
            + loggamma(self.alpha1)
            + (self.alpha0 - self.alpha1) * self.mu_ln_lmbda_bar
            - (self.beta0 - self.beta1) * self.mu_lmbda_bar
        )
        c_events = self.c_t[self.time_indices]
        EQ_pq_omega = 0.5 * numpy.sum(c_events**2 * mu_omega_events) - numpy.sum(
            numpy.log(numpy.cosh(0.5 * c_events))
        )
        EQ_pq_Pi = numpy.mean(
            (
                self.mu_ln_lmbda_bar
                - numpy.log(self.lmbda_2)
                - numpy.log(2.0 * numpy.cosh(0.5 * self.c_t))
                + 0.5 * self.c_t**2 * self.mu_omega_t
            )
            * self.lmbda_2
        ) * self.T - (self.T * self.mu_lmbda_bar - numpy.mean(self.lmbda_2) * self.T)
        elbo = EQ_ln_L + EQ_pq_alpha + EQ_pq_lmbda_bar + EQ_pq_omega + EQ_pq_Pi
        # print(EQ_ln_L, EQ_pq_alpha, EQ_pq_lmbda_bar, EQ_pq_omega, EQ_pq_Pi)
        return elbo

    def compute_coupling_convolution(self):
        h_couplings = numpy.zeros((len(self.trange), self.N_coupled))
        spk_times, unit_indices = self.history
        for nidx, coupled_unit in enumerate(self.coupled_neurons):
            spk_times_unit = spk_times[numpy.where(unit_indices == coupled_unit)[0]]
            if coupled_unit != self.unit_id:
                h_couplings[:, nidx] = self.convolve_spk_trains(
                    spk_times_unit, self.coupling_tau
                )[:, 0]
            else:
                h_couplings[:, nidx] = self.convolve_spk_trains(
                    spk_times_unit, self.coupling_tau
                )[:, 0]
        self.h_couplings = h_couplings

    def compute_pop_convolution(self):
        h_pop = numpy.zeros((len(self.trange), self.D_pop))
        spk_times, unit_indices = self.history
        for didx, tau in enumerate(self.coupling_tau_pop):
            spks_to_convolve = numpy.copy(spk_times)
            h_pop[:, didx] /= self.N_tot
        self.h_pop = h_pop

    def convolve_spk_trains(self, spk_times, tau, unit_indices=[]):
        # TODO: Could be done in tensorflow-scientific
        spk_times = spk_times[spk_times < self.trange[-1]]
        spk_idx = numpy.searchsorted(self.trange, spk_times, side="left")
        spk_idx_unique, counts = numpy.unique(spk_idx, return_counts=True)
        delta_t = numpy.arange(0, 20 * tau, self.dt)
        conv_filter = numpy.exp(-delta_t / tau)
        y = numpy.zeros((len(self.trange)))
        y[spk_idx_unique] = counts
        y = numpy.array(
            [numpy.convolve(y, conv_filter, mode="full")[1 : -len(conv_filter) + 2]]
        ).T
        y[y <= 0] = 1e-10
        return y


class GLMPP(SpikeConnectivityInference):
    """
    Implements the point process GLM method.

    Args:
        params (dict): A dictionary containing the following parameters:

            - 'tau' (float): Time constant of the history kernel. Default is 5e-3.
            - 'alpha' (float): Threshold. Default is 1e-2.

    """

    def __init__(self, params: dict = {}):

        super().__init__(params)
        self.method = "glmpp"
        self.default_params = {"tau": 5e-2, "alpha": 0.001}

    def _infer_connectivity(
        self, times: numpy.ndarray, ids: numpy.ndarray, pairs: numpy.ndarray
    ) -> tuple:
        """
        GLM connectivity inference.

        Args:
            times (numpy.ndarray): Spike times in seconds.
            ids (numpy.ndarray): Unit IDs corresponding to spike times.
            pairs (numpy.ndarray): Array of [pre, post] pair node IDs for which inference will be done.

        Returns:
            tuple: A tuple containing the following elements:
                - nodes (numpy.ndarray): An array of node labels with shape [number_of_nodes].
                - weights (numpy.ndarray): An array of graded strengths of connections with shape [number_of_edges].
                - stats (numpy.ndarray): A 2D array representing a fully connected graph with shape [number_of_edges, 3].
                The columns are as follows:
                    - The first column represents outgoing nodes.
                    - The second column represents incoming nodes.
                    - The third row contains the statistic used to decide whether it is an edge or not.
                        A higher value indicates that an edge is more probable.
                - threshold (float): A float value that considers an edge to be a connection if stats > threshold.

        """

        alpha = self.params.get("alpha", self.default_params["alpha"])
        nodes = numpy.unique(ids)
        weights = []
        stats = []
        num_connections_to_test = pairs.shape[0]
        conn_count = 0
        target_nodes = numpy.unique(pairs[:, 1])
        for id1 in tqdm(target_nodes):
            unit_weights, unit_stats, id1_tmp = self._test_unit(
                numpy.copy(times), numpy.copy(ids), id1
            )
            for junit, id2 in enumerate(nodes):
                if id1_tmp != id2 and any(numpy.prod(pairs == [id2, id1_tmp], axis=1)):
                    weights.append(unit_weights[junit])
                    stats.append(numpy.array([id2, id1_tmp, unit_stats[junit]]))
            conn_count += len(nodes) - 1
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
        threshold = norm.isf(alpha)
        return nodes, weights, stats, threshold

    def _infer_connectivity_parallel(
        self,
        times: numpy.ndarray,
        ids: numpy.ndarray,
        pairs: numpy.ndarray,
        num_cores: int,
    ) -> tuple:
        """
        GLM connectivity inference. Parallel version.

        Args:
            times (numpy.ndarray): Spike times in seconds.
            ids (numpy.ndarray): Unit IDs corresponding to spike times.
            pairs (numpy.ndarray): Array of [pre, post] pair node IDs for which inference will be done.
            num_cores (int): How many cores should be used for multiprocessing.

        Returns:
            tuple: A tuple containing the following elements:
                - nodes (numpy.ndarray): An array of node labels with shape [number_of_nodes].
                - weights (numpy.ndarray): An array of graded strengths of connections with shape [number_of_edges].
                - stats (numpy.ndarray): A 2D array representing a fully connected graph with shape [number_of_edges, 3].
                The columns are as follows:
                    - The first column represents outgoing nodes.
                    - The second column represents incoming nodes.
                    - The third row contains the statistic used to decide whether it is an edge or not.
                        A higher value indicates that an edge is more probable.
                - threshold (float): A float value that considers an edge to be a connection if stats > threshold.

        """

        alpha = self.params.get("alpha", self.default_params["alpha"])
        nodes = numpy.unique(ids)
        weights = []
        stats = []
        target_nodes = numpy.unique(pairs[:, 1])
        job_arguments = zip(repeat(times), repeat(ids), target_nodes)
        pool = multiprocessing.Pool(processes=num_cores)
        results = pool.starmap(self._test_unit, job_arguments)
        pool.close()
        for result in results:
            unit_weights, unit_stats, id1 = result
            for junit, id2 in enumerate(nodes):
                if id1 != id2 and any(numpy.prod(pairs == [id2, id1], axis=1)):
                    weights.append(unit_weights[junit])
                    stats.append(numpy.array([id2, id1, unit_stats[junit]]))
        weights = numpy.vstack(weights)
        stats = numpy.vstack(stats)
        threshold = norm.isf(alpha / len(nodes))
        return nodes, weights, stats, threshold

    def _test_unit(self, times: numpy.ndarray, ids: numpy.ndarray, id1: int) -> float:
        """
        Test connections in both directions.

        Args:
            times (numpy.ndarray): Spike times in seconds.
            ids (numpy.ndarray): Unit IDs corresponding to spike times.
            id1 (int): ID of the first node.

        Returns:
            tuple: A tuple containing the following elements:
                - pval_id1_to_id2 (float): p-value for the edge id1->id2.
                - weight_id1_to_id2 (float): Weight for the edge id1->id2.
                - pval_id2_to_id1 (float): p-value for the edge id2->id1.
                - weight_id2_to_id1 (float): Weight for the edge id2->id1.

        """

        ids_tmp = numpy.empty(ids.shape)
        id_set = numpy.unique(ids)
        for iunit, unit_id in enumerate(id_set):
            ids_tmp[ids == unit_id] = iunit
        nidx = numpy.where(id_set == id1)[0][0]
        tau = self.params.get("tau", self.default_params["tau"])
        times_tmp = times - times[0]
        T = times_tmp[-1]
        history = [times_tmp, ids_tmp]
        unit_times = times_tmp[ids == id1]
        unit_vi = UnitModel_VI(nidx, T, unit_times, history, numpy.unique(ids_tmp))
        unit_vi.run(update_hyperparams=False)
        weights = unit_vi.mu1[1:, 0]
        std = numpy.sqrt(unit_vi.Sigma1.diagonal()[1:])
        del unit_vi
        unit_stats = numpy.abs(weights) / std
        return weights, unit_stats, id1

    def _return_unit_vi(
        self, times: numpy.ndarray, ids: numpy.ndarray, id1: int
    ) -> float:
        """
        Fit model for one unit.

        Args:
            times (numpy.ndarray): Spike times in seconds.
            ids (numpy.ndarray): Unit IDs corresponding to spike times.
            id1 (int): ID of the first node.

        Returns:
            tuple: A tuple containing the following elements:
                - pval_id1_to_id2 (float): p-value for the edge id1->id2.
                - weight_id1_to_id2 (float): Weight for the edge id1->id2.
                - pval_id2_to_id1 (float): p-value for the edge id2->id1.
                - weight_id2_to_id1 (float): Weight for the edge id2->id1.

        """

        ids_tmp = numpy.empty(ids.shape)
        id_set = numpy.unique(ids)
        for iunit, unit_id in enumerate(id_set):
            ids_tmp[ids == unit_id] = iunit
        nidx = numpy.where(id_set == id1)[0][0]
        times_tmp = times - times[0]
        T = times_tmp[-1]
        history = [times_tmp, ids_tmp]
        unit_times = times_tmp[ids == id1]
        unit_vi = UnitModel_VI(nidx, T, unit_times, history, numpy.unique(ids_tmp))
        return unit_vi
