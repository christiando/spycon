from spycon.spycon_inference import SpikeConnectivityInference
import elephant
import neo
import quantities
import numpy
import multiprocessing
from itertools import repeat
from scipy.stats import norm


class CoincidenceIndex(SpikeConnectivityInference):
    """
    Coincidence index (CI) method as described in Eq. 5 of the paper:

    de Abril, Ildefons Magrans, Junichiro Yoshimoto, and Kenji Doya.
    "Connectivity inference from neural recording data: Challenges, mathematical bases and research directions."
    Neural Networks 102 (2018): 120-137.

    Args:
        params (dict, optional): Parameters for the CI method:
            - 'binsize' (float): Time step (in seconds) used for time-discretization. Default is 0.4e-3.
            - 'syn_tau' (float): Lag on the cross-correlogram (CCG) taken into account for the denominator in seconds. Default is 0.6e-3.
            - 'ccg_tau' (float): The maximum lag for which the CCG is calculated in seconds. Default is 50e-3.
            - 'alpha' (float): Value used for thresholding p-values (0 < alpha < 1). Default is 0.01.
            - 'jitter_factor' (int): Maximum number of time bins the spikes can be jittered. Default is 7.
            - 'num_surrogates' (int): Number of surrogates to be created. Default is 50.
            - 'jitter' (bool): If True, spikes are uniformly jittered; otherwise, spikes are randomly selected from the population. Default is False.

    Returns:
        None
    """

    def __init__(self, params: dict = {}):

        super().__init__(params)
        self.method = "ci"
        self.default_params = {
            "binsize": 0.4e-3,
            "syn_tau": 6e-3,
            "ccg_tau": 50e-3,
            "alpha": 0.001,
            "jitter_factor": 3,
            "num_surrogates": 50,
            "jitter": False,
        }

    def _infer_connectivity(
        self, times: numpy.ndarray, ids: numpy.ndarray, pairs: numpy.ndarray
    ) -> tuple:
        """
        CCG connectivity inference.

        Args:
            times (numpy.ndarray): Spike times in seconds.
            ids (numpy.ndarray): Unit IDs corresponding to spike times.
            pairs (numpy.ndarray): Array of [pre, post] pair node IDs, for which inference will be done.

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
        nodes = numpy.unique(ids)
        weights = []
        stats = []
        num_connections_to_test = pairs.shape[0]
        conn_count = 0
        pairs_already_computed = numpy.empty((0, 2))
        for pair in tqdm(pairs):
            id1, id2 = pair
            if not any(numpy.prod(pairs_already_computed == [id2, id1], axis=1)):
                logpval1, weight1, logpval2, weight2, pair = self._test_connection_pair(
                    times, ids, pair
                )
                weights.append(weight1)
                stats.append(numpy.array([id1, id2, logpval1]))
                pairs_already_computed = numpy.vstack(
                    [pairs_already_computed, numpy.array([id1, id2])]
                )
                if any(numpy.prod(pairs == [id2, id1], axis=1)):
                    weights.append(weight2)
                    stats.append(numpy.array([id2, id1, logpval2]))
                    conn_count += 2
                    pairs_already_computed = numpy.vstack(
                        [pairs_already_computed, numpy.array([id2, id1])]
                    )
                else:
                    conn_count += 1

        weights = numpy.vstack(weights)
        stats = numpy.vstack(stats)
        threshold = norm.ppf(1 - 0.5 * alpha)
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
            pairs (numpy.ndarray): Array of [pre, post] pair node IDs, for which inference will be done.
            num_cores (int): How many CPU cores should be used for multiprocessing.

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
        nodes = numpy.unique(ids)
        weights = []
        stats = []
        pairs_to_compute = []
        pairs_already_computed = numpy.empty((0, 2))
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

        job_arguments = zip(repeat(times), repeat(ids), pairs_to_compute)
        pool = multiprocessing.Pool(processes=num_cores)
        results = pool.starmap(self._test_connection_pair, job_arguments)
        pool.close()

        for result in results:
            logpval1, weight1, logpval2, weight2, pair = result
            id1, id2 = pair
            weights.append(weight1)
            stats.append(numpy.array([id1, id2, logpval1]))
            if any(numpy.prod(pairs == [id2, id1], axis=1)):
                weights.append(weight2)
                stats.append(numpy.array([id2, id1, logpval2]))

        weights = numpy.vstack(weights)
        stats = numpy.vstack(stats)
        threshold = norm.ppf(1 - 0.5 * alpha)
        return nodes, weights, stats, threshold

    def _test_connection_pair(
        self, times: numpy.ndarray, ids: numpy.ndarray, pair: tuple
    ) -> tuple:
        """
        Test connections in both directions.

        Args:
            times (numpy.ndarray): Spike times in seconds.
            ids (numpy.ndarray): Unit IDs corresponding to spike times.
            pair (tuple): Node pair represented as a tuple (id1, id2).

        Returns:
            tuple: A tuple containing the following elements:
                - pval_id1_to_id2 (float): p-value for the edge from id1 to id2.
                - weight_id1_to_id2 (float): Weight or strength of the edge from id1 to id2.
                - pval_id2_to_id1 (float): p-value for the edge from id2 to id1.
                - weight_id2_to_id1 (float): Weight or strength of the edge from id2 to id1.
        """
        id1, id2 = pair
        binsize = self.params.get("binsize", self.default_params["binsize"])
        syn_tau = self.params.get("syn_tau", self.default_params["syn_tau"])
        jitter_factor = self.params.get(
            "jitter_factor", self.default_params["jitter_factor"]
        )
        t_start, t_stop = numpy.amin(times) - binsize, numpy.amax(times) + binsize
        num_surrogates = self.params.get(
            "num_surrogates", self.default_params["num_surrogates"]
        )

        times1 = times[ids == id1]
        times2 = times[ids == id2]
        ci1_H0, ci2_H0 = numpy.empty(num_surrogates), numpy.empty(num_surrogates)
        for ishuffle in range(num_surrogates):
            if self.params.get("jitter", self.default_params["jitter"]):
                times1_shuffled = numpy.sort(
                    times1
                    + jitter_factor * syn_tau * (numpy.random.rand(len(times1)) - 0.5)
                )
            else:
                times1_shuffled = numpy.sort(
                    numpy.random.choice(times[ids != id1], len(times2), replace=False)
                )
            t_start_tmp, t_stop_tmp = numpy.amin(
                [times1_shuffled[0] - binsize, t_start]
            ), numpy.amax([times1_shuffled[-1] + binsize, t_stop])
            ci1_H0[ishuffle], ci2_H0[ishuffle] = self._compute_ci(
                times1_shuffled, times2, t_start_tmp, t_stop_tmp
            )
        ci1, ci2 = self._compute_ci(times1, times2, t_start, t_stop)
        mu1_H0, std1_H0 = numpy.mean(ci1_H0), numpy.amax([numpy.std(ci1_H0), 1e-10])
        mu2_H0, std2_H0 = numpy.mean(ci2_H0), numpy.amax([numpy.std(ci2_H0), 1e-10])
        zval1 = numpy.abs(ci1 - mu1_H0) / std1_H0
        zval2 = numpy.abs(ci2 - mu2_H0) / std2_H0
        return zval1, ci1, zval2, ci2, pair

    def _compute_ci(
        self,
        times1: numpy.ndarray,
        times2: numpy.ndarray,
        t_start: float,
        t_stop: float,
    ) -> numpy.ndarray:
        """
        Compute the coincidence index.

        Args:
            times1 (numpy.ndarray): Spike times of neuron 1 in seconds.
            times2 (numpy.ndarray): Spike times of neuron 2 in seconds.
            t_start (float): Starting time of the recording in seconds.
            t_stop (float): Stopping time of the recording in seconds.

        Returns:
            numpy.ndarray: Coincidence index values for both directions, 1->2 and 2->1.
        """
        binsize = self.params.get("binsize", self.default_params["binsize"])

        neo_spk_train1 = neo.SpikeTrain(
            times1, units=quantities.second, t_start=t_start, t_stop=t_stop
        )
        neo_spk_train2 = neo.SpikeTrain(
            times2, units=quantities.second, t_start=t_start, t_stop=t_stop
        )
        st1 = elephant.conversion.BinnedSpikeTrain(
            neo_spk_train1, bin_size=binsize * quantities.second, tolerance=None
        )
        st2 = elephant.conversion.BinnedSpikeTrain(
            neo_spk_train2, bin_size=binsize * quantities.second, tolerance=None
        )
        ccg_tau = self.params.get("ccg_tau", self.default_params["ccg_tau"])
        syn_tau = self.params.get("syn_tau", self.default_params["syn_tau"])
        effective_bins = int(ccg_tau / binsize)
        ccg = elephant.spike_train_correlation.cross_correlation_histogram(
            st1,
            st2,
            window=[-effective_bins, effective_bins],
            border_correction=False,
            binary=False,
            kernel=None,
            method="memory",
        )
        # normalization = numpy.sqrt(len(times1) * len(times2))
        # ccg_normed = ccg[0].as_array()/normalization
        st1_mu, st1_std = numpy.mean(st1.to_array()), numpy.std(st1.to_array())
        st2_mu, st2_std = numpy.mean(st2.to_array()), numpy.std(st2.to_array())
        ccg_normed = (
            (ccg[0].as_array() / st1.n_bins - st1_mu * st2_mu) / st1_std / st2_std
        )
        tau = ccg[1] * binsize
        delay_window1 = numpy.where(numpy.logical_and(tau > 0, tau <= syn_tau))[0]
        delay_window2 = numpy.where(numpy.logical_and(tau >= -syn_tau, tau < 0))[0]
        ci1 = numpy.sum(ccg_normed[delay_window1]) / numpy.sum(ccg_normed[tau > 0])
        ci2 = numpy.sum(ccg_normed[delay_window2]) / numpy.sum(ccg_normed[tau < 0])
        return ci1, ci2
