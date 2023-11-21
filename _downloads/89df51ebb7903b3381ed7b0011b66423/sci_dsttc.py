from spycon.spycon_inference import SpikeConnectivityInference
import numpy
from scipy.stats import norm
from itertools import repeat
import multiprocessing
from tqdm import tqdm


class directed_STTC(SpikeConnectivityInference):
    """
    Directed version of the spike time tiling coefficient (STTC), as originally proposed in:

    Cutts, Catherine S., and Stephen J. Eglen. "Detecting pairwise correlations in spike trains: an objective comparison of methods and application to the study of retinal waves." Journal of Neuroscience 34.43 (2014): 14288-14303.

    Args:
        params (dict, optional): Parameters for the STTC method:

            - 'delta_t' (float): Synaptic time window in seconds. Default is 7e-3.
            - 'jitter_factor' (int): Maximum number of time bins the spikes can be jittered. Default is 7.
            - 'num_surrogates' (int): Number of surrogates to be created. Default is 50.
            - 'jitter' (bool): If True, spikes are uniformly jittered; otherwise, spikes are randomly selected from the population. Default is False.
            - 'alpha' (float): Threshold. Default is 1e-3.
    """

    def __init__(self, params: dict = {}):
        super().__init__(params)
        self.method = "dsttc"
        # In this dictionary specify all default values for the algorithm, ideally taken from the corresponding publication.
        self.default_params = {
            "delta_t": 7e-3,
            "alpha": 1e-3,
            "num_surrogates": 50,
            "jitter": False,
            "jitter_factor": 7.0,
        }

    def _infer_connectivity(
        self, times: numpy.ndarray, ids: numpy.ndarray, pairs: numpy.ndarray
    ) -> tuple:
        """
        Compute the directed spike time tiling coefficient estimation.

        Args:
            times (numpy.ndarray): Spike times in seconds.
            ids (numpy.ndarray): Unit IDs corresponding to spike times.
            pairs (numpy.ndarray): Array of [pre, post] pair node IDs for which inference will be done.

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
                zval1, weight1, zval2, weight2, pair = self._test_connection_pair(
                    times, ids, pair
                )
                weights.append(weight1)
                stats.append(numpy.array([id1, id2, zval1]))
                pairs_already_computed = numpy.vstack(
                    [pairs_already_computed, numpy.array([id1, id2])]
                )
                if any(numpy.prod(pairs == [id2, id1], axis=1)):
                    weights.append(weight2)
                    stats.append(numpy.array([id2, id1, zval2]))
                    conn_count += 2
                    pairs_already_computed = numpy.vstack(
                        [pairs_already_computed, numpy.array([id2, id1])]
                    )
                else:
                    conn_count += 1

        weights = numpy.vstack(weights)
        stats = numpy.vstack(stats)
        # change pvalues such that predicted edges have positive value
        # stats[:,2] = 1 - stats[:,2]
        threshold = norm.ppf(1 - 0.5 * alpha)
        return nodes, weights, stats, threshold

    def _infer_connectivity_parallel(
        self,
        times: numpy.ndarray,
        ids: numpy.ndarray,
        pairs: numpy.ndarray,
        num_cores: int,
    ) -> numpy.ndarray:
        """
        Compute the directed spike time tiling coefficient estimation in parallel.

        Args:
            times (numpy.ndarray): Spike times in seconds.
            ids (numpy.ndarray): Unit IDs corresponding to spike times.
            pairs (numpy.ndarray): Array of [pre, post] pair node IDs for which inference will be done.

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

        job_arguments = zip(repeat(times), repeat(ids), pairs_to_compute)
        pool = multiprocessing.Pool(processes=num_cores)
        results = pool.starmap(self._test_connection_pair, job_arguments)
        pool.close()
        for result in results:
            zval1, weight1, zval2, weight2, pair = result
            id1, id2 = pair
            weights.append(weight1)
            stats.append(numpy.array([id1, id2, zval1]))
            if any(numpy.prod(pairs == [id2, id1], axis=1)):
                weights.append(weight2)
                stats.append(numpy.array([id2, id1, zval2]))

        weights = numpy.vstack(weights)
        stats = numpy.vstack(stats)
        # change pvalues such that predicted edges have positive value
        # stats[:,2] = - stats[:,2]
        threshold = norm.ppf(1 - 0.5 * alpha)
        return nodes, weights, stats, threshold

    def _test_connection_pair(
        self, times: numpy.ndarray, ids: numpy.ndarray, pair: tuple
    ):
        idB, idA = pair
        delta_t = self.params.get("delta_t", self.default_params["delta_t"])
        T = times[-1] - times[0]
        timesA = times[ids == idA]
        timesB = times[ids == idB]
        num_surrogates = self.params.get(
            "num_surrogates", self.default_params["num_surrogates"]
        )
        STTC_H0 = numpy.empty(num_surrogates)
        for ishuffle in range(num_surrogates):
            if self.params.get("jitter", self.default_params["jitter"]):
                jitter_factor = self.params.get(
                    "jitter_factor", self.default_params["jitter_factor"]
                )
                timesB_shuffled = numpy.sort(
                    timesB
                    + jitter_factor * delta_t * (numpy.random.rand(len(timesB)) - 0.5)
                )
            else:
                timesB_shuffled = numpy.sort(
                    numpy.random.choice(times[ids != idA], len(timesB), replace=False)
                )
            STTC_H0[ishuffle] = self._compute_directed_STTC(timesB_shuffled, timesA, T)
        STTC_BA = self._compute_directed_STTC(timesB, timesA, T)
        mu_H0, std_H0 = numpy.mean(STTC_H0), numpy.amax([numpy.std(STTC_H0), 1e-10])

        zval_BA = numpy.abs(STTC_BA - mu_H0) / std_H0
        STTC_H0 = numpy.empty(num_surrogates)
        for ishuffle in range(num_surrogates):
            if self.params.get("jitter", self.default_params["jitter"]):
                jitter_factor = self.params.get(
                    "jitter_factor", self.default_params["jitter_factor"]
                )
                timesA_shuffled = numpy.sort(
                    timesA
                    + jitter_factor * delta_t * (numpy.random.rand(len(timesA)) - 0.5)
                )
            else:
                timesA_shuffled = numpy.sort(
                    numpy.random.choice(times[ids != idB], len(timesA), replace=False)
                )
            STTC_H0[ishuffle] = self._compute_directed_STTC(timesA_shuffled, timesB, T)
        STTC_AB = self._compute_directed_STTC(timesA, timesB, T)
        mu_H0, std_H0 = numpy.mean(STTC_H0), numpy.amax([numpy.std(STTC_H0), 1e-10])
        zval_AB = numpy.abs(STTC_AB - mu_H0) / std_H0
        return zval_BA, STTC_BA, zval_AB, STTC_AB, pair

    def _compute_directed_STTC(
        self, timesB: numpy.ndarray, timesA: numpy.ndarray, T: float
    ) -> float:
        timesA_tmp = numpy.copy(timesA)
        delta_t = self.params.get("delta_t", self.default_params["delta_t"])
        interval_duration_bw = delta_t * numpy.ones(len(timesA_tmp))
        small_intervals_exist = True
        while small_intervals_exist:
            isisA = numpy.diff(timesA_tmp)
            small_isis = numpy.where(isisA < delta_t)[0]
            small_intervals_exist = len(small_isis) != 0
            timesA_tmp = numpy.delete(timesA_tmp, small_isis)
            interval_duration_bw[small_isis + 1] += isisA[small_isis]
            interval_duration_bw = numpy.delete(interval_duration_bw, small_isis)
        windowsA_bw = numpy.zeros((len(timesA_tmp), 2))
        windowsA_bw[:, 0] = timesA_tmp - interval_duration_bw
        windowsA_bw[:, 1] = timesA_tmp
        Ta_bw = numpy.sum(interval_duration_bw) / T
        possible_interval = numpy.searchsorted(
            windowsA_bw[:, 1], timesB[timesB < timesA_tmp[-1]]
        )
        Pb_bw = numpy.sum(
            windowsA_bw[possible_interval, 0] < timesB[timesB < timesA_tmp[-1]]
        ) / len(timesB)

        timesB_tmp = numpy.copy(timesB)
        delta_t = 5e-3
        interval_duration_fw = delta_t * numpy.ones(len(timesB))
        small_intervals_exist = True
        while small_intervals_exist:
            isisB = numpy.diff(timesB_tmp)
            small_isis = numpy.where(isisB < delta_t)[0]
            small_intervals_exist = len(small_isis) != 0
            timesB_tmp = numpy.delete(timesB_tmp, small_isis + 1)
            interval_duration_fw[small_isis] += isisB[small_isis]
            interval_duration_fw = numpy.delete(interval_duration_fw, small_isis + 1)
        windowsB_fw = numpy.zeros((len(timesB_tmp), 2))
        windowsB_fw[:, 0] = timesB_tmp
        windowsB_fw[:, 1] = timesB_tmp + interval_duration_fw
        Tb_fw = numpy.sum(interval_duration_fw) / T
        possible_interval = numpy.searchsorted(
            windowsB_fw[:, 1], timesA[timesA < windowsB_fw[-1, 1]]
        )
        Pa_fw = numpy.sum(
            windowsB_fw[possible_interval, 0] < timesA[timesA < windowsB_fw[-1, 1]]
        ) / len(timesA)
        directed_STTC = 0.5 * (
            (Pb_bw - Ta_bw) / (1.0 - Pb_bw * Ta_bw)
            + (Pa_fw - Tb_fw) / (1.0 - Pa_fw * Tb_fw)
        )
        return directed_STTC
