from spycon.spycon_inference import SpikeConnectivityInference

# Import classes
from idtxl.estimators_jidt import JidtDiscreteTE

from itertools import repeat
import multiprocessing
import numpy
from scipy.stats import norm
from tqdm import tqdm


class TE_IDTXL(SpikeConnectivityInference):
    """
    Transfer entropy method.

    This method is based on the following reference:
    - P. Wollstadt, J. T. Lizier, R. Vicente, C. Finn, M. Martinez-Zarzuela, P. Mediano, L. Novelli, M. Wibral (2019).
    IDTxl: The Information Dynamics Toolkit xl: a Python package for the efficient analysis of multivariate information dynamics in networks.
    Journal of Open Source Software, 4(34), 1081.

    Args:
        params (dict): A dictionary containing the following parameters:

            - 'binsize' (float): Time step (in seconds) used for time-discretization. Default is 5e-3.
            - 'k' (int): Length of history embedding. Default is 2.
            - 'alpha' (float): Threshold. Default is 1e-2.
            - 'jitter_factor' (int): Maximum number of time bins for spike jittering. Default is 7.
            - 'num_surrogates' (int): Number of surrogates to be created. Default is 50.
            - 'jitter' (bool): If True, spikes are uniformly jittered; otherwise, spikes are randomly selected from the population. Default is False.
    """

    def __init__(self, params: dict = {}):

        super().__init__(params)
        self.method = "idtxl"
        self.default_params = {
            "binsize": 5e-3,
            "history_target": 1,
            "history_source": 2,
            "num_surrogates": 50,
            "alpha": 0.01,
            "max_num_spikes_per_bin": 5,
            "jitter": False,
            "jitter_factor": 7.0,
            "source_target_delay": 1,
        }

    def _infer_connectivity(
        self, times: numpy.ndarray, ids: numpy.ndarray, pairs: numpy.ndarray
    ) -> tuple:
        """
        TE connectivity inference.

        This method performs connectivity inference using Transfer Entropy (TE) based on spike times.

        Args:
            times (numpy.ndarray): Spike times in seconds.
            ids (numpy.ndarray): Unit ids corresponding to spike times.
            pairs (numpy.ndarray): Array of [pre, post] pair node IDs for which inference will be done.

        Returns:
            tuple: A tuple containing the following elements:
                1) nodes (numpy.ndarray): An array of node labels [number_of_nodes].
                2) weights (numpy.ndarray): An array of graded strength of connections [number_of_edges].
                3) stats (numpy.ndarray): A 2D array containing a fully connected graph, where the first column represents
                outgoing nodes, the second column represents incoming nodes, and the third column contains the statistic
                used to decide whether it is an edge or not. A higher value indicates that an edge is more probable
                [number_of_edges, 3].
                4) threshold (float): A threshold value that considers an edge to be a connection if stats > threshold.

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
                te1, zval1, te2, zval2, pair = self._test_connection_pair(
                    times, ids, pair
                )
                weights.append(te1)
                stats.append(numpy.array([id1, id2, zval1]))
                pairs_already_computed = numpy.vstack(
                    [pairs_already_computed, numpy.array([id1, id2])]
                )
                if any(numpy.prod(pairs == [id2, id1], axis=1)):
                    weights.append(te2)
                    stats.append(numpy.array([id2, id1, zval2]))
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
        TE connectivity inference. Parallel version.

        This method performs parallelized connectivity inference using Transfer Entropy (TE) based on spike times.

        Args:
            times (numpy.ndarray): Spike times in seconds.
            ids (numpy.ndarray): Unit ids corresponding to spike times.
            pairs (numpy.ndarray): Array of [pre, post] pair node IDs for which inference will be done.

        Returns:
            tuple: A tuple containing the following elements:
                1) nodes (numpy.ndarray): An array of node labels [number_of_nodes].
                2) weights (numpy.ndarray): An array of graded strength of connections [number_of_edges].
                3) stats (numpy.ndarray): A 2D array containing a fully connected graph, where the first column represents
                outgoing nodes, the second column represents incoming nodes, and the third column contains the statistic
                used to decide whether it is an edge or not. A higher value indicates that an edge is more probable
                [number_of_edges, 3].
                4) threshold (float): A threshold value that considers an edge to be a connection if stats > threshold.

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
            te1, zval1, te2, zval2, pair = result
            id1, id2 = pair
            weights.append(te1)
            stats.append(numpy.array([id1, id2, zval1]))
            if any(numpy.prod(pairs == [id2, id1], axis=1)):
                weights.append(te2)
                stats.append(numpy.array([id2, id1, zval2]))

        weights = numpy.vstack(weights)
        stats = numpy.vstack(stats)
        # change pvalues such that predicted edges have positive value
        # stats[:,2] = 1 - stats[:,2]
        threshold = norm.ppf(1 - 0.5 * alpha)
        return nodes, weights, stats, threshold

    def _test_connection_pair(
        self, times: numpy.ndarray, ids: numpy.ndarray, pair: tuple
    ) -> (float):
        """
        Test connections in both directions.

        This method tests connections in both directions for a given node pair based on spike times and unit IDs.

        Args:
            times (numpy.ndarray): Spike times in seconds.
            ids (numpy.ndarray): Unit ids corresponding to spike times.
            pair (tuple): Node pair represented as a tuple (id1, id2).

        Returns:
            tuple: A tuple containing the following elements:
                1) pval1 (float): P-value for the edge from id1 to id2.
                2) weight1 (float): Weight or strength of the edge from id1 to id2.
                3) pval2 (float): P-value for the edge from id2 to id1.
                4) weight2 (float): Weight or strength of the edge from id2 to id1.

        Return Type:
            tuple of floats

        """
        id1, id2 = pair
        binsize = self.params.get("binsize", self.default_params["binsize"])
        max_num_spikes = self.params.get(
            "max_num_spikes_per_bin", self.default_params["max_num_spikes_per_bin"]
        )
        settings = {
            "history_target": self.params.get(
                "history_target", self.default_params["history_target"]
            ),
            "history_sources": self.params.get(
                "history_source", self.default_params["history_source"]
            ),
            "source_target_delay": self.params.get(
                "source_target_delay", self.default_params["source_target_delay"]
            ),
            "alph1": max_num_spikes + 1,
            "alph2": max_num_spikes + 1,
            "alphc": max_num_spikes + 1,
        }
        num_surrogates = self.params.get(
            "num_surrogates", self.default_params["num_surrogates"]
        )
        bins = numpy.arange(
            numpy.floor(1e3 * times[0]) * 1e-3,
            numpy.ceil(1e3 * times[-1]) * 1e-3,
            binsize,
        )
        times1 = times[ids == id1]
        times2 = times[ids == id2]
        spk_train1 = numpy.histogram(times1, bins=bins)[0]
        spk_train1[spk_train1 > max_num_spikes] = max_num_spikes
        spk_train2 = numpy.histogram(times2, bins=bins)[0]
        spk_train2[spk_train2 > max_num_spikes] = max_num_spikes
        te_estimator = JidtDiscreteTE(settings)
        te1 = te_estimator.estimate(spk_train1, spk_train2)
        te1_H0 = numpy.empty(num_surrogates)
        for isurr in range(num_surrogates):
            if self.params.get("jitter", self.default_params["jitter"]):
                jitter_factor = self.params.get(
                    "jitter_factor", self.default_params["jitter_factor"]
                )
                times1_shuffled = numpy.sort(
                    times1
                    + jitter_factor * binsize * (numpy.random.rand(len(times1)) - 0.5)
                )
            else:
                times1_shuffled = numpy.sort(
                    numpy.random.choice(times[ids != id2], len(times1), replace=False)
                )
            spk_train1_shuffled = numpy.histogram(times1_shuffled, bins=bins)[0]
            spk_train1_shuffled[spk_train1_shuffled > max_num_spikes] = max_num_spikes
            te1_H0[isurr] = te_estimator.estimate(spk_train1_shuffled, spk_train2)
        mu_H0, std_H0 = numpy.mean(te1_H0), numpy.amax([numpy.std(te1_H0), 1e-10])
        # pval_tmp = norm.cdf(te1, loc=mu_H0, scale=std_H0)
        # pval1 = numpy.amin([pval_tmp, 1. - pval_tmp])
        zval1 = numpy.abs(te1 - mu_H0) / std_H0

        te2 = te_estimator.estimate(spk_train2, spk_train1)
        te2_H0 = numpy.empty(num_surrogates)
        for isurr in range(num_surrogates):
            if self.params.get("jitter", self.default_params["jitter"]):
                jitter_factor = self.params.get(
                    "jitter_factor", self.default_params["jitter_factor"]
                )
                times2_shuffled = numpy.sort(
                    times2
                    + jitter_factor * binsize * (numpy.random.rand(len(times2)) - 0.5)
                )
            else:
                times2_shuffled = numpy.sort(
                    numpy.random.choice(times[ids != id1], len(times2), replace=False)
                )
            spk_train2_shuffled = numpy.histogram(times2_shuffled, bins=bins)[0]
            spk_train2_shuffled[spk_train2_shuffled > max_num_spikes] = max_num_spikes
            te2_H0[isurr] = te_estimator.estimate(spk_train2_shuffled, spk_train1)
        mu_H0, std_H0 = numpy.mean(te2_H0), numpy.amax([numpy.std(te2_H0), 1e-10])
        # pval_tmp = norm.cdf(te2, loc=mu_H0, scale=std_H0)
        # pval2 = numpy.amin([pval_tmp, 1. - pval_tmp])
        zval2 = numpy.abs(te2 - mu_H0) / std_H0
        return te1, zval1, te2, zval2, pair
