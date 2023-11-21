from spycon.spycon_inference import SpikeConnectivityInference
import elephant
import neo
import quantities
import numpy
from scipy.stats import poisson
import multiprocessing
from itertools import repeat
from tqdm import tqdm


class Smoothed_CCG(SpikeConnectivityInference):
    """Smoothed CCG method from

    English, Daniel Fine, et al. "Pyramidal cell-interneuron circuit architecture and dynamics in hippocampal networks." Neuron 96.2 (2017): 505-520.

    Args:
        params (dict): A dictionary containing the following parameters:

            - 'binsize': Time step (in seconds) that is used for time-discretization. (Default=.4e-3)
            - 'hf': Half fraction of the Gaussian kernel at zero lag (0<=hf<=1). (Default=0.6)
            - 'gauss_std': Standard deviation of Gaussian kernel (in seconds). (Default=0.01)
            - 'syn_window': Time window, which the CCG is check for peaks (in seconds). (Default=(.8e-3,5.8e-3))
            - 'ccg_tau': The maximum lag which the CCG is calculated for (in seconds). (Default=50e-3)
            - 'alpha': Value, that is used for thresholding p-values (0<alpha<1). (Default=.01)
            - 'deconv_ccg': Boolean value, that determines whether the CCG is deconvolved [Spivak, 22]. (Default=False)
    """

    def __init__(self, params: dict = {}):
        super().__init__(params)
        self.method = "sccg"
        self.default_params = {
            "binsize": 0.4e-3,
            "hf": 0.6,
            "gauss_std": 0.01,
            "syn_window": (0.8e-3, 5.8e-3),
            "ccg_tau": 50e-3,
            "alpha": 0.001,
            "deconv_ccg": False,
        }

    def _infer_connectivity(
        self, times: numpy.ndarray, ids: numpy.ndarray, pairs: numpy.ndarray
    ) -> tuple:
        """
        Smoothed CCG (Cross-Correlation Histogram) connectivity inference.

        This method performs connectivity inference using the CCG method. It calculates cross-correlation histograms between spike
        trains for the specified node pairs and uses statistics to determine whether there is a connection between nodes.

        Args:
            times (numpy.ndarray): Spike times in seconds.
            ids (numpy.ndarray): Unit ids corresponding to spike times.
            pairs (numpy.ndarray): Array of [pre, post] pair node IDs for which inference will be performed.

        Returns:
            tuple: A tuple containing the following elements:
                1) nodes (numpy.ndarray): Array of node labels [number_of_nodes].
                2) weights (numpy.ndarray): Array of graded strength of connections [number_of_edges].
                3) stats (numpy.ndarray): Array of connection statistics [number_of_edges, 3]. The first column represents the
                                        outgoing nodes, the second column represents the incoming nodes, and the third column
                                        contains the statistic used to decide whether it is an edge or not. A higher value
                                        indicates a more probable edge.
                4) threshold (float): A threshold value for considering an edge to be a connection if stats > threshold.

        Return Type:
            tuple

        """

        alpha = self.params.get("alpha", self.default_params["alpha"])
        nodes = numpy.unique(ids)
        weights = []
        stats = []
        num_connections_to_test = pairs.shape[0]
        conn_count = 0
        binsize = self.params.get("binsize", self.default_params["binsize"])
        syn_window = self.params.get("syn_window", self.default_params["syn_window"])
        bonf_corr = numpy.round((syn_window[1] - syn_window[0]) / binsize)
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
        # change pvalues such that predicted edges have positive value
        stats[:, 2] = -stats[:, 2]
        threshold = -numpy.log(alpha / bonf_corr)
        return nodes, weights, stats, threshold

    def _infer_connectivity_parallel(
        self,
        times: numpy.ndarray,
        ids: numpy.ndarray,
        pairs: numpy.ndarray,
        num_cores: int,
    ) -> tuple:
        """
        Parallel CCG (Cross-Correlation Histogram) connectivity inference.

        This method performs parallel connectivity inference using the CCG method. It calculates cross-correlation histograms
        between spike trains for the specified node pairs and uses statistics to determine whether there is a connection
        between nodes. Parallel processing is used to speed up the computation.

        Args:
            times (numpy.ndarray): Spike times in seconds.
            ids (numpy.ndarray): Unit ids corresponding to spike times.
            pairs (numpy.ndarray): Array of [pre, post] pair node IDs for which inference will be performed.
            num_cores (int): Number of CPU cores to be used for multiprocessing.

        Returns:
            tuple: A tuple containing the following elements:
                1) nodes (numpy.ndarray): Array of node labels [number_of_nodes].
                2) weights (numpy.ndarray): Array of graded strength of connections [number_of_edges].
                3) stats (numpy.ndarray): Array of connection statistics [number_of_edges, 3]. The first column represents the
                                        outgoing nodes, the second column represents the incoming nodes, and the third column
                                        contains the statistic used to decide whether it is an edge or not. A higher value
                                        indicates a more probable edge.
                4) threshold (float): A threshold value for considering an edge to be a connection if stats > threshold.

        Return Type:
            tuple

        """
        alpha = self.params.get("alpha", self.default_params["alpha"])
        nodes = numpy.unique(ids)
        weights = []
        stats = []
        binsize = self.params.get("binsize", self.default_params["binsize"])
        syn_window = self.params.get("syn_window", self.default_params["syn_window"])
        bonf_corr = numpy.round((syn_window[1] - syn_window[0]) / binsize)
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
            logpval1, weight1, logpval2, weight2, pair = result
            id1, id2 = pair
            weights.append(weight1)
            stats.append(numpy.array([id1, id2, logpval1]))
            if any(numpy.prod(pairs == [id2, id1], axis=1)):
                weights.append(weight2)
                stats.append(numpy.array([id2, id1, logpval2]))

        weights = numpy.vstack(weights)
        stats = numpy.vstack(stats)
        # change pvalues such that predicted edges have positive value
        stats[:, 2] = -stats[:, 2]
        threshold = -numpy.log(alpha / bonf_corr)
        return nodes, weights, stats, threshold

    def _test_connection_pair(
        self, times: numpy.ndarray, ids: numpy.ndarray, pair: tuple
    ) -> tuple:
        """
        Test connections in both directions.

        This method tests the connections between two nodes in both directions by analyzing spike times and unit IDs.

        Args:
            times (numpy.ndarray): Spike times in seconds.
            ids (numpy.ndarray): Unit IDs corresponding to spike times.
            pair (tuple): Node pair for which the connections are tested. The tuple should contain two elements representing
                        the IDs of the two nodes.

        Returns:
            tuple: A tuple containing the following elements:
                - pval_id1_to_id2 (float): p-value for the connection from node id1 to id2.
                - weight_id1_to_id2 (float): Weight or strength of the connection from node id1 to id2.
                - pval_id2_to_id1 (float): p-value for the connection from node id2 to id1.
                - weight_id2_to_id1 (float): Weight or strength of the connection from node id2 to id1.

        Return Type:
            tuple
        """
        id1, id2 = pair
        binsize = self.params.get("binsize", self.default_params["binsize"])
        kernel = self._partially_hollow_gauss_kernel()
        t_start, t_stop = numpy.amin(times) - binsize, numpy.amax(times) + binsize
        times1 = times[ids == id1]
        times2 = times[ids == id2]
        counts_ccg, counts_ccg_convolved, times_ccg = self._compute_ccg(
            times1, times2, kernel, t_start, t_stop
        )

        pval1, weight1 = self._test_connection(
            counts_ccg, counts_ccg_convolved, times_ccg, len(times1)
        )
        pval2, weight2 = self._test_connection(
            counts_ccg[::-1], counts_ccg_convolved[::-1], times_ccg, len(times2)
        )
        return pval1, weight1, pval2, weight2, pair

    def _test_connection(
        self,
        counts_ccg: numpy.ndarray,
        counts_ccg_convolved: numpy.ndarray,
        times_ccg: numpy.ndarray,
        num_presyn_spikes: int,
    ) -> tuple:
        """
        Test a connection using cross-correlograms (CCG) and convolved CCG.

        This method tests a connection by analyzing the provided CCG and convolved CCG data, as well as additional parameters.

        Args:
            counts_ccg (numpy.ndarray): The CCG (cross-correlogram) data.
            counts_ccg_convolved (numpy.ndarray): The convolved CCG data.
            times_ccg (numpy.ndarray): Spike times corresponding to the CCG data.
            num_presyn_spikes (int): The number of presynaptic spikes.

        Returns:
            tuple: A tuple containing the following elements:
                - pval_id1_to_id2 (float): p-value for the connection from node id1 to id2.
                - weight_id1_to_id2 (float): Weight or strength of the connection from node id1 to id2.
                - pval_id2_to_id1 (float): p-value for the connection from node id2 to id1.
                - weight_id2_to_id1 (float): Weight or strength of the connection from node id2 to id1.

        Return Type:
            tuple

        """

        syn_window = self.params.get("syn_window", self.default_params["syn_window"])
        pos_time_points = numpy.logical_and(
            times_ccg >= syn_window[0], times_ccg < syn_window[1]
        )
        lmbda_slow = numpy.amax(counts_ccg_convolved[pos_time_points])
        num_spikes_max = numpy.amax(counts_ccg[pos_time_points])
        num_spikes_min = numpy.amin(counts_ccg[pos_time_points])
        #
        # pfast_max = 1. - poisson.cdf(num_spikes_max - 1., mu=lmbda_slow) - .5 * poisson.pmf(num_spikes_max, mu=lmbda_slow)
        # pfast_min = 1. - poisson.cdf(num_spikes_min - 1., mu=lmbda_slow) - .5 * poisson.pmf(num_spikes_min, mu=lmbda_slow)
        logpfast_max = numpy.logaddexp(
            poisson.logcdf(num_spikes_max - 1, mu=lmbda_slow),
            poisson.logpmf(num_spikes_max, mu=lmbda_slow) - numpy.log(2),
        )
        if numpy.abs(logpfast_max) > 1e-4:
            logpfast_max = numpy.log(-numpy.expm1(logpfast_max))
        else:
            if logpfast_max > -1e-20:
                logpfast_max = -1e-20
            logpfast_max = numpy.log(-logpfast_max)
        logpfast_min = numpy.logaddexp(
            poisson.logcdf(num_spikes_min - 1, mu=lmbda_slow),
            poisson.logpmf(num_spikes_min, mu=lmbda_slow) - numpy.log(2),
        )
        # if numpy.abs(logpfast_max) > 1e-4:
        #    logpfast_max = numpy.log(-numpy.expm1(x))
        # else:
        #    logpfast_max = numpy.log(-x)
        # num_pos_time_points = numpy.sum(pos_time_points)
        # anticausal_idx = numpy.where(times_ccg <= 0)[0][-num_pos_time_points:]
        # lmbda_anticausal = numpy.amax(counts_ccg_convolved[anticausal_idx])
        # pcausal = 1. - poisson.cdf(num_spikes - 1, mu=lmbda_anticausal) - .5 * poisson.pmf(num_spikes,
        #                                                                                    mu=lmbda_anticausal)
        # pval = max(pfast, pcausal, 0)
        pval_max = logpfast_max  # max(pfast_max,0)
        pval_min = logpfast_min  # max(1. - pfast_min,0)
        pval = numpy.amin([pval_max, pval_min])
        weight = (
            numpy.sum(
                counts_ccg[pos_time_points] - counts_ccg_convolved[pos_time_points]
            )
            / num_presyn_spikes
        )
        return pval, weight

    def _partially_hollow_gauss_kernel(self) -> numpy.ndarray:
        """
        Compute the partially hollow Gaussian kernel.

        Returns:
            numpy.ndarray: 1D array with the kernel.
        """

        binsize = self.params.get("binsize", self.default_params["binsize"])
        hollow_fraction = self.params.get("hf", self.default_params["hf"])
        std = self.params.get("gauss_std", self.default_params["gauss_std"])
        kernel_tau = 5.0 * std
        delta = numpy.arange(-kernel_tau, kernel_tau + binsize, binsize)
        kernel = numpy.exp(-0.5 * (delta / std) ** 2.0)
        zero_idx = numpy.where(numpy.isclose(delta, 0))[0]
        kernel[zero_idx] = hollow_fraction * kernel[zero_idx]
        kernel /= numpy.sum(kernel)
        return kernel

    def _compute_ccg(
        self,
        times1: numpy.ndarray,
        times2: numpy.ndarray,
        kernel: numpy.ndarray,
        t_start: float,
        t_stop: float,
    ):
        """
        Compute the cross-correlogram (CCG) and convolved CCG between two spike trains.

        Parameters:
            times1 (numpy.ndarray): Spike times of neuron 1 (in seconds).
            times2 (numpy.ndarray): Spike times of neuron 2 (in seconds).
            kernel (numpy.ndarray): Kernel used for smoothing the CCG.
            t_start (float): Starting time of recording (in seconds).
            t_stop (float): Stopping time of recording (in seconds).

        Returns:
            tuple: A tuple containing three elements:
                - CCG (numpy.ndarray): The cross-correlogram.
                - Convolved CCG (numpy.ndarray): The convolved cross-correlogram.
                - Times of CCG (numpy.ndarray): The time values associated with the CCG.

        Note:
            The CCG represents the cross-correlation between spike times of neuron 1 and neuron 2.
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
        ccg_bins = int(numpy.ceil(ccg_tau / binsize))
        ccg_bins_eff = numpy.amax([int(numpy.ceil(len(kernel) / 2)), ccg_bins])

        kernel_hw = len(kernel) // 2
        ccg_full = elephant.spike_train_correlation.cross_correlation_histogram(
            st1,
            st2,
            window=[-ccg_bins_eff - kernel_hw, ccg_bins_eff + kernel_hw],
            border_correction=False,
            binary=False,
            kernel=None,
            method="memory",
        )
        ccg_times = ccg_full[0].times.magnitude[
            ccg_bins_eff
            - ccg_bins
            + kernel_hw : ccg_bins_eff
            + ccg_bins
            + kernel_hw
            + 1
        ]

        ccg_full = ccg_full[0][:, 0].magnitude.T[0]
        deconv_ccg = self.params.get("deconv_ccg", self.default_params["deconv_ccg"])

        if deconv_ccg:
            nspks1, nspks2 = len(times1), len(times2)
            acg1 = elephant.spike_train_correlation.cross_correlation_histogram(
                st1,
                st1,
                window=[-ccg_bins_eff - kernel_hw, ccg_bins_eff + kernel_hw],
                border_correction=False,
                binary=False,
                kernel=None,
                method="memory",
            )

            acg1 = acg1[0][:, 0].magnitude.T[0]
            m = len(acg1)
            hw = (m - 1) // 2
            acg1 = acg1 - numpy.sum(acg1) / m
            acg1 /= nspks1
            hidx = list(range(0, m + 0, hw * (hw + 2)))
            acg1[hw] = 1 - numpy.sum(acg1[hidx])

            acg2 = elephant.spike_train_correlation.cross_correlation_histogram(
                st2,
                st2,
                window=[-ccg_bins_eff - kernel_hw, ccg_bins_eff + kernel_hw],
                border_correction=False,
                binary=False,
                kernel=None,
                method="memory",
            )
            acg2 = acg2[0][:, 0].magnitude.T[0]
            m = len(acg2)
            hw = (m - 1) // 2
            acg2 = acg2 - numpy.sum(acg2) / m
            acg2 /= nspks2
            hidx = list(range(0, m + 0, hw * (hw + 2)))
            acg2[hw] = 1 - numpy.sum(acg2[hidx])

            den = numpy.fft.fft(acg1, m) * numpy.fft.fft(acg2, m)
            num = numpy.fft.fft(ccg_full, m)
            dccg_fft = num
            dccg_fft[1:] /= den[1:]
            dccg = numpy.real(numpy.fft.ifft(dccg_fft, m))
            dccg[dccg < 0] = 0.0
            ccg_convolved = numpy.convolve(dccg, kernel, mode="valid")[1:-1]
            ccg = dccg[
                ccg_bins_eff
                - ccg_bins
                + kernel_hw
                + 1 : ccg_bins_eff
                + ccg_bins
                + kernel_hw
                + 2
            ]
        else:
            ccg_convolved = numpy.convolve(ccg_full, kernel, mode="valid")[1:-1]
            ccg = ccg_full[
                ccg_bins_eff
                - ccg_bins
                + kernel_hw : ccg_bins_eff
                + ccg_bins
                + kernel_hw
                + 1
            ]

        return ccg, ccg_convolved, ccg_times
