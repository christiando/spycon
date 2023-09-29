import numpy
import time
import warnings
from spycon.spycon_result import SpikeConnectivityResult
from itertools import combinations


class SpikeConnectivityInference(object):
    """
    Base class for spike connectivity inference. It will infer a fully connected graph.

    Args:
        params (dict): Dictionary with necessary parameters as keywords. Keys should be strings.

    """

    def __init__(self, params: dict = {str: object}):

        self.method = "full graph"
        self.params = params
        self.default_params = {}

    def infer_connectivity(
        self,
        times: numpy.ndarray,
        ids: numpy.ndarray,
        pairs: numpy.ndarray = None,
        parallel: bool = False,
        **kwargs
    ) -> SpikeConnectivityResult:
        """
        Full connectivity inference.

        Args:
            times (numpy.ndarray): Spike times in seconds.
            ids (numpy.ndarray): Unit IDs corresponding to spike times.
            pairs (numpy.ndarray): Array of [pre, post] pair node IDs, for which inference will be done.
            parallel (bool): Whether the parallel version is used, if implemented.

        Returns:
            SpikeConnectivityResult: Connectivity result object.
        """
        if pairs is None:
            nodes = numpy.unique(ids)
            pairs = numpy.array(list(combinations(nodes, 2)))
            pairs = numpy.vstack([pairs, pairs[:, ::-1]])

        remove_bursts = self.params.get("remove_bursts", False)
        if remove_bursts:
            ISI_N_burst = self.params.get("ISI_N_burst", 0.1)
            N_burst = self.params.get("N_burst", int(0.5 * len(numpy.unique(ids))))
            times, ids = self.burst_removal(times, ids, N=N_burst, ISI_N=ISI_N_burst)
        tstart = time.perf_counter()
        if parallel:
            try:
                num_cores = self.params.get("num_cores", 4)
                nodes, weights, stats, threshold = self._infer_connectivity_parallel(
                    times, ids, pairs, num_cores, **kwargs
                )
            except NotImplementedError:
                warnings.warn(
                    "Parallel computation not implemented. Fall back to sequential computation.",
                    category=RuntimeWarning,
                )
                parallel = False
        if not parallel:
            nodes, weights, stats, threshold = self._infer_connectivity(
                times, ids, pairs, **kwargs
            )
        runtime = time.perf_counter() - tstart
        spycon_result = SpikeConnectivityResult(
            self.method, self.params, nodes, weights, stats, threshold, runtime
        )
        return spycon_result

    def _infer_connectivity(
        self, times: numpy.ndarray, ids: numpy.ndarray, pairs: numpy.ndarray, **kwargs
    ) -> tuple:
        """
        Full connectivity inference.

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
                4) threshold (float): A float value representing the threshold used to consider an edge as a connection,
                based on the stats.
        """
        nodes = numpy.unique(ids)
        num_nodes = len(nodes)
        mesh = numpy.meshgrid(nodes, nodes)
        edges = numpy.empty((num_nodes**2, 2), dtype=int)
        edges[:, 0] = mesh[0].flatten()
        edges[:, 1] = mesh[1].flatten()
        weights = numpy.ones(num_nodes)
        stats = numpy.empty((num_nodes**2, 3))
        stats[:, 0] = mesh[0].flatten()
        stats[:, 1] = mesh[1].flatten()
        stats[:, 2] = 1.0
        threshold = 0.0
        return nodes, edges, weights, threshold

    def _infer_connectivity_parallel(
        self,
        times: numpy.ndarray,
        ids: numpy.ndarray,
        pairs: numpy.ndarray,
        num_cores: int,
        **kwargs
    ) -> tuple:
        raise NotImplementedError("Parallel computing not implemented.")

    def burst_detection(self, T, C, N=None, ISI_N=0.1):
        """Burst detection algorithm. Credits to Adrian Bertagnoli.

        [Citation needed]
        """
        # Credit to adrian
        if N is None:
            rec_length = numpy.amax(T) - numpy.amin(T)
            N = int(len(T) / rec_length)
        # Credit to adrian
        min_condition = numpy.zeros([len(T)]) + float("inf")
        for j in range(N):
            dT_tmp = numpy.ndarray.flatten(
                T[N - 1 + j : -1 - (N - 2) + j or None]
                - T[j : -1 - ((N - 1) * 2 - 1) + j]
            )
            min_idx = numpy.where(min_condition[N - 1 : -N + 1] > dT_tmp)[0]
            min_condition[N - 1 : -N + 1][min_idx] = dT_tmp[
                min_idx
            ]  # Initialize to zero
        Criteria = min_condition <= ISI_N
        # Spike passes condition if it is
        # included in a set of N spikes
        # with ISI_N <= threshold.

        # Assign burst numbers to each spike
        SpikeBurstNumber = numpy.zeros(len(T)) - 1
        # Initialize to '-1'
        INBURST = 0
        # In a burst (1) or not (0)
        NUM_ = 0
        # Burst Number iterator
        NUMBER = -1
        # Burst Number assigned
        BL = 0
        # Burst Length

        for i in range(N - 1, len(T)):

            if INBURST == 0:  # Was not in burst.
                if Criteria[i]:  # Criteria met, now in new burst.
                    INBURST = 1
                    # Update.
                    NUM_ = NUM_ + 1
                    NUMBER = NUM_
                    BL = 1
                else:  # Still not in burst, continue.
                    continue

            else:  # Was in burst.

                if not Criteria[i]:  # Criteria no longer met.
                    INBURST = 0
                    # Update.
                    if BL < N:  # Erase if not big enough.
                        SpikeBurstNumber[numpy.where(SpikeBurstNumber == NUMBER)] = -1
                        NUM_ = NUM_ - 1

                    NUMBER = -1
                elif ((T[i - 1] - T[i - N]) > ISI_N) and (BL >= N):

                    # This conditional statement is necessary to split apart
                    # consecutive bursts that are not interspersed by a tonic spike
                    # (i.e. Criteria == 0). Occasionally in this case, the second
                    # burst has fewer than 'N' spikes and is therefore deleted in
                    # the above conditional statement (i.e. 'if BL<N').
                    #
                    # Skip this if at the start of a new burst (i.e. 'BL>=N'
                    # requirement).
                    #

                    NUM_ = NUM_ + 1
                    # New burst, update number.
                    NUMBER = NUM_
                    BL = 1
                    # Reset burst length.
                else:  # Criteria still met.

                    BL = BL + 1
                    # Update burst length.

            SpikeBurstNumber[i] = NUMBER
            # Assign a burst number to
            # each spike.

        # Assign Burst information
        MaxBurstNumber = max(SpikeBurstNumber)
        T_start = numpy.zeros([int(MaxBurstNumber)])
        T_end = numpy.zeros([int(MaxBurstNumber)])
        S = numpy.zeros([int(MaxBurstNumber)])
        BC = numpy.zeros([int(MaxBurstNumber)])
        # Burst.T_start = zeros(1,MaxBurstNumber); # Burst start time [sec]
        # Burst.T_end = zeros(1,MaxBurstNumber); # Burst end time [sec]
        # Burst.S = zeros(1,MaxBurstNumber); #Size (total spikes)
        # Burst.C = zeros(1,MaxBurstNumber); # Size (total channels)
        for i in range(int(MaxBurstNumber)):
            ID = numpy.array([numpy.where(SpikeBurstNumber == i + 1)]).flatten()
            startID = int(ID[0])
            endID = int(ID[-1])
            T_start[i] = T[startID]
            T_end[i] = T[endID]
            S[i] = len(ID)
            BC[i] = len(numpy.unique(C[ID]))
        return SpikeBurstNumber, T_start, T_end, S, BC

    def burst_removal(self, T, C, N=None, ISI_N=0.1):
        # TODO: This needs to be cleaned!
        try:
            burst_result = self.burst_detection(T, C, N, ISI_N)
        except ValueError:
            raise RuntimeWarning(
                "No bursts detected with current parameter configuration."
            )
            return T, C
        SpikeBurstNumber, T_start, T_end, S, BC = burst_result
        num_bursts = len(T_start)
        T_new, C_new = [], []
        cum_burst_duration = 0
        T_new.append(T[T < T_start[0]])
        C_new.append(C[T < T_start[0]])
        for iburst in range(num_bursts):
            burst_start, burst_stop = T_start[iburst], T_end[iburst]
            burst_duration = burst_stop - burst_start
            cum_burst_duration += burst_duration
            if iburst < (num_bursts - 1):
                spks2append = numpy.where(
                    numpy.logical_and(T > T_end[iburst], T < T_start[iburst + 1])
                )[0]
            else:
                spks2append = numpy.where(T > T_end[iburst])[0]
            spks = T[spks2append]
            T_new.append(spks - cum_burst_duration)
            C_new.append(C[spks2append])
        T_new, C_new = numpy.concatenate(T_new), numpy.concatenate(C_new)
        return T_new, C_new
