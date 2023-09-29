from spycon_inference import SpikeConnectivityInference
import elephant
import neo
import quantities
import numpy
from scipy.stats import poisson


class MyConnectivityMethod(SpikeConnectivityInference):
    def __init__(self, params: dict = {}):
        """
        Implements my connectivity inference method.

        Parameters:
            params (dict): A dictionary containing configuration parameters for the inference method.

        Returns:
            tuple: A tuple containing the results of the inference method.

        Note:
            Provide a detailed description of the XXX method and the specific parameters expected in the 'params' dictionary.
        """

        super().__init__(params)
        self.method = "method name"
        # In this dictionary specify all default values for the algorithm, ideally taken from the corresponding publication.
        self.default_params = {}

    def _infer_connectivity(
        self, times: numpy.ndarray, ids: numpy.ndarray, pairs: numpy.ndarray
    ) -> tuple:
        """The working horse.

        Args:
            times (numpy.ndarray): Spike times in seconds.
            ids (numpy.ndarray): Unit ids corresponding to spike times.
            pairs (numpy.ndarray): Array of [pre, post] pair node IDs, for which inference will be done.

        Returns:
            tuple: A tuple containing four arrays:
                - nodes (numpy.ndarray): [number_of_nodes], with the node labels.
                - edges (numpy.ndarray): [number_of_edges, 2], where the first column is the outgoing node, and the second is the incoming node.
                - weights (numpy.ndarray): [number_of_edges], with graded strength of connection.
                - stats (numpy.ndarray): [number_of_edges, 3], containing a fully connected graph. The first column represents outgoing nodes, the second represents incoming nodes, and the third row contains the statistic used to decide whether it is an edge or not. A higher value indicates a more probable edge.
        """

        # Here has to happen the magic!
        # If you access a parameter from the params-dict do it in the follow form:
        #
        # >>> self.params.get('param_name', self.default_params['param_name'])
        #
        # where the default value is set ideally to the value recommended in the publication.

        return nodes, edges, weights, stats

    def _infer_connectivity_parallel(
        self, times: numpy.ndarray, ids: numpy.ndarray, pairs: numpy.ndarray
    ) -> tuple:
        """The working horse. Parallel version.

        Args:
            times (numpy.ndarray): Spike times in seconds.
            ids (numpy.ndarray): Unit ids corresponding to spike times.
            pairs (numpy.ndarray): Array of [pre, post] pair node IDs, for which inference will be done.

        Returns:
            tuple: A tuple containing four arrays:
                - nodes (numpy.ndarray): [number_of_nodes], with the node labels.
                - edges (numpy.ndarray): [number_of_edges, 2], where the first column is the outgoing node, and the second is the incoming node.
                - weights (numpy.ndarray): [number_of_edges], with graded strength of connection.
                - stats (numpy.ndarray): [number_of_edges, 3], containing a fully connected graph. The first column represents outgoing nodes, the second represents incoming nodes, and the third row contains the statistic used to decide whether it is an edge or not. A higher value indicates a more probable edge.
        """

        # Here can be implemented a parallel version of your algorithm. If you can't/don't want to do it, just delete it.

        return nodes, edges, weights, stats
