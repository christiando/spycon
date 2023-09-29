import numpy
from sklearn import metrics
from spycon.spycon_inference import SpikeConnectivityInference
from spycon.spycon_result import SpikeConnectivityResult
import pandas
import networkx as nx
import itertools
from matplotlib import pyplot
import pickle


class ConnectivityTest(object):
    """
    Object that helps to easily test a spike connectivity inference algorithm.

    Args:
        name (str): Name of the test.
        times (numpy.ndarray): Spike times (in seconds).
        ids (numpy.ndarray): Node IDs corresponding to spike times.
        nodes (numpy.ndarray): List of node IDs.
        marked_edges (numpy.ndarray): A 2D array with shape [num_of_edges, 3], containing ground truth edges.
            - The first column represents outgoing nodes.
            - The second column represents incoming nodes.
            - The third column is 1 for an existing edge and 0 otherwise.
            If an edge is not indicated, it is assumed that ground truth is unknown.
        params (dict): Parameters for the test run. If unspecified, the whole recording is used. Possible entries are:
            - 'subset': Unit IDs of the subnetwork that should be analyzed.
            - 'T_start': From which time recording is considered.
            - 'T_stop': Up to which time recording is considered.
        info (dict, optional): Dictionary with additional information. Default is an empty dictionary ({}).
    """

    def __init__(
        self,
        name: str,
        times: numpy.ndarray,
        ids: numpy.ndarray,
        nodes: numpy.ndarray,
        marked_edges: numpy.ndarray,
        params: dict = {},
        info: dict = {},
    ):
        self.name = name
        self.times = times
        self.ids = ids
        self.nodes = nodes
        self.marked_edges = marked_edges
        self.params = params
        self.info = info

    def run_test(
        self,
        coninf_method: SpikeConnectivityInference,
        only_metrics: bool = True,
        parallel: bool = False,
        **kwargs
    ) -> pandas.DataFrame:
        """
        Run the test for a specific spike connectivity algorithm.

        Args:
            coninf_method (SpikeConnectivityInference): Inference method.
            only_metrics (bool, optional): If True, only metrics are returned; otherwise, the result object is also returned. Default is True.
            parallel (bool, optional): Whether the parallel version is used, if implemented. Default is False.

        Returns:
            pandas.Dataframe or (SpikeConnectivityResult, pandas.Dataframe): A DataFrame with all the metrics, and if indicated
            (when only_metrics is False), also the result object.
        """
        T_start = self.params.get("T_start", numpy.amin(self.times))
        T_stop = self.params.get("T_stop", numpy.amax(self.times))
        unique_ids = numpy.unique(self.ids)
        N = self.params.get("N", len(unique_ids))
        seed = self.params.get("seed", None)
        if seed is not None:
            numpy.random.seed(seed)
        numpy.random.shuffle(unique_ids)
        ids_to_consider = unique_ids[:N]
        times_to_consider = numpy.where(
            numpy.logical_and(
                numpy.logical_and(self.times >= T_start, self.times <= T_stop),
                numpy.isin(self.ids, ids_to_consider),
            )
        )[0]
        times, ids = self.times[times_to_consider], self.ids[times_to_consider]
        spycon_result = coninf_method.infer_connectivity(
            times, ids, parallel=parallel, **kwargs
        )
        if only_metrics:
            return self.eval_performance(spycon_result)
        else:
            return spycon_result, self.eval_performance(spycon_result)

    def eval_performance(
        self, spycon_result: SpikeConnectivityResult
    ) -> pandas.DataFrame:
        """
        Calculate the metrics for a given result.

        Metrics calculated:
            - 'runtime': The algorithm runtime (in seconds).
            - 'fpr', 'tpr', 'thresholds': Data for the ROC curve (false positive, true positive rate, with corresponding thresholds).
            - 'auc': Area under curve.
            - 'f1': F1-score.
            - 'precision': Precision score.
            - 'accuracy': Accuracy.

        Args:
            spycon_result (SpikeConnectivityResult): The result object returned by the inference methods.

        Returns:
            pandas.Dataframe: A DataFrame with all the metrics.
        """

        metrics_dict = {}
        metrics_dict["runtime"] = spycon_result.runtime
        true_con_mat = self.create_connectivity_matrix(spycon_result.nodes)
        score_con_mat = spycon_result.create_connectivity_matrix(conn_type="stats")
        gt_edge_idx = numpy.where(numpy.logical_not(numpy.isnan(true_con_mat)))
        y_true = numpy.zeros(len(gt_edge_idx[0]))
        y_true[numpy.nonzero(true_con_mat[gt_edge_idx[0], gt_edge_idx[1]])] = 1
        y_score = score_con_mat[gt_edge_idx[0], gt_edge_idx[1]]
        (
            metrics_dict["fpr"],
            metrics_dict["tpr"],
            metrics_dict["thresholds"],
        ) = metrics.roc_curve(y_true, y_score)
        metrics_dict["auc"] = metrics.roc_auc_score(y_true, y_score)
        metrics_dict["aps"] = metrics.average_precision_score(y_true, y_score)
        (
            metrics_dict["prc_precision"],
            metrics_dict["prc_recall"],
            metrics_dict["prc_thresholds"],
        ) = metrics.precision_recall_curve(y_true, y_score)
        pred_con_mat = spycon_result.create_connectivity_matrix(conn_type="binary")
        y_pred = pred_con_mat[gt_edge_idx[0], gt_edge_idx[1]]
        metrics_dict["f1"] = metrics.f1_score(y_true, y_pred)
        metrics_dict["precision"] = metrics.precision_score(
            y_true, y_pred, zero_division=0
        )
        metrics_dict["recall"] = metrics.recall_score(y_true, y_pred)
        metrics_dict["accuracy"] = metrics.accuracy_score(y_true, y_pred)
        metrics_dict["mcc"] = metrics.matthews_corrcoef(y_true, y_pred)
        metrics_df = pandas.DataFrame(
            [metrics_dict.values()], index=[0], columns=metrics_dict.keys()
        )
        return metrics_df

    def create_connectivity_matrix(self, nodes: numpy.ndarray = None) -> numpy.ndarray:
        """
        Create a binary connectivity matrix. Non-observed edges are represented as NaN.

        Returns:
            numpy.ndarray: The connectivity matrix with NaN values for non-observed edges.
        """
        if nodes is None:
            nodes = self.nodes
        pairs = numpy.array(list(itertools.combinations(nodes, 2)))
        pairs = numpy.vstack([pairs, pairs[:, ::-1]])
        con_matrix = numpy.empty((len(nodes), len(nodes)))
        con_matrix[:, :] = numpy.nan
        edges_to_consider = numpy.where(
            numpy.logical_and(
                numpy.isin(self.marked_edges[:, 0], nodes),
                numpy.isin(self.marked_edges[:, 1], nodes),
            )
        )[0]
        idx1 = numpy.searchsorted(nodes, self.marked_edges[edges_to_consider, 0])
        idx2 = numpy.searchsorted(nodes, self.marked_edges[edges_to_consider, 1])
        con_matrix[idx1, idx2] = self.marked_edges[edges_to_consider, 2]
        return con_matrix

    def create_nx_graph(self) -> nx.DiGraph:
        """
        Create a NetworkX graph.

        Returns:
            nx.DiGraph: A directed NetworkX graph.
        """
        nxgraph = nx.DiGraph()
        nxgraph.add_nodes_from(self.nodes)
        conns = numpy.where(
            numpy.logical_and(
                self.marked_edges[:, 2] != 0,
                numpy.logical_not(numpy.isnan(self.marked_edges[:, 2])),
            )
        )[0]
        nxgraph.add_edges_from(self.marked_edges[conns, :2])
        return nxgraph

    def draw_graph(
        self,
        ax: pyplot.Axes = None,
    ):
        """
        Draw a NetworkX graph.

        Args:
            ax (pyplot.Axes, optional): Axis where the graph should be drawn. Default is None.

        Returns:
            nx.DiGraph: A directed NetworkX graph.
        """
        graph = self.create_nx_graph()
        nx.draw_circular(graph, ax=ax, with_labels=True, node_size=500, node_color="C1")
        return graph

    def save(self, path=""):
        """
        Save the test object.

        Args:
            path (str, optional): Path to the saving location. Default is an empty string ('').

        """

        numpy.savez(
            path + self.name + ".npz",
            times=self.times,
            ids=self.ids,
            nodes=self.nodes,
            marked_edges=self.marked_edges,
        )
        if len(self.info) > 0:
            with open(path + self.name + ".pkl", "wb") as handle:
                pickle.dump(self.info, handle)


def load_test(name: str, path: str = "", params: dict = {}) -> ConnectivityTest:
    """
    Loads a test object.

    Args:
        name (str): Name of the test.
        path (str, optional): Path to the saving location. Default is an empty string ('').
        params (dict, optional): Parameters for the test such as 'seed', 'T_start', 'T_stop', and 'N'. Default is an empty dictionary ({}) .

    Returns:
        ConnectivityTest: The loaded test object.
    """
    data = numpy.load(path + name + ".npz", allow_pickle=True)
    try:
        with open(path + name + ".pkl", "rb") as handle:
            info = pickle.load(handle)
    except FileNotFoundError:
        info = {}
    subset = params.get("subset", None)
    if subset is None:
        con_test = ConnectivityTest(
            name,
            data["times"],
            data["ids"],
            data["nodes"],
            data["marked_edges"],
            params=params,
            info=info,
        )
    else:
        times, ids, nodes, marked_edges = (
            data["times"],
            data["ids"],
            data["nodes"],
            data["marked_edges"],
        )
        valid_spikes = numpy.where(numpy.isin(ids, subset))[0]
        valid_nodes = numpy.where(numpy.isin(nodes, subset))[0]
        valid_edges = numpy.where(
            numpy.logical_and(
                numpy.isin(marked_edges[:, 0], subset),
                numpy.isin(marked_edges[:, 1], subset),
            )
        )[0]
        con_test = ConnectivityTest(
            name,
            times[valid_spikes],
            ids[valid_spikes],
            nodes[valid_nodes],
            marked_edges[valid_edges],
            params=params,
            info=info,
        )
    return con_test
