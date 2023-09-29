import numpy
from sklearn import metrics
from spycon_inference import SpikeConnectivityInference
from spycon_result import SpikeConnectivityResult
import pandas
import networkx as nx
import itertools
from matplotlib import pyplot
import pickle


class ConnectivityTest(object):

    def __init__(self, name: str, times: numpy.ndarray, ids: numpy.ndarray, nodes: numpy.ndarray,
                 marked_edges: numpy.ndarray, params: {str: object} = {}, info: {str:object} = {}):
        """ Object, that helps to easily test a spycon_inference algorithm.

        :param name: Name of test
        :type name: str
        :param times: Spike times (in seconds).
        :type times: numpy.ndarray
        :param ids: Node ids corresponding to spike times.
        :type ids: numpy.ndarray
        :param nodes: List of node ids.
        :type nodes: numpy.ndarray
        :param marked_edges: [num_of_edges, 3], containing ground truth edges. First and second column are outgoing and
                                incoming nodes, respectively. The third column is 1 for an existing edge and 0
                                otherwise. If an edge is not indicated, it is assumed, that ground truth is unknown.
        :type marked_edges: numpy.ndarray
        :param params: Parameters for test run. If unspecified whole recording is used. Possible entries are
                        'subset': Unit ids of subnetwork that should be analyzed.
                        'T_start': From which time recording is considered.
                        'T_stop': Up to which time recording is considered.
        :type params: dict
        :param info: Dictionary with additional information. (Default={})
        :type info: dict
        """
        self.name = name
        self.times = times
        self.ids = ids
        self.nodes = nodes
        self.marked_edges = marked_edges
        self.params = params
        self.info = info

    def run_test(self, coninf_method: SpikeConnectivityInference, only_metrics: bool=True, 
                 parallel: bool=False, **kwargs) -> pandas.DataFrame:
        """ Runs the test for an specific spike connectivity algorithm.

        :param coninf_method: Inference method.
        :type coninf_method: SpikeConnectivityInference
        :param only_metrics: If true, only metrics are returned, otherwise also the result object. (Default=True)
        :type only_metrics: bool
        :param parallel: Whether parallel version is used, if implemented
        :type parallel: bool
        :return: Dataframe with all the metrics, and if indicated also the result object.
        :rtype: pandas.Dataframe or (SpikeConnectivityResult, pandas.Dataframe)
        """
        T_start = self.params.get('T_start', numpy.amin(self.times))
        T_stop = self.params.get('T_stop', numpy.amax(self.times))
        unique_ids = numpy.unique(self.ids)
        N = self.params.get('N', len(unique_ids))
        seed = self.params.get('seed', None)
        if seed is not None:
            numpy.random.seed(seed)
        numpy.random.shuffle(unique_ids)
        ids_to_consider = unique_ids[:N]
        times_to_consider = numpy.where(numpy.logical_and(
            numpy.logical_and(self.times >= T_start, self.times <= T_stop),
            numpy.isin(self.ids, ids_to_consider)))[0]
        times, ids = self.times[times_to_consider], self.ids[times_to_consider]
        spycon_result = coninf_method.infer_connectivity(times, ids, parallel=parallel, **kwargs)
        if only_metrics:
            return self.eval_performance(spycon_result)
        else:
            return spycon_result, self.eval_performance(spycon_result)

    def eval_performance(self, spycon_result: SpikeConnectivityResult) -> pandas.DataFrame:
        """ Calculates the metrics for a given result. At the moment that is

            'runtime': The algorithm runtime (in seconds).
            'fpr', 'tpr', 'thresholds': Data for the ROC curve (false positive, true positive rate, with corresponding
                                        thresholds)
            'auc': Area under curve.
            'f1': F1-score.
            'precision': precision score.
            'accuracy': accuracy.

        :param spycon_result: The result object returned by the inference methods.
        :type spycon_result: SpikeConnectivityResult
        :return: Dataframe with all the metrics.
        :rtype: pandas.Dataframe
        """
        metrics_dict = {}
        metrics_dict['runtime'] = spycon_result.runtime
        true_con_mat = self.create_connectivity_matrix(spycon_result.nodes)
        score_con_mat = spycon_result.create_connectivity_matrix(conn_type='stats')
        gt_edge_idx = numpy.where(numpy.logical_not(numpy.isnan(true_con_mat)))
        y_true = numpy.zeros(len(gt_edge_idx[0]))
        y_true[numpy.nonzero(true_con_mat[gt_edge_idx[0], gt_edge_idx[1]])] = 1
        y_score = score_con_mat[gt_edge_idx[0], gt_edge_idx[1]]
        metrics_dict['fpr'], metrics_dict['tpr'], metrics_dict['thresholds'] = metrics.roc_curve(y_true, y_score)
        metrics_dict['auc'] = metrics.roc_auc_score(y_true, y_score)
        metrics_dict['aps'] = metrics.average_precision_score(y_true, y_score)
        metrics_dict['prc_precision'], metrics_dict['prc_recall'], metrics_dict['prc_thresholds'] = metrics.precision_recall_curve(y_true, y_score)
        pred_con_mat = spycon_result.create_connectivity_matrix(conn_type='binary')
        y_pred = pred_con_mat[gt_edge_idx[0], gt_edge_idx[1]]
        metrics_dict['f1'] = metrics.f1_score(y_true, y_pred)
        metrics_dict['precision'] = metrics.precision_score(y_true, y_pred)
        metrics_dict['recall'] = metrics.recall_score(y_true, y_pred)
        metrics_dict['accuracy'] = metrics.accuracy_score(y_true, y_pred)
        metrics_dict['mcc'] = metrics.matthews_corrcoef(y_true, y_pred)
        metrics_df = pandas.DataFrame([metrics_dict.values()], index=[0], columns=metrics_dict.keys())
        return metrics_df
    
    def create_connectivity_matrix(self, nodes: numpy.ndarray=None) -> numpy.ndarray:
        """ Creates a binary connectivity matrix. Non-observed edges are nan.

        :return: Connectivity matrix.
        :rtype: numpy.ndarray
        """
        if nodes is None:
            nodes = self.nodes
        pairs = numpy.array(list(itertools.combinations(nodes, 2)))
        pairs = numpy.vstack([pairs, pairs[:, ::-1]])
        con_matrix = numpy.empty((len(nodes), len(nodes)))
        con_matrix[:,:] = numpy.nan
        edges_to_consider = numpy.where(numpy.logical_and(numpy.isin(self.marked_edges[:,0], nodes),
                                                          numpy.isin(self.marked_edges[:,1], nodes)))[0]
        idx1 = numpy.searchsorted(nodes, self.marked_edges[edges_to_consider,0])
        idx2 = numpy.searchsorted(nodes, self.marked_edges[edges_to_consider,1])
        con_matrix[idx1, idx2] = self.marked_edges[edges_to_consider,2]
        return con_matrix
    
    def create_nx_graph(self) -> nx.DiGraph:
        """ Creates networkx graph

        :return: Directed networkx graph.
        :rtype: nx.DiGraph
        """
        # ToDo: Figure out how to treat unobserved!
        nxgraph = nx.DiGraph()
        nxgraph.add_nodes_from(self.nodes)
        conns = numpy.where(numpy.logical_and(self.marked_edges[:,2] != 0, numpy.logical_not(numpy.isnan(self.marked_edges[:,2]))))[0]
        nxgraph.add_edges_from(self.marked_edges[conns,:2])
        return nxgraph
    
    def draw_graph(self, ax: pyplot.Axes = None,):
        """ Draws networkx graph.

        :param ax: Axis, where graph should be drawn. (Default=None)
        :type ax: pyplot.Axes
        :return: Directed networkx graph.
        :rtype: nx.DiGraph
        """
        graph = self.create_nx_graph()
        nx.draw_circular(graph, ax=ax, with_labels=True, node_size=500, node_color='C1')
        return graph

    def save(self, path=''):
        """ Saves test object.

        :param path: Path to saving location. (Default='')
        :type path: str
        """
        numpy.savez(path + self.name + '.npz', times=self.times, ids=self.ids, nodes=self.nodes,
                    marked_edges=self.marked_edges)
        if len(self.info) > 0:
            with open(path + self.name + '.pkl', 'wb') as handle:
                pickle.dump(self.info, handle)


def load_test(name: str, path: str = '', params: {str: object} = {}) -> ConnectivityTest:
    """ Loads test object.

    :param name: Name of test.
    :type name: str
    :param path: Path to saving location. (Default='')
    :type path: str
    :param params: Parameter for the test like 'seed', 'T_start', 'T_stop', and 'N'.
    :type params: dict
    :return: Returns test object.
    :rtype: ConnectivityTest
    """
    data = numpy.load(path + name + '.npz', allow_pickle=True)
    try:
        with open(path + name + '.pkl', 'rb') as handle:
            info = pickle.load(handle)
    except FileNotFoundError:
        info = {}
    subset = params.get('subset', None)
    if subset is None:
        con_test = ConnectivityTest(name, data['times'], data['ids'], data['nodes'], data['marked_edges'], 
                                    params=params, info=info)
    else:
        times, ids, nodes, marked_edges = data['times'], data['ids'], data['nodes'], data['marked_edges']
        valid_spikes = numpy.where(numpy.isin(ids, subset))[0]
        valid_nodes = numpy.where(numpy.isin(nodes, subset))[0]
        valid_edges = numpy.where(numpy.logical_and(numpy.isin(marked_edges[:,0], subset),
                                                    numpy.isin(marked_edges[:,1], subset)))[0]
        con_test = ConnectivityTest(name, times[valid_spikes], ids[valid_spikes], nodes[valid_nodes], 
                                    marked_edges[valid_edges], params=params, info=info)
    return con_test
