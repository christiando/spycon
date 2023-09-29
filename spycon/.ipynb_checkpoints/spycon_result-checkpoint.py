import networkx as nx
import numpy
import pickle
from matplotlib import pyplot
import matplotlib as mpl


class SpikeConnectivityResult(object):

    def __init__(self, method: str, params: dict, nodes: numpy.ndarray,  all_weights: numpy.ndarray,
                 stats: numpy.ndarray, threshold: float, runtime: float) -> object:
        """

        :param method: Name of the methods that was used to obtain the result.
        :type method: str
        :param params: Dictionary with the corresponding parameters.
        :type params: dict
        :param nodes: [number_of_nodes], with the node labels.
        :type nodes: numpy.ndarray
        :param weights: [number_of_edges], with a graded strength of an edge.
        :type weights: numpy.ndarray
        :param stats: [number_of_edges, 3], containing a fully connected graph, where the first columns are outgoing
                        nodes, the second the incoming node, and the third row contains the statistic, which was used to
                        decide, whether it is an edge or not. A higher value indicates, that an edge is more probable.
        :type stats: numpy.ndarray
        :param threshold: Thresholding the stats, where a connection is defined as those, where stats > threshold.
        :type threshold: float
        :param runtime: Runtime of the inference algorithm.
        :type runtime: float
        """

        self.method = method
        self.params = params
        self.nodes = nodes
        self.all_weights = all_weights
        self.stats = stats
        self.runtime = runtime
        self.set_threshold(threshold)
        
    def set_threshold(self, new_threshold: float):
        """ Method to change the thresholding of the result.
        
        :param threshold: Thresholding the stats, where a connection is defined as those, where stats > threshold.
        :type threshold: float
        """
        self.threshold = new_threshold
        self.edges = self.stats[self.stats[:,2] > self.threshold,:2]
        self.weights = self.all_weights[self.stats[:,2] > self.threshold,:2]
        if len(self.edges) == 0:
            self.edges = numpy.zeros((0,2))
        if len(self.weights) == 0:
            self.weights = numpy.zeros((0,2))
    
    def create_nx_graph(self, graph_type: str = 'binary') -> nx.DiGraph:
        """ Creates a networkx graph from the results.

        :param graph_type:
            'binary': Creates an unweighted graph with the inferred connections. (Default)
            'stats': Creates a fully connected graph, with the decision stats as edge weights.
            'weighted': Creates a weighted graph, where weights are the inferred strengths.
        :type graph_type: str
        :return: NetworkX directed graph
        :rtype: nx.DiGraph
        """
        nxgraph = nx.DiGraph()
        nxgraph.add_nodes_from(self.nodes)
        if graph_type == 'binary':
            nxgraph.add_edges_from(self.edges)
        elif graph_type == 'stats':
            nxgraph.add_weighted_edges_from(self.stats)
        elif graph_type == 'weighted':
            weighted_edges = numpy.hstack([self.edges, self.weights])
            nxgraph.add_weighted_edges_from(weighted_edges)
        else:
            raise RuntimeError('Graph type unrecognized.')
        return nxgraph
    
    def draw_graph(self, graph_type: str = 'binary', ax: pyplot.Axes=None, cax: pyplot.Axes=None):
        """

        :param graph_type:
            'binary': Creates an unweighted graph with the inferred connections. (Default)
            'stats': Creates a fully connected graph, with the decision stats as edge weights.
            'weighted': Creates a weighted graph, where weights are the inferred strengths.
        :type graph_type: str
        :param ax: matplotlib axis the graph should be plotted in. (Default=None)
        :type ax: pyplot.Axes
        :param cax: matplotlib axis the colorbar should be plotted in. (Default=None)
        :type cax: pyplot.Axes
        :return: NetworkX directed graph
        :rtype: nx.DiGraph
        """
        graph = self.create_nx_graph(graph_type=graph_type)
        if graph_type == 'binary':
            nx.draw_circular(graph, ax=ax, with_labels=True, node_size=500, node_color='C1')
        elif graph_type == 'stats':
            cmap = pyplot.get_cmap('inferno')
            weights = list(nx.get_edge_attributes(graph, 'weight').values())
            min_weight, max_weight =  numpy.amin(weights), numpy.amax(weights)
            nx.draw_circular(graph, ax=ax, with_labels=True, node_size=500, node_color='C1', 
                             edge_color=weights, edge_cmap=cmap, edge_vmin=min_weight, edge_vmax=max_weight
                             )
            norm = mpl.colors.Normalize(vmin=min_weight, vmax=max_weight)
            pyplot.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='stats')
        elif graph_type is 'weighted':
            cmap = pyplot.get_cmap('BrBG')
            weights = list(nx.get_edge_attributes(graph, 'weight').values())
            max_weight = numpy.amax(numpy.absolute(weights))
            nx.draw_circular(graph, ax=ax, with_labels=True, node_size=500, node_color='C1', 
                             edge_color=weights, edge_vmin=-max_weight, edge_vmax=max_weight, edge_cmap=cmap)
            norm = mpl.colors.Normalize(vmin=-max_weight, vmax=max_weight)
            pyplot.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='weights')
        return graph

    def create_connectivity_matrix(self, conn_type: str = 'stats') -> numpy.ndarray:
        """ Creates the connectivity matrix from the edges.

        :param conn_type:
            'binary': Creates a binary connectivity matrix with the inferred connections. (Default)
            'stats': Creates a connectivity matrix, with the decision stats as edge weights.
            'weighted': Creates a weighted connectivity matrix, where weights are the inferred strengths.
        :type conn_type: str
        :return: The connectivity matrix [num_of_nodes, num_of_nodes]
        :rtype: numpy.ndarray
        """
        con_matrix = numpy.zeros((len(self.nodes), len(self.nodes)))
        if conn_type == 'stats':
            idx1 = numpy.searchsorted(self.nodes, self.stats[:,0])
            idx2 = numpy.searchsorted(self.nodes, self.stats[:,1])
            con_matrix[idx1, idx2] = self.stats[:,2]
        elif conn_type == 'binary':
            idx1 = numpy.searchsorted(self.nodes, self.edges[:,0])
            idx2 = numpy.searchsorted(self.nodes, self.edges[:,1])
            con_matrix[idx1, idx2] = 1
        elif conn_type == 'weighted':
            idx1 = numpy.searchsorted(self.nodes, self.edges[:,0])
            idx2 = numpy.searchsorted(self.nodes, self.edges[:,1])
            con_matrix[idx1, idx2] = self.weights[:,0]
        return con_matrix

    def save(self, name: str, path: str = ''):
        """ Save the results object.

        :param name: Name of the result object.
        :type name: str
        :param path: Path to saving location. (Default='')
        :type path: str
        """
        pkl_filename = path + name + '.pkl'
        with open(pkl_filename, 'wb') as f:
            pickle.dump({'method': self.method, 'params': self.params, 'runtime': self.runtime}, f)
        npz_filename = path + name + '.npz'
        numpy.savez(npz_filename, nodes=self.nodes, all_weights=self.all_weights, stats=self.stats,
                    threshold=numpy.array([self.threshold]))


def load_connectivity_result(name: str, path: str = '') -> SpikeConnectivityResult:
    """ Loads a result object.

    :param name: Name of the result object.
    :type name: str
    :param path: Path to where the result object is saved. (Default='')
    :type path: str
    :return: Result object.
    :rtype: SpikeConnectivityResult
    """
    pkl_filename = path + name + '.pkl'
    with open(pkl_filename, 'rb') as f:
        pkl_data = pickle.load(f)
    method = pkl_data['method']
    params = pkl_data['params']
    runtime = pkl_data['runtime']
    npz_filename = path + name + '.npz'
    npz_data = numpy.load(npz_filename)
    nodes = npz_data['nodes']
    all_weights = npz_data['all_weights']
    stats = npz_data['stats']
    threshold = npz_data['threshold']
    con_result = SpikeConnectivityResult(method, params, nodes, all_weights, stats, threshold, runtime)
    return con_result
