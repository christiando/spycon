from spycon_inference import SpikeConnectivityInference
import pyinform
from itertools import repeat
import multiprocessing
import numpy
from scipy.stats import norm


class TE_PyInform(SpikeConnectivityInference):

    def __init__(self, params: dict = {}):
        """ Implements the transfer entropy method from
        
        Moore, Douglas G., et al. "Inform: efficient information-theoretic analysis of collective behaviors." Frontiers in Robotics and AI 5 (2018): 60.

        :param params:
            'binsize': Time step (in seconds) that is used for time-discretization. (Default=5e-3)
            'k': Length of history embedding. (Default=2)
            'alpha': Threshold. (Default=1e-2)
        :type params: dict
        """
        super().__init__(params)
        self.method = 'pyinform'
        self.default_params = {'binsize': 5e-3,
                               'k': 2,
                               'num_surrogates': 50,
                               'alpha': .01,
                               'jitter': False,
                               'jitter_factor': 7.,
                               'source_target_delay': 1}

    def _infer_connectivity(self, times: numpy.ndarray, ids: numpy.ndarray, pairs: numpy.ndarray) -> (numpy.ndarray):
        """ TE connectivity inference.

        :param times: Spike times in seconds.
        :type times: numpy.ndarray
        :param ids: Unit ids corresponding to spike times.
        :type ids: numpy.ndarray
        :param pairs: Array of [pre,post] pair node IDs, which inference will be done for.
        :type pairs: numpy.ndarray
        
        :return: Returns
            1) nodes:   [number_of_nodes], with the node labels.
            2) weights: [number_of_edges], with a graded strength of connections.
            3) stats:   [number_of_edges, 3], containing a fully connected graph, where the first columns are outgoing
                        nodes, the second the incoming node, and the third row contains the statistic, which was used to
                        decide, whether it is an edge or not. A higher value indicates, that an edge is more probable.
            4) threshold: a float that considers and edge to be a connection, if stats > threshold.
        :rtype: tuple
        """
        alpha = self.params.get('alpha', self.default_params['alpha'])
        nodes = numpy.unique(ids)
        weights = []
        stats = []
        num_connections_to_test = pairs.shape[0]
        conn_count = 0
        print_step = numpy.amin([1000,numpy.round(num_connections_to_test / 10.)])
        pairs_already_computed = numpy.empty((0,2))
        for pair in pairs:
            id1, id2 = pair
            if not any(numpy.prod(pairs_already_computed == [id2, id1], axis=1)):
                if conn_count % print_step == 0:
                    print('Test connection %d of %d (%d %%)' %(conn_count, num_connections_to_test,
                                                               100*conn_count/num_connections_to_test))
                te1, zval1, te2, zval2, pair = self.test_connection_pair(times, ids, pair)
                weights.append(te1)
                stats.append(numpy.array([id1, id2, zval1]))
                pairs_already_computed = numpy.vstack([pairs_already_computed, numpy.array([id1, id2])])
                if any(numpy.prod(pairs == [id2, id1], axis=1)):
                    weights.append(te2)
                    stats.append(numpy.array([id2, id1, zval2]))
                    conn_count += 2
                    pairs_already_computed = numpy.vstack([pairs_already_computed, numpy.array([id2, id1])])
                else:
                    conn_count += 1
        print('Test connection %d of %d (%d %%)' %(conn_count, num_connections_to_test,
                                                               100*conn_count/num_connections_to_test))
        weights = numpy.vstack(weights)
        stats = numpy.vstack(stats)
        # change pvalues such that predicted edges have positive value
        #stats[:,2] = 1 - stats[:,2]
        threshold = norm.ppf(1-.5*alpha)
        return nodes, weights, stats, threshold


    def _infer_connectivity_parallel(self, times: numpy.ndarray, ids: numpy.ndarray,
                                     pairs: numpy.ndarray, num_cores: int) -> (numpy.ndarray,):
        """ TE connectivity inference. Parallel version.

        :param times: Spike times in seconds.
        :type times: numpy.ndarray
        :param ids: Unit ids corresponding to spike times.
        :type ids: numpy.ndarray
        :param pairs: Array of [pre,post] pair node IDs, which inference will be done for.
        :type pairs: numpy.ndarray
        
        :return: Returns
            1) nodes:   [number_of_nodes], with the node labels.
            2) weights: [number_of_edges], with a graded strength of connections.
            3) stats:   [number_of_edges, 3], containing a fully connected graph, where the first columns are outgoing
                        nodes, the second the incoming node, and the third row contains the statistic, which was used to
                        decide, whether it is an edge or not. A higher value indicates, that an edge is more probable.
            4) threshold: a float that considers and edge to be a connection, if stats > threshold.
        :rtype: tuple
        """
        alpha = self.params.get('alpha', self.default_params['alpha'])
        nodes = numpy.unique(ids)
        weights = []
        stats = []
        pairs_already_computed = numpy.empty((0, 2))
        pairs_to_compute = []
        for pair in pairs:
            id1, id2 = pair
            if not any(numpy.prod(pairs_already_computed == [id2, id1], axis=1)):
                pairs_to_compute.append(pair)
                pairs_already_computed = numpy.vstack([pairs_already_computed, numpy.array([id1, id2])])
                if any(numpy.prod(pairs == [id2, id1], axis=1)):
                    pairs_already_computed = numpy.vstack([pairs_already_computed, numpy.array([id2, id1])])

        job_arguments = zip(repeat(times), repeat(ids), pairs_to_compute)
        pool = multiprocessing.Pool(processes=num_cores)
        results = pool.starmap(self.test_connection_pair, job_arguments)
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
        #stats[:,2] = 1 - stats[:,2]
        threshold = norm.ppf(1-.5*alpha)
        return nodes, weights, stats, threshold


    def test_connection_pair(self, times: numpy.ndarray, ids: numpy.ndarray, pair: tuple) -> (float):
        """ Test connections in both directions.

        :param times: Spike times in seconds.
        :type times: numpy.ndarray
        :param ids: Unit ids corresponding to spike times.
        :type ids: numpy.ndarray
        :param pair: Node pair.
        :type pair: tuple
        :return: pval and weight for edge id1->id2, followed by pval and weight for id2->id1
        :rtype: (float)
        """
        id1, id2 = pair
        binsize = self.params.get('binsize', self.default_params['binsize'])
        k = self.params.get('k', self.default_params['k'])
        num_surrogates = self.params.get('num_surrogates', self.default_params['num_surrogates'])
        bins = numpy.arange(numpy.floor(1e3 * times[0]) * 1e-3, numpy.ceil(1e3 * times[-1]) * 1e-3, binsize)
        times1 = times[ids == id1]
        times2 = times[ids == id2]
        spk_train1 = numpy.histogram(times1, bins=bins)[0]
        spk_train2 = numpy.histogram(times2, bins=bins)[0]
        times_bg = times[numpy.logical_not(numpy.logical_or(ids == id1, ids == id2))]
        spk_train_bg = numpy.histogram(times_bg, bins=bins)[0]
        te1 = pyinform.transfer_entropy(spk_train1, spk_train2, k=k)
        te1_H0 = numpy.empty(num_surrogates)
        #surr_spk_train1 = self.get_surrogates(source=spk_train1, conditional=spk_train_bg)
        for isurr in range(num_surrogates):
            if self.params.get('jitter', self.default_params['jitter']):
                jitter_factor = self.params.get('jitter_factor', self.default_params['jitter_factor'])
                times1_shuffled = numpy.sort(times1 + jitter_factor * binsize * (numpy.random.rand(len(times1)) - .5))
            else:
                times1_shuffled = numpy.sort(numpy.random.choice(times[ids != id2], len(times1), replace=False))
            spk_train1_shuffled = numpy.histogram(times1_shuffled, bins=bins)[0]
            te1_H0[isurr] = pyinform.transfer_entropy(spk_train1_shuffled, spk_train2, k=k)
        mu_H0, std_H0 = numpy.mean(te1_H0), numpy.amax([numpy.std(te1_H0), 1e-10])
        #pval_tmp = norm.cdf(te1, loc=mu_H0, scale=std_H0)
        #pval1 = numpy.amin([pval_tmp, 1. - pval_tmp])
        zval1 = numpy.abs(te1 - mu_H0) / std_H0
        te2 = pyinform.transfer_entropy(spk_train2, spk_train1, k=k)
        te2_H0 = numpy.empty(num_surrogates)
        #surr_spk_train2 = self.get_surrogates(source=spk_train2)
        for isurr in range(num_surrogates):
            if self.params.get('jitter', self.default_params['jitter']):
                jitter_factor = self.params.get('jitter_factor', self.default_params['jitter_factor'])
                times2_shuffled = numpy.sort(times2 + jitter_factor * binsize * (numpy.random.rand(len(times2)) - .5))
            else:
                times2_shuffled = numpy.sort(numpy.random.choice(times[ids != id1], len(times2), replace=False))
            spk_train2_shuffled = numpy.histogram(times2_shuffled, bins=bins)[0]
            te2_H0[isurr] = pyinform.transfer_entropy(spk_train2_shuffled, spk_train1, k=k)
        mu_H0, std_H0 = numpy.mean(te2_H0), numpy.amax([numpy.std(te2_H0), 1e-10])
        #pval_tmp = norm.cdf(te2, loc=mu_H0, scale=std_H0)
        #pval2 = numpy.amin([pval_tmp, 1. - pval_tmp])
        zval2 = numpy.abs(te2 - mu_H0) / std_H0
        return te1, zval1, te2, zval2, pair
    
    def get_surrogates(self, spk_t):
        conditional_set = numpy.unique(conditional)
        num_surrogates = self.params.get('num_surrogates', self.default_params['num_surrogates'])
        surr_source = numpy.empty([source.shape[0], num_surrogates])
        for conditional_elem in conditional_set:
            elem_indices = numpy.where(conditional == conditional_elem)[0]
            surr_source[elem_indices] = numpy.random.choice(source[elem_indices], size=(len(elem_indices), num_surrogates))
        return surr_source