from spycon.spycon_inference import SpikeConnectivityInference
from spycon.spycon_result import SpikeConnectivityResult
from spycon.spycon_tests import ConnectivityTest, load_test
from scipy.stats import mstats
from torch import nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy, pandas

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


def create_dataset(
    spycon_result_dict: dict,
    labels: numpy.ndarray = None,
):
    """Creates a dataset from the results of the connectivity inference methods.

    Args:
        spycon_result_dict: Dictionary with the connectivity inference methods.
        labels: Array of [pre, post, label] pairs, where label is 1 for excitatory, -1 for inhibitory, and 0 for no connection.

    Returns:
        X: Dataframe with the features.
        y: Array with the labels.
        pair_ids: Array of [pre, post] pairs, which are the IDs of the nodes.
    """
    X_dict = {}

    feature_names = []
    for con_inf_name, spycon_result in spycon_result_dict.items():
        X_tmp = []
        nodes = spycon_result.nodes
        feature_names = feature_names + [
            "stats_%s" % con_inf_name,
            "weight_%s" % con_inf_name,
        ]
        pair_ids = []
        y = []
        for pre_syn in nodes:
            for post_syn in nodes:
                if not post_syn == pre_syn:
                    result_conn_idx = numpy.where(
                        numpy.logical_and(
                            spycon_result.stats[:, 0] == pre_syn,
                            spycon_result.stats[:, 1] == post_syn,
                        )
                    )[0]
                    pred_stat = spycon_result.stats[result_conn_idx, 2][0]
                    pred_weight = spycon_result.all_weights[result_conn_idx][0, 0]

                    x_tmp = numpy.array([pred_stat, pred_weight])

                    X_tmp.append(numpy.array([x_tmp]))
                    pair_ids.append(numpy.array([[pre_syn, post_syn]]))
                    if labels is not None:
                        label_conn_idx = numpy.where(
                            numpy.logical_and(
                                labels[:, 0] == pre_syn, labels[:, 1] == post_syn
                            )
                        )[0]
                        if labels[label_conn_idx, 2] > 0:
                            true_label = numpy.array([1])
                        elif labels[label_conn_idx, 2] < 0:
                            true_label = numpy.array([2])
                        else:
                            true_label = numpy.array([0])
                        y.append(true_label)

        X_dict[con_inf_name] = numpy.concatenate(X_tmp)
    pair_ids = numpy.concatenate(pair_ids)
    X = pandas.DataFrame(numpy.hstack(list(X_dict.values())), columns=feature_names)
    if labels is not None:
        y = numpy.concatenate(y)
        return X, y, pair_ids
    else:
        return X, pair_ids


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        "Initialization"
        self.y = y
        self.X = X

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.X)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample

        # Load data and get label
        X = self.X[index]
        y = self.y[index]

        return X, y


class ConnectivityEnsemble(nn.Module):
    def __init__(self, num_inputs: int):
        """Implements a neural network ensemble.

        Args:
            num_inputs: Number of input features.
        """
        super(ConnectivityEnsemble, self).__init__()
        # self.net = nn.Sequential(
        #    nn.Linear(22, 10), nn.ReLU(), nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 3)
        # )
        self.net = nn.Sequential(
            nn.Linear(num_inputs, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 3),
        )

    def forward(self, x):
        scores = self.net(x)
        return scores


def _train(dataloader, model, loss_fn, optimizer):
    for batch, (X, y) in enumerate(dataloader):
        # print(batch)
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)


class NNEnsemble(SpikeConnectivityInference):
    """
    Neural network ensemble.

    Args:
        con_inf (SpikeConnectivityInference): Connectivity method that should be corrected.
        params (dict): Parameter dictionary with the following keys:
            - 'name' (str): Name to identify the model and load if already trained.
            - 'model_path' (str): Path to the saved model.
            - 'threshold' (float): Threshold value between 0 and 1 for thresholding the network output. Default is 0.5.
            - 'con_inf_dict' (dict): Dictionary with the connectivity inference methods. Default is an empty dictionary ({}).
        save_training_models (bool, optional): Whether the models during training should be saved. Default is False.

    Returns:
        None
    """

    def __init__(self, params: dict, save_training_models: bool = False):

        super().__init__(params)
        self.default_params = {
            "name": None,
            "model_path": "../data/nn_models/",
            "threshold": 0.66,
            "con_inf_dict": {},
            "save_test": True,
        }
        self.con_inf_dict = self.params.get(
            "con_inf_dict", self.default_params["con_inf_dict"]
        )
        if len(self.con_inf_dict) == 0:
            raise RuntimeError("No original inference method specified!")
        self.method = "NNEnsemble"
        self.nn_model = ConnectivityEnsemble(
            int(len(self.con_inf_dict) * 2)
        )  # .to(device)
        model_path = self.params.get("model_path", self.default_params["model_path"])
        name = self.params.get("name", self.default_params["name"])
        if name is None:
            raise RuntimeError("No model name specified!")
        self.model_path = model_path + name
        try:
            self.nn_model.load_state_dict(torch.load(self.model_path + ".net"))
            print("##### Trained model successfully loaded #####")
        except FileNotFoundError:
            print("##### Could not find trained model: Training from skretch #####")
            self._train_network(save_training_models=save_training_models)

    def _load_train_dataset(self, noncon_fold: int = 10):
        example_params = numpy.array(
            [[0.34, 0.2], [0.34, 0.16], [0.36, 0.08], [0.42, 0.08], [0.4, 0.18]]
        )
        chip, experiment_name = 2602, "cell3VC"
        X, y = None, None
        for example_param in example_params:
            mu_noise, std_noise = example_param
            name = "ren_simulation_%d_%s_long_%d_%d" % (
                chip,
                experiment_name,
                numpy.around(mu_noise * 1e3),
                numpy.around(std_noise * 1e2),
            )
            for i in range(5):
                N = 100  # 2 * numpy.random.randint(30, 50)
                # N = 2 * numpy.random.randint(5, 10)
                N_exc = int(N / 2)
                # T_stop = numpy.random.randint(1800, 3600)
                print(i)
                exc_neurons, inh_neurons = numpy.arange(0, 150), numpy.arange(150, 300)
                numpy.random.shuffle(exc_neurons), numpy.random.shuffle(inh_neurons)
                subset = numpy.concatenate([exc_neurons[:N_exc], inh_neurons[:N_exc]])
                spycon_test = load_test(
                    name,
                    params={"T_stop": 3600, "subset": subset},
                    path="../data/gt_data/",
                )
                spycon_result_dict = {}
                for con_inf_name, con_inf in self.con_inf_dict.items():
                    spycon_result = con_inf.infer_connectivity(
                        spycon_test.times, spycon_test.ids, parallel=True
                    )
                    spycon_result_dict[con_inf_name] = spycon_result
                X_tmp, y_tmp, pair_ids = create_dataset(
                    spycon_result_dict,
                    spycon_test.marked_edges,
                )
                if X is None or y is None:
                    X, y = X_tmp.to_numpy(), y_tmp
                else:
                    X = numpy.concatenate([X, X_tmp.to_numpy()])
                    y = numpy.concatenate([y, y_tmp])
        pos_samples = numpy.where(y != 0)[0]
        neg_samples = numpy.where(y == 0)[0]
        numpy.random.shuffle(neg_samples)
        neg_samples = neg_samples[: noncon_fold * len(pos_samples)]
        sample_idx = numpy.concatenate([pos_samples, neg_samples])
        numpy.random.shuffle(sample_idx)
        X_select, y_select = X[sample_idx], y[sample_idx]
        y_select = torch.tensor(y_select, dtype=torch.int64)
        X_select = torch.tensor(X_select, dtype=torch.float32)
        return X_select, y_select

    def _train_network(self, save_training_models: bool = False):
        print("##### Loads Training Dataset #####")
        X, y = self._load_train_dataset()
        dataset = Dataset(X, y)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.nn_model.parameters(), lr=1e-3)
        dataloader = DataLoader(dataset, 200)
        print("##### Model training #####")
        epochs = 10000
        for t in tqdm(range(epochs)):
            # print(f"Epoch {t+1}\n-------------------------------")
            _train(dataloader, self.nn_model, loss_fn, optimizer)
            if save_training_models and t % 10 == 0:
                torch.save(
                    self.nn_model.state_dict(), f"{self.model_path}_epoch_{t}.net"
                )
        print("##### Save trained model #####")
        torch.save(self.nn_model.state_dict(), self.model_path + ".net")
        numpy.savez(self.model_path + "_trainset.npz", X=X.numpy(), y=y.numpy())

    def _infer_connectivity(
        self,
        times: numpy.ndarray,
        ids: numpy.ndarray,
        pairs: numpy.ndarray,
        spycon_result_dict: dict = None,
    ) -> tuple:
        """
        Ensemble connectivity inference.

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
        input_dict = {}
        for con_inf_name, con_inf in self.con_inf_dict.items():
            try:
                input_dict[con_inf_name] = spycon_result_dict[con_inf_name]
            except KeyError:
                print("No result provided for %s. Method runs." % con_inf_name)
                input_dict[con_inf_name] = con_inf.infer_connectivity(
                    times, ids, pairs, parallel=False
                )
        nodes = numpy.unique(ids)
        print("##### Creating dataset #####")
        X, pair_ids = create_dataset(input_dict)
        X = X.to_numpy()
        valid_pair_idx = []
        for pair in pairs:
            valid_idx = numpy.where(
                numpy.logical_and(pair_ids[:, 0] == pair[0], pair_ids[:, 1] == pair[1])
            )[0]
            if len(valid_idx) != 0:
                valid_pair_idx.append(valid_idx)
        valid_pair_idx = numpy.concatenate(valid_pair_idx)
        X_valid = X[valid_pair_idx]
        pairs_valid = pair_ids[valid_pair_idx]
        print("##### Making predictions #####")
        X_valid = torch.tensor(X_valid, dtype=torch.float32)
        softmax = torch.nn.Softmax(dim=1)
        y_score = softmax(self.nn_model(X_valid)).detach().numpy()
        stats = numpy.empty((pairs_valid.shape[0], 3))
        stats[:, :2] = pairs_valid
        stats[:, 2] = numpy.amax([y_score[:, 1], y_score[:, 2]])
        weights = []
        spycon_result = list(spycon_result_dict.values())[0]
        for ipair, edge in enumerate(pairs_valid):
            edge_idx = numpy.where(
                numpy.logical_and(
                    spycon_result.stats[:, 0] == edge[0],
                    spycon_result.stats[:, 1] == edge[1],
                )
            )[0]
            if len(edge_idx) != 0:
                weight_idx = numpy.argmax(y_score[ipair, 1:])
                if weight_idx == 0:
                    weight = y_score[ipair, weight_idx + 1]
                else:
                    weight = -y_score[ipair, weight_idx + 1]
                weights.append(numpy.array([[edge[0], edge[1], weight]]))
            else:
                weights.append(numpy.array([[edge[0], edge[1], numpy.nan]]))
        weights = numpy.concatenate(weights)
        threshold = self.params.get("threshold", self.default_params["threshold"])
        if self.params.get("save_test", self.default_params["save_test"]):
            numpy.savez(self.model_path + "_testset.npz", X=X, pair_ids=pair_ids)
        return nodes, weights, stats, threshold

    def _infer_connectivity_parallel(
        self,
        times: numpy.ndarray,
        ids: numpy.ndarray,
        pairs: numpy.ndarray,
        num_cores: int,
        spycon_result_dict: dict = None,
    ) -> numpy.ndarray:
        """
        Ensemble connectivity inference. Original inference methods are attempted to be executed in parallel.

        Args:
            times (numpy.ndarray): Spike times in seconds.
            ids (numpy.ndarray): Unit IDs corresponding to spike times.
            pairs (numpy.ndarray): Array of [pre, post] pair node IDs for which inference will be done.
            num_cores (int): Number of CPU cores to be used for multiprocessing.

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

        input_dict = {}
        for con_inf_name, con_inf in self.con_inf_dict.items():
            try:
                input_dict[con_inf_name] = spycon_result_dict[con_inf_name]
            except KeyError:
                print("No result provided for %s. Method runs." % con_inf_name)
                input_dict[con_inf_name] = con_inf.infer_connectivity(
                    times, ids, pairs, parallel=True
                )

        nodes = numpy.unique(ids)
        print("##### Creating dataset #####")
        X, pair_ids = create_dataset(input_dict)
        X = X.to_numpy()
        valid_pair_idx = []
        for pair in pairs:
            valid_idx = numpy.where(
                numpy.logical_and(pair_ids[:, 0] == pair[0], pair_ids[:, 1] == pair[1])
            )[0]
            if len(valid_idx) != 0:
                valid_pair_idx.append(valid_idx)
        valid_pair_idx = numpy.concatenate(valid_pair_idx)
        X_valid = X[valid_pair_idx]
        pairs_valid = pair_ids[valid_pair_idx]
        print("##### Making predictions #####")
        X_valid = torch.tensor(X_valid, dtype=torch.float32)
        softmax = torch.nn.Softmax(dim=1)
        y_score = softmax(self.nn_model(X_valid)).detach().numpy()
        stats = numpy.empty((pairs_valid.shape[0], 3))
        stats[:, :2] = pairs_valid
        stats[:, 2] = y_score[:, 1] + y_score[:, 2]
        weights = []
        spycon_result = list(input_dict.values())[0]
        for edge in pairs_valid:
            edge_idx = numpy.where(
                numpy.logical_and(
                    spycon_result.stats[:, 0] == edge[0],
                    spycon_result.stats[:, 1] == edge[1],
                )
            )[0]
            if len(valid_idx) != 0:
                weights.append(spycon_result.all_weights[edge_idx])
            else:
                weights.append(numpy.nan)
        weights = numpy.concatenate(weights)
        threshold = self.params.get("threshold", self.default_params["threshold"])
        if self.params.get("save_test", self.default_params["save_test"]):
            numpy.savez(self.model_path + "_testset.npz", X=X, pair_ids=pair_ids)
        return nodes, weights, stats, threshold
