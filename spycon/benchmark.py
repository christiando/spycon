import pandas
import datetime
from spycon import coninf
from spycon.spycon_tests import load_test
from spycon.spycon_inference import SpikeConnectivityInference


class ConnectivityBenchmark(object):
    """Benchmark object, that gets a list of datasets and methods, and tests each method on each dataset.

    Args:
        name (str): Name of the pipeline.
        data_path (str): Path where data is located.
        data_names (list of tuple): List of tuples, where each tuple has the data name (first entry) and a dictionary
            with parameter specification.
        methods (list of tuple): List of tuples, where each tuple has the method's name (first entry) and a dictionary
            with parameter specification.
    """

    def __init__(
        self,
        name: str,
        data_path: str,
        data_sets: list[tuple[str, dict]],
        methods: list[tuple[str, dict]],
    ):
        """Benchmark object, that gets a list of datasets and methods, and tests each method on each dataset.

        Args:
            name (str): Name of the pipeline.
            data_path (str): Path where data is located.
            data_names (list of tuple): List of tuples, where each tuple has the data name (first entry) and a dictionary
                with parameter specification.
            methods (list of tuple): List of tuples, where each tuple has the method's name (first entry) and a dictionary
                with parameter specification.
        """
        self.name = name
        self.data_path = data_path
        self.data_sets = data_sets
        self.methods = methods
        self.methods_dict = {
            "full graph": SpikeConnectivityInference,
            "ci": coninf.CoincidenceIndex,
            "sccg": coninf.Smoothed_CCG,
            "dsttc": coninf.directed_STTC,
            "pyinform": coninf.TE_PyInform,
            "glmpp": coninf.GLMPP,
            "glmcc": coninf.GLMCC,
            "idtxl": coninf.TE_IDTXL,
            "nnensemble": coninf.NNEnsemble,
        }

    def run_benchmarks(self, parallel: bool = False) -> pandas.DataFrame:
        """
        Runs the benchmark sequentially.

        Args:
            parallel (bool): Whether the parallel version is used, if implemented.

        Returns:
            pandas.DataFrame: A DataFrame with a row for each test containing metrics.
        """
        benchmark_df = pandas.DataFrame(
            columns=[
                "method_name",
                "method_params",
                "data_name",
                "data_params",
                "time_stamp",
                "runtime",
                "fpr",
                "tpr",
                "thresholds",
                "auc",
                "aps",
                "pcr_precision",
                "pcr_recall",
                "pcr_thresholds",
                "f1",
                "precision",
                "recall",
                "accuracy",
                "mcc",
            ]
        )
        test_count = 0
        num_tests = len(self.data_sets) * len(self.methods)

        for data_name, data_params in self.data_sets:
            for method_name, method_params in self.methods:
                print("+----------------------------------------------+")
                print(
                    "%d of %d tests: Currently method '%s' with dataset '%s'"
                    % (test_count, num_tests, method_name, data_name)
                )
                print("+----------------------------------------------+")
                test_df = self.run_test(
                    data_name, data_params, method_name, method_params, parallel
                )
                test_df.index = [test_count]
                benchmark_df = pandas.concat([benchmark_df, test_df], sort=False)
                test_count += 1
        return benchmark_df

    def run_test(
        self,
        data_name: str,
        data_params: dict,
        method_name: str,
        method_params: dict,
        parallel: bool,
    ) -> pandas.DataFrame:
        """
        Runs a single test.

        Args:
            data_name (str): Name of the test.
            method_name (str): Name of the method.
            method_params (dict): Dictionary with parameter specifications.

        Returns:
            pandas.DataFrame: A DataFrame with result metrics for the test.
        """
        test = load_test(data_name, self.data_path, params=data_params)
        coninf_method = self.methods_dict[method_name](method_params)
        metrics_df = test.run_test(coninf_method, parallel=parallel)
        ts = datetime.datetime.now(tz=None).strftime("%d-%b-%Y (%H:%M:%S)")
        meta_df = pandas.DataFrame(
            [[method_name, method_params, data_name, data_params, ts]],
            columns=[
                "method_name",
                "method_params",
                "data_name",
                "data_params",
                "time_stamp",
            ],
            index=[0],
        )
        test_df = pandas.concat([meta_df, metrics_df], axis=1, sort=False)
        return test_df
