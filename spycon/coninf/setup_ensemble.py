from spycon import coninf


def get_eann_coninf_dict(
    num_cores, remove_bursts=False, N_burst=50, ISI_N_burst=0.1
) -> tuple:
    """Setup the base connectivity inference methods for the eANN model.

    Note: This is a specific function for the model presented in the paper.
    If you want to train a different model, you need to write your own function.

    Args:
        num_cores: Number of cores used for parallelization.
        remove_bursts: Whether bursts should be removed in the data. Defaults to False.
        N_burst: _description_. Defaults to 50.
        ISI_N_burst: _description_. Defaults to 0.1.

    Returns:
        tuple: A tuple containing the connectivity inference dictionary and the model name.
    """
    model_name = "eANN"
    coninf_dict = {
        "ci": coninf.directed_STTC(
            params={
                "num_cores": num_cores,
                "remove_bursts": remove_bursts,
                "N_burst": N_burst,
                "ISI_N_burst": ISI_N_burst,
                "jitter": True,
            }
        ),
        "sccg": coninf.Smoothed_CCG(
            params={
                "syn_window": (0.8e-3, 5.8e-3),
                "alpha": 1e-3,
                "num_cores": num_cores,
                "remove_bursts": remove_bursts,
                "N_burst": N_burst,
                "ISI_N_burst": ISI_N_burst,
            }
        ),
        "dsttc": coninf.directed_STTC(
            params={
                "num_surrogates": 50,
                "delta_t": 7e-3,
                "alpha": 1e-3,
                "num_cores": num_cores,
                "N_burst": N_burst,
                "ISI_N_burst": ISI_N_burst,
                "jitter": True,
            }
        ),
        "glmcc": coninf.GLMCC(
            params={
                "num_cores": num_cores,
                "remove_bursts": remove_bursts,
                "N_burst": N_burst,
                "ISI_N_burst": ISI_N_burst,
            }
        ),
        "glmpp": coninf.GLMPP(
            params={
                "num_cores": num_cores,
                "remove_bursts": remove_bursts,
                "N_burst": N_burst,
                "ISI_N_burst": ISI_N_burst,
            }
        ),
        "idtxl": coninf.TE_IDTXL(
            params={
                "num_cores": num_cores,
                "remove_bursts": remove_bursts,
                "N_burst": N_burst,
                "ISI_N_burst": ISI_N_burst,
                "jitter": True,
            }
        ),
    }
    return coninf_dict, model_name
