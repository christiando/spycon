# spycon: A toolbox for benchmarking neuronal connectivity inference
## Introduction

Numerous methods have been proposed that attempt to infer neuronal connectivity from parallel recorded spike trains. 
Here we provide the `spycon` framework, that aims at making a comparable framework among these methods. We provide several 
connectivity inference algorithms, and a template, such that costum algorithms can be intergrated into `spycon`. `spycon`
then provides a unified output of the methods, that methods can be benchmarked on different datasets.

[**Basic Usage**](#basics) | [**Install guide**](#installation) | [**Citing**](#citation) | [**Documentation**]()

## Get started

You can either install `spycon` locally. If you just want to try it out quickly, you can also start and interactive 
session on [renku](https://renkulab.io/projects/christian.donner/spycon) and directly start coding. Check the [`notebooks`](notebooks) folder for some examples.

Here is a simple example, that shows how easy it is to do connectivity inference.

```python
from spycon.coninf import Smoothed_CCG

times, ids = ...                           # Load your data as simple 1D numpy arrays
spycon_method = Smoothed_CCG()             # Initialize your method with default parameters
spycon_result = spycon_method.infer_connectivity(times, ids) # Run inference
spycon_result.draw_graph()                 # Draw inferred graph
```

## Installation

Clone the repository into a directory and go into the folder. Just do the following

```bash
pip install git+https://gitlab.renkulab.io/christian.donner/spycon
```

For code development do
```bash
git clone https://gitlab.renkulab.io/christian.donner/spycon.git
cd gaussian-toolbox/
pip install -r requirements.txt
pip install -e .
```

## Implemented algorithms

All the algorithms that are discussed in our paper are implemented in this toolbox.

| Method             | Abbrevation | Remarks                                                            | Reference |
|--------------------|-------------|--------------------------------------------------------------------|-----------|
| [`CoincidenceIndex`](spycon/coninf/sci_ci.py) | CI          | Assess significance of coincidence index via surrogate data        |           |
| [`Smoothed_CCG`](spycon/coninf/sci_sccg.py)     | sCCG        | Assess significance of peaks of CCG by comparing to smoothed CCG   |           |
| [`directed_STTC`](spycon/coninf/sci_dsttc.py)     | dSTTC       | Assess signficance of directed STTC via surrogate data             |           |
| [`GLMCC`](spycon/coninf/sci_glmcc.py)             | GLMCC       | Assess significance of peaks of CCG via parametric GLM fit         |           |
| [`TE_IDTXL`](spycon/coninf/sci_idtxl.py)          | TE          | Assess significance of transfer entropy via surrogate data         |           |
| [`GLMPP`](spycon/coninf/sci_glmpp.py)             | GLMPP       | Assess significance of GLM coupling by posterior approximation     |           |
| [`NNEnsmble`](spycon/coninf/sci_ensemble.py)         | eANN        | Predict (type of) connection based on the outcome of other methods |           |

__Disclaimer__: `NNEnsemble` is a supervised method. In our paper, we carefully designed training data, that is approximately close to the experimental conditions. If your data varies from this, performance is expected to degrade. Furthermore, each algorithm has different parameters, and depending on your dataset the default one might not be appropriate. Some benchmarking on some surrogate datasets, that resemble the regime of your experimental data will be helpful to assess this.

## Using you own data

`spycon` aims at keeping the bar low for using the connectivity algorithms for your data. We voluntarily choose, that it is enough to provide the data in the form of two simple `numpy` arrays, one containing the spike times (in seconds), and one the unit IDs. Note, that this makes the code very simple to use, but gives the user more responsibility of checking, that the data is in appropriately scaled, ordered, etc, compared to more sophisticated data formats. 

## Creating your own test

`spycon` contains some functionalities to make benchmarking easier on labeled data. If you have spiking data & information about the underlying connectivity, you can create an [`ConnectivityTest`](spycon/spycon_tests.py). On these objects, all implemented connectivity inference methods can be run, and provide you with some performance metrics, for benchmarking and allowing you to decide for the appropriate algorithm.

## Contributing your own algorithm

If you want to use other methods than the one implemented look at the [`sci_template.py`](spycon/coninf/sci_template.py)

## Citation

