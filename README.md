# Overview

This repository contains code for reproducing the results of our eLife paper "Automated, high-dimensional evaluation of 
physiological aging and resilience in outbred mice." (DOI: https://doi.org/10.7554/eLife.72664110e72664)


## Structure

`data_processing` contains the paper-specific code for pre-processing the data from its raw form off the metabolic
cage into a per-run / per-trace feature matrix which we use for most of the results in the paper.

`caspar` contains the code for the Combined Age and Survival Prediction of Aging Rate (CASPAR) model - a regression
model that combines both chronological age and mortality data as signals and takes advantage of longitudinal
measurements. A standalone version of CASPAR containing only the general purpose model will be released shortly.

`network` contains the code to reproduce our results on applying Time Varying Graph Lasso (TVGL) and the nonparanormal
SKEPTIC estimator to infer a sparse time-varying graphical model on our data and to run consensus (spectral) clustering
on the inferred networks as described in our paper.

`data` contains per-run feature data in HDF format. Features are namespaced by keys. Important keys include per-mouse metadata (`mouse metadata`), per-run metadata (`trace metadata`) and all per-trace features (`all features`). You will require `git-lfs` to download the data.

To access specific subsets of features / keys, you can use the short snippet below:

```
def read_results_from_h5(filename, keys=None):
  if keys is None:
    f = h5py.File(filename, 'r')    
    keys = list(f.keys())
  result = {}
  for key in keys:
    result[key] = pd.read_hdf(filename, key=key)
  return result
```

Where `filename` points to the location of the `trace_features.h5` file.
