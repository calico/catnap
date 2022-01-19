# Overview

This is the code needed to reproduce the results from the paper "Automated, high-dimensional evaluation of 
physiological aging and resilience in outbred mice."

## Structure

`data_processing` contains the paper-specific code for pre-processing the data from its raw form off the metabolic
cage into a per-run / per-trace feature matrix which we use for most of the results in the paper.

`caspar` contains the code for the Combined Age and Survival Prediction of Aging Rate (CASPAR) model - a regression
model that combines both chronological age and mortality data as signals and takes advantage of longitudinal
measurements. A standalone version of CASPAR containing only the general purpose model is found at <link>.

`network` contains the code to reproduce our results on applying Time Varying Graph Lasso (TVGL) and the nonparanormal
SKEPTIC estimator to infer a sparse time-varying graphical model on our data and to run consensus (spectral) clustering
on the inferred networks as described in our paper.

