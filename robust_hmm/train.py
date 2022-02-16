"""
The MIT License (MIT)

Copyright (c) 2022 Calico Life Sciences

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from __future__ import absolute_import
from __future__ import print_function

import sys
sys.path.append('')

import os
import argparse
import random
import numpy as np
import pandas as pd
from functools import partial
from collections import Counter
from tqdm import tqdm
from hmmlearn.utils import iter_from_X_lengths


from catnap.data_processing.prepare_hmm_data import prepare_traces
from catnap.data_processing.prepare_hmm_data import split_train_validation
from catnap.data_processing.prepare_hmm_data import HMM_FEATURES
from catnap.robust_hmm.utils import str_to_bool
from catnap.robust_hmm.utils import flatten_traces
from catnap.robust_hmm.utils import nancov
from catnap.robust_hmm import rhmm


DB_NAME = 'calico_catnap_2020_03_09_QCed_v2'
MODEL_DIR = 'hmm_models_batch'
INTERVAL = 3
START_K = 1
STOP_K = 11
STEP_K = 1
TRAIN_FRACTION = 0.9
N_PROCESSES = 8
SEED = 42
COMPLETE = False

def parse_args():
  """Pass the arguments required to run the hmm batch model

  Returns:
      An object with two attributes, integers and accumulate
  """
  parser = argparse.ArgumentParser(
      description="Fit HMM models for a number of different component counts. ")

  parser.add_argument("--db_name", type=str, default=DB_NAME,
    help=f'Database from which get the data. Default: {DB_NAME}')
  parser.add_argument("--interval", type=str, default=INTERVAL,
    help=f'Aggregation interval in minutes. Default: {INTERVAL}')
  parser.add_argument("--model_dir", type=str, default=MODEL_DIR,
    help=f'Directory where to store fit HMM models. Default: {MODEL_DIR}')
  parser.add_argument("--start_k", type=int, default=START_K,
    help=f'Starting number of components. Default: {START_K}')
  parser.add_argument("--step_k", type=int, default=STEP_K,
    help=f'Increment in number of components. Default: {STEP_K}')
  parser.add_argument("--stop_k", type=int, default=STOP_K,
    help=f'Bound on the no. of components in a an HMM model. Default:{STOP_K}')
  parser.add_argument("--n_processes", type=int, default=N_PROCESSES,
    help=f'Number of parallel processes. Default: {N_PROCESSES}')
  parser.add_argument("--train_fraction", type=float, default=TRAIN_FRACTION,
    help=f'Fraction of training set used for HMM fitting, rest is used for ' \
      f'model selection. Default: {TRAIN_FRACTION}')
  parser.add_argument("--seed", type=int, default=SEED,
    help=f'Random seed for hmm initialization. Default: {SEED}')
  parser.add_argument("--path_to_cnf", type=str, default='mylogin.cnf',
    help='Path to configuration file for the MySQL database.')
  parser.add_argument("--complete", type=str_to_bool, default=False, 
    const=True, nargs='?', 
    help=f'Using only fraction of the data for training. Default: {COMPLETE}')

  args = parser.parse_args()
  return args

def load_all_data(db_name, interval, path_to_cnf):
  """Load all data from the databased using the interval length

  Args:
      db_name (str): name of the database
      interval (int): sampling interval in mins
      path_to_cnf (str): password to access database

  Returns:
      all_traces_flat (pandas): flattened dataframe
      all_traces_lengths (ndarray): contains lengths of each trace
  """
  all_traces_grouped = prepare_traces(db_name, 
                                      interval, 
                                      path_to_cnf)
  all_traces_flat, all_traces_lengths = flatten_traces(all_traces_grouped)
  return (all_traces_flat, all_traces_lengths)

def hmm_train_fn(k, 
                 X_train, 
                 lengths_train, 
                 batches_train, 
                 n_processes, 
                 seed):
  """Function to train the traces for a given model order.
  It is important to include `b` in the params. Here, `b`
  stands for batch. Batches are defined with respect to
  rotation, track, and gas analyzer.

  Args:
      k (int): model order of HMM
      X_train (ndarray): flattened trace data
      lengths_train (ndarray): lengths of traces
      batches_train (ndarray): batch ids of traces
      n_processes (int): run train in parallel
      seed (int): random seed value

  Returns:
      hmm_model (object): trained robust HMM
  """
  data_cov = np.diag(nancov(X_train))
  data_mean = np.nanmean(X_train, axis=0)
  amean = np.expand_dims(data_mean, 0)
  acov = np.expand_dims(np.diag(data_cov), 0)
  garbage_means = np.concatenate((0 * amean, amean), axis=0)
  garbage_covs = np.concatenate((1e-5 * acov, 2 * acov), axis=0)
  hmm_model = rhmm.RobustGaussianHMM(algorithm='viterbi',
                                     n_components=k,
                                     params='mtscb',
                                     covariance_type='diag',
                                     min_covar=1e-2 * data_cov,
                                     covars_prior=0.0,
                                     n_iter=100,
                                     verbose=True,
                                     startprob_prior=2.0,
                                     transmat_prior=2.0 * np.eye(k, k) +
                                     1.0 * np.ones((k, k)),
                                     init_params='mtscb',
                                     random_state=seed)
  hmm_model.set_garbage_component(garbage_means, garbage_covs, 0.99)
  if n_processes>1:
    hmm_model.fit(X_train, 
                  lengths=lengths_train, 
                  batch_ids=batches_train)
  else:
    hmm_model.fit_parallel(X_train, 
                           lengths=lengths_train,
                           batch_ids=batches_train, 
                           n_processes=n_processes)
  return hmm_model

def fit_hmm(db_name, 
            interval, 
            start_k, 
            step_k, 
            stop_k, 
            model_dir, 
            n_processes,
            train_fraction, 
            seed, 
            path_to_cnf, 
            complete=False):
  """Function used to train the robust HMM model. The model

  Args:
      db_name (str): name of the database
      interval (int): sampling interval in mins
      start_k (int): min model order
      step_k (int): step size of model order increments
      stop_k (int): max model order
      model_dir (str): location where trained models are stored
      n_processes (int): run train in parallel
      train_fraction (float): fraction of training data
      seed (int): random seed value
      path_to_cnf (str): password to access database
      complete (bool, optional): use complete databse. Defaults to False.

  Returns:
      hmm_models (list): list of trained robust HMM
  """

  print(f'Using data from {db_name} with interval of {interval} minute(s).')
  print(f'Fitting models with k=range(start={start_k}, stop={stop_k}, '
        f'step={step_k}) using {n_processes} processes.')

  ks = range(start_k, stop_k, step_k)
  print(f'Fitting models for component counts:\n {ks}')
  print(f'Using {n_processes} processes.')
  os.makedirs(model_dir, exist_ok=True)
  print(f'Storing in {model_dir}.')

  all_traces_flat, _ = load_all_data(db_name, interval, path_to_cnf)
  train_traces, batch_ids_train, _, _, all_traces, all_batch_ids = \
    split_train_validation(all_traces_flat, 
                           train_fraction=train_fraction, 
                           seed=seed)

  if not complete:
    X_train, lengths_train = flatten_traces(train_traces)
    print('Average age in training set:', np.mean(X_train.age_in_months))
    n_train_samples = len(lengths_train)
    print(f'Using {n_train_samples} samples for training. ')
    X_train = np.asarray(X_train[HMM_FEATURES], dtype=np.float64)
  else:
    X_train, lengths_train = flatten_traces(all_traces)
    print('Average age in training set:', np.mean(X_train.age_in_months))
    n_train_samples = len(lengths_train)
    print(f'Using {n_train_samples} samples for training. ')
    X_train = np.asarray(X_train[HMM_FEATURES], dtype=np.float64)
    batch_ids_train = all_batch_ids

  _hmm_train_fn = partial(hmm_train_fn, n_processes=n_processes, seed=seed)
  hmm_models = rhmm.train_batch_models(hmm_train_fn=_hmm_train_fn,
                                              X_train=X_train,
                                              lengths_train=lengths_train,
                                              batches_train=batch_ids_train,
                                              ks=ks,
                                              model_dir=model_dir)

  return hmm_models

def main():
  args = parse_args()
  print(args)
  fit_hmm(db_name=args.db_name,
          interval=args.interval,
          start_k=args.start_k,
          stop_k=args.stop_k,
          step_k=args.step_k,
          model_dir=args.model_dir,
          n_processes=args.n_processes,
          train_fraction=args.train_fraction,
          seed=args.seed,
          complete=args.complete,
          path_to_cnf=args.path_to_cnf)

if __name__ == "__main__":
  main()