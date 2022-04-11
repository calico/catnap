import os
import tqdm
import pickle
import numpy as np
import pandas as pd

from datetime import datetime
from datetime import  timedelta
from itertools import chain
from multiprocessing import Pool, Value

import hmmlearn
import hmmlearn.hmm
from hmmlearn.base import DECODER_ALGORITHMS
from hmmlearn.utils import iter_from_X_lengths

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from catnap.robust_hmm.hmm import _log_densities_diag_components
from catnap.robust_hmm.hmm import DEFAULT_N_PROCESSES
from catnap.robust_hmm.hmm import GaussianPerFeatureGarbageHMM


class RobustGaussianHMM(GaussianPerFeatureGarbageHMM):
  """HMM with garbage components and batch correction

  Args:
      GaussianPerFeatureGarbageHMM (class): base class of HMM learn with 
      garbage states
  """

  def _init(self, 
            X, 
            lengths=None, 
            batch_ids=None):
    """Initialize class

    Args:
        X (ndarray): flattened data aray
        lengths (list, optional): list containing lengths of each trace. 
        Defaults to None.
        batch_ids (list, optional): batch id assigned to the group. 
        Defaults to None.
    """
    X = X[np.where(np.all(~np.isnan(X), axis=1))[0], :]
    self.n_batches = np.max(batch_ids) + 1
    if self.n_batches > 1:
      self.beta_ = np.zeros((self.n_batches-1, X.shape[1]))
    else:
      self.beta_ = 0
    super(RobustGaussianHMM, self)._init(X, lengths=lengths)


  def _compute_log_likelihood_complete(self, 
                                       X, 
                                       batch_id=None):
    """Compute log-likelihoods accounting for garbage model.

    Args:
        X (ndarray): flattened data aray
        batch_id (int, optional): batch id assigned to the trace. 
        Defaults to None.

    Returns:
        log_prob (ndarray): total log-likelihood after marginalizing over data 
        quality variable
        log_garbage (ndarray): total log-likelihood of the garbage components
        log_appearance (ndarray): total log-likelihood of the data with good 
        quality
    """
    if batch_id is None or batch_id == 0:
      b_shift = 0
    else:
      b_shift = self.beta_[int(batch_id)-1, :]

    log_appearance = _log_densities_diag_components(X, 
                                                    self.means_ + b_shift, 
                                                    self.covars_)
    log_garbage = []
    for i in range(self.n_garbage_components):
      mean = self.garbage_means_[[i],:]
      covar = self.garbage_covars_[[i],:,:]
      log_garbage.append(_log_densities_diag_components(X, mean, covar))

    # marginalize over masking variable z
    log_prob_factored = self._log_marginalize_z(log_garbage, log_appearance)
    log_prob = np.sum(log_prob_factored, axis=1)
    return log_prob, log_garbage, log_appearance

  def _compute_log_likelihood(self, 
                              X, 
                              batch_id=None):
    """Function to compute the complete log-likelihood

    Args:
        X (ndarray): flattened data aray
        batch_id (int, optional): batch id assigned to the trace. 
        Defaults to None.

    Returns:
        log_prob (ndarray): total log-likelihood after marginalizing over data 
        quality variable
    """

    log_prob, _, _ = self._compute_log_likelihood_complete(X, batch_id=batch_id)
    return log_prob


  def _decode_viterbi(self, 
                      X, 
                      batch_id):
    """Calls hmmlearn Viterbi decoding function

    Args:
        X (ndarray): flattened trace
        batch_id (int): corresponding batch id of the trace

    Returns:
        : forward-backward algorithm generated probabilities
    """
    framelogprob = self._compute_log_likelihood(X, batch_id=batch_id)
    return self._do_viterbi_pass(framelogprob)

  def _decode_map(self, 
                  X, 
                  batch_id):
    """Call decoder

    Args:
        X (ndarray): flattened trace
        batch_id (int): corresponding batch id of the trace

    Returns:
        log_prob (float): log-likelihood
        state_sequence (ndarray): arg max of posteriors
    """
    _, posteriors = self.score_samples(X, batch_ids=batch_id)
    logprob = np.max(posteriors, axis=1).sum()
    state_sequence = np.argmax(posteriors, axis=1)
    return logprob, state_sequence


  def decode(self, 
             X, 
             lengths=None, 
             batch_ids=None, 
             algorithm=None):
    """Find most likely state sequence corresponding to X.

    Args:
        X (ndarray): flattended data (n_samples x n_features)
        lengths (ndarray, optional): Lengths of the individual sequences in X. 
        The sum of these should be n_samples. Defaults to None.
        batch_ids (list, optional): batch id assigned to the group. 
        Defaults to None.
        algorithm (string, optional): Decoder algorithm. Must be one of 
        "viterbi" or "map". Defaults to None.

    Raises:
        ValueError: An error occured if the decoder algorithm is not known

    Returns:
        logprob (float): log-likelihood of the input data sequence 
        state_sequence (ndarray): decoded states 
    """

    check_is_fitted(self, "startprob_")
    self._check()

    algorithm = algorithm or self.algorithm
    if algorithm not in DECODER_ALGORITHMS:
      raise ValueError("Unknown decoder {!r}".format(algorithm))

    decoder = {
        "viterbi": self._decode_viterbi,
        "map": self._decode_map
    }[algorithm]

    X = check_array(X, force_all_finite=False)
    n_samples = X.shape[0]
    logprob = 0
    state_sequence = np.empty(n_samples, dtype=int)
    for (i, j), batch_id in zip(iter_from_X_lengths(X, lengths), batch_ids):
      # XXX decoder works on a single sample at a time!
      logprobij, state_sequenceij = decoder(X[i:j], batch_id)
      logprob += logprobij
      state_sequence[i:j] = state_sequenceij

    return logprob, state_sequence

  def _compute_sufficient_statistcs(self, 
                                    X, 
                                    lengths,
                                    batch_ids=None):
    """Compute sufficient stats

    Args:
        X (ndarray): flattended data (n_samples x n_features)
        lengths (ndarray, optional): Lengths of the individual sequences in X. 
        The sum of these should be n_samples. Defaults to None.
        batch_ids (list, optional): batch id assigned to the group. 
        Defaults to None.

    Returns:
        stats (dict): dictionary containing sufficient stats
        curr_logprob (ndarray): log-prob obtained by forward + backward pass
    """
    stats = self._initialize_sufficient_statistics()
    curr_logprob = 0
    for (i, j), batch_id in zip(iter_from_X_lengths(X, lengths), batch_ids):
      framelogmarginalprob, frameloggarbage, framelogappearance = \
        self._compute_log_likelihood_complete(X[i:j], batch_id=batch_id)
      logprob, fwdlattice = self._do_forward_pass(framelogmarginalprob)
      curr_logprob += logprob
      bwdlattice = self._do_backward_pass(framelogmarginalprob)
      posteriorh, posteriorz = \
        self._compute_posteriors_complete(fwdlattice, 
                                          bwdlattice, 
                                          framelogmarginalprob,
                                          frameloggarbage, 
                                          framelogappearance)

      # construct a unit vector in the direction of the batch
      b_ic = np.zeros((self.n_batches-1, 1))
      if self.n_batches > 1 and batch_id != 0:
        b_ic[int(batch_id)-1,0] = 1

      self._accumulate_sufficient_statistics(stats, 
                                             X[i:j], 
                                             b_ic, 
                                             framelogmarginalprob, 
                                             posteriorh, 
                                             posteriorz, 
                                             fwdlattice, 
                                             bwdlattice)
    return stats, curr_logprob

  def fit(self, 
          X, 
          lengths=None, 
          batch_ids=None):
    """Estimate model parameters.

    Args:
        X (ndarray): flattended data (n_samples x n_features)
        lengths (ndarray, optional): Lengths of the individual sequences in X. 
        The sum of these should be n_samples. Defaults to None.
        batch_ids (list, optional): batch id assigned to the group. 
        Defaults to None.

    Returns:
        self (object): returns self with updated parameters
    """
    X = check_array(X, force_all_finite=False)
    self._init(X, lengths=lengths, batch_ids=batch_ids)
    #self._check()

    self.monitor_._reset()
    for i in range(self.n_iter):
      stats, curr_logprob = \
        self._compute_sufficient_statistcs(X, 
                                           lengths, 
                                           batch_ids=batch_ids)
      # XXX must be before convergence check, because otherwise
      #     there won't be any updates for the case ``n_iter=1``.
      self._do_mstep(stats, i)

      self.monitor_.report(curr_logprob)
      if i > self.n_iter / 2 + 5 and self.monitor_.converged:
        break

    return self

  def _split_X_and_lengths(self, 
                           X, 
                           lengths, 
                           batch_ids, 
                           n_parts):
    """Split flattened data.

    Parallelize the training process of the HMM by splitting the dataset into 
    multiple batches based on the number of processes

    Args:
        X (ndarray): flattended data (n_samples x n_features)
        lengths (ndarray, optional): Lengths of the individual sequences in X. 
        The sum of these should be n_samples. Defaults to None.
        batch_ids (list, optional): batch id assigned to the group. 
        Defaults to None.
        n_parts (int): for parallel processing

    Returns:
        X_parts (list): flattened data split into parts
        length_parts (list): length of each trace corresponding to X_parts
    """

    # initialize variables
    assert len(lengths) == len(batch_ids)
    part = np.sum(lengths)/n_parts
    lengths_parts = []
    batch_ids_parts = []
    X_parts = []
    start = 0
    end = 0
    lengths_part = []
    batch_ids_part = []
    
    # split the training data into batches
    for i, l in enumerate(lengths):
      end = end + l
      lengths_part.append(l)
      batch_ids_part.append(batch_ids[i])
      if end - start >= part or i == len(lengths)-1:
        X_part = X[start:end]
        X_parts.append(X_part)
        lengths_parts.append(lengths_part)
        batch_ids_parts.append(batch_ids_part)
        assert np.sum(lengths_part) == X_part.shape[0]
        assert len(lengths_part) == len(batch_ids_part)
        lengths_part = []
        batch_ids_part = []
        start = end

    # check if the sum of lengths are equal
    assert sum([sum(lengths_part) for lengths_part in lengths_parts]) == sum(lengths)
    assert end == sum(lengths)
    return X_parts, lengths_parts, batch_ids_parts


  def fit_parallel(self, 
                   X, 
                   lengths=None, 
                   batch_ids=None, 
                   n_processes=DEFAULT_N_PROCESSES):
    """Estimate model parameters

    HMM fit using a parallelization. Split data matrix X into X_parts
    based on the number of processes and learn the HMM parameters in parallel

    Args:
        X (ndarray): flattended data (n_samples x n_features)
        lengths (ndarray, optional): Lengths of the individual sequences in X. 
        The sum of these should be n_samples. Defaults to None.
        batch_ids (list, optional): batch id assigned to the group. 
        Defaults to None.
        n_processes (int, optional): number of parallel proces. 
        Defaults to DEFAULT_N_PROCESSES.

    Returns:
        self (object): returns self with updated parameters
    """
    X = check_array(X, force_all_finite=False)
    self._init(X, lengths=lengths, batch_ids=batch_ids)
    X_parts, lengths_parts, batch_ids_parts = self._split_X_and_lengths(
      X, lengths, batch_ids=batch_ids, n_parts=n_processes)
    self.monitor_._reset()

    # run parallelization
    with Pool(n_processes) as pool:
      # em algorithm iteration
      for i in range(self.n_iter):
        batched_stats_and_log_probs = pool.starmap(
          self._compute_sufficient_statistcs,
          zip(X_parts, lengths_parts, batch_ids_parts))
        batched_stats, log_probs = zip(*batched_stats_and_log_probs)
        stats = self._aggregate_sufficient_statistics(batched_stats)
        curr_logprob = np.sum(log_probs)
        # XXX must be before convergence check, because otherwise
        #     there won't be any updates for the case ``n_iter=1``.
        self._do_mstep(stats, i)

        self.monitor_.report(curr_logprob)
        if i > self.n_iter / 2 + 5 and self.monitor_.converged:
          break
    return self


  def score(self, 
            X, 
            lengths=None, 
            batch_ids=None):
    """Compute the log probability under the model aka score

    Args:
        X (ndarray): flattended data (n_samples x n_features)
        lengths (ndarray, optional): Lengths of the individual sequences in X. 
        The sum of these should be n_samples. Defaults to None.
        batch_ids (list, optional): batch id assigned to the group. 
        Defaults to None.

    Returns:
        logprob (float): log-likelihood of the input data sequence
    """
    check_is_fitted(self, "startprob_")
    #self._check()

    X = check_array(X, force_all_finite=False)
    # XXX we can unroll forward pass for speed and memory efficiency.
    logprob = 0
    for (i, j), batch_id in zip(iter_from_X_lengths(X, lengths), batch_ids):
        framelogprob = self._compute_log_likelihood(X[i:j], batch_id)
        logprobij, _fwdlattice = self._do_forward_pass(framelogprob)
        logprob += logprobij
    return logprob


  def score_parallel(self, 
                     X, 
                     lengths=None, 
                     batch_ids=None,
                     n_processes=DEFAULT_N_PROCESSES):
    """Score in parallel

    Args:
        X (ndarray): flattended data (n_samples x n_features)
        lengths (ndarray, optional): Lengths of the individual sequences in X. 
        The sum of these should be n_samples. Defaults to None.
        batch_ids (list, optional): batch id assigned to the group. 
        Defaults to None.
        n_processes (int, optional): number of parallel proces. 
        Defaults to DEFAULT_N_PROCESSES.

    Returns:
        logprob (float): log-likelihood of the input data sequence
    """

    if lengths is not None:
      logprobs = self.score_parallel_per_sample(X, 
                                                lengths, 
                                                batch_ids, 
                                                n_processes)
      return np.sum(logprobs)
    else:
      return self.score(X, batch_ids=batch_ids)


  def score_per_sample(self, 
                       X, 
                       lengths, 
                       batch_ids=None):
    """Compute the log probability under the model aka score
    per sample

    Args:
        X (ndarray): flattended data (n_samples x n_features)
        lengths (ndarray, optional): Lengths of the individual sequences in X. 
        The sum of these should be n_samples. Defaults to None.
        batch_ids (list, optional): batch id assigned to the group. 
        Defaults to None.

    Returns:
        logprobs (list): list of log-probability of every trace
    """
    check_is_fitted(self, "startprob_")
    #self._check()

    X = check_array(X, force_all_finite=False)
    # XXX we can unroll forward pass for speed and memory efficiency.
    logprobs = []
    for (i, j), batch_id in zip(iter_from_X_lengths(X, lengths), batch_ids):
      framelogprob = self._compute_log_likelihood(X[i:j], batch_id=batch_id)
      logprobij, _fwdlattice = self._do_forward_pass(framelogprob)
      logprobs.append(logprobij)
    return logprobs


  def score_samples(self, 
                    X, 
                    lengths=None, 
                    batch_ids=None):
    """Compute the log probability under the model and compute posteriors

    Args:
        X (ndarray): flattended data (n_samples x n_features)
        lengths (ndarray, optional): Lengths of the individual sequences in X. 
        The sum of these should be n_samples. Defaults to None.
        batch_ids (list, optional): batch id assigned to the group. 
        Defaults to None.

    Returns:
        logprob (float): log-likelihood of the input data sequence
        posteriors (ndarray): posterior probability of being in a state
    """

    check_is_fitted(self, "startprob_")
    self._check()

    #X = check_array(X)
    n_samples = X.shape[0]
    logprob = 0
    posteriors = np.zeros((n_samples, self.n_components))
    for (i, j), batch_id in zip(iter_from_X_lengths(X, lengths), batch_ids):
      framelogprob = self._compute_log_likelihood(X[i:j], batch_id=batch_id)
      logprobij, fwdlattice = self._do_forward_pass(framelogprob)
      logprob += logprobij

      bwdlattice = self._do_backward_pass(framelogprob)
      posteriors[i:j] = self._compute_posteriors(fwdlattice, bwdlattice)
    return logprob, posteriors

  def score_parallel_per_sample(self, 
                                X, 
                                lengths, 
                                batch_ids=None, 
                                n_processes=DEFAULT_N_PROCESSES):
    """Score sample in parallel

    Args:
        X (ndarray): flattended data (n_samples x n_features)
        lengths (ndarray, optional): Lengths of the individual sequences in X. 
        The sum of these should be n_samples. Defaults to None.
        batch_ids (list, optional): batch id assigned to the group. 
        Defaults to None.
        n_processes (int, optional): number of parallel proces. 
        Defaults to DEFAULT_N_PROCESSES.

    Returns:
        logprobs (list): list of log-probability of every trace
    """
    check_is_fitted(self, "startprob_")
    #self._check()

    X = check_array(X, force_all_finite=False)
    X_parts, lengths_parts, batch_ids_parts = self._split_X_and_lengths(
      X, lengths, batch_ids=batch_ids, n_parts=n_processes)
    # XXX we can unroll forward pass for speed and memory efficiency.
    logprob = 0
    with Pool(n_processes) as pool:
      batched_log_probs = pool.starmap(
        self.score_per_sample, zip(X_parts, lengths_parts, batch_ids_parts))
      logprob = np.sum(batched_log_probs)
    return list(chain.from_iterable(batched_log_probs))

  def predict_z(self, 
                X, 
                lengths=None, 
                batch_ids=None):
    """Predict for each observation whether it is real for a particular 
    component.

    Args:
        X (ndarray): flattended data (n_samples x n_features)
        lengths (ndarray, optional): Lengths of the individual sequences in X. 
        The sum of these should be n_samples. Defaults to None.
        batch_ids (list, optional): batch id assigned to the group. 
        Defaults to None.

    Returns:
        zs (ndarray): shape (n_samples, n_features, n_components)
                      For each sample, given that it came from a particular 
                      component, what is the probability that it is real data: 
                      p(z_{i,j}=1 | x_i, h_i=c)
    """
    X = check_array(X, force_all_finite=False)
    self._check()
    zs = np.zeros([*X.shape, self.n_components])
    for (i, j), batch_id in zip(iter_from_X_lengths(X, lengths), batch_ids):
      framelogmarginalprob, frameloggarbage, framelogappearance = \
        self._compute_log_likelihood_complete(X[i:j], batch_id=batch_id)
      assert np.all(np.isfinite(framelogmarginalprob))
      logprob, fwdlattice = self._do_forward_pass(framelogmarginalprob)
      assert np.all(np.isfinite(fwdlattice))
      assert np.all(np.isfinite(logprob))
      bwdlattice = self._do_backward_pass(framelogmarginalprob)
      posteriorh, posteriorz = \
        self._compute_posteriors_complete(fwdlattice, 
                                          bwdlattice, 
                                          framelogmarginalprob,
                                          frameloggarbage, 
                                          framelogappearance)
      idx = np.any(~np.isnan(X[i:j]), axis=1)
      z = np.zeros([*X[i:j].shape, self.n_components])
      z[idx, :, :] = posteriorz[idx,:,:]
      zs[i:j] = z
    return zs


  def _initialize_sufficient_statistics(self):
    """Initialize sufficient stats

    Raises:
        ValueError: if the covariance type is "full" or "tied", raise error

    Returns:
        stats (dict): the keys are the expressions of sufficient stats
    """

    stats = \
      super(hmmlearn.hmm.GaussianHMM, self)._initialize_sufficient_statistics()
    stats['totobs'] = 0
    stats['post'] = np.zeros((self.n_components, self.n_features))
    stats['garbagepost'] = np.zeros((1, self.n_features))
    stats['obs'] = np.zeros((self.n_components, self.n_features))
    stats['garbageobs'] = np.zeros((1, self.n_features))
    stats['obs**2'] = np.zeros((self.n_components, self.n_features))
    stats['garbageobs**2'] = np.zeros((1, self.n_features))

    stats['bbeta'] = np.zeros((self.n_components, self.n_features))
    stats['mbeta'] = np.zeros((self.n_components, self.n_features))
    stats['bbeta**2'] = np.zeros((self.n_components, self.n_features))
    stats['bmatrix'] = np.zeros((self.n_batches-1, self.n_features))
    stats['bvec'] = np.zeros((self.n_batches-1, self.n_features))
    stats['bvecmean'] = np.zeros((self.n_batches-1, self.n_features))

    if self.covariance_type in ('tied', 'full'):
      raise ValueError('unsupported')
      stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
                                     self.n_features))
    return stats


  def _accumulate_sufficient_statistics(self, 
                                        stats, 
                                        obs, 
                                        batch_indicator, 
                                        framelogprob,
                                        posteriorh, 
                                        posteriorz, 
                                        fwdlattice, 
                                        bwdlattice):
    """Accumulate sufficient statistics for the E-step of the EM algorithm

    Args:
        stats (dict): contains sufficient statistics
        obs (ndarray): flattened data matrix
        framelogprob (ndarray): log-probability of the data matrix
        posteriorh (ndarray): posterior probability of hidden states (h)
        posteriorz (ndarray): posterior probability of data quality (z)
        fwdlattice (ndarray): forward probabilities 
        bwdlattice (ndarray): backward probabilities 

    Raises:
        ValueError: if the covariance type is "full" or "tied",
                    raise value error
    """
    super(hmmlearn.hmm.GaussianHMM, self)._accumulate_sufficient_statistics(
      stats, obs, framelogprob, posteriorh, fwdlattice, bwdlattice)

    # n_samples x n_features
    observation_mask = np.isnan(obs) # n x d
    # n_samples x n_components
    h_mask = np.repeat(np.expand_dims(np.all(observation_mask, axis=1),
                                      axis=1),
                                      self.n_components, axis=1)
    # n_samples x n_features x n_components
    z_mask = np.repeat(np.expand_dims(observation_mask, axis=2),
                       self.n_components, axis=2)


    masked_observations = np.ma.masked_array(obs, observation_mask)
    expanded_masked_observations = np.expand_dims(masked_observations, axis=2)
    expanded_squared_masked_observations = np.expand_dims(
      masked_observations**2.0, axis=2)


    expanded_means = np.expand_dims(self.means_.T, axis=0)
    # expanded covariances along sample dimensions to facilatate broadcasting
    covars = [np.diag(self.covars_[i, :, :]) for i in range(self.n_components)]
    expanded_covars = np.expand_dims(np.stack(covars, axis=1), axis=0)

    masked_posteriorh = np.ma.masked_array(posteriorh, h_mask)
    # n_samples x n_features x n_components
    # storing q(z_{ic}=1 | h_i, x_i)
    masked_posteriorz = np.ma.masked_array(posteriorz, z_mask)

    # n_traces x n_features x n_components
    # each entry is q(h_i, z_{ic}=0) computed by
    #   q(h_i) * q(z_{ic}=0 | h_i)
    posterior_h_z0 = np.expand_dims(masked_posteriorh, axis=1) \
      * (1.0-masked_posteriorz)

    # n_traces x n_features x n_components
    # each entry is q(h_i, z_{ic}=1) computed by
    #   q(h_i) * q(z_{ic}=1 | h_i)
    posterior_h_z1 = np.expand_dims(masked_posteriorh, axis=1)*masked_posteriorz

    bbeta = np.dot(batch_indicator.T, self.beta_)

    if 'z' in self.params:
      stats['totobs'] += np.sum(~observation_mask)
      stats['garbagepost'] += np.sum(posterior_h_z0, axis=(0,2)).T

      if 'm' in self.params or 'c' in self.params:
        stats['garbageobs'] += np.sum(
          posterior_h_z0*expanded_masked_observations, axis=(0, 2)).T
      if 'c' in self.params:
        stats['garbageobs**2'] += np.sum(
          posterior_h_z0*expanded_squared_masked_observations, axis=(0,2)).T


    if 'm' in self.params or 'c' in self.params:
      # sum across samples
      #   sum_i q(h_i, z_{ic})
      stats['post'] += np.sum(posterior_h_z1, axis=0).T
      # sum across nonmasked samples
      #    sum_i q(h_i, z_{ic}=1) x_{ic}
      stats['obs'] += np.sum(
        posterior_h_z1*expanded_masked_observations, axis=0).T

    if self.n_batches > 1:
      if 'm' in self.params or 'c' in self.params :
        # sum across samples
        #   sum_i q(h_i, z_{ic}=1)*(b_i^T*beta)
        stats['bbeta'] += np.sum(posterior_h_z1, axis=0).T*bbeta

      if 'b' in self.params :
        inv_covars = 1./expanded_covars

        # q(h_i, z_ic=1)/sigma_{h_i, c}^2
        scaled_posterior = posterior_h_z1*inv_covars
        # sum_i sum_h 1/sigma_{h_i}^2 q(h_i=h, z_{ic}=1)
        summed_scaled_posterior = np.sum(scaled_posterior, axis=(0,2))
        # sum_i sum_h 1/sigma_{h_i=h}^2 q(h_i=h, z_{ic}=1) b
        stats['bmatrix'] += np.dot(
          np.expand_dims(summed_scaled_posterior, axis=-1), batch_indicator.T).T

        # q(h_i, z_{ic}=1) y_{ic}/sigma_{h_i}^2
        scaled_observations = posterior_h_z1 \
          * inv_covars * expanded_masked_observations
        # sum_i sum_h 1/sigma_{h_i}^2 q(h_i=h, z_{ic}=1) y_{ic}
        summed_scaled_observations = np.sum(scaled_observations, axis=(0,2))
        # sum_i sum_h 1/sigma_{h_i=h}^2 q(h_i=h, z_{ic}=1)  y_{ic} b
        stats['bvec'] += np.dot(np.expand_dims(summed_scaled_observations, \
          axis=-1), batch_indicator.T).T

        # q(h_i, z_{ic}=1) mu_{h_i}/sigma_{h_i}^2
        scaled_means = posterior_h_z1*inv_covars*expanded_means
        # sum_i sum_h 1/sigma_{h_i}^2 q(h_i, z_{ic}=1) mu_{h_i,c}
        summed_scaled_means = np.sum(scaled_means, axis=(0,2))
        # sum_i sum_h 1/sigma_{h_i}^2 q(h_i, z_{ic}=1) mu_{h_i,c} b
        stats['bvecmean'] += np.dot(
          np.expand_dims(summed_scaled_means, axis=-1), batch_indicator.T).T

    if 'c' in self.params:
      if self.covariance_type in ('spherical', 'diag'):
        stats['obs**2'] += np.sum(
          posterior_h_z1*expanded_squared_masked_observations, axis=0).T
        if self.n_batches > 1:
          stats['mbeta'] += np.sum(
            posterior_h_z1*expanded_masked_observations*bbeta.T, axis=(0)).T
          stats['bbeta**2'] += np.sum(posterior_h_z1*bbeta.T**2, axis=0).T
      elif self.covariance_type in ('tied', 'full'):
        raise ValueError('unsupported')


  def _do_mstep(self, 
                stats, 
                i):
    """M-step of the EM algorithm

    Args:
        stats (dict): contains sufficient statistics
        i (int): iteration index

    Raises:
        ValueError: if the covariance type is "full" or "tied",
                    raise value error
    """
    super(hmmlearn.hmm.GaussianHMM, self)._do_mstep(stats)

    means_prior = self.means_prior
    means_weight = self.means_weight

    # TODO: find a proper reference for estimates for different
    #       covariance models.
    # Based on Huang, Acero, Hon, "Spoken Language Processing",
    # p. 443 - 445
    denom = stats['post']

    if 'm' in self.params:
      self.means_ = ((means_weight*means_prior + stats['obs'] - stats['bbeta'])
                        / (means_weight + denom))
    if 'b' in self.params:
      if self.n_batches > 1:
        # self.beta_ = (stats['bvec'] - stats['bvecmean']) / stats['bmatrix']
        # bmatrix_mask = np.ma.masked_where(stats['bmatrix']==0, stats
        # ['bmatrix'])
        bmatrix_mask = np.ma.masked_values(stats['bmatrix'], 0)
        self.beta_ = (stats['bvec'] - stats['bvecmean']) / bmatrix_mask
        self.beta_ = self.beta_.filled(fill_value = 0)

    if 'z' in self.params:
      garbagedenom = stats['garbagepost']
      self.prior_z = np.sum(stats['post'])/stats['totobs']

      if 'm' in self.params:
        self.garbage_means_[0, :] = ((stats['garbageobs'])
                              / (garbagedenom))
      if 'c' in self.params:
        covars_prior = self.covars_prior
        cv_num = (stats['garbageobs**2']
                  - 2 * self.garbage_means_ * stats['garbageobs']
                  + self.garbage_means_**2 * garbagedenom)
        cv_den = garbagedenom
        self.garbage_covars_[0, :, :] = \
            np.diag(np.squeeze((covars_prior + cv_num) \
              / np.maximum(cv_den, 1e-5)))

    if i > self.n_iter / 2:
      if 'c' in self.params:
        covars_prior = self.covars_prior
        covars_weight = self.covars_weight
        meandiff = self.means_ - means_prior

        if self.covariance_type in ('spherical', 'diag'):
          cv_num = (means_weight * meandiff**2
                    + stats['obs**2']
                    - 2 * self.means_ * stats['obs']
                    + self.means_**2 * denom)
          if self.n_batches > 1:
            cv_num = (cv_num
                    - 2 * stats['mbeta']
                    + 2 * self.means_ * stats['bbeta']
                    + stats['bbeta**2'])
          cv_den = max(covars_weight - 1, 0) + denom
          self._covars_ = \
            (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)

          self._covars_ = np.max(
            np.stack((self._covars_, 
                      self.min_covar*np.ones(self._covars_.shape)), 
                      axis=-1), axis=-1)
          if self.covariance_type == 'spherical':
            self._covars_ = np.tile(
              self._covars_.mean(1)[:, np.newaxis],
              (1, self._covars_.shape[1]))
        elif self.covariance_type in ('tied', 'full'):
          raise ValueError('uninmplemented')

def train_batch_models(hmm_train_fn, 
                       X_train, 
                       lengths_train, 
                       batches_train, 
                       ks, 
                       model_dir=None):
  """Train different models 

  Args:
      hmm_train_fn (function): function to train
      X_train (ndarray): flattened training data
      lengths_train (list): list of lengths of each trace
      batches_train (list): contains batch ids of the train data
      ks (ndarray): numpy range for model order
      model_dir (str, optional): folder to store models. Defaults to None.

  Returns:
      hmm_models (list): list of training models for different model
      orders
  """
  hmm_models = []
  for k in ks:
    cached_hmm_model_filename = f'{model_dir}/hmm_model_{k}.pkl'
    if not os.path.isfile(cached_hmm_model_filename):
      hmm_model = hmm_train_fn(k, X_train, lengths_train, batches_train)
      with open(cached_hmm_model_filename, 'wb') as f:
        pickle.dump(hmm_model, f)
    else:
      with open(cached_hmm_model_filename, 'rb') as f:
        hmm_model = pickle.load(f)
    print(f'Saved {cached_hmm_model_filename}.')
    hmm_models.append(hmm_model)
  return hmm_models


def score_all_models(hmm_models, 
                     X_validation, 
                     lengths_validation, 
                     batches_validation, 
                     n_processes):
  """Function to score all models

  Args:
      hmm_models (list): contains list of objects
      X_validation (ndarray): flattened data matrix
      lengths_validation (list): list containing length of each trace
      batches_validation (list): batch ids of traces in the val set
      n_processes (int, optional): number of parallel process

  Returns:
      mean_log_probs (list): mean value of log-probabilities
      serr_log_probs (list): std error of log-probabilities
      hmm_models (lsit): contains list of objects
  """
  mean_log_probs = []
  serr_log_probs = []
  best_log_prob = -np.inf
  for hmm_model in hmm_models:
    print(f'Scoring a model with {hmm_model.n_components} components.')
    log_prob = hmm_model.score_parallel_per_sample(X_validation,
                                                   lengths=lengths_validation, 
                                                   batch_ids=batches_validation,
                                                   n_processes=n_processes)
    mean_log_probs.append(np.mean(log_prob))
    serr_log_probs.append(np.std(log_prob)/np.sqrt(len(lengths_validation)))
  return mean_log_probs, serr_log_probs, hmm_models