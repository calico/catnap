import os
import numpy as np
import pandas as pd
import tqdm
import pickle

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from datetime import datetime
from datetime import  timedelta
from itertools import chain
from multiprocessing import Pool, Value

import hmmlearn
import hmmlearn.hmm
from hmmlearn.utils import iter_from_X_lengths
from hmmlearn.base import DECODER_ALGORITHMS

import matplotlib.pyplot as plt
import seaborn as sns

REALMAX = np.finfo(np.float64).max
DEFAULT_N_PROCESSES = 8


def _log_densities_diag(X, 
                        mean, 
                        covar):
  """Compute log-likelihood of a Gaussian distrbution

  Args:
      X (ndarray): sample points
      mean (ndarray): mean vector
      covar (ndarray): covariance matrix

  Returns:
      val (ndarray): log-likelihood of a Gaussian distribution
  """

  val = -0.5*(np.log(2*np.pi) + np.log(covar)
              + (mean ** 2) / covar
              - 2 * (X * (mean / covar))
              + (X ** 2) * (1.0 / covar))
  val[np.isnan(val)] = 0
  return val

def _log_densities_diag_components(X, 
                                   means, 
                                   covars):
  """Compute log-likelihood of Gaussian distrbution with the assumption 
  that the covariance matrix is diagonal across all features

  Args:
      X (ndarray): sample points
      means (ndarray): matrix of means with columns representing the 
      features/channels
      covars (ndarray): 3d matrix of covariances with the first dimention
      representing feature/channel

  Returns:
      lpr (ndarray): the log-likelihood matrix
  """

  n_samples, n_dim = X.shape
  n_components = means.shape[0]
  lpr = np.zeros((n_samples, n_dim, n_components))
  for c in range(n_components):
    lpr[:, :, c] = _log_densities_diag(X, means[c, :], np.diag(covars[c, :, :]))
  return lpr

class GaussianPerFeatureGarbageHMM(hmmlearn.hmm.GaussianHMM):
  """HMM with garbage components

  Args:
      hmmlearn (class): base class of a Gaussian distributed 
      hmmlearn package
  """

  def set_garbage_component(self, 
                            means_, 
                            covars_, 
                            prior_z):
    """Set garbage model parameters

    Args:
        means_ (ndarray): mean of garbage component shape
        (n_garbage_components, n_features)
        covars_ (ndarray): diag covariance matrix of garbage components shape
        (n_garbage_components, n_features)
        prior_z (ndarray): prior probability that an observation is not garbage
    """

    # initialize means and covariances
    self.garbage_means_ = means_.copy()
    self.garbage_covars_ = covars_.copy()
    # diagonal covariance matrix
    assert means_.shape[0] == covars_.shape[0]
    assert means_.shape[1] == covars_.shape[1]
    # measurement quality prior
    self.prior_z = prior_z
    # initialize garbage components
    self.n_garbage_components = means_.shape[0]


  def _init(self, 
            X, 
            lengths=None):
    """Make the GaussianPerFeatureGarbageHMM super class

    Args:
        X (ndarray): flattened data array
        lengths (list, optional): list containing lengths of each trace. 
        Defaults to None.
    """

    X = X[np.where(np.all(~np.isnan(X), axis=1))[0], :]
    super(GaussianPerFeatureGarbageHMM, self)._init(X, lengths=lengths)


  def _log_marginalize_z(self, 
                         log_joint_z0s, 
                         log_joint_z1):
    """Compute marginal probability for each sample over indicator whether 
       data is garbage or real

    Args:
        log_joint_z0s (ndarray): a list of n_garbage_components arrays, each of 
        shape (n_samples, n_features) storing log probability of each sample's
        observations under the corresponding garbage component model.
        log_joint_z1 (ndarray): array-like, shape (n_samples, n_features, 
        n_components) log probability of each sample's observations under the
        corresponding real data model's components

    Returns:
        log_marginal_prob (ndarray): shape (n_samples, n_features, n_components 
        marginal log probability of each observation given component log p(x_ij 
        | h_i = c)

        Remark: The inputs are not symmetric, we marginalize over garbage 
        components but not over the real data components.
    """
 
    m = log_joint_z1
    n_garbage_components = len(log_joint_z0s)
    for log_joint_z0 in log_joint_z0s:
      m = np.maximum(log_joint_z0, m)
    ret = self.prior_z*np.exp(log_joint_z1 - m)
    for log_joint_z0 in log_joint_z0s:
      ret += (1.-self.prior_z)/n_garbage_components*np.exp(log_joint_z0 - m)
    return np.log(ret) + m


  def _compute_log_likelihood_complete(self, X):
    """Compute log-likelihoods accounting for garbage model.

    Args:
        X (ndarray): flattened data aray

    Returns:
        log_prob (ndarray): total log-likelihood after marginalizing over data 
        quality variable
        log_garbage (ndarray): total log-likelihood of the garbage components
        log_appearance (ndarray): total log-likelihood of the data with good 
        quality
    """

    log_appearance = _log_densities_diag_components(X, 
                                                    self.means_, 
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


  def _compute_log_likelihood(self, X):
    """Function to compute the complete log-likelihood

    Args:
        X (ndarray): flattened data aray

    Returns:
        log_prob (ndarray): total log-likelihood after marginalizing over data 
        quality variable
    """

    log_prob, _, _ = self._compute_log_likelihood_complete(X)
    return log_prob


  def _compute_posteriors_complete(self, 
                                   fwdlattice, 
                                   bwdlattice,
                                   framelogmarginalprob, 
                                   frameloggarbage, 
                                   framelogappearance):
    """Compute the posteriors of the hidden states and data quality indicator 
    variables

    Args:
        fwdlattice (ndarray): forward probabilities
        bwdlattice (ndarray): backward probabilities
        framelogmarginalprob (ndarray): complete frame log-marginal probs
        frameloggarbage (ndarray): frame log probabilities of garbage comps
        framelogappearance (ndarray): frame log probabilities of good data

    Returns:
        posteriors (ndarray): posterior probability of hidden states
        posteriorz1 (ndarray): posterior probability of data quality indicator 
    """

    posteriors = \
      super(GaussianPerFeatureGarbageHMM, self)._compute_posteriors(fwdlattice,
                                                                    bwdlattice)
    log_marginal_z = np.expand_dims(fwdlattice + bwdlattice, axis=1)

    log_joint_appearance = log_marginal_z - \
      np.expand_dims(framelogmarginalprob, axis=1) + framelogappearance

    log_joint_garbage = []
    for frameloggarbagecomponent in frameloggarbage:
      log_joint_garbage.append(log_marginal_z - \
                               np.expand_dims(framelogmarginalprob, axis=1) + \
                               frameloggarbagecomponent)
    log_joint = self._log_marginalize_z(log_joint_garbage, log_joint_appearance)
    posteriorz1 = self.prior_z*np.exp(log_joint_appearance - log_joint)
    s = posteriorz1
    for log_joint_garbage_component in log_joint_garbage:
      prior_z0 = (1 - self.prior_z)/len(log_joint_garbage)
      s = s + prior_z0*np.exp(log_joint_garbage_component - log_joint)
    np.testing.assert_almost_equal(s, 1)
    return posteriors, posteriorz1

  def decode(self, 
             X, 
             lengths=None, 
             algorithm=None):
    """Find most likely state sequence corresponding to X.

    Args:
        X (ndarray): flattended data (n_samples x n_features)
        lengths (ndarray, optional): Lengths of the individual sequences in X. 
        The sum of these should be n_samples. Defaults to None.
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
    for i, j in iter_from_X_lengths(X, lengths):
      # XXX decoder works on a single sample at a time!
      logprobij, state_sequenceij = decoder(X[i:j])
      logprob += logprobij
      state_sequence[i:j] = state_sequenceij

    return logprob, state_sequence

  def _compute_sufficient_statistcs(self, 
                                    X, 
                                    lengths):
    """Compute sufficient stats

    Args:
        X (ndarray): flattended data (n_samples x n_features)
        lengths (ndarray, optional): Lengths of the individual sequences in X. 
        The sum of these should be n_samples. Defaults to None.

    Returns:
        stats (dict): dictionary containing sufficient stats
        curr_logprob (ndarray): log-prob obtained by forward + backward pass
    """

    stats = self._initialize_sufficient_statistics()
    curr_logprob = 0
    for i, j in iter_from_X_lengths(X, lengths):
        framelogmarginalprob, frameloggarbage, \
          framelogappearance = self._compute_log_likelihood_complete(X[i:j])
        logprob, fwdlattice = self._do_forward_pass(framelogmarginalprob)
        curr_logprob += logprob
        bwdlattice = self._do_backward_pass(framelogmarginalprob)
        posteriorh, posteriorz = self._compute_posteriors_complete(
          fwdlattice, bwdlattice, framelogmarginalprob, 
          frameloggarbage, framelogappearance)

        self._accumulate_sufficient_statistics(
          stats, X[i:j], framelogmarginalprob, posteriorh, posteriorz, \
            fwdlattice, bwdlattice)
    return stats, curr_logprob

  def fit(self, 
          X, 
          lengths=None):
    """Estimate model parameters.

    Args:
        X (ndarray): flattended data (n_samples x n_features)
        lengths (ndarray, optional): Lengths of the individual sequences in X. 
        The sum of these should be n_samples. Defaults to None.

    Returns:
        self (object): returns self with updated parameters
    """

    X = check_array(X, force_all_finite=False)
    self._init(X, lengths=lengths)
    #self._check()

    self.monitor_._reset()
    for iter in range(self.n_iter):
      stats, curr_logprob = self._compute_sufficient_statistcs(X, lengths)
      # XXX must be before convergence check, because otherwise
      #     there won't be any updates for the case ``n_iter=1``.
      self._do_mstep(stats)

      self.monitor_.report(curr_logprob)
      if self.monitor_.converged:
        break

    return self

  def _aggregate_sufficient_statistics(self, 
                                       batched_stats):
    """Aggregate complete sufficient stats

    Args:
        batched_stats (dict): batch of complete stats

    Returns:
        aggregated_stats (dict): aggregated statistics obtained by adding 
        complete statistics of each trace
    """

    aggregated_stats = None
    for stats in batched_stats:
      if aggregated_stats is None:
        aggregated_stats = stats.copy()
      else:
        for k in stats.keys():
          aggregated_stats[k] += stats[k]
    return aggregated_stats

  def _split_X_and_lengths(self, 
                           X, 
                           lengths, 
                           n_parts):
    """Split flattened data

        X (ndarray): flattended data (n_samples x n_features)
        lengths (ndarray, optional): Lengths of the individual sequences in X. 
        The sum of these should be n_samples. Defaults to None.
        n_parts (int): for parallel processing

    Returns:
        X_parts (list): flattened data split into parts
        length_parts (list): length of each trace corresponding to X_parts
    """

    part = np.sum(lengths)/n_parts
    lengths_parts = []
    X_parts = []
    start = 0
    end = 0
    lengths_part = []
    for i, l in enumerate(lengths):
      end = end + l
      lengths_part.append(l)
      if end - start >= part or i == len(lengths)-1:
        X_part = X[start:end]
        X_parts.append(X_part)
        lengths_parts.append(lengths_part)
        assert(np.sum(lengths_part) == X_part.shape[0])
        lengths_part = []
        start = end

    assert sum([sum(lengths_part) for lengths_part in lengths_parts]) \
      == sum(lengths)
    assert end == sum(lengths)
    return X_parts, lengths_parts


  def fit_parallel(self, 
                   X, 
                   lengths=None, 
                   n_processes=DEFAULT_N_PROCESSES):
    """Estimate model parameters

    Args:
        X (ndarray): flattended data (n_samples x n_features)
        lengths (ndarray, optional): Lengths of the individual sequences in X. 
        The sum of these should be n_samples. Defaults to None.
        n_processes (int, optional): number of parallel proces. Defaults to 
        DEFAULT_N_PROCESSES.

    Returns:
        self (object): returns self with updated parameters
    """

    X = check_array(X, force_all_finite=False)
    self._init(X, lengths=lengths)

    X_parts, lengths_parts = \
      self._split_X_and_lengths(X, lengths, n_processes)
    self.monitor_._reset()
    with Pool(n_processes) as pool:
      for iter in range(self.n_iter):
        batched_stats_and_log_probs = pool.starmap(
          self._compute_sufficient_statistcs, zip(X_parts, lengths_parts))
        batched_stats, log_probs = zip(*batched_stats_and_log_probs)
        stats = self._aggregate_sufficient_statistics(batched_stats)
        curr_logprob = np.sum(log_probs)
        # XXX must be before convergence check, because otherwise
        #     there won't be any updates for the case ``n_iter=1``.
        self._do_mstep(stats)

        self.monitor_.report(curr_logprob)
        if self.monitor_.converged:
            break
    return self


  def score(self, 
            X, 
            lengths=None):
    """Compute the log probability under the model aka score

    Args:
        X (ndarray): flattended data (n_samples x n_features)
        lengths (ndarray, optional): Lengths of the individual sequences in X. 
        The sum of these should be n_samples. Defaults to None.

    Returns:
        logprob (float): log-likelihood of the input data sequence
    """

    check_is_fitted(self, "startprob_")
    #self._check()

    X = check_array(X, force_all_finite=False)
    # XXX we can unroll forward pass for speed and memory efficiency.
    logprob = 0
    for i, j in iter_from_X_lengths(X, lengths):
        framelogprob = self._compute_log_likelihood(X[i:j])
        logprobij, _fwdlattice = self._do_forward_pass(framelogprob)
        logprob += logprobij
    return logprob


  def score_parallel(self, 
                     X, 
                     lengths=None, 
                     n_processes=DEFAULT_N_PROCESSES):
    """Score in parallel

    Args:
        X (ndarray): flattended data (n_samples x n_features)
        lengths (ndarray, optional): Lengths of the individual sequences in X. 
        The sum of these should be n_samples. Defaults to None.
        n_processes (int, optional): number of parallel proces. Defaults to 
        DEFAULT_N_PROCESSES.

    Returns:
        logprob (float): log-likelihood of the input data sequence
    """
    if lengths is not None:
      logprobs = self.score_parallel_per_sample(X, lengths, n_processes)
      return np.sum(logprobs)
    else:
      return self.score(X)


  def score_per_sample(self, 
                       X, 
                       lengths):
    """Compute the log probability under the model aka score
    per sample

    Args:
        X (ndarray): flattended data (n_samples x n_features)
        lengths (ndarray, optional): Lengths of the individual sequences in X. 
        The sum of these should be n_samples. Defaults to None.

    Returns:
        logprobs (list): list of log-probability of every trace
    """

    check_is_fitted(self, "startprob_")
    #self._check()

    X = check_array(X, force_all_finite=False)
    # XXX we can unroll forward pass for speed and memory efficiency.
    logprobs = []
    for i, j in iter_from_X_lengths(X, lengths):
      framelogprob = self._compute_log_likelihood(X[i:j])
      logprobij, _fwdlattice = self._do_forward_pass(framelogprob)
      logprobs.append(logprobij)
    return logprobs


  def score_samples(self, 
                    X, 
                    lengths=None):
    """Compute the log probability under the model and compute posteriors

    Args:
        X (ndarray): flattended data (n_samples x n_features)
        lengths (ndarray, optional): Lengths of the individual sequences in X. 
        The sum of these should be n_samples. Defaults to None.

    Returns:
        logprob (float): log-likelihood of the input data sequence posteriors 
        posteriors (ndarray): posterior probability of being in a state
    """

    check_is_fitted(self, "startprob_")
    self._check()

    #X = check_array(X)
    n_samples = X.shape[0]
    logprob = 0
    posteriors = np.zeros((n_samples, self.n_components))
    for i, j in iter_from_X_lengths(X, lengths):
      framelogprob = self._compute_log_likelihood(X[i:j])
      logprobij, fwdlattice = self._do_forward_pass(framelogprob)
      logprob += logprobij

      bwdlattice = self._do_backward_pass(framelogprob)
      posteriors[i:j] = self._compute_posteriors(fwdlattice, bwdlattice)

    return logprob, posteriors

  def score_parallel_per_sample(self, 
                                X, 
                                lengths, 
                                n_processes=DEFAULT_N_PROCESSES):
    """Score sample in parallel

    Args:
        X (ndarray): flattended data (n_samples x n_features)
        lengths (ndarray, optional): Lengths of the individual sequences in X. 
        The sum of these should be n_samples. Defaults to None.
        n_processes (int, optional): number of parallel proces. Defaults to 
        DEFAULT_N_PROCESSES.

    Returns:
        logprobs (list): list of log-probability of every trace
    """

    check_is_fitted(self, "startprob_")
    #self._check()

    X = check_array(X, force_all_finite=False)
    X_parts, lengths_parts = self._split_X_and_lengths(X, lengths, n_processes)
    # XXX we can unroll forward pass for speed and memory efficiency.
    logprob = 0
    with Pool(n_processes) as pool:
      batched_log_probs = pool.starmap(
        self.score_per_sample, zip(X_parts, lengths_parts))
      logprob = np.sum(batched_log_probs)
    return list(chain.from_iterable(batched_log_probs))

  def predict_z(self, 
                X, 
                lengths=None):
    """Predict for each observation whether it is real for a particular 
    component.

    Args:
        X (ndarray): flattended data (n_samples x n_features)
        lengths (ndarray, optional): Lengths of the individual sequences in X. 
        The sum of these should be n_samples. Defaults to None.

    Returns:
        zs (ndarray): shape (n_samples, n_features, n_components). For each 
        sample, given that it came from a particular component, what is the 
        probability that it is real data: p(z_{i,j}=1 | x_i, h_i=c)
    """

    X = check_array(X, force_all_finite=False)
    self._check()
    zs = np.zeros([*X.shape, self.n_components])

    for i, j in iter_from_X_lengths(X, lengths):
      framelogmarginalprob, frameloggarbage, framelogappearance = self._compute_log_likelihood_complete(X[i:j])
      assert np.all(np.isfinite(framelogmarginalprob))
      logprob, fwdlattice = self._do_forward_pass(framelogmarginalprob)
      assert np.all(np.isfinite(fwdlattice))
      assert np.all(np.isfinite(logprob))
      bwdlattice = self._do_backward_pass(framelogmarginalprob)
      posteriorh, posteriorz = \
        self._compute_posteriors_complete(fwdlattice, 
                                          bwdlattice, 
                                          framelogmarginalprob, frameloggarbage, 
                                          framelogappearance)
      idx = np.any(~np.isnan(X[i:j]), axis=1)
      z = np.zeros([*X[i:j].shape, self.n_components])
      z[idx, :, :] = posteriorz[idx,:,:]
      zs[i:j] = z
    return zs


  def _initialize_sufficient_statistics(self):
    """Initialize sufficient stats

    Raises:
        ValueError: if the covariance type is "full" or "tied", raise value 
        error

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
    if self.covariance_type in ('tied', 'full'):
      stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
                                     self.n_features))
      raise ValueError('unsupported')
      
    return stats


  def _accumulate_sufficient_statistics(self, 
                                        stats, 
                                        obs, 
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
        ValueError: if the covariance type is "full" or "tied", raise error
    """

    super(hmmlearn.hmm.GaussianHMM, self)._accumulate_sufficient_statistics(
        stats, obs, framelogprob, posteriorh, fwdlattice, bwdlattice)


    omask = np.isnan(obs) # n x d
    zmask = np.repeat(np.expand_dims(omask, axis=2), \
      self.n_components, axis=2) # n x d x c
    hmask = np.repeat(np.expand_dims(np.all(omask, axis=1), axis=1), \
      self.n_components, axis=1) # n x c

    mobs = np.ma.masked_array(obs, omask)
    mposteriorh = np.ma.masked_array(posteriorh, hmask)
    mposteriorz = np.ma.masked_array(posteriorz, zmask)

    if 'z' in self.params:
      stats['totobs'] += np.sum(~omask)
      postz0 = np.expand_dims(mposteriorh, axis=1)*(1.0-mposteriorz)
      stats['garbagepost'] += np.sum(postz0, axis=(0,2)).T

      if 'm' in self.params or 'c' in self.params:
        stats['garbageobs'] += \
          np.sum(postz0*np.expand_dims(mobs, axis=2),axis=(0,2)).T
      if 'c' in self.params:
        stats['garbageobs**2'] += \
          np.sum(postz0*np.expand_dims(mobs**2.0, axis=2), axis=(0,2)).T

    if 'm' in self.params or 'c' in self.params:
      postz1 = np.expand_dims(mposteriorh, axis=1)*mposteriorz
      stats['post'] += np.sum(postz1, axis=0).T
      stats['obs'] += np.sum(postz1*np.expand_dims(mobs, axis=2), axis=0).T

    if 'c' in self.params:
      if self.covariance_type in ('spherical', 'diag'):
        stats['obs**2'] += \
          np.sum(postz1*np.expand_dims(mobs**2.0, axis=2), axis=0).T
      elif self.covariance_type in ('tied', 'full'):
        raise ValueError('unsupported')


  def _do_mstep(self, stats):
    """M-step of the EM algorithm

    Args:
        stats (dict): contains sufficient statistics

    Raises:
        ValueError: if the covariance type is "full" or "tied", raise error
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
      self.means_ = ((means_weight * means_prior + stats['obs'])
                     / (means_weight + denom))

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
          np.diag(np.squeeze((covars_prior + cv_num)/np.maximum(cv_den, 1e-5)))

    if 'c' in self.params:
      covars_prior = self.covars_prior
      covars_weight = self.covars_weight
      meandiff = self.means_ - means_prior

      if self.covariance_type in ('spherical', 'diag'):
        cv_num = (means_weight * meandiff**2
                  + stats['obs**2']
                  - 2 * self.means_ * stats['obs']
                  + self.means_**2 * denom)
        cv_den = max(covars_weight - 1, 0) + denom
        self._covars_ = \
            (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)

        self._covars_ = np.max(
          np.stack((self._covars_, 
                    self.min_covar*np.ones(self._covars_.shape)),
                    axis=-1), axis=-1)
        if self.covariance_type == 'spherical':
          self._covars_ = np.tile(self._covars_.mean(1)[:, np.newaxis],
                                  (1, self._covars_.shape[1]))
      elif self.covariance_type in ('tied', 'full'):
        raise ValueError('uninmplemented')

def score_all_models(hmm_models, 
                     X_validation, 
                     lengths_validation, 
                     n_processes):
  """Function to score all models

  Args:
      hmm_models (list): contains list of objects
      X_validation (ndarray): flattened data matrix
      lengths_validation (list): list containing length of each trace
      n_processes (int, optional): number of parallel process 

  Returns:
      mean_log_probs (list): mean value of log-probabilities
      serr_log_probs (list): std error of log-probabilities
      hmm_models (lsit): contains list of objects
  """
  mean_log_probs = []
  serr_log_probs = []
  # best_log_prob = -np.inf
  for hmm_model in hmm_models:
    print(f'Scoring a model with {hmm_model.n_components} components.')
    log_prob = hmm_model.score_parallel_per_sample(
      X_validation, lengths=lengths_validation, n_processes=n_processes)
    mean_log_probs.append(np.mean(log_prob))
    serr_log_probs.append(np.std(log_prob)/np.sqrt(len(lengths_validation)))
  return mean_log_probs, serr_log_probs, hmm_models

def score_all_samples_by_all_models(hmm_models, 
                                    X_validation, 
                                    lengths_validation, 
                                    n_processes):
  """Function to score all models

  Args:
      hmm_models (list): contains list of objects
      X_validation (ndarray): flattened data matrix
      lengths_validation (list): list containing length of each trace
      n_processes (int, optional): number of parallel proces. 

  Returns:
      logprobs (ndarray): log probability for each model
  """
  log_probs = np.zeros((len(hmm_models), len(lengths_validation)))
  for i, hmm_model in enumerate(hmm_models):
    print(f'Scoring a model with {hmm_model.n_components} components.')
    log_prob = hmm_model.score_parallel_per_sample(
      X_validation, lengths=lengths_validation, n_processes=n_processes)
    log_probs[i, :] = log_prob
  return log_probs

def train_models(hmm_train_fn, 
                 X_train, 
                 lengths_train, 
                 ks, 
                 model_dir=None):
  """Train different models 

  Args:
      hmm_train_fn (function): function to train
      X_train (ndarray): flattened training data
      lengths_train (list): list of lengths of each trace
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
      hmm_model = hmm_train_fn(k, X_train, lengths_train)
      with open(cached_hmm_model_filename, 'wb') as f:
        pickle.dump(hmm_model, f)
    else:
      with open(cached_hmm_model_filename, 'rb') as f:
        hmm_model = pickle.load(f)
    print(f'Saved {cached_hmm_model_filename}.')
    hmm_models.append(hmm_model)
  return hmm_models

