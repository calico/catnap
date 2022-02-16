from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd

from datetime import  timedelta
from tqdm import tqdm
from hmmlearn.utils import iter_from_X_lengths

import matplotlib.pyplot as plt
import seaborn as sns

# from catnap.data_processing.prepare_hmm_data import HMM_FEATURES

HMM_FEATURES = [
    'VO2', 'VCO2', 'VH2O', 'KCal_hr', 'RQ', 'Food', 'Water', 'PedMeters',
    'WheelMeters', 'XBreak', 'YBreak', 'ZBreak'
]

def str_to_bool(value):
  """Convert string to bool for arg parsers

  Args:
      value (str): value of the boolean

  Raises:
      ValueError: if the value does not indicate bool type

  Returns:
      (boolean): convert the string to boolean
  """
  if value.lower() in {'false', 'f', '0', 'no', 'n'}:
    return False
  elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
    return True
  raise ValueError(f'{value} is not a valid boolean value')


def flatten_traces(all_traces):
  """Flatten traces

  Args:
      all_traces (list): list of all traces

  Returns:
      (dataframe): traces stacked on over other (flattened)
  """
  lengths = [trace.shape[0] for trace in all_traces]
  return pd.concat(all_traces, axis=0, sort=True), np.array(lengths)


def nancov(M):
  """Compute nan covariance

  Args:
      M (matrix): covariance matrix

  Returns:
      (masked array): covariance matrix
  """
  masked_arr = np.ma.array(M, mask=np.isnan(M))
  return np.ma.cov(masked_arr, rowvar=0, allow_masked=True, ddof=1).data


def turn_constant_zero_to_missing(traces, modeled_features):
  """If a trace has constant zero or small values, turn them to nan values

  Args:
      traces (list): list of all traces
      modeled_features (list): list of features
  """
  for i, trace in enumerate(traces):
    for c in modeled_features:
      try:
          idx = ~np.isnan(trace[c].astype('float'))
          if np.all(np.abs(trace[c][idx])<1e-6):
            traces[i][c] = np.NAN
      except Exception as e:
          print(i, c, e)


def populate_missing(trace, 
                     cutoff=timedelta(minutes=3), 
                     features=[], 
                     tol=timedelta(minutes=1.5)):
  """Populate missing with nans

  Args:
      trace (ndarray): Trace of metcage run
      cutoff (numpy timedelta64, optional): sampling time interval. 
                                            Defaults to timedelta(minutes=3).
      features (list, optional): feature selected. Defaults to [].
      tol (numpy timedelta64, optional): tolerance value of timedelta
                                         Defaults to timedelta(minutes=1.5).

  Raises:
      ValueError: overshot

  Returns:
      trace (pandas): dataframe
  """
  fmt = '%Y-%m-%d %H:%M:%S'
  # last = datetime.strptime(trace.iloc[0]['time'], fmt)
  last = trace.iloc[0]['time']
  missing = []
  for i, row in trace.iterrows():
    # current = datetime.strptime(row['time'], fmt)
    current = row['time']
    if last > current:
      raise ValueError('overshot')
    while (current - last) > cutoff + tol:
      last += cutoff
      arow = row.copy()
      arow['time'] = last  # .strftime(fmt)
      arow[features] = np.nan
      missing.append(arow)
    last = current
  if len(missing):
    trace = pd.concat((trace, pd.concat(missing,axis=1).T))
    trace = trace.sort_values(by='time')
  return trace

def select_model(scored_models):
  """Select best model using one-sigma rule

  Args:
      scored_models (list): list of scored models

  Returns:
      best_model (object): best model selected using one-sigma rule
  """
  best_mean = -np.inf
  for mean, serr, model in zip(*scored_models):
    if mean - serr > best_mean:
      best_mean = mean
      best_model = model
    else:
      return best_model


def add_states_residuals_posterior(model, traces, batch_ids):
  r"""Add the state and residual columns to every trace

  Args:
      hmm_model: The best hmm model along with its parameters
      test_traces: List of traces to be appended with states and residuals
      batch_ids_test: Batch ids of the traces

  Returns:
      A list of trace ids appended with residuals and states
  """

  _, lengths = flatten_traces(traces)

  # for all batches
  for i in tqdm(np.arange(np.max(batch_ids)+1)):

    # select batch id
    batch_id_interest = i

    # update means and covars
    if batch_id_interest == 0:
      m = model.means_
    else:
      m = model.means_ + model.beta_[batch_id_interest-1, :]
    s = np.stack([np.sqrt(np.diag(c)) for c in model.covars_], axis=0)

    # identify indicies of batch id interest
    ind_batch_ids = np.argwhere(np.array(batch_ids) == batch_id_interest)

    # append traces for the identified batches
    traces_selected = []
    for j in range(np.shape(ind_batch_ids)[0]):
      traces_selected.append(traces[ind_batch_ids[j, 0]])

    # flatten traces
    X_selected, _ = flatten_traces(traces_selected)
    X_selected = X_selected[HMM_FEATURES]

    # lengths of the selected traces
    lengths_selected = np.array(lengths)[np.squeeze(ind_batch_ids)]
    if lengths_selected.size == 1:
      lengths_selected = np.expand_dims(lengths_selected, axis=0)

    assert np.shape(X_selected)[0] == np.sum(lengths_selected)

    # batch ids of the selcted traces
    batch_ids_selected = np.array(batch_ids)[np.squeeze(ind_batch_ids)]
    if batch_ids_selected.size == 1:
      batch_ids_selected = np.expand_dims(batch_ids_selected, axis=0)

    # decoding algorithm
    _, h = model.decode(X=X_selected, 
                        lengths=lengths_selected, 
                        batch_ids=batch_ids_selected)
    z = model.predict_z(X=X_selected, 
                        lengths=lengths_selected, 
                        batch_ids=batch_ids_selected)
    _, posteriors = model.score_samples(X=X_selected, 
                                        lengths=lengths_selected, 
                                        batch_ids=batch_ids_selected)
    
    for (p, q), j in zip(iter_from_X_lengths(X_selected, 
                                             lengths_selected), 
                                             range(np.shape(ind_batch_ids)[0])):

      # add residuals, posteriors for all channels
      for r, feature in enumerate(X_selected.columns):
        X = X_selected[p:q]
        
        # add residual
        traces[ind_batch_ids[j, 0]]['res_' + feature] = X[feature]-m[h[p:q], r]
        
        # add posterior
        z_val = np.zeros((q-p,))
        for c in range(np.max(h)+1):
          ind = np.where(h[p:q] == c)
          z_sel = z[p:q, r, c]
          z_val[ind] = z_sel[ind]
        traces[ind_batch_ids[j, 0]]['pos_' + feature] = z_val

      for r in range(model.n_components):
        traces[ind_batch_ids[j, 0]]['pos_state_' + str(r)] = posteriors[p:q, r]

      # add states and batch ids
      traces[ind_batch_ids[j, 0]]['states'] = h[p:q]
      traces[ind_batch_ids[j, 0]]['batch_id'] = [i] * (q-p)
      traces[ind_batch_ids[j, 0]] = \
        traces[ind_batch_ids[j, 0]].reset_index(drop=True)
    
  return traces

def plot_atrace(atrace, h=None, z=None, m=None, s=None, fig_scale=1):
  """Plot a single trace

  Args:
      atrace (ndarray): all channels of metcage for one trace
      h (ndarray, optional): assigned latent state. Defaults to None.
      z (ndarray, optional): data good or garbage. Defaults to None.
      m (ndarray, optional): matrix containing means. Defaults to None.
      s (ndarray, optional): matrix containing covariances. Defaults to None.
      fig_scale (int, optional): figure scale. Defaults to 1.
  """
  colors = sns.color_palette(n_colors=z.shape[2])
  n = np.ceil(np.sqrt(len(atrace.columns))).astype('int')
  num_plots = len(atrace.columns)
  plt.figure(figsize=(fig_scale*10, fig_scale*1 * num_plots))

  for i, f in enumerate(atrace.columns):
    plt.subplot(num_plots, 1, i + 1)
    if m is not None and s is not None:
      y1 = np.maximum(0, m[h,i] - 2*s[h,i])
      y2 = m[h,i] + 2*s[h,i]
      plt.fill_between(range(len(h)), y1, y2, alpha=0.1)
    if z is not None:
      for c in range(np.max(h)+1):
        ind = np.logical_and((h==c), z[:,i,c]>=0.5)
        plt.plot(np.where(ind)[0], atrace[ind][f],'.', label=c, c=colors[c])
        ind = np.logical_and((h==c), z[:,i,c]<0.5)
        plt.plot(np.where(ind)[0], atrace[ind][f],'o', 
                          label=c, markerfacecolor='none',
                          markeredgecolor=colors[c], markersize=10)
    else:
      plt.plot(atrace[f], '.')
    plt.title(f)
  plt.legend()
  plt.show()

  return

def plot_scores_all_models(scored_models):
  """Plot model selection scores

  Args:
      scored_models (list): contains log-likelihood values
  """
  mean_log_probs, serr_log_probs, hmm_models = scored_models
  mean_log_probs = np.asarray(mean_log_probs)
  serr_log_probs = np.asarray(serr_log_probs)
  ks = [hmm_model.n_components for hmm_model in hmm_models]
  plt.plot(ks, mean_log_probs - serr_log_probs, c='b',
    label='mean +/- std.err. of log prob')
  plt.plot(ks, mean_log_probs, c='r', label='mean log prob')
  plt.plot(ks, mean_log_probs + serr_log_probs, c='b', label='_nolegend_')
  plt.gca().set_xticks(ks)
  plt.show()
  
  return