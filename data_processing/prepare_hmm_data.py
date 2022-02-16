from __future__ import absolute_import
from __future__ import print_function

import os
import time
import argparse
import datetime
import numpy as np
import pandas as pd
import pickle
import collections

from functools import partial
from multiprocessing import Pool
import pymysql.cursors

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from catnap.robust_hmm.utils import turn_constant_zero_to_missing
from catnap.robust_hmm.utils import populate_missing
from catnap.data_processing import preprocess_traces


DB_NAME = 'calico_catnap_2020_03_09_QCed_v2'

TRACES_PER_ANALYZER = 8
MIN_TRACES_PER_GA = 3
SEED = 42
BATCH = True
TRAIN_FRACTION = 0.9

ALL_FEATURES = [
    'VO2', 'VCO2', 'VH2O', 'KCal_hr', 'RQ', 'Food', 'Water', 'BodyMass',
    'PedSpeed', 'PedMeters', 'PedMeters_Rwc', 'AllMeters', 'AllMeters_Rwc',
    'WheelMeters', 'WheelSpeed', 'XBreak', 'YBreak', 'ZBreak', 'age_in_months',
    'time_period'
]

HMM_FEATURES = [
    'VO2', 'VCO2', 'VH2O', 'KCal_hr', 'RQ', 'Food', 'Water', 'PedMeters',
    'WheelMeters', 'XBreak', 'YBreak', 'ZBreak'
]


def parse_args():
  """Argument parser function.

  Returns:
      args (object): contains info about the arguments
  """
  parser = argparse.ArgumentParser(
      description="Downloads, process, and save CATNAP data to a pickle file.")

  parser.add_argument("--db_name", type=str, default=DB_NAME,
    help='Database from which get the data.')
  parser.add_argument("--interval", type=int, default=3,
    help='Aggregation interval in minutes.')
  parser.add_argument("--path_to_cnf", type=str, default='mylogin.cnf',
    help='Path to configuration file for the MySQL database.')
  args = parser.parse_args()
  return args


def get_dataframes(db_name, path_to_cnf):
  """Connect to the database and fetch the database.

  Args:
      db_name (str): name of the database
      path_to_cnf (str): contains password to access database

  Returns:
      dataframe (pandas): contains dataframe with raw measurments
  """

  start = time.time()
  cache_data_name = f'{db_name}_df.pkl'
  print(f'Loading data from {db_name} and caching it in {cache_data_name} ...')

  if not os.path.isfile(cache_data_name):
    c = pymysql.connect(read_default_file=path_to_cnf, database=db_name)
    db = c.cursor()
    db.execute('SHOW TABLES')
    tables = db.fetchall()
    dataframes = {}
    for table in tables:
      table_name = table[0]
      dataframes[table_name] = pd.read_sql('SELECT * from ' + table_name, con=c)
    dataframes['mouse'] = dataframes['mouse'].set_index('mouse_id')
    with open(cache_data_name, 'wb') as f:
      pickle.dump(dataframes, f)

  else:
    with open(cache_data_name, 'rb') as f:
      dataframes = pickle.load(f)

  print('Downloaded data. Time elapsed:', time.time() - start)
  return dataframes


def get_processed_dataframes(db_name, interval, path_to_cnf):
  """Calls postprocess traces function

  Args:
      db_name (str): name of the database
      interval (time): sampling time (in mins)
      path_to_cnf (str): contains password to access database

  Returns:
      _data (pandas): post processed dataframe
  """

  start = time.time()
  cache_data_name = f'{db_name}.pkl'

  if not os.path.isfile(cache_data_name):
    dataframes = get_dataframes(db_name, path_to_cnf)
    start = time.time()
    _data = preprocess_traces.process_dataframes(
      dataframes, interval=f'{interval}T', resample=False)
    with open(cache_data_name, 'wb') as f:
      pickle.dump(_data, f)

  else:
    with open(cache_data_name, 'rb') as f:
      _data = pickle.load(f)

  print(f'Postprocessed data from {db_name}. Time elapsed:',
        time.time() - start)

  return _data

def _process_grouped_entry(entry, interval):
  _, data = entry
  cutoff = datetime.timedelta(minutes=interval)
  return populate_missing(data, cutoff=cutoff, features=ALL_FEATURES)


def group_traces(interval, df, parallel=True):
  print('>>> Grouping traces')
  grouped = df.groupby(['trace_id'])
  _process_entry = partial(_process_grouped_entry, interval=interval)
  if parallel:
    with Pool(8) as pool:
      all_traces = list(
          tqdm(pool.imap(_process_entry, grouped), total=len(grouped)))
  else:
    all_traces = [_process_entry(entry) for entry in tqdm(grouped)]

  all_traces_df = pd.concat(all_traces)
  grouped = all_traces_df.groupby(['trace_id'])
  all_traces_grouped = [data for _, data in tqdm(grouped)]
  return all_traces_grouped


def prepare_traces(db_name, interval, path_to_cnf):
  """Prepare traces for training robust HMM

  Args:
      db_name (str): name of the database
      interval (int): sampling interval in mins
      path_to_cnf (str): password to access database

  Returns:
      all_traces_grouped (list): list of all traces
  """
  all_traces_filename = f'{db_name}_{interval}_all.pkl'
  if os.path.isfile(all_traces_filename):
    print('>>> Loading saved traces')
    with open(all_traces_filename, 'rb') as f:
      all_traces_grouped = pickle.load(f)
  else:
    _data = get_processed_dataframes(db_name, interval, path_to_cnf)
    (hourly_aggregates, circadian_aggregates, complete_aggregates, \
      runs, windows) = _data

    df = complete_aggregates.copy()
    if 'level_1' in df.columns:
      df = complete_aggregates.drop('level_1', axis=1)
      
    all_traces_grouped = group_traces(interval=interval, df=df, parallel=True)
    print('>>> Replacing constant zero with missing')
    turn_constant_zero_to_missing(all_traces_grouped, HMM_FEATURES)
    with open(all_traces_filename, 'wb') as f:
      pickle.dump(all_traces_grouped, f)

  return all_traces_grouped

def split_train_validation(all_traces_flat, 
                           batch=BATCH, 
                           train_fraction=TRAIN_FRACTION, 
                           seed=SEED):
  """Split all traces into train/validation/test

  Args:
    all_traces_flat: A dataframe containing all the traces together

  Returns:
    A list of train/validaton/test traces along with the batch IDs
  """
  
  
  # initalize variables
  train_traces = []
  validation_traces = []
  batch_ids_train = []
  batch_ids_validation = []
  all_traces = []
  all_batch_ids = []
  batch_id = 0
  
  _g1 = all_traces_flat.groupby(['rotation', 'rack'])
  rot_rat_grouped = [group for _, group in _g1]
  
  for g in range(len(rot_rat_grouped)):
      
    _g2 = rot_rat_grouped[g].groupby('trace_id')
    traces_grouped = [group for _, group in _g2]
    n_traces = len(traces_grouped)
    
    # get slot_ids
    all_slot_ids =  [int(traces_grouped[k]['slot'].iloc[0]) \
      for k in range(n_traces)]

    # check duplicates
    dup_slot_ids = [item for item, count in \
      collections.Counter(all_slot_ids).items() if count > 1]

    # remove duplicates from slot_ids and traces
    if len(dup_slot_ids) > 0:
      slot_ids = [x for x in all_slot_ids if x not in dup_slot_ids]
      del_indices = [s for s, slot in enumerate(all_slot_ids) \
        if slot in dup_slot_ids]
      for i in del_indices:
        print("Duplicate runs: %s" \
          %(str(traces_grouped[i]['trace_id'].iloc[0])))
      traces_grouped = [traces_grouped[i] for i in range(n_traces) \
        if i not in del_indices]
    else:
      slot_ids = all_slot_ids.copy()
        
    # split slot ids by gas analyzer
    ga_ids = [slot for slot in slot_ids if slot <= TRACES_PER_ANALYZER]
    gb_ids = [slot for slot in slot_ids if slot > TRACES_PER_ANALYZER]
    
    # check if length of slot ids statisfies min number of traces per batch
    if (len(ga_ids) < MIN_TRACES_PER_GA) and (len(ga_ids) != 0):
      del_indices = [slot_ids.index(slot) for slot in ga_ids]
      for i in del_indices:
        print("Insufficient runs: %s" \
          %(str(traces_grouped[i]['trace_id'].iloc[0])))
        ga_ids = []
        
    if (len(gb_ids) < MIN_TRACES_PER_GA) and (len(gb_ids) != 0):
      del_indices = [slot_ids.index(slot) for slot in gb_ids]
      for i in del_indices:
        print("Insufficient runs: %s" \
          %(str(traces_grouped[i]['trace_id'].iloc[0])))
      gb_ids = []
    
    # split train validation for GZ-A
    if len(ga_ids) != 0:
      ga_train, ga_val = train_test_split(ga_ids, 
                                          shuffle=False, 
                                          random_state=seed, 
                                          test_size=1-train_fraction)
      ga_train_indices = [slot_ids.index(slot) for slot in ga_train]
      ga_val_indices = [slot_ids.index(slot) for slot in ga_val]
      ga_all_indices = [slot_ids.index(slot) for slot in ga_ids]
      
      # add batche ids
      batch_ids_train += [batch_id] * len(ga_train_indices)
      batch_ids_validation += [batch_id] * len(ga_val_indices)
      all_batch_ids += [batch_id] * len(ga_all_indices)
      
      # add traces
      for i in ga_train_indices:
        train_traces.append(traces_grouped[i])
      for i in ga_val_indices:
        validation_traces.append(traces_grouped[i])
      for i in ga_all_indices:
        all_traces.append(traces_grouped[i])
      
      # increment batch
      if batch:
        batch_id += 1
    
    # split train validation for GZ-B
    if len(gb_ids) != 0:
      gb_train, gb_val = train_test_split(gb_ids, 
                                          shuffle=False, 
                                          random_state=seed, 
                                          test_size=1-train_fraction)
      gb_train_indices = [slot_ids.index(slot) for slot in gb_train]
      gb_val_indices = [slot_ids.index(slot) for slot in gb_val]
      gb_all_indices = [slot_ids.index(slot) for slot in gb_ids]
      
      # add batch ides
      batch_ids_train += [batch_id] * len(gb_train_indices)
      batch_ids_validation += [batch_id] * len(gb_val_indices)
      all_batch_ids += [batch_id] * len(gb_all_indices)
      
      # add traces
      for i in gb_train_indices:
        train_traces.append(traces_grouped[i])
      for i in gb_val_indices:
        validation_traces.append(traces_grouped[i])
      for i in gb_all_indices:
        all_traces.append(traces_grouped[i])
      
      # increment batch ids
      if batch:
        batch_id += 1

  return (train_traces, batch_ids_train, validation_traces, \
    batch_ids_validation, all_traces, all_batch_ids)


def main():
  args = parse_args()
  prepare_traces(args.db_name, args.interval, args.path_to_cnf)


if __name__ == "__main__":
  main()
