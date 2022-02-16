import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb


def make_feature_matrices(trace_features, train_trace_ids, test_trace_ids):
    train_features = trace_features.loc[train_trace_ids]
    test_features = trace_features.loc[test_trace_ids]
    dtrain = xgb.DMatrix(train_features, label=train_trace_ids)
    dtest = xgb.DMatrix(test_features, label=test_trace_ids)
    return dtrain, dtest


def make_splits(trace_features, trace_meta, mouse_meta, random_state=0):
    mouse_ids = trace_meta["mouse_id"].unique()
    trace_meta = trace_meta[trace_meta["age_in_months"].between(0, 33)]
    mouse_generation = mouse_meta.loc[mouse_ids]["generation"].values
    train_mouse_ids, test_mouse_ids = train_test_split(
        mouse_ids, stratify=mouse_generation, test_size=0.1,
        random_state=random_state)

    def _subset_traces(mouse_ids):
        subset = trace_meta[trace_meta["mouse_id"].isin(mouse_ids)]
        return subset

    train_trace_ids = _subset_traces(train_mouse_ids).index.values
    test_trace_ids = _subset_traces(test_mouse_ids).index.values
    dtrain, dtest = make_feature_matrices(
        trace_features, train_trace_ids, test_trace_ids)
    return dtrain, dtest


def get_bounds(trace_meta, mouse_meta):
    trace_mouse_ids = trace_meta["mouse_id"].values
    is_dead = mouse_meta.loc[trace_mouse_ids]["is_dead"].values
    date_of_birth = mouse_meta.loc[trace_mouse_ids]["date_of_birth"]
    age_at_death = mouse_meta.loc[trace_mouse_ids]["age_at_death"]
    last_recorded_date = (
        mouse_meta.loc[trace_mouse_ids]["last_recorded_date"]
                  .astype(np.datetime64))
    survival_time = age_at_death.combine_first(
            (last_recorded_date - date_of_birth).dt.days).values
    survival_time[~is_dead] *= -1
    age = trace_meta["age"].values
    return trace_mouse_ids, age, survival_time


def get_labels(trace_meta, mouse_meta, trace_ids):
    subset = trace_meta.loc[trace_ids]
    mouse_ids, age, survival_time = get_bounds(
            subset, mouse_meta)
    return mouse_ids, age, survival_time


def make_label_df(trace_meta, mouse_meta):
    mouse_ids, age, survival_time = get_bounds(
            trace_meta, mouse_meta)
    return pd.DataFrame({
        'age': age,
        'survival': survival_time,
        'animal_id': mouse_ids}, index=trace_meta.index)
