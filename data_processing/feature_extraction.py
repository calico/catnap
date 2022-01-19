import datetime
import functools
# import logging
import numpy as np
import pandas as pd

from . import smoothing
from . import study_constants as constants

LENGTH_OF_COMPLETE_DAY = 24 * 60 / 3
ONE_MONTH_IN_DAYS = 365.0 / 12

BASE_FEATURES = [
    'AllMeters', 'BodyMass', 'Food', 'KCal_hr', 'PedMeters', 'PedSpeed',
    'RQ', 'VCO2', 'VH2O', 'VO2', 'Water', 'WheelMeters', 'WheelSpeed',
    'XBreak', 'YBreak', 'ZBreak']
CUMULATIVE_FEATURES = [
    'AllMeters', 'Food', 'PedMeters', 'Water', 'WheelMeters', 'XBreak',
    'YBreak', 'ZBreak']
HMM_FEATURES = [
    'VO2', 'VCO2', 'VH2O', 'KCal_hr', 'RQ', 'Food', 'Water', 'PedMeters',
    'WheelMeters', 'XBreak', 'YBreak', 'ZBreak']
STATE_FEATURES = ['state-%d' % x for x in range(6)]
AGGREGATE_FEATURES = BASE_FEATURES + STATE_FEATURES
FEATURE_ORDER = [
    'VO2', 'VCO2', 'RQ', 'KCal_hr', 'VH2O', 'Food', 'Water',
    'WheelMeters', 'AllMeters', 'PedMeters',
    'XBreak', 'YBreak', 'ZBreak']
STATE_ORDER = ['SLEEP', 'REST', 'ACTIVE', 'RUN', 'EAT', 'EAT&DRINK']
STATE2NAME = constants.STATE2DESCRIPTOR


def make_age_group(age_in_months, interval=3, max_age=33):
    if age_in_months >= max_age:
        return '%02d months or older' % max_age
    lower_bound = (age_in_months // interval) * interval
    return '%02d-%02d months' % (lower_bound, lower_bound + interval)


def make_trace_metadata(complete_df, mouse_df):
    """Compute trace metadata such as age at run etc."""
    def _get_run_number(trace_id):
        run_number = int(trace_id.split('_')[-1])
        return run_number

    trace_meta_df = (
        complete_df.groupby(['mouse_id', 'trace_id'])['date']
                   .max().reset_index().copy())
    trace_meta_df = trace_meta_df.sort_values(['mouse_id', 'date'])

    # Add run number
    run_numbers = trace_meta_df['trace_id'].apply(_get_run_number)
    trace_meta_df.loc[:, 'run_number'] = run_numbers.values

    # Add age fields
    trace_meta_df.loc[:, 'age'] = pd.Series(
        trace_meta_df['date'].values
        - mouse_df.loc[trace_meta_df['mouse_id']]['date_of_birth'].values
    ).dt.days.values
    trace_meta_df.loc[:, 'age_in_months'] = (
        trace_meta_df['age'] / (365 / 12.0)
    ).astype(int)
    trace_meta_df.loc[:, 'age_group'] = trace_meta_df['age_in_months'].apply(
        make_age_group)

    # Add lifespan information
    days_to_death = (
        mouse_df.loc[trace_meta_df['mouse_id']]['age_at_death'].values
        - trace_meta_df['age'])
    fraction_of_lifespan = trace_meta_df['age'].astype(float).div(
        mouse_df.loc[trace_meta_df['mouse_id']]['age_at_death'].values,
        fill_value=np.nan)
    trace_meta_df.loc[:, 'fraction_of_lifespan'] = fraction_of_lifespan
    trace_meta_df.loc[:, 'days_to_death'] = days_to_death

    # Remove traces that have deaths within a week of met cage measurement
    mask = ~(days_to_death <= 7).values
    trace_meta_df = trace_meta_df[mask]

    trace_meta_df = trace_meta_df.rename({'date': 'trace_date'}, axis=1)
    trace_meta_df = trace_meta_df.set_index('trace_id')
    return trace_meta_df


def add_trace_meta(df, trace_meta_df):
    for col in trace_meta_df:
        df.loc[:, col] = trace_meta_df.loc[df['trace_id']][col].values
    return df


def add_derived_features(complete_df, trace_meta_df):
    """ Make base features from 3-minute resolution level data.
    """
    state_dummies = pd.get_dummies(
        complete_df['states'], prefix='state-', prefix_sep='')
    complete_df = pd.concat([complete_df, state_dummies], axis=1)
    # complete_df = complete_df.drop('level_0', axis=1)
    complete_df = complete_df[
        (complete_df['trace_id'].isin(trace_meta_df.index)).values
    ]

    # Add flag for complete day to time_df - exclude incomplete days
    timepoints_by_trace_date = complete_df.groupby(['trace_id', 'date']).size()
    timepoints_by_trace_date = timepoints_by_trace_date[
        timepoints_by_trace_date == LENGTH_OF_COMPLETE_DAY]
    complete_trace_dates = timepoints_by_trace_date.reset_index().copy()
    complete_trace_dates = complete_trace_dates[['trace_id', 'date']]
    complete_trace_dates.loc[:, 'part_of_complete_day'] = True
    complete_df = complete_df.merge(complete_trace_dates,
                                    on=['trace_id', 'date'], how='left')
    complete_df['part_of_complete_day'] = (
            complete_df['part_of_complete_day'].fillna(False))
    return complete_df


def make_hourly_average_df(complete_df):
    hourly_df = complete_df.copy()

    def _round_dwn_to_hour(t):
        hour = (int(t.total_seconds() // 3600) + 1) % 24
        return datetime.time(hour=hour)

    hourly_df['time_hour'] = hourly_df['time_period'].apply(_round_dwn_to_hour)
    hourly_average_df = hourly_df.groupby(['trace_id', 'time_hour'])
    hourly_average_df = hourly_average_df[AGGREGATE_FEATURES].agg(np.mean)
    hourly_average_df = hourly_average_df.reset_index()
    return hourly_average_df


def make_time_window_and_lights_on_off_ratio_dfs(complete_df):
    time_window_df = complete_df.copy()

    def _subset_time_window(time_window_df, time_window):
        mask = time_window_df.index.get_level_values(1) == time_window
        df = time_window_df[mask]
        df.index = df.index.droplevel(1)
        return df

    def _make_ratio_df(pre_df, post_df, suffix):
        ratio_df = pre_df / post_df
        # mask = post_df < 1e-3
        # ratio_df[mask] = None
        ratio_df = ratio_df.clip(0, 1e2)
        ratio_df.columns = ['%s(%s)' % (col, suffix)
                            for col in ratio_df.columns]
        return ratio_df.reset_index()

    # Partition timepoint into 4 hour time windows around light transitions
    time_boundaries = [
        datetime.timedelta(hours=0, seconds=-1), datetime.timedelta(hours=3),
        datetime.timedelta(hours=7), datetime.timedelta(hours=11),
        datetime.timedelta(hours=15), datetime.timedelta(hours=19),
        datetime.timedelta(hours=23), datetime.timedelta(hours=24, seconds=1)]
    time_window = pd.cut(time_window_df.time_period, bins=time_boundaries,
                         labels=['dark', 'late-dark', 'early-light',
                                 'light', 'late-light', 'early-dark',
                                 'dark2'])
    time_window[time_window == 'dark2'] = 'dark'
    time_window = time_window.cat.remove_unused_categories()
    time_window_df['time_window'] = time_window.astype(str)

    time_window_df = time_window_df.groupby(['trace_id', 'time_window'])
    time_window_df = time_window_df[AGGREGATE_FEATURES].agg(np.mean)

    # Make lights on/off ratio features
    pre_lights_on_df = _subset_time_window(time_window_df, 'late-dark')
    post_lights_on_df = _subset_time_window(time_window_df, 'early-light')
    lights_on_ratio_df = _make_ratio_df(
            pre_lights_on_df, post_lights_on_df, 'lights-on-ratio')

    pre_lights_off_df = _subset_time_window(time_window_df, 'late-light')
    post_lights_off_df = _subset_time_window(time_window_df, 'early-dark')
    lights_off_ratio_df = _make_ratio_df(
            pre_lights_off_df, post_lights_off_df, 'lights-off-ratio')

    time_window_df = time_window_df.unstack()
    time_window_df.columns = ['%s(%s)' % (var, time_window)
                              for var, time_window in time_window_df.columns]
    time_window_df = time_window_df.reset_index()
    return time_window_df, lights_on_ratio_df, lights_off_ratio_df


def make_trace_total_and_average_dfs(complete_df):
    trace_average_df = complete_df.copy()

    def _aggregate_by_day(df, agg_features, reduce_fn):
        agg_df = df.copy()
        agg_df = agg_df.groupby(['trace_id', 'date']).agg(reduce_fn)
        return agg_df.reset_index()

    # Only keep complete days
    mask = trace_average_df['part_of_complete_day']
    trace_average_df = trace_average_df[mask]

    # Sums of features of complete days
    total_by_date_df = _aggregate_by_day(
            trace_average_df, AGGREGATE_FEATURES, 'sum')

    # Averages of features of complete days
    average_by_date_df = _aggregate_by_day(
            trace_average_df, AGGREGATE_FEATURES, 'mean')

    # Aggregate by trace now to get the average per-day sum or
    # mean across the run
    agg_fns = {col: 'mean' for col in AGGREGATE_FEATURES}
    trace_daily_total_df = (total_by_date_df.groupby('trace_id')
                                            .agg(agg_fns).reset_index())
    trace_daily_average_df = (average_by_date_df.groupby('trace_id')
                                                .agg(agg_fns).reset_index())
    return trace_daily_total_df, trace_daily_average_df


def make_state_means_df(complete_df):
    state_means_df = complete_df.copy()
    state_means_df = (state_means_df.groupby(['trace_id', 'states'])
                      [BASE_FEATURES].mean())
    state_means_df = state_means_df.unstack()
    state_means_df.columns = [x + '(' + STATE2NAME[y] + ')'
                              for x, y in state_means_df.columns]
    state_means_df = state_means_df.reset_index()
    return state_means_df


def make_circadian_means_and_ratio_dfs(complete_df):
    circadian_means_df = complete_df.copy()
    circadian_means_df = (circadian_means_df
                          .groupby(['day_night', 'trace_id'])[BASE_FEATURES]
                          .mean())
    circadian_ratio_df = (circadian_means_df.loc['night'] /
                          circadian_means_df.loc['day'])
    mask = circadian_means_df.loc['day'] < 1e-3
    circadian_ratio_df[mask] = None
    circadian_ratio_df = circadian_ratio_df.reset_index()
    return circadian_means_df, circadian_ratio_df


def compute_window_stats(data, indicator_fn, name):
    index = ['num period', 'median period', 'median interval', 'max period',
             'max interval']
    index = [name + ' ' + col for col in index]
    missing = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan], index=index)

    if isinstance(data, pd.Series):
        if ((~np.isnan(data)).mean() < 0.7 or
           len(data) < LENGTH_OF_COMPLETE_DAY):
            return missing
    else:
        for col in data.columns:
            if ((~np.isnan(data[col])).mean() < 0.7 or
               len(data[col]) < LENGTH_OF_COMPLETE_DAY):
                return missing

    indicators = indicator_fn(data)
    labels = indicators.ne(indicators.shift()).cumsum()
    period_lengths = labels[indicators].value_counts()
    interval_lengths = labels[~indicators].value_counts()

    if len(period_lengths) == 0:
        num_periods = np.nan
        median_period = np.nan
        max_period = np.nan
    else:
        num_periods = len(period_lengths)
        median_period = period_lengths.quantile(0.5) / 20
        max_period = period_lengths.quantile(0.95) / 20

    if len(interval_lengths) == 0:
        median_interval = np.nan
        max_interval = np.nan
    else:
        median_interval = interval_lengths.quantile(0.5) / 20
        max_interval = interval_lengths.quantile(0.95) / 20

    return pd.Series([num_periods, median_period, median_interval,
                      max_period, max_interval], index=index)


def make_window_stats_df(complete_df, columns, indicator_fn, name):
    map_fn = functools.partial(
        compute_window_stats, indicator_fn=indicator_fn, name=name)
    complete_df = complete_df[complete_df['part_of_complete_day']]
    window_stats_df = (complete_df.groupby(['trace_id', 'date'])[columns]
                       .apply(map_fn).reset_index())
    window_stats_df = window_stats_df.drop('date', axis=1)
    window_stats_df = window_stats_df.groupby('trace_id').mean().reset_index()
    return window_stats_df


def make_feeding_window_stats_df(complete_df):
    def _indicator_fn(data):
        return (data['Food'] > 1e-3) | (data['states'].isin([4, 5]))
    return make_window_stats_df(
        complete_df, ['Food', 'states'], _indicator_fn, 'feeding')


def make_sleeping_window_stats_df(complete_df):
    def _indicator_fn(data):
        return data['states'].isin([1])
    return make_window_stats_df(
        complete_df, ['states'], _indicator_fn, 'sleeping')


def make_exercising_window_stats_df(complete_df):
    def _indicator_fn(data):
        return data['states'].isin([2, 3])
    return make_window_stats_df(
        complete_df, ['states'], _indicator_fn, 'exercising')


def generate_all_feature_dfs(complete_df, mouse_df):
    trace_meta_df = make_trace_metadata(complete_df, mouse_df)
    complete_df = add_derived_features(complete_df, trace_meta_df)

    # Generate feature sets
    hourly_average_df = make_hourly_average_df(complete_df)

    t_window_dfs = make_time_window_and_lights_on_off_ratio_dfs(complete_df)
    time_window_df, lights_on_ratio_df, lights_off_ratio_df = t_window_dfs

    circadian_features_dfs = make_circadian_means_and_ratio_dfs(complete_df)
    circadian_means_df, circadian_ratio_df = circadian_features_dfs

    daily_totals_and_avgs_dfs = make_trace_total_and_average_dfs(complete_df)
    trace_daily_total_df, trace_daily_average_df = daily_totals_and_avgs_dfs

    state_means_df = make_state_means_df(complete_df)

    feeding_window_stats_df = make_feeding_window_stats_df(complete_df)
    sleeping_window_stats_df = make_sleeping_window_stats_df(complete_df)
    exercising_window_stats_df = make_exercising_window_stats_df(complete_df)

    def _stack_features(feature_dfs):
        feature_subsets = [df.copy().set_index('trace_id')
                           for df in feature_dfs]
        return pd.concat(feature_subsets, axis=1, sort=True).reset_index()

    all_trace_features_df = _stack_features([
        trace_daily_average_df, state_means_df,
        sleeping_window_stats_df, exercising_window_stats_df,
        feeding_window_stats_df, time_window_df,
        lights_on_ratio_df, lights_off_ratio_df])

    trace_ids = all_trace_features_df['trace_id']
    trace_dates = trace_meta_df.loc[trace_ids]['trace_date']
    all_trace_features_df['trace_date'] = trace_dates

    mouse_ids = trace_meta_df.loc[trace_ids]['mouse_id']
    smoothed_trace_features_df = (
        all_trace_features_df.set_index('trace_id')
                             .sort_values('trace_date')
                             .groupby(mouse_ids)
                             .apply(smoothing.l1_trend_filter_agg,
                                    'trace_date', lmbda=40))
    smoothed_trace_features_df.index.name = 'trace_id'
    smoothed_trace_features_df = smoothed_trace_features_df.reset_index()

    trace_feature_sets = {
        'trace average': trace_daily_average_df,
        'trace sum': trace_daily_total_df,
        'state means': state_means_df,
        'sleeping window': sleeping_window_stats_df,
        'exercise window': exercising_window_stats_df,
        'feeding window': feeding_window_stats_df,
        'hourly means': hourly_average_df,
        'time window means': time_window_df,
        'circadian ratio': circadian_ratio_df,
        'lights on ratio': lights_on_ratio_df,
        'lights off ratio': lights_off_ratio_df,
        'all features': all_trace_features_df,
        'smoothed all features': smoothed_trace_features_df
    }

    for key, df in trace_feature_sets.items():
        trace_feature_sets[key] = (
                add_trace_meta(df, trace_meta_df).set_index('trace_id'))

    complete_df.loc[:, 'age_group'] = complete_df['age_in_months'].apply(
            make_age_group)
    trace_feature_sets.update({
        'complete data': complete_df,
        'trace metadata': trace_meta_df,
        'mouse metadata': mouse_df})

    return trace_feature_sets
