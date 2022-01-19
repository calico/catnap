from functools import partial
import numpy as np
import pandas as pd
import scipy
import scipy.stats

DT_HOUR_LIGHTS_ON = 6
DT_HOUR_LIGHTS_OFF = 18
ZERO_THRESHOLD = 1e-6


def resample_to_constant_interval(trace_df, interval='3T', limit=4):
    """Resamples a ts to constant interval.
    
    This resamples a given time series to constant interval which
    is required for various downstream steps such as hybrid ESD
    or STL decomposition.
    
    The resampling averages time-points that are downsampled and
    upsamples via bfill().
    
    Args:
        trace_df: Dataframe of a single run.
        interval: New spacing between time-points.
    Returns:
        trace_df but resampled to have uniform interval between time-points.
    """
    try:
        resampled_df = trace_df.copy().set_index('time')
        resampled_df = resampled_df.resample(interval).mean().interpolate(method='nearest', limit=limit)
        return resampled_df        
    except:
        return None


def find_lights_on_off(trace_time):
    t = trace_time.apply(pd.Timestamp)
    hours = (
        (t.values - t.values.astype("datetime64[D]"))
        .astype("timedelta64[h]")
        .astype(float)
    )
    lights_on = (hours[:-1] < DT_HOUR_LIGHTS_ON) & (hours[1:] >= DT_HOUR_LIGHTS_ON)
    lights_off = (hours[:-1] < DT_HOUR_LIGHTS_OFF) & (hours[1:] >= DT_HOUR_LIGHTS_OFF)
    breakpoints = [False] + (lights_on | lights_off).tolist()
    return np.array(breakpoints).astype(int).cumsum()


def get_metadata(processed_cage_traces, mouse_metadata):
    duplicated = processed_cage_traces.duplicated(
        ["trace_id", "time"], keep=False
    ).any()
    assert duplicated == False, "Duplicated entries found in processed cage traces!"

    metadata = processed_cage_traces[["trace_id", "mouse_id", "time"]].copy()
    timestamps = metadata.time
    metadata.loc[:, "time_period"] = timestamps.dt.time
    metadata.loc[:, "date"] = timestamps.dt.date.astype("datetime64[ns]")
    metadata.loc[:, "day_of_week"] = timestamps.dt.day_name()
    metadata.loc[:, "day_night"] = timestamps.dt.hour.between(
        DT_HOUR_LIGHTS_ON, DT_HOUR_LIGHTS_OFF - 1
    ).apply(lambda x: "day" if x else "night")
    mouse_dobs = mouse_metadata["date_of_birth"]
    metadata.loc[:, "date_of_birth"] = mouse_dobs.loc[metadata["mouse_id"]].values
    metadata.loc[:, "age"] = metadata["date"] - metadata["date_of_birth"]
    metadata.loc[:, "age_in_months"] = (
        metadata.date.dt.year - metadata["date_of_birth"].dt.year
    ) * 12 + (metadata.date.dt.month - metadata["date_of_birth"].dt.month)
    metadata["birth_cohort"] = metadata["date_of_birth"].apply(lambda x: str(x)[:7])
    return metadata


def get_outlier_mask(measurements, threshold=5):
    robust_std = scipy.stats.iqr(measurements.values, nan_policy="omit") / 1.349
    if robust_std < 1e-3:
        return np.ones_like(measurements.values).astype(bool)
    mask = ((measurements - np.nanmedian(measurements.values)) / robust_std) < threshold
    return mask


def get_range_mask(measurements, valid_range):
    mask = measurements.between(*valid_range).values
    return mask


def filter_interval_data(interval_aggregates, mouse_metadata):
    filtered_interval_aggregates = interval_aggregates.copy()
    all_mask = np.ones(len(filtered_interval_aggregates)).astype(bool)

    valid_interval_ranges = {
        "VO2": (1e-2, 4),
        "VCO2": (1e-2, 4),
        "VH2O": (1e-2, 4),
        "RQ": (0.4, 1.1),
        "Food": (0, 1e10),
        "Water": (0, 1e10),
        "WheelMeters": (0, 1e10),
        "PedMeters": (0, 1e10),
        "AllMeters": (0, 1e10),
        "XBreak": (0, 500),
        "YBreak": (0, 500),
        "ZBreak": (0, 180),
    }

    # Range check
    for agg_data_col, valid_range in valid_interval_ranges.items():
        measurements = filtered_interval_aggregates[agg_data_col]
        mask = get_range_mask(measurements, valid_range)
        filtered_interval_aggregates.loc[~mask, agg_data_col] = None
        print(
            "%s - range check removes %d points (%0.2f %% of data)"
            % (agg_data_col, np.sum(~mask), 100.0 * np.mean(~mask))
        )
        all_mask &= mask

    print(
        "Final mask - removing %d points (%0.2f %% of data)"
        % (np.sum(~all_mask), 100.0 * np.mean(~all_mask))
    )

    # Remove outliers
    for agg_data_col in ["VO2", "VCO2", "VH2O", "KCal_hr", "RQ", "BodyMass"]:
        measurements = filtered_interval_aggregates[agg_data_col]
        mask = get_outlier_mask(measurements)
        filtered_interval_aggregates.loc[~mask, agg_data_col] = None
        print(
            "%s - outlier check removes %d points (%0.2f %% of data)"
            % (agg_data_col, np.sum(~mask), 100.0 * np.mean(~mask))
        )
        all_mask &= mask

    print(
        "Final mask - removing %d points (%0.2f %% of data)"
        % (np.sum(~all_mask), 100.0 * np.mean(~all_mask))
    )

    # Get metadata
    metadata = get_metadata(interval_aggregates, mouse_metadata)
    return pd.merge(
        filtered_interval_aggregates, metadata, on=["trace_id", "mouse_id", "time"]
    )


def filter_circadian_data(
    circadian_agg_cage_traces, mouse_metadata, filter_hard_zeros=False
):
    # Filter circadian data
    filtered_circadian_agg_cage_traces = circadian_agg_cage_traces.copy()
    all_mask = np.ones(len(filtered_circadian_agg_cage_traces)).astype(bool)
    if filter_hard_zeros:
        zero = 0.0
    else:
        zero = ZERO_THRESHOLD

    valid_circadian_ranges = {
        "VO2_(mean)": (1e-2, 5),
        "VCO2_(mean)": (1e-2, 5),
        "VH2O_(mean)": (1e-2, 5),
        "RQ_(mean)": (0.4, 1.1),
        "Food_(mean)": (zero, 1e2),
        "Water_(mean)": (zero, 1e2),
        "WheelMeters_(mean)": (zero, 1e3),
        "VO2_(pp99)": (1e-2, 6),
        "VCO2_(pp99)": (1e-2, 6),
        "WheelMeters_(pp99)": (zero, 1e3),
        "Food_(pp99)": (zero, 1e2),
        "Water_(pp99)": (zero, 1e2),
    }

    # Range check
    for agg_data_col, valid_range in valid_circadian_ranges.items():
        measurements = filtered_circadian_agg_cage_traces[agg_data_col]
        mask = get_range_mask(measurements, valid_range)
        filtered_circadian_agg_cage_traces.loc[~mask, agg_data_col] = None
        print(
            "%s - range check removes %d points (%0.2f %% of data)"
            % (agg_data_col, np.sum(~mask), 100.0 * np.mean(~mask))
        )
        all_mask &= mask

    print(
        "Final mask - removing %d points (%0.2f %% of data)"
        % (np.sum(~all_mask), 100.0 * np.mean(~all_mask))
    )

    # Remove circadian outliers
    for agg_data_col in [
        "VO2_(mean)",
        "VCO2_(mean)",
        "VH2O_(mean)",
        "KCal_hr_(mean)",
        "RQ_(mean)",
        "BodyMass_(mean)",
        "VO2_(pp99)",
        "VCO2_(pp99)",
        "VH2O_(pp99)",
        "KCal_hr_(pp99)",
        "RQ_(pp99)",
        "BodyMass_(pp99)",
    ]:
        measurements = filtered_circadian_agg_cage_traces[agg_data_col]
        mask = get_outlier_mask(measurements)
        filtered_circadian_agg_cage_traces.loc[~mask, agg_data_col] = None
        print(
            "%s - outlier check removes %d points (%0.2f %% of data)"
            % (agg_data_col, np.sum(~mask), 100.0 * np.mean(~mask))
        )
        all_mask &= mask

    print(
        "Final mask - removing %d points (%0.2f %% of data)"
        % (np.sum(~all_mask), 100.0 * np.mean(~all_mask))
    )

    metadata = get_metadata(filtered_circadian_agg_cage_traces, mouse_metadata)
    filtered_circadian_agg_cage_traces = pd.merge(
        filtered_circadian_agg_cage_traces,
        metadata,
        on=["trace_id", "mouse_id", "time"],
    )
    return filtered_circadian_agg_cage_traces


def aggregate_circadian_data(complete_processed_cage_traces):
    def pp99(x):
        if np.nanvar(x) < 1e-10:
            return np.nanmean(x)
        else:
            return np.nanpercentile(x, 99)

    def pp5(x):
        if np.nanvar(x) < 1e-10:
            return np.nanmean(x)
        else:
            return np.nanpercentile(x, 99)

    data_columns = [
        "VO2",
        "VCO2",
        "VH2O",
        "KCal_hr",
        "RQ",
        "Food",
        "Water",
        "BodyMass",
        "PedSpeed",
        "PedMeters",
        "AllMeters",
        "WheelMeters",
        "WheelSpeed",
        "XBreak",
        "YBreak",
        "ZBreak",
    ]
    aggregate_fns = [np.mean, np.nanstd, pp99]
    circadian_agg_cage_traces = complete_processed_cage_traces.copy()
    circadian_groups = circadian_agg_cage_traces.groupby(
        ["trace_id", "circadian_period", "mouse_id", "rack", "slot"]
    )
    circadian_timestamp = circadian_groups.agg({"time": np.min})["time"]
    circadian_agg_cage_traces = circadian_groups.aggregate(
        {k: aggregate_fns for k in data_columns}
    )
    circadian_agg_cage_traces.loc[:, "time"] = circadian_timestamp
    agg_data_cols = [
        ("%s_(%s)" % (a, b) if b != "" else a)
        for a, b in circadian_agg_cage_traces.columns
    ]
    circadian_agg_cage_traces.columns = agg_data_cols
    return circadian_agg_cage_traces.reset_index()


MOUSE_METADATA_MAPPINGS = {"date_born": "date_of_birth"}


def add_canonical_mouse_columns(dataframes):
    mouse_df = dataframes["mouse"]
    for field_jax, field_calico in MOUSE_METADATA_MAPPINGS.items():
        if field_jax in mouse_df.columns and field_calico not in mouse_df.columns:
            mouse_df[field_calico] = mouse_df[field_jax]


def process_dataframes(dataframes, filter_hard_zeros=False, interval="3T",
                       resample=True):
    # Partition each trace into day / night periods
    # Lights come on at 6am and turn off at 6pm
    add_canonical_mouse_columns(dataframes)
    mouse_metadata = dataframes["mouse"]

    complete_aggregates = (dataframes["complete"].drop("index", axis="columns")
                                                 .copy())
    complete_aggregates = (
        complete_aggregates.groupby(["trace_id", "mouse_id"])
                           .apply(pd.DataFrame))
    if resample:
        _resample = partial(resample_to_constant_interval, interval=interval)
        complete_aggregates = (
            complete_aggregates.groupby(["trace_id", "mouse_id"])
            .apply(_resample)
            .reset_index()
        )
    if "level_2" in complete_aggregates.columns:
        complete_aggregates = complete_aggregates.drop(
            "level_2", axis="columns")
    complete_aggregates = filter_interval_data(
        complete_aggregates, mouse_metadata)
    complete_aggregates.loc[:, "circadian_period"] = complete_aggregates.groupby(
        "trace_id"
    ).time.transform(find_lights_on_off)
    runs = complete_aggregates["trace_id"].unique()

    hourly_processed_cage_traces = dataframes["hourly"].copy()
    hourly_aggregates = filter_interval_data(
        hourly_processed_cage_traces, mouse_metadata
    )
    hourly_aggregates.loc[:, "circadian_period"] = hourly_aggregates.groupby(
        "trace_id"
    ).time.transform(find_lights_on_off)

    circadian_aggregates = complete_aggregates.copy()
    circadian_aggregates.loc[:, "circadian_period"] = (
        circadian_aggregates.groupby(["trace_id", "mouse_id"]
    ).time.transform(find_lights_on_off))
    circadian_aggregates = aggregate_circadian_data(circadian_aggregates)
    circadian_aggregates = filter_circadian_data(
        circadian_aggregates, mouse_metadata,
        filter_hard_zeros=filter_hard_zeros)
    circadian_periods = (
        circadian_aggregates[["trace_id", "time_period"]].drop_duplicates()
                                                         .values)

    return (hourly_aggregates, circadian_aggregates, complete_aggregates,
            runs, circadian_periods)
