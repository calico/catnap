import os
import pandas as pd
import pickle
import pymysql.cursors
import time
from . import study_constants as constants


# Feature sets
STATE_PROPS_FEATURES = [
    f"state_props~state-{state}~{phase}"
    for state in constants.STATES
    for phase in constants.DAY_NIGHTS
]

STATE_AGG_FEATURES = [
    f"state_agg~state-{state}~{var}_({agg})~{phase}"
    for state in constants.STATES
    for var in constants.STATE_AGG_BASE_FEATURES
    for agg in constants.STATE_AGGREGATE_FNS
    for phase in constants.DAY_NIGHTS
]

STATE_HOURLY_PROPS_FEATURES = [
    "state_props_hourly~state-%s~hour-%02d" % (state, hour)
    for state in constants.STATES
    for hour in range(24)
]

STATE_TRANSITION_FEATURES = [
    f"state_transitions~{src}_to_{dst}_{phase}"
    for src in constants.STATES
    for dst in constants.STATES
    for phase in constants.DAY_NIGHTS
]

CIRCADIAN_AGG_FEATURES = [
    "%s_(%s)~%s" % (var, agg, phase)
    for var in constants.CIRCADIAN_AGG_BASE_FEATURES
    for agg in constants.CIRCADIAN_AGGREGATE_FNS
    for phase in constants.DAY_NIGHTS
]


def load_dataframes(db_name, ignore_cache=False):
    start = time.time()
    cache_path = "./%s.cache" % db_name
    if os.path.exists(cache_path) and not ignore_cache:
        with open(cache_path, "rb") as f:
            dataframes = pickle.load(f)
            elapsed = time.time() - start
            print("Using cache %s, (%0.2s elapsed)" % (cache_path, elapsed))
            return dataframes

    c = pymysql.connect(read_default_file="mylogin.cnf", database=db_name)
    db = c.cursor()
    db.execute("SHOW TABLES")
    tables = db.fetchall()
    dataframes = {}
    for table in tables:
        table_name = table[0]
        dataframes[table_name] = pd.read_sql("SELECT * from " + table_name, con=c)
        elapsed = time.time() - start
        print("Loaded table %s, (%0.2s elapsed)" % (table, elapsed))
    print(time.time() - start)

    dataframes["mouse"] = dataframes["mouse"].set_index("mouse_id")

    mouse_id2generation = dataframes["mouse"].home_cage.apply(lambda x: x.split("-")[0])
    mouse_id2generation.name = "generation"
    dataframes["mouse"].loc[:, "generation"] = mouse_id2generation.values

    is_dead = ~dataframes["mouse"].date_of_death.isnull()
    dataframes["mouse"].loc[:, "is_dead"] = is_dead

    last_recorded_date = (
        dataframes["complete"]
        .groupby("mouse_id")
        .time.max()
        .dt.date.loc[dataframes["mouse"].index]
    )
    dataframes["mouse"].loc[:, "last_recorded_date"] = last_recorded_date
    return dataframes
