import datetime


END_DATE = datetime.date(2019, 4, 25)
STATE2DESCRIPTOR = {0: 'REST', 1: 'SLEEP', 2: 'ACTIVE', 3: 'RUN', 4: 'EAT&DRINK', 5: 'EAT'}
DAY_NIGHTS = ['day', 'night']
STATES = list(STATE2DESCRIPTOR.values())
STATE_AGG_BASE_FEATURES = ['VO2', 'VCO2', 'VH2O', 'KCal_hr', 'RQ', 'Food', 'PedMeters', 'AllMeters']
STATE_AGGREGATE_FNS = ['nanmean', 'pp99']
CIRCADIAN_AGG_BASE_FEATURES = ["VO2", "RQ", "KCal_hr", "Food", "Water", "BodyMass", "WheelSpeed",
                               "WheelMeters", 'PedMeters', "AllMeters", "XBreak", "YBreak", "ZBreak"]
CIRCADIAN_AGGREGATE_FNS = ['mean', 'pp99']
INDEX_VARS = ['mouse_id', 'trace_id', 'date', 'age_in_months', 'day_of_week']
AGE_ORDER = ['00-03 months', '03-06 months', '06-09 months', '09-12 months',
             '12-15 months', '15-18 months', '18-21 months', '21-24 months',
             '24-27 months', '27-30 months', '30-33 months',
             '33 months or older']

