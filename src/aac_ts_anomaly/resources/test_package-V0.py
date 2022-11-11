
import os, warnings
#warnings.filterwarnings("ignore")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.core.defchararray import rpartition
import seaborn as sns
import pandas as pd
#from statsmodels.tsa.stattools import adfuller, kpss, acf, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, month_plot, quarter_plot
import numpy as np
from copy import deepcopy
import datetime
#pd.set_option('display.max_rows', 10**5)
pd.set_option('display.max_columns', 10**5)
from importlib import reload
import adtk

#from aac_ts_anomaly.utils import tsa_utils as tsa
from aac_ts_anomaly.utils import utils_func as util
from aac_ts_anomaly.config import global_config as glob
from aac_ts_anomaly.services import file
from aac_ts_anomaly.resources import (config, preprocessor, trainer)

reload(trainer)
reload(util)
reload(file)
reload(config)
reload(glob)
reload(preprocessor)

#os.getcwd()

filename = util.get_newest_file(search_for = "agg_time_series_52",  src_dir=glob.UC_DATA_DIR)
filename

csv = file.CSVService(path=filename, dtype= {'time_index': str}, delimiter=',')

data_orig = csv.doRead() ; data_orig.shape
data_orig.head()

data_orig.shape

ts_col='target'

periodicity = 52

target_col = ts_col  #ts_col: column name of target time series in df
#periodicity = periodicity    # 12 or 52 : seasonal period in the data. Currently: monthly, weekly (i.e. calendar weeks)

reload(config)

# Get parameters from I/O yaml
if periodicity == 12 : 
    config_input = config.in_out12['input']
    config_output = config.in_out12['output']
    config_detect = config.in_out12['detection']

if periodicity == 52 : 
    config_input = config.in_out52['input']
    config_output = config.in_out52['output']
    config_detect = config.in_out52['detection']
#------------------------------------------------------------------

config_input

hyper_para = config_detect['training']['hyper_para']
transformers = config_detect['training']['transformers']
stat_transform = config_detect['training']['stat_transform']
outlier_filter = config_detect['training']['outlier_filter']
aggreg_level, pre_filter, ignore_lag, min_sample_size, min_median_cnts = list(config_detect['preprocessing'].values())
tbl_name = config_output['database']['tbl_name']
detect_thresh = config_detect['prediction']['detect_thresh']

from datetime import datetime, date
from dateutil.relativedelta import relativedelta

age = 6
if outlier_filter is None:
        six_months_ago = date.today() - relativedelta(months=age)
        outlier_filter = six_months_ago.strftime("%Y-%m")

outlier_filter


# Instantiate class:
#--------------------
claims = preprocessor.claims_reporting(periodicity=periodicity)

#aggreg_level, pre_filter, ignore_lag, min_sample_size, min_median_cnts = list(config_detect['preprocessing'].values())

gen = claims.process_data(data_orig, aggreg_level = 'all_combi')

#get_all = dict(gen)
#get_all['all']

# # Get next series
label, sub_set = next(gen)

print(label, sub_set.shape[0])

df = deepcopy(sub_set)

# df.head(10)
# #ts_bag = claims.tseries
# #ts_bag.get("all")

# multi_ts = claims.multi_ts

#---------------------------------------------------

reload(trainer)
reload(preprocessor)

train = trainer.trainer(verbose=False)

# #for i in dir(train): print(i)
fitted = train.fit(df = df)
out = fitted.predict(detect_thresh = None)
fitted.anomalies

# y = fitted.val_series
# y.head()

results_all, results_new = train.run_all(data_orig = data_orig, verbose=True)

results_final = deepcopy(results_new)      # only show new outliers excluding ones shown before
#results_final = deepcopy(results_all)      # show all detected outliers potentially including ones shown before

results = deepcopy(results_final)
results.rename(columns={'time_anomaly': 'Time', 'time_series_name': 'Time series', 'target': 'Claim counts'}, inplace=True)
results.reset_index(inplace=True, drop=True)

all_series = train.all_series

len(all_series)

new_anomalies = list(set(results_final['time_series_name']))
new_anomalies


# Change plots to take original series
# after having used transformed data
#######################################################################

from adtk.data import validate_series
from adtk.transformer import DoubleRollingAggregate, RollingAggregate, Retrospect, ClassicSeasonalDecomposition
from adtk.pipe import Pipeline, Pipenet

df['year_period_ts'] = df.apply(lambda row: claims._year_week(row.year, row.period), axis=1)
# Remove calendar week 53 if there! Frequ. = 52 from now on.
df = claims._correct_cweek_53(df, time_col = 'year_period_ts')

df.head()

transform = 'diff'
target_col = "target"

if transform in ['log', 'diff_log']:
    ts_index, ts_values = df['year_period_ts'], np.log(1 + df[target_col])   # log transform
else:
    ts_index, ts_values = df['year_period_ts'], df[target_col]
ts_values.index = pd.to_datetime(ts_index) 

val_series = validate_series(ts_values)

if transform in ['diff', 'diff_log']:
    y_lag = Retrospect(n_steps=2, step_size=1).transform(val_series)
    y_lag.dropna(inplace=True)              
    val_series = validate_series(y_lag["t-0"] - y_lag["t-1"])   # first differences
    #df = df.iloc[1: , :]           # drop first row so dimension of orig. dataframe is up-to-date after first diff. 

df.head(10)

val_series

y_lag = Retrospect(n_steps=2, step_size=1).transform(val_series)
y_lag

# No cweek 53 allowed in the following due to the following and other subsequent
# specifications in time series methods! 
s_deseasonal = deepcopy(val_series)    # instantiate