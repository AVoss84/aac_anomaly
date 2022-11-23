
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

csv = file.CSVService(path=filename, delimiter=',')

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

df.shape

df = claims.upsample_time_series(df)
df.shape

#df_new = df.resample('W-MON', on = 'year_period_ts') # .median().ffill()
#df_upsample = df.resample('W-MON', on = 'year_period_ts').sum().ffill().reset_index()

#a = df.set_index('year_period_ts').resample('W-MON').mean().head(10)
#df_upsample = df.resample('W-MON', on = 'year_period_ts').ffill(limit=1)


transform = 'diff'
target_col = "target"

if transform in ['log', 'diff_log']:
    ts_index, ts_values = df['year_period_ts'], np.log(1 + df[target_col])   # log transform
else:
    ts_index, ts_values = df['year_period_ts'], df[target_col]
ts_values.index = pd.to_datetime(ts_index) 

print(ts_values.shape)

#ts_values.to_period('W-MON')
#ts_values.asfreq('W-MON', method="bfill")
#ts_values.asfreq('W-MON', method="ffill")  # propagate last valid observation forward to next valid

#ts_values = ts_values.resample('W-MON').mean().ffill()

print(ts_values.shape)
print(df.shape)

#df_upsample.merge(ts_values, how= "inner", left_index=True, right_index=True).shape  #check

val_series = validate_series(ts_values)

if transform in ['diff', 'diff_log']:
    y_lag = Retrospect(n_steps=2, step_size=1).transform(val_series)
    y_lag.dropna(inplace=True)              
    val_series = validate_series(y_lag["t-0"] - y_lag["t-1"])   # first differences
    #df = df.iloc[1: , :]           # drop first row so dimension of orig. dataframe is up-to-date after first diff. 

# y_lag = Retrospect(n_steps=2, step_size=1).transform(val_series)
# y_lag

# No cweek 53 allowed in the following due to the following and other subsequent
# specifications in time series methods! 
s_deseasonal = deepcopy(val_series)    # instantiate

# Have transfomations been specified?
#--------------------------------------
# if transformers is not None:
#     model_transf = list(transformers.keys())[0]         # take first, only one transformation allowed for now
#     transf_hyper_para = transformers[model_transf]
#     try:                
#         anomaly_transformer = eval(model_transf+"("+"**transf_hyper_para)")
#         s_deseasonal = anomaly_transformer.fit_transform(val_series)
#     except Exception as e0:
#         print(e0)
#         print("No seasonal adjustment used.")    

val_series
y_lag

import statsmodels.api as sm
from statsmodels.tsa import seasonal as sea

# https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.DecomposeResult.html#statsmodels.tsa.seasonal.DecomposeResult

result_mul = sea.seasonal_decompose(ts_values, extrapolate_trend='freq', **transformers['ClassicSeasonalDecomposition'])

s_deseasonal = result_mul.observed - result_mul.seasonal
s_deseasonal 
obs = result_mul.resid + result_mul.seasonal + result_mul.trend
obs

# plt.rcParams.update({'figure.figsize': (5,5)})
# fig = result_mul.plot()#.title('Multiplicative Decompose', fontdict = {'fontsize' : 17})
# axes_dc = fig.get_axes()
# axes_dc[0].set_title("Plot 4: Forecast number of claims - "+label, fontsize=14.5)
# axes_dc[0].set_xlabel('time index')
# #plt.title('subplot 2')
# #result_add.plot().suptitle('Additive Decompose', fontsize=15)
# plt.show()


# from sktime.transformations.series.detrend import Deseasonalizer
# from sktime.datasets import load_airline

# y = load_airline()
# transformer = Deseasonalizer()
# y_hat = transformer.fit_transform(y)

hyper_para

from adtk.visualization import plot
from adtk.detector import ThresholdAD, InterQuartileRangeAD, GeneralizedESDTestAD, PersistAD, QuantileAD
from adtk.detector import LevelShiftAD, VolatilityShiftAD, SeasonalAD, AutoregressionAD
from adtk.transformer import DoubleRollingAggregate, RollingAggregate, Retrospect, ClassicSeasonalDecomposition
from adtk.pipe import Pipeline, Pipenet

# Loop over all base learners for building the ensemble learner    
#---------------------------------------------------------------
for z, model in enumerate(hyper_para.keys()):            
    anom_detector = eval(model+"("+"**hyper_para['"+model+"'])")               # evaluate estimator expressions
    model_abstr = [(model, anom_detector)]
    pipe = Pipeline(model_abstr)
    try:
        train_res = pipe.fit_detect(s_deseasonal).rename(model, inplace=False).to_frame()     # fit estimator and predict
    except Exception as e1:
        print(e1) ; train_res = None
    if z==0:
        anomalies = deepcopy(train_res)    # instantiate
    else:
        anomalies = pd.concat([anomalies,train_res], axis=1)
    anomalies = anomalies*1.    

anomalies.fillna(0, inplace=True)         
if 'anom_detector' in dir(): 
    del anom_detector

anomaly_counts = anomalies.sum(axis=1)    
anomaly_proba = anomalies.mean(axis=1)
anomalies_or = anomalies.max(axis=1)  # union/or operation
del anomalies
