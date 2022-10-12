
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from copy import deepcopy
pd.set_option('display.max_rows', 10**5)
pd.set_option('display.max_columns', 10**5)
import os
from importlib import reload
import adtk
import inspect

from claims_reporting.utils import tsa_utils as tsa
from claims_reporting.utils import utils_func as util
from claims_reporting.services import file
from claims_reporting.config import global_config as glob
from claims_reporting.services import file
#from claims_reporting.services import base
from claims_reporting.resources import config

reload(tsa)
reload(file)
reload(config)
reload(glob)
reload(util)

#os.getcwd() 

# config_input = config.in_out['input']
# config_input
# config_output = config.in_out['output']
# config_detect = config.in_out['detection']

filename = util.get_newest_file(search_for = "AGCS CCO CRA - Monthly Incurred amounts",  src_dir=glob.UC_DATA_DIR)
xls = file.XLSXService(path=filename, root_path=glob.UC_DATA_DIR, dtype= {'time': str}, sheetname='data', index_col=None, header=0)
filename

data_orig = xls.doRead()

data_orig.shape

data_orig.head()

##############

from claims_reporting.resources import preprocessor_incurred as pre
from claims_reporting.resources import trainer

reload(pre)

from claims_reporting.utils import aggregation_functions

reload(aggregation_functions)


periodicity = 12

if periodicity == 52:
    config_input = config.in_out52['input']
    config_output = config.in_out52['output']
    config_detect = config.in_out52['detection']
if periodicity == 12:    
    config_input = config.in_out12['input']
    config_output = config.in_out12['output']
    config_detect = config.in_out12['detection']

hyper_para = config_detect['training']['hyper_para']
stat_transform = config_detect['training']['stat_transform']

"""

# Instantiate class:
#--------------------
claims = pre.claims_reporting(ts_col = 'target')

outlier_filter = claims.outlier_filter
print(outlier_filter)

#aggreg_level, pre_filter, ignore_week_lag, min_sample_size, min_median_cnts = list(config_detect['preprocessing'].values())

gen = claims.process_data(data_orig, aggreg_level = 'lob_only', ignore_lag = 1)

#get_all = dict(gen)
#get_all['all']

# all_series = dict(gen)
# for k in all_series.keys(): 
#     print(k)

# For time series clustering:
#----------------------------------
#y_names = list(get_all.keys())[:4]
#dw = tsa.timewarp(normalize=True)
#res, dist = dw.fit(y1,y2, 5)
#distmat = dw.make_distmatrix(X = get_all, series_list = y_names)
# if distmat is numpy array:
#ind = np.unravel_index(np.argsort(distmat, axis=None), distmat.shape)
#distmat[ind]
#-----------------------------------------------------------------

# Get next series
#-------------------
label, sub_set = next(gen)

print(label, sub_set.shape[0])

df = deepcopy(sub_set)

df.head(10)
#ts_bag = claims.tseries
#ts_bag.get("all")

#multi_ts = claims.multi_ts

train = trainer.trainer(verbose=False)

#for i in dir(train): print(i)
fitted = train.fit(df = df)
out = fitted.predict()
#fitted.anomalies

y = fitted.val_series
y.head()


"""

train0 = trainer.trainer(periodicity = 12, verbose=False)

# gen = train0.process_data(data_orig, aggreg_level = 'lob_only', ignore_lag = 1)

# all_series = dict(gen)

# for k in all_series.keys(): 
#     print(k)

results_all, results_new = train0.run_all(data_orig = data_orig, verbose=True, aggreg_level = 'region_only', write_table = True)  

results_all.head()
results_new

train0.all_series

for k in train0.all_series: 
    print(k[0])




#######
claims = util.claims_reporting()
claims

gen0 = claims.process_data(data_orig, ignore_week_lag = 1, min_sample_size = 30, min_median_cnts = 50)

all_series = list(gen0)

suspects, filt_suspects, filt_suspects_values = {}, {}, {}

outlier_filter = config_detect['training']['outlier_filter']

where = np.where(np.array(claims.time_index) == outlier_filter)[0][0]
outlier_search_list = claims.time_index[where:]

for i in range(len(all_series)):
    
    label, sub_set = all_series[i]
    df = deepcopy(sub_set)
    fitted = train.fit(df = df, ts_col = 'clm_cnt', stat_transform = ["none"])
    y = fitted.ts_values
    out = fitted.predict()
    
    if out.nof_outliers > 0:
        outlier_dates = out.outlier_dates
        filt = [outl in outlier_search_list for outl in outlier_dates]
        filtered_outliers = np.array(outlier_dates)[filt].tolist()
        suspects[label] = outlier_dates      # all
        
        if len(filtered_outliers) > 0:
            filt_suspects[label] = filtered_outliers
            anom_val = [df['clm_cnt'].values[df['time'].values == fdates][0] for fdates in filtered_outliers]
            filt_suspects_values[label] = {'anomaly_dates': filtered_outliers, 'anomaly_values': anom_val}
            
filt_suspects_values

res_to_pg = pd.DataFrame(columns=['time_anomaly', 'time_series_name', 'clm_cnt'])
for s_name,v in filt_suspects_values.items():
    dates_values = list(v.values())
    dates_list, values_list = dates_values[0], dates_values[1]
    assert len(dates_list) == len(values_list), 'Unequal number of anomaly dates and values!'
    tmp = pd.DataFrame(columns=['time_anomaly', 'time_series_name', 'clm_cnt'])
    for ii in range(len(dates_list)):
        combi = [dates_list[ii], s_name, values_list[ii]]
        tmp.loc[ii] = combi
    res_to_pg = res_to_pg.append(tmp, ignore_index=True)

res_to_pg

# For timestamps:
#from datetime import date
#today = date.today().strftime("%d/%m/%Y")
#today
#---------------------------------------------------------------------------------------------------------------------
pd.to_datetime(fitted.ts_index.values)
pd.DatetimeIndex(fitted.ts_index.values, freq) 

x = 0
a = {'nein' if x>.5 else 'ja' : 'ha' if x>.5 else 2}
a
os.getcwd() 

textList = ["Alex", "writes", "to file."]
#glob.UC_DATA_DIR_PKG+"myOutFile.txt"

outF = open("../data/myOutFile.txt", "a")
for line in textList:
  outF.write(line)
  outF.write("\n")
outF.close()



#------------------------------------------------------------------------------
def embed_py(data, lags, dropnan=True):
  df = pd.DataFrame(data) 
  colnames = data.columns
  cols, names = list(), list()
  k = data.shape[1]
  for j in range(0,k):
    ts = df.iloc[:,j]
    for i in range(0,lags+1):
      cols.append(ts.shift(i))                                  # lag series/shift series up
      names.append(str(colnames[j]) + '_lag' + str(i))             # make names
  agg = pd.concat(cols, axis=1)                                 # concatenate the matrix list elements to dataframe -> cbind
  agg.columns = names  
  if dropnan:
    agg.dropna(inplace=True)              	# drop rows with NaN values
  return(agg)  
#------------------------------------------------------------------------------
df = pd.DataFrame(np.random.random((20,2)), columns=['Var1','Var2']);df
embed_py(df, 2, True).head(5)



from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import ThresholdAD, InterQuartileRangeAD, GeneralizedESDTestAD, PersistAD
from adtk.detector import LevelShiftAD, VolatilityShiftAD, SeasonalAD, AutoregressionAD

from adtk.transformer import DoubleRollingAggregate, ClassicSeasonalDecomposition
from adtk.pipe import Pipeline, Pipenet
from adtk.aggregator import AndAggregator, OrAggregator
from adtk.data import split_train_test

# Get next series
label, sub_set, size = next(gen)

print(label)
print(size)

df = deepcopy(sub_set)
df['year_week_ts'] = df.apply(lambda row: util.year_week(row.year, row.week), axis=1)
df['year_week_str'] = df.apply(lambda row: row.year_week_ts.strftime('%G-%V'), axis=1)

df.head()

x, y = df['year_week_ts'], df['clm_cnt']

y.index = x
s = validate_series(y)

quantile_ad = QuantileAD(high=0.99, low=0.01)
anomalies = quantile_ad.fit_detect(s)

#  generalized extreme Studentized deviate (ESD) test
esd_ad = GeneralizedESDTestAD(alpha=0.3)
anomalies = esd_ad.fit_detect(s)

iqr_ad = InterQuartileRangeAD(c=1.5)
anomalies = iqr_ad.fit_detect(s)

lags = 1
autoregression_ad = AutoregressionAD(n_steps=lags, step_size=1, c=3.0, side = "both")
anomalies = autoregression_ad.fit_detect(s)


class trainer:

    """Trains anomaly detector ensemble""" 

    def __init__(self, verbose=True, 
                       hyper_para = {'QuantileAD': {'high': 0.98, 'low' : 0.01}, 
                                     'GESD_test': {'alpha' : 0.3}, 
                                     'InterQuartileRangeAD': {'c': 3.0}, 
                                     'AutoregressionAD': {'n_steps': 1, 'step_size':1, 'c': 3.0, 'side':  "both"}}
                     ):

        self.verbose = verbose
        self.hyper_para = hyper_para
        self.steps = {
            #"roll_transf": {
            #    "model": DoubleRollingAggregate(
            #        agg="mean",
            #        window=1,
            #        center=True,
            #        diff="diff"
            #    ),
            #    "input": "original"
            #},
            "GESD_test": {
                "model": GeneralizedESDTestAD(**hyper_para['GESD_test']),   # generalized extreme Studentized deviate (ESD) test
                "input": "original"
            },
            "AR_res_shift": {
                "model": AutoregressionAD(**hyper_para['AutoregressionAD']),
                "input": "original"
            },
            "IQ_shift": {
                "model": InterQuartileRangeAD(**hyper_para['InterQuartileRangeAD']),
                "input": "original"
                #"input": "roll_transf"
            },
            "quantile_shift": {
                "model": QuantileAD(**hyper_para['QuantileAD']),
                "input": "original"
                #"input": "roll_transf"
            },
            "positive_level_shift": {
                #"model": AndAggregator(),
                "model": OrAggregator(),
                "input": ["quantile_shift", "IQ_shift", "AR_res_shift", "GESD_test"]
            }
        }

    def __del__(self):
        class_name = self.__class__.__name__

    def fit(self, ts_index, ts_values): 

        if self.verbose:
            print("Training outlier ensemble...")
        ts_values.index = ts_index
        s = validate_series(ts_values)
        self.pipenet = Pipenet(self.steps)
        self.anomalies = self.pipenet.fit_detect(s)
        return self

    def predict(self):

        self.nof_outliers = np.nansum(self.anomalies.tolist())
        self.outliers = np.where(self.anomalies.to_numpy())[0].tolist()
        self.outlier_dates = df.iloc[self.outliers,:].time.tolist()
        if self.verbose:
            print('\n->', self.nof_outliers,"anomalies detected!") 
            print("Occured at year-calendar week(s):\n", self.outlier_dates)
        return self


#--------------------------------------------------------------------------------------

reload(util)

claims = util.claims_reporting()

gen = claims.process_data(data_orig, min_sample_size = 15)

train = util.trainer(verbose=True)

fitted = train.fit(x,y)
out = fitted.predict()



