
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

y = fitted.val_series
y.head()

res1, res2_new = train.run_all(data_orig = data_orig, verbose=True)
res1.tail()
res2_new


# Create plots as in JUypter notebook....
###############################################################################

from adtk.visualization import plot
import statsmodels.api as sm
import matplotlib.ticker as ticker

# Get next series
#-------------------
label, sub_set = next(gen)

print('Claims from period {} to {}.'.format(claims.min_year_period, claims.max_year_period)) 

print(label, sub_set.shape[0])
df = deepcopy(sub_set)

df.head()

train = trainer.trainer(periodicity = periodicity, verbose=False)

fitted = train.fit(df = df)

y = fitted.ts_values
#y = fitted.val_series
out = fitted.predict(detect_thresh = None)

outlier_filter

where = np.where(np.array(claims.time_index) == outlier_filter)[0][0]
outlier_search_list = claims.time_index[where:]

filtered_outliers = []
if out.nof_outliers > 0:
    outlier_dates = out.outlier_dates
    filt = [outl in outlier_search_list for outl in outlier_dates]
    filtered_outliers = np.array(outlier_dates)[filt].tolist()
    
    if len(filtered_outliers) > 0:
        #print("\nSeries",i)
        #print(label, sub_set.shape[0])
        print("Anomaly found!")
        print(filtered_outliers)
    
#lag = 1
#y_diff = util.difference(y, lag)
# First diff.
#util.ts_plot(x=x[lag:], y=y_diff, title='Weekly claim counts (First diff.): '+label) 


# Detect anomalies:
#----------------------
inside = ''    
if label in list(claims.level_wise_aggr.keys()):

    inside = claims.level_wise_aggr[label]       # then shows over which set it was aggregated    
    #new_inside = [str(i)+'\n' for i in inside] 
    
    #main = label +':\n\n '+ str(len(filtered_outliers)) + \
    #    ' outlier(s) detected!\n' + 'Occured at year-calendar week(s): '+ \
    #    ', '.join(filtered_outliers)+'\n'+'Aggregated over:'+str(new_inside)+'\n'
    
    main = label +':\n\n '+ str(len(filtered_outliers)) + \
            ' outlier(s) detected!\n' + 'Occured at year-period(s): '+ \
            ', '.join(filtered_outliers)+'\n'+'\nAggregated over: '
    for i in inside: main += str(i)+'\n'
    
else:
    main = label +':\n\n '+ str(len(filtered_outliers)) + \
        ' outlier(s) detected!\n' + 'Occured at year-period(s): '+ \
        ', '.join(filtered_outliers)+'\n'

    
pp = plot(fitted.val_series, anomaly = fitted.anomalies, ts_linewidth=1.2, ts_markersize=6, 
     anomaly_markersize=5, anomaly_color='red', freq_as_period=False, ts_alpha=0.8, anomaly_alpha=0.5)


# other change point detection algos
#model = "rbf"  # "l2", "rbf"
#signal = train.s_deseasonal.to_numpy()
#signal = y.to_numpy()
#algo = rpt.Pelt(model=model, min_size=5).fit(signal)
#algo = rpt.Binseg(model=model).fit(signal)
#my_bkps = algo.predict(pen=np.log(len(signal))*np.std(signal)**2)

#ind = np.zeros(len(fitted.anomalies))
#ind[np.array(my_bkps)-1] = 1
#anomalies_algo = pd.Series(ind,name=fitted.anomalies.name, index=fitted.anomalies.index)
#bb = plot(fitted.val_series, anomaly_true = anomalies_algo, ts_linewidth=1.2, ts_markersize=6, 
#     at_markersize=5, at_color='red', freq_as_period=False, ts_alpha=0.8, at_alpha=0.5, title = "Algo")

#ticklabels = pp.get_xticks().tolist()
#pp.set_xticklabels(df['time'].tolist())
#pp.get_xticklabels()
#pp.get_xticks().tolist()
#pp.set_xticklabels(a)

# Anomaly probabilities:
#-------------------------
plt.figure(figsize=(12,4), dpi=100)
pro = plt.plot(fitted.anomaly_proba.index, fitted.anomaly_proba, color='tab:blue',label="prob. of anomaly", linestyle='--', marker='o', markerfacecolor='orange', linewidth=1)
plt.plot(fitted.anomaly_proba.index, [fitted.detect_thresh]*len(fitted.anomaly_proba.index), label="decision threshold",  linewidth=.5)
plt.gca().set(title="", xlabel="time", ylabel="probability", ylim = plt.ylim(-0.02, 1.05))   #plt.xlim(left=0)
locs, labels = plt.xticks()
#plt.xticks(y.index[0::3], labels=df['time'].to_numpy()[0::3], rotation=60)
plt.title(r'Anomaly probabilities $\pi_{t}, t=1,...,T$', fontdict = {'fontsize' : 14})
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()  

#start, end = pp.get_xlim()
#pp.xaxis.set_ticks(np.linspace(start, end, num=len(fitted.df['time'].values.tolist()[0::3])))
#pp.set_xticklabels(fitted.df['time'].values.tolist()[0::3], rotation=65)

#pp.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

# Density plots
#----------------------------------------------------
plt.figure(figsize=(12,3), dpi=100)
#plt.subplot(211)
y.hist(bins=40)
plt.gca().set(title=label, xlabel="counts", ylabel="abs. frequency")
#--------------------------------------------------------------------
#plt.subplot(212)
#log_y = np.log(1 + y)
#log_y.hist(bins=15)
#plt.gca().set(title='', xlabel="log counts", ylabel="abs. frequency")
#plt.subplot(212)
#y.plot(kind='kde')
#plt.gca().set(title='', xlabel="counts", ylabel="density")
#------------------------------------------------------------------------
# Draw Boxplot
if periodicity == 52 :
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,6), dpi= 80)
if periodicity == 12 :    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,6), dpi= 80)
sns.boxplot(x='year', y='target', data=sub_set, ax=axes[0])
sns.boxplot(x='month', y='target', data=sub_set, ax=axes[1]).set(ylabel="counts")
if periodicity == 52 :
    sns.boxplot(x='period', y='target', data=sub_set, ax=axes[2], orient='v').set(
    xlabel='week', ylabel="counts")
#------------------------------------------------------------------------------------------

# Set Titles
axes[0].set_title('Yearly box plots\n(Trend)', fontsize=18) 
axes[1].set_title('Monthly box plots\n(Seasonality)', fontsize=18)
if periodicity == 52 :
    axes[2].set_title('Weekly box plots\n(Seasonality)', fontsize=18)
#plt.yticks(rotation=15)
plt.xticks(rotation=45)

plt.show()

# Bayesian:
#trim = 3
#probm = exact_post_cp(y.values[trim:], alpha = .2, beta = .2, gamma = .2, delta = .2)

#x = np.arange(0, len(y))
#fig, ax = plt.subplots()
#rects1 = ax.bar(x, np.append(np.zeros(3), probm), label='posterior', width = 4)
#plt.title('Posterior p.m.f. of change point')
#plt.show()
#plt.plot(x,y,'b')
#plt.axvline(x[np.argmax(probm)], color='k', linestyle='--', lw=.65)


################################################################




anomaly_history = pd.DataFrame(columns=['time_anomaly', 'time_series_name', 'clm_cnt'])
anomaly_history

# Use pickle file as substitute for Postgres:
pkl = file.PickleService(path = "anomaly_history.pkl")

pkl.doWrite(anomaly_history)

pkl.doRead()


###### PLAYGROUND #########################################

def print_string(string : str = "")-> str:
    return string

print_string(string="ICE sucks so much...")

# def capitalize(my_func):
#     def wrapper_function(some_string : str = ""):
#         return my_func(some_string).upper()    # modify input function....
#     return wrapper_function

def capitalize(my_func):
    def wrapper_function(*args, **kwargs):       # accept any number of arbitrary arguments and keyword arguments
        return my_func(*args, **kwargs).upper()    # modify input function....
    return wrapper_function


def format_it(another_func):
    def wrapper_function2(*args, **kwargs):       # accept any number of arbitrary arguments and keyword arguments
        return another_func(*args, **kwargs).split()    # modify input function....
    return wrapper_function2


@format_it     # step 2
@capitalize    # step 1
def print_string(string : str = "")-> str:
    return string

# Call:
print_string(string="ICE sucks so much...")

# Add arguments to second decorator:
def format_new(my_arg):
    def format_it(another_func):
        def wrapper_function2(*args, **kwargs):       # accept any number of arbitrary arguments and keyword arguments
            return another_func(*args, **kwargs).split()*my_arg    # modify input function....
        return wrapper_function2
    return format_it


@format_new(my_arg = 2)     # step 2
@capitalize    # step 1
def print_string(string : str = "")-> str:
    return string

# Call:
print_string(string="ICE sucks so much...")


