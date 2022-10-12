
########## Note: This is for incurred only, not for the deviations from expected and planned (see V1)
############################################################################################################

import os, warnings
#warnings.filterwarnings("ignore")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
#import pweave     # for markdown reports
#os.chdir("..")
from claims_reporting.utils import tsa_utils as tsa
from claims_reporting.utils import utils_func as util
from claims_reporting.services import file

from claims_reporting.config import global_config as glob
#from claims_reporting.service import base
from claims_reporting.resources import config

reload(tsa)
reload(file)
reload(config)
reload(glob)

#os.getcwd()
#reload(util)

filename = util.get_newest_file(search_for = "AGCS CCO CRA - Monthly Incurred amounts",  src_dir=glob.UC_DATA_DIR)
xls = file.XLSXService(path=filename, root_path=glob.UC_DATA_DIR, dtype= {'time': str}, sheetname='data', index_col=None, header=0)
filename

data_orig = xls.doRead()

data_orig.shape

data_orig.head()

#pg = file.PostgresService(verbose=False)

#data_orig = pg.doRead(qry = 'select * from "Incurred_Expected_CAYPAY"')   
#data_orig.shape
#data_orig.head()

import pandas as pd
import numpy as np
from copy import deepcopy
import glob as gl
import subprocess, os
from datetime import datetime, date
import dateutil.parser as dateparser

from dateutil.relativedelta import relativedelta

from textdistance import jaro_winkler
from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import ThresholdAD, InterQuartileRangeAD, GeneralizedESDTestAD, PersistAD, QuantileAD
from adtk.detector import LevelShiftAD, VolatilityShiftAD, SeasonalAD, AutoregressionAD
from adtk.transformer import DoubleRollingAggregate, RollingAggregate, Retrospect, ClassicSeasonalDecomposition
from adtk.pipe import Pipeline, Pipenet
from adtk.aggregator import AndAggregator, OrAggregator
from adtk.data import split_train_test
from importlib import reload
from claims_reporting.config import global_config as glob
from claims_reporting.resources import config
from claims_reporting.services import file


ts_col='target'
periodicity = 12

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

hyper_para = config_detect['training']['hyper_para']
transformers = config_detect['training']['transformers']
stat_transform = config_detect['training']['stat_transform']
outlier_filter = config_detect['training']['outlier_filter']
aggreg_level, pre_filter, ignore_lag, min_sample_size, min_median_cnts = list(config_detect['preprocessing'].values())
tbl_name = config_output['database']['tbl_name']
detect_thresh = config_detect['prediction']['detect_thresh']


if outlier_filter is None:
        six_months_ago = date.today() - relativedelta(months=+6)
        outlier_filter = six_months_ago.strftime("%Y-%m")
        print('Detect anomalies not older than {}.'.format(outlier_filter))


# model_transf = list(transformers.keys())[0]
# model_transf

# from adtk.detector import OutlierDetector
# from sklearn.neighbors import LocalOutlierFactor
# from sklearn.ensemble import IsolationForest

# reload(config)

# for z, model in enumerate(hyper_para.keys()): 
#         print(z, model)

# model+"("+"**hyper_para['"+model+"'])"
# model
# hyper_para['OutlierDetector']

# anom_detector = eval(model+"("+"**hyper_para['"+model+"'])")               # evaluate estimator expressions
# model_abstr = [(model, anom_detector)]
# pipe = Pipeline(model_abstr)
#train_res = pipe.fit_detect(self.s_deseasonal).rename(model, inplace=False).to_frame()     # fit estimator and predict



def _convert2date(y, t):
        return datetime.strptime(f'{y} {t} 1', '%Y %m %d')  # %G, %V, %u are ISO equivalents of %Y, %W, %w

def _year_week(y, w):
        return datetime.strptime(f'{y} {w} 1', '%G %V %u')  # %G, %V, %u are ISO equivalents of %Y, %W, %w

def _correct_cweek_53(dat : pd.core.frame.DataFrame, time_col : str = 'year_period_ts', target_col : str = 'clm_cnt', verbose=False):
        """
        This aggregates non-unique dates which were generated by 
        transforming from calendar week to a representative day 
        of that week (using _year_week function)
        Example: 
        2019-53 : 2019-12-30 and 2020-01: 2019-12-30. 
        These two will be summed up to have unique date points 
        for time series analysis. Result will remove 2019-53 for example
        """
        my_ts = deepcopy(dat)
        ssa = my_ts.groupby([time_col]) 
        #time_duplicates = ssa.agg(size=(target_col,'count')).reset_index()
        deduplicated_ts = ssa.agg(target = (target_col,'sum')).reset_index()
        my_ts.drop(columns=[target_col], inplace = True)
        my_ts_new = my_ts.merge(deduplicated_ts, how='left', left_on=[time_col], right_on=[time_col])
        my_ts_new.drop_duplicates(subset=[time_col, target_col], keep='last', inplace=True)
        my_ts_new.reset_index(inplace=True)
        if verbose:
            print('{} rows aggregated'.format(my_ts.shape[0]-my_ts_new.shape[0]))
        return my_ts_new

def q_50(x):
    return x.quantile(0.5)

def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)

#########################################################################################
############################### START
######################################################################################

aggreg_level = 'all_combi'
ignore_lag = 1 
min_sample_size = 20
min_median_target = 0 
verbose = True
aggreg_dimensions = ['Region', 'OE', 'Lob',  'LossCat']


# Overwrite I/O yaml spec. if user arguments are given:
#--------------------------------------------------------
if ignore_lag is not None:
    ignore_lag = ignore_lag
if min_sample_size is not None:    
    min_sample_size = min_sample_size
if min_median_target is not None:
    min_median_target = min_median_target    
if aggreg_level is not None:
    aggreg_level = aggreg_level   
if verbose:
    print('Periodicity: {}'.format(periodicity)) 
    print("Aggregation type: '{}'".format(aggreg_level)) 
print('Parameters: ignore_lag {}, min_sample_size {}, min_median_target {}'.format(ignore_lag, min_sample_size, min_median_target)) 

df0 = deepcopy(data_orig)
df0.rename(columns={'Line of Business': 'Lob', 'Source System': 'source_sys', 'Sub Line of Business': 'SubLob', 'Loss Category': 'LossCat'}, inplace=True)

df0.columns

# Apply a prefiltering step, 
# most often to filter out Brazil, South America data
if pre_filter is not None:
    try:
        df0 = df0.query(pre_filter)
        if verbose: print('pre-filter applied.')
    except Exception as e:
        if verbose: print(pre_filter,e)  

df = deepcopy(df0)
print(df.shape)
#df.dropna(inplace=True)
print(df.shape)

years = df['time'].apply(lambda x: x[:4]).astype(int)
periods = df['time'].apply(lambda x: x[5:]).astype(int)


max_year = max(years)
min_year = min(years)
max_years = (years == max_year)
min_years = (years == min_year)
max_period = max(periods[max_years])
min_period = min(periods[min_years])

thresh_period = max_period - ignore_lag
thresh_year = max_year

min_year_period = str(int(min_year))+'-'+'{0:02d}'.format(int(min_period))
max_year_period = str(int(max_year))+'-'+'{0:02d}'.format(int(max_period))
#min_calendar_week = str(int(min_year))+'-'+'{0:02d}'.format(int(min_week))
#max_calendar_week = str(int(max_year))+'-'+'{0:02d}'.format(int(max_week))
if verbose:
    print("Ignoring claims younger than:",str(int(thresh_year))+'-'+'{0:02d} in detection'.format(int(thresh_period)))

time_filter = ~((periods > thresh_period) & (years == thresh_year))
#df_orig = deepcopy(df)

# Apply time filter to prefiltered data:
df = df[time_filter]
periods, years = periods[time_filter], years[time_filter]
period_seq = np.arange(min(periods), max(periods)+1)
year_seq = np.arange(min(years), max(years)+1)

# For multivariate time series creation below:
#----------------------------------------------
time_index = []
for y in year_seq:
        for p in period_seq:
            time_index.append(str(int(y))+'-'+'{0:02d}'.format(int(p)))  # calendar weeks
#max_cweek_index = np.where(max_calendar_week == np.array(time_index))[0][0]
max_period_index = np.where(max_year_period == np.array(time_index))[0][0]

df.columns

agg_func = 'sum'

# Deduplicate:
#---------------
#gr0 = df.groupby(aggreg_dimensions + ['time'])
#gr0 = df.groupby(['Region', 'OE', 'Lob', 'SubLob', 'LossCat'] + ['time'])
gr0 = df.groupby(['Region', 'OE', 'Lob', 'LossCat'] + ['time'])
data_lev1 = gr0.agg(target = (target_col, agg_func)).reset_index()

#data_lev1 = gr0.agg(target = (target_col,'sum'), reference0 = (reference_col0,'sum')).reset_index()
data_lev1

ts_bag = {}

##################################################
# Level 1 (LOB - OE - Region - LossCat)
##################################################

#aggreg_dimensions1 = aggreg_dimensions.copy()
#print("Aggregation level 1: {}".format(aggreg_dimensions1))

#gr = data_lev1.groupby(aggreg_dimensions1)
#gr = data_lev1.groupby(['Region', 'OE', 'Lob', 'SubLob', 'LossCat'])
gr = data_lev1.groupby(['Region', 'OE', 'Lob', 'LossCat'])

# Summary wrt to target:
gr[target_col].describe().unstack()

sample1 = gr.agg(size=(target_col, 'count'), q_50_target=(target_col, q_50)
                #, iqr_cnt=('clm_cnt',iqr)
                ).reset_index()

sample1['pooling'] = (((sample1['size'] < min_sample_size) | (sample1['q_50_target'] < min_median_target))*1).astype(object)

lookup_level1_single = sample1.query('pooling == 0')      # singles
lookup_level1_singles = lookup_level1_single.drop(columns='pooling', inplace=False)
lookup_level1_ag = sample1.query('pooling == 1')        # aggregate
lookup_level1_agg = lookup_level1_ag.drop(columns='pooling', inplace=False)

# for _, ts_info in lookup_level1_singles.iterrows():
#         #combi = tuple(ts_info[:len(aggreg_dimensions1)])
#         #combi = tuple(ts_info[:len(['Region', 'OE', 'Lob', 'SubLob', 'LossCat'])])
#         combi = tuple(ts_info[:len(['Region', 'OE', 'Lob', 'LossCat'])])
#         ts_slice = gr.get_group(combi)#.reset_index()
#         ts_slice['year'] = ts_slice['time'].apply(lambda x: x[:4]).astype(int)
#         ts_slice['period'] = ts_slice['time'].apply(lambda x: x[5:]).astype(int)
#         if periodicity == 52: 
#                 ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _year_week(row.year, row.period), axis=1)            
#         if periodicity == 12:
#                 ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _convert2date(row.year, row.period), axis=1)
#         ts_slice['month'] = [x.month for x in ts_slice['year_period_ts'].tolist()]
#         ts_bag['-'.join(combi)] = ts_slice[['time', 'target', 'year', 'month','period']]
#         print("Sample size {}".format(ts_bag['-'.join(combi)].shape))

for _, ts_info in lookup_level1_singles.iterrows():
        #combi = tuple(ts_info[:len(aggreg_dimensions1)])
        combi = tuple(ts_info[:len(['Region', 'OE', 'Lob', 'LossCat'])])
        ts_slice_grouped = gr.get_group(combi)
        ts_slice = ts_slice_grouped.set_index('time', inplace=False)['target'].reset_index()
        # Format dates from 'YYYY-period' to 'YYYY-MM-01'
        # needed so asfreq can recognize it as date index:
        ts_slice['period'] = ts_slice['time'].apply(lambda x: x[5:]).astype(int)
        ts_slice['year'] = ts_slice['time'].apply(lambda x: x[:4]).astype(int)

        # Adjust frequencies in case of irregular time series 
        # (i.e. missing timestamps, not full cycle)
        #------------------------------------------------------
        if periodicity == 12:
                ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _convert2date(row.year, row.period), axis=1)
                my_series = ts_slice[['year_period_ts', 'target']].set_index('year_period_ts', inplace=False)
                ts_slice = my_series.asfreq(freq = 'MS', fill_value = 0.).reset_index().rename(columns={'year_period_ts' : 'time'})
                ts_slice['year'] = pd.DatetimeIndex(ts_slice['time']).year
                ts_slice['period'] = pd.DatetimeIndex(ts_slice['time']).month
                ts_slice['month'] = ts_slice['period']
                ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _convert2date(row.year, row.period), axis=1)

        if periodicity == 52:
                #ts_slice = my_series.asfreq(freq = 'W', fill_value = 0.).reset_index()
                #ts_slice['year'] = pd.DatetimeIndex(ts_slice['time']).year
                ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _year_week(row.year, row.period), axis=1)
                ts_slice['month'] = [x.month for x in ts_slice['year_period_ts'].tolist()]

        ts_bag['-'.join(combi)] = ts_slice[['time', 'target', 'year', 'month','period']]
        #print("Sample size {}".format(ts_bag['-'.join(combi)].shape))
ts_bag

ts_bag['RUL-UK-Property-4-Cat'].tail(20)

#print("Aggregation level 1: {}".format(aggreg_dimensions1))

# Next take the left-overs (having too short time series) 
# and aggregate further -> level 2
#----------------------------------------------------------
data_lev2 = data_lev1.merge(lookup_level1_agg[['Region', 'OE', 'Lob', 'LossCat']], how='right', left_on=['Region', 'OE', 'Lob', 'LossCat'], right_on=['Region', 'OE', 'Lob', 'LossCat'])
#data_lev2 = data_lev1.merge(lookup_level1_agg[aggreg_dimensions1], how='right', left_on=aggreg_dimensions1, right_on=aggreg_dimensions1)
data_lev2.shape


##################################################
# Level 2 (OE - Region - LossCat)
##################################################

#aggreg_dimensions2 = aggreg_dimensions1.copy()

#agg_variable_level2 = 'Lob'
#aggreg_dimensions2.remove(agg_variable_level2)    # removes inplace!

#print("Aggregation level 2: {}".format(aggreg_dimensions2))

#gr2 = data_lev2.groupby(aggreg_dimensions2+['time'])
gr2 = data_lev2.groupby(['Region', 'OE', 'LossCat']+['time'])

data_lev2_new = gr2.agg(target = (target_col,'sum')).reset_index()     # aggregate

# Check cluster size again for filtering:
#-------------------------------------------
#gr2b = data_lev2_new.groupby(aggreg_dimensions2)
gr2b = data_lev2_new.groupby(['Region', 'OE', 'LossCat'])
sample2 = gr2b.agg(size=(target_col,'count'), q_50_target=(target_col,q_50)).reset_index()

sample2['pooling'] = (((sample2['size'] < min_sample_size) | (sample2['q_50_target'] < min_median_target))*1).astype(object)

lookup_level2_single = sample2.query('pooling == 0')      # singles, constitute unpooled time series on that level
lookup_level2_singles = lookup_level2_single.drop(columns='pooling', inplace=False)
lookup_level2_ag = sample2.query('pooling == 1')        # aggregate further in next level!
lookup_level2_agg = lookup_level2_ag.drop(columns='pooling', inplace=False)


# # Append to already existing dictionary:
# #-----------------------------------------
# level_wise_aggr = {}
# for _, ts_info in lookup_level2_singles.iterrows():
#         #combi = tuple(ts_info[:len(aggreg_dimensions2)])
#         combi = tuple(ts_info[:len(['Region', 'OE', 'LossCat'])])
#         #oe_region = list(combi)

#         ts_slice = gr2b.get_group(combi).reset_index()
#         ts_slice['year'] = ts_slice['time'].apply(lambda x: x[:4]).astype(int)
#         ts_slice['period'] = ts_slice['time'].apply(lambda x: x[5:]).astype(int) 

#         if periodicity == 52: 
#                 ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _year_week(row.year, row.period), axis=1)            
#         if periodicity == 12:
#                 ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _convert2date(row.year, row.period), axis=1)

#         ts_slice['month'] = [x.month for x in ts_slice['year_period_ts'].tolist()]
#         ts_bag['-'.join(combi)] = ts_slice[['time', 'target', 'year', 'month','period']]

#         # Find all level combinations over which we aggregated for titles in plots! 
#         # Iteratively compute boolean filter 
#         # vector via chain of 'and' operations per column:
#         #-------------------------------------------------------------------------- 
#         temp_mask = True 
#         for i in range(len(combi)):
#             #temp_mask *= (lookup_level1_agg[aggreg_dimensions2[i]] == combi[i])
#             temp_mask &= (lookup_level1_agg[['Region', 'OE', 'LossCat'][i]] == combi[i])

#         # Select the levels of agg_variable_level over which we have aggregated (for figure captions only)
#         aggregation_categories = lookup_level1_agg[temp_mask]['Lob'] #[agg_variable_level2]
#         level_wise_aggr['-'.join(combi)] = tuple(aggregation_categories)

#         # Get OE-Region_Lob combination that was aggregated:
#         # for titles in plots!
#         #aggreg_lobs = lookup_level1_agg[(lookup_level1_agg['OE'] == oe_region[0]) & (lookup_level1_agg['Region'] == oe_region[1])]['Lob'] 
#         #level_wise_aggr['-'.join(tuple(oe_region))] = tuple(aggreg_lobs)


# Append to already existing dictionary:
#-----------------------------------------
level_wise_aggr = {}
for _, ts_info in lookup_level2_singles.iterrows():
        #combi = tuple(ts_info[:len(aggreg_dimensions2)])

        combi = tuple(ts_info[:len(['Region', 'OE', 'LossCat'])])
        ts_slice_grouped = gr2b.get_group(combi)
        ts_slice = ts_slice_grouped.set_index('time', inplace=False)['target'].reset_index()

        # Format dates from 'YYYY-period' to 'YYYY-MM-01'
        # needed so asfreq can recognize it as date index:
        ts_slice['period'] = ts_slice['time'].apply(lambda x: x[5:]).astype(int)
        ts_slice['year'] = ts_slice['time'].apply(lambda x: x[:4]).astype(int)

        # Adjust frequencies in case of irregular time series 
        # (i.e. missing timestamps, not full cycle)
        #------------------------------------------------------
        if periodicity == 12:
                ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _convert2date(row.year, row.period), axis=1)
                my_series = ts_slice[['year_period_ts', 'target']].set_index('year_period_ts', inplace=False)
                ts_slice = my_series.asfreq(freq = 'MS', fill_value = 0.).reset_index().rename(columns={'year_period_ts' : 'time'})
                ts_slice['year'] = pd.DatetimeIndex(ts_slice['time']).year
                ts_slice['period'] = pd.DatetimeIndex(ts_slice['time']).month
                ts_slice['month'] = ts_slice['period']
                ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _convert2date(row.year, row.period), axis=1)

        if periodicity == 52:
                #ts_slice = my_series.asfreq(freq = 'W', fill_value = 0.).reset_index()
                #ts_slice['year'] = pd.DatetimeIndex(ts_slice['time']).year
                ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _year_week(row.year, row.period), axis=1)
                ts_slice['month'] = [x.month for x in ts_slice['year_period_ts'].tolist()]

        ts_bag['-'.join(combi)] = ts_slice[['time', 'target', 'year', 'month','period']]

        # Find all level combinations over which we aggregated for titles in plots! 
        # Iteratively compute boolean filter 
        # vector via chain of 'and' operations per column:
        #-------------------------------------------------------------------------- 
        temp_mask = True 
        for i in range(len(combi)):
            #temp_mask *= (lookup_level1_agg[aggreg_dimensions2[i]] == combi[i])
            temp_mask &= (lookup_level1_agg[['Region', 'OE', 'LossCat'][i]] == combi[i])

        # Select the levels of agg_variable_level over which we have aggregated (for figure captions only)
        aggregation_categories = lookup_level1_agg[temp_mask]['Lob'] #[agg_variable_level2]
        level_wise_aggr['-'.join(combi)] = tuple(aggregation_categories)

ts_bag

# Complementary set to loop above
# Which (Lob/OE/Region/..) combis remain after level2 (aggre. w.r.t. agg_variable_level2), 
# thus has to be further aggreg. in level3:
#------------------------------------------------------------------------
#lookup_agg_1_2 = lookup_level1_agg.merge(lookup_level2_agg, how='right', left_on=aggreg_dimensions2, right_on=aggreg_dimensions2)
lookup_agg_1_2 = lookup_level1_agg.merge(lookup_level2_agg, how='right', left_on=['Region', 'OE', 'LossCat'], right_on=['Region', 'OE', 'LossCat'])

# -> join LoB levels back, since in the orginal aggregation you have LoB too (was only aggregated wrt SubLob)  

# Next take the left-overs you have to aggregate:
#--------------------------------------------------
#data_lev3 = data_lev2.merge(lookup_agg_1_2, how='right', left_on=aggreg_dimensions1, right_on=aggreg_dimensions1) 
data_lev3 = data_lev2.merge(lookup_agg_1_2, how='right', left_on=['Region', 'OE', 'Lob','LossCat'], right_on=['Region', 'OE', 'Lob','LossCat']) 
data_lev3


##################################################
# Level 3 (Region - LossCat)
##################################################

#aggreg_dimensions3 = aggreg_dimensions2.copy()

#agg_variable_level3 = ['Lob','OE']
#aggreg_dimensions3.remove(agg_variable_level3)    # removes inplace!

#print("Aggregation level 3: {}".format(aggreg_dimensions3))

gr3 = data_lev3.groupby(['Region','LossCat']+['time'])
#gr3 = data_lev3.groupby(aggreg_dimensions3+['time'])
data_lev3_new = gr3.agg(target = (target_col,'sum')).reset_index()     # aggregate

# Check cluster size again for filtering:
#-------------------------------------------
gr3b = data_lev3_new.groupby(['Region','LossCat'])
#gr3b = data_lev3_new.groupby(aggreg_dimensions3)
sample3 = gr3b.agg(size=(target_col,'count'), q_50_target=(target_col,q_50)).reset_index()

sample3['pooling'] = (((sample3['size'] < min_sample_size) | (sample3['q_50_target'] < min_median_target))*1).astype(object)

lookup_level3_single = sample3.query('pooling == 0')      # singles
lookup_level3_singles = lookup_level3_single.drop(columns='pooling', inplace=False)
lookup_level3_ag = sample3.query('pooling == 1')        # aggregate
lookup_level3_agg = lookup_level3_ag.drop(columns='pooling', inplace=False)

aggr_after_lev2 = data_lev3.merge(lookup_level3_singles, how='right', left_on=['Region','LossCat'], right_on=['Region','LossCat']) 
#aggr_after_lev2 = data_lev3.merge(lookup_level3_singles, how='right', left_on=aggreg_dimensions3, right_on=aggreg_dimensions3) 

# Get all corresponding combinations to loop over next,
# Based on singles in level3
#-------------------------------------------------------
gr4 = aggr_after_lev2.groupby(['Region', 'OE', 'Lob','LossCat'])
#gr4 = aggr_after_lev2.groupby(aggreg_dimensions1)
data_lev4 = gr4.agg(size4 = (target_col,'count')).reset_index()     # aggregate

# Append to already existing dictionary:
#--------------------------------------------------
# for _, ts_info in lookup_level3_singles.iterrows():
#         #combi = tuple(ts_info[:1])     
#         #combi = tuple(ts_info[:len(aggreg_dimensions3)])                   
#         combi = tuple(ts_info[:len(['Region','LossCat'])])

#         ts_slice = gr3b.get_group(combi).reset_index()
#         ts_slice['year'] = ts_slice['time'].apply(lambda x: x[:4]).astype(int)
#         ts_slice['period'] = ts_slice['time'].apply(lambda x: x[5:]).astype(int)

#         if periodicity == 52: 
#                 ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _year_week(row.year, row.period), axis=1)            
#         if periodicity == 12:
#                 ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _convert2date(row.year, row.period), axis=1)

#         ts_slice['month'] = [x.month for x in ts_slice['year_period_ts'].tolist()]
#         ts_bag['-'.join(combi)] = ts_slice[['time', 'target', 'year', 'month','period']]

#         # For figure captions: collect what was aggregated    
#         #aggreg_oe_lob = data_lev4[(data_lev4['Region'] == combi[0])][['Lob','OE']]
#         #aggreg = aggreg_oe_lob.to_records(index=False)
#         #level_wise_aggr[combi] = tuple(aggreg) if len(aggreg)>1 else tuple(aggreg)[0]

#         # Find all level combinations over which we aggregated for titles in plots! 
#         # Iteratively compute boolean filter 
#         # vector via chain of 'and' operations per column:
#         #-------------------------------------------------------------------------- 
#         temp_mask = True 
#         for i in range(len(combi)):
#             #temp_mask *= (lookup_level2_agg[aggreg_dimensions3[i]] == combi[i])
#             #temp_mask *= (lookup_level2_agg[['Region','LossCat'][i]] == combi[i])
#             temp_mask &= (data_lev4[['Region','LossCat'][i]] == combi[i])

#         # Select the levels of agg_variable_level over which we have aggregated (for figure captions only)
#         #aggregation_levels = lookup_level2_agg[temp_mask][agg_variable_level3]   
#         aggregation_levels = data_lev4[temp_mask][['Lob','OE']]
#         level_wise_aggr['-'.join(combi)] = tuple(aggregation_levels)


# Append to already existing dictionary:
#-------------------------------------------
for _, ts_info in lookup_level3_singles.iterrows():
        
        combi = tuple(ts_info[:len(['Region','LossCat'])])
        ts_slice_grouped = gr3b.get_group(combi)
        ts_slice = ts_slice_grouped.set_index('time', inplace=False)['target'].reset_index()

        # Format dates from 'YYYY-period' to 'YYYY-MM-01'
        # needed so asfreq can recognize it as date index:
        ts_slice['period'] = ts_slice['time'].apply(lambda x: x[5:]).astype(int)
        ts_slice['year'] = ts_slice['time'].apply(lambda x: x[:4]).astype(int)

        # Adjust frequencies in case of irregular time series 
        # (i.e. missing timestamps, not full cycle)
        #------------------------------------------------------
        if periodicity == 12:
                ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _convert2date(row.year, row.period), axis=1)
                my_series = ts_slice[['year_period_ts', 'target']].set_index('year_period_ts', inplace=False)
                ts_slice = my_series.asfreq(freq = 'MS', fill_value = 0.).reset_index().rename(columns={'year_period_ts' : 'time'})
                ts_slice['year'] = pd.DatetimeIndex(ts_slice['time']).year
                ts_slice['period'] = pd.DatetimeIndex(ts_slice['time']).month
                ts_slice['month'] = ts_slice['period']
                ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _convert2date(row.year, row.period), axis=1)
        if periodicity == 52:
                #ts_slice = my_series.asfreq(freq = 'W', fill_value = 0.).reset_index()
                #ts_slice['year'] = pd.DatetimeIndex(ts_slice['time']).year
                ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _year_week(row.year, row.period), axis=1)
                ts_slice['month'] = [x.month for x in ts_slice['year_period_ts'].tolist()]

        ts_bag['-'.join(combi)] = ts_slice[['time', 'target', 'year', 'month','period']]

        # Find all level combinations over which we aggregated for titles in plots! 
        # Iteratively compute boolean filter 
        # vector via chain of 'and' operations per column:
        #-------------------------------------------------------------------------- 
        temp_mask = True 
        for i in range(len(combi)):
            temp_mask &= (data_lev4[['Region','LossCat'][i]] == combi[i])

        # Select the levels of agg_variable_level over which we have aggregated (for figure captions only)
        aggregation_levels = data_lev4[temp_mask][['Lob','OE']]
        level_wise_aggr['-'.join(combi)] = tuple(aggregation_levels)

ts_bag


# Next take the left-overs you have to aggregate:
#--------------------------------------------------
data_leftover = data_lev3_new.merge(lookup_level3_agg[['Region','LossCat']], how='right', left_on=['Region','LossCat'], right_on=['Region','LossCat'])
if verbose:
    print("Finished data preprocessing.")
    print("Number of left over aggregations:", lookup_level3_agg.shape[0])


# Finally do a an aggregation over only LossCat:
#-------------------------------------------------
gr_full0 = df.groupby(['LossCat']+['time'])
full_agg_series = gr_full0.agg(target = (ts_col,'sum')).reset_index()
full_agg_series['year'] = full_agg_series['time'].apply(lambda x: x[:4]).astype(int)
full_agg_series['period'] = full_agg_series['time'].apply(lambda x: x[5:]).astype(int)

if periodicity == 52: 
        full_agg_series['year_period_ts'] = full_agg_series.apply(lambda row: _year_week(row.year, row.period), axis=1)
        full_agg_series['month'] = [x.month for x in full_agg_series['year_period_ts'].tolist()]
        for lc in full_agg_series['LossCat'].unique():
            ts_bag['all-'+str(lc)] = full_agg_series.loc[full_agg_series['LossCat'] == lc, ['time', 'target', 'year', 'month','period']]

if periodicity == 12:
        full_agg_series['year_period_ts'] = full_agg_series.apply(lambda row: _convert2date(row.year, row.period), axis=1)
        for lc in full_agg_series['LossCat'].unique():
                ts_slice0 = full_agg_series.loc[full_agg_series['LossCat'] == lc, ['year_period_ts', 'target']].set_index('year_period_ts', inplace=False)
                ts_slice = ts_slice0.asfreq(freq = 'MS', fill_value = 0.).reset_index().rename(columns={'year_period_ts' : 'time'})
                ts_slice['year'] = pd.DatetimeIndex(ts_slice['time']).year
                ts_slice['month'] = pd.DatetimeIndex(ts_slice['time']).month
                ts_slice['period'] = ts_slice['month']
                ts_bag['all-'+str(lc)] = ts_slice


# Finally do a full aggregation over ALL dimensions:
#----------------------------------------------------
#gr_full = df0.groupby(['time'])     # unfiltered wrt to time filter and nans, reference for xls
gr_full = df.groupby(['time'])     # filtered wrt to time filter and nans
full_agg_series = gr_full.agg(target = (ts_col,'sum')).reset_index()
full_agg_series['year'] = full_agg_series['time'].apply(lambda x: x[:4]).astype(int)
full_agg_series['period'] = full_agg_series['time'].apply(lambda x: x[5:]).astype(int)

if periodicity == 52: 
        full_agg_series['year_period_ts'] = full_agg_series.apply(lambda row: _year_week(row.year, row.period), axis=1)
        full_agg_series['month'] = [x.month for x in full_agg_series['year_period_ts'].tolist()]

if periodicity == 12:
        full_agg_series['year_period_ts'] = full_agg_series.apply(lambda row: _convert2date(row.year, row.period), axis=1)
        ts_slice0 = full_agg_series[['year_period_ts', 'target']].set_index('year_period_ts', inplace=False)
        full_agg_series = ts_slice0.asfreq(freq = 'MS', fill_value = 0.).reset_index().rename(columns={'year_period_ts' : 'time'})
        full_agg_series['year'] = pd.DatetimeIndex(full_agg_series['time']).year
        full_agg_series['month'] = pd.DatetimeIndex(full_agg_series['time']).month
        full_agg_series['period'] = full_agg_series['month']

ts_bag['all'] = full_agg_series[['time', 'target', 'year', 'month','period']]


# Construct multivariate time series:
#---------------------------------------
tseries = deepcopy(ts_bag)

# Make proper proper date index
# in case you want to use multivariate ts methods:        
#---------------------------------------------------
# Format dates from 'YYYY-period' to proper date index 'YYYY-MM-01'
# needed e.g. so seasonal decomp. can recognize it as date index:
multi_ts_tmp = pd.DataFrame(time_index, columns = ['time'])
multi_ts_tmp['year'] = multi_ts_tmp['time'].apply(lambda x: x[:4]).astype(int)
multi_ts_tmp['period'] = multi_ts_tmp['time'].apply(lambda x: x[5:]).astype(int)

if periodicity == 12:
        multi_ts_tmp['year_period_ts'] = multi_ts_tmp.apply(lambda row: _convert2date(row.year, row.period), axis=1)
        multi_ts = multi_ts_tmp.set_index('year_period_ts', inplace=False).drop(columns=['time', 'year', 'period'])
if periodicity == 52: 
        multi_ts_tmp['year_period_ts'] = multi_ts_tmp.apply(lambda row: _year_week(row.year, row.period), axis=1)
        multi_ts = multi_ts_tmp.set_index('year_period_ts', inplace=False).drop(columns=['time', 'year', 'period']) 

for name, ts in tseries.items():
    ts.index = ts['time']    
    multi_ts[name] = ts[target_col]   

multi_ts = multi_ts.loc[multi_ts.index[:max_period_index+1],:]
multi_ts.fillna(value=0, inplace=True)   # replace missings by zeros!
multi_ts

# Make proper proper date index
# in case you want to use multivariate ts methods:        
#---------------------------------------------------
# multi_ts = multi_ts.reset_index().rename(columns={'index': 'time'})
# multi_ts['year'] = multi_ts['time'].apply(lambda x: x[:4]).astype(int)
# multi_ts['period'] = multi_ts['time'].apply(lambda x: x[5:]).astype(int)

# if periodicity == 52: 
#         multi_ts.index = multi_ts.apply(lambda row: _year_week(row.year, row.period), axis=1)
# if periodicity == 12:
#         multi_ts.index = multi_ts.apply(lambda row: _convert2date(row.year, row.period), axis=1)

# multi_ts.drop(columns=['year','period'], inplace=True)
all_time_series = multi_ts.columns

multi_ts

### Hier weitermachen, alles davor wurde bereits angepasst!
####################################################################################

#--------------------------------------------------
# Next make aggregation over groups: Region, LOB:
#--------------------------------------------------

#--------------------
# 1.) Region only:
#--------------------
#if self.aggreg_level == 'region_only':

# df has been time filtered already!!!
gr0_region = df.groupby(['Region', 'LossCat', 'time'])

data_region = gr0_region.agg(target = (target_col, agg_func)).reset_index()

gr_region = data_region.groupby(['Region', 'LossCat'])
sample_region = gr_region.agg(size=(target_col,'count'), q_50_target=(target_col, q_50)).reset_index()
sample_region['pooling'] = (((sample_region['size'] < min_sample_size) | (sample_region['q_50_target'] < min_median_target))*1).astype(object)

# Region aggregation:
#-----------------------
lookup_level_region_singles = sample_region.query('pooling == 0').drop(columns='pooling', inplace=False)
lookup_level_region_agg = sample_region.query('pooling == 1').drop(columns='pooling', inplace=False)
data_leftover_region = lookup_level_region_agg

ts_bag_region_only = {}

# Append to already existing dictionary:
#----------------------------------------------------------
# for _, ts_info in lookup_level_region_singles.iterrows():
#         #ts_slice = gr_region.get_group(ts_info).reset_index()
#         combi = tuple(ts_info[:len(['Region', 'LossCat'])])
#         ts_slice = gr_region.get_group(combi).reset_index()
        
#         ts_slice['year'] = ts_slice['time'].apply(lambda x: x[:4]).astype(int)
#         ts_slice['period'] = ts_slice['time'].apply(lambda x: x[5:]).astype(int)

#         if periodicity == 52: 
#                 ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _year_week(row.year, row.period), axis=1)            
#         if periodicity == 12:
#                 ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _convert2date(row.year, row.period), axis=1)

#         #ts_slice['year_period_ts'] = ts_slice.apply(lambda row: self._convert2date(row.year, row.period), axis=1)
#         ts_slice['month'] = [x.month for x in ts_slice['year_period_ts'].tolist()]
#         ts_bag_region_only['-'.join(combi)] = ts_slice[['time', target_col, 'year', 'month','period']]


for _, ts_info in lookup_level_region_singles.iterrows():
        #ts_slice = gr_region.get_group(ts_info).reset_index()
        combi = tuple(ts_info[:len(['Region', 'LossCat'])])
        ts_slice = gr_region.get_group(combi).reset_index()
        
        ts_slice['year'] = ts_slice['time'].apply(lambda x: x[:4]).astype(int)
        ts_slice['period'] = ts_slice['time'].apply(lambda x: x[5:]).astype(int)

        # Adjust frequencies in case of irregular time series 
        # (i.e. missing timestamps, not full cycle)
        #------------------------------------------------------
        if periodicity == 12:
                ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _convert2date(row.year, row.period), axis=1)
                my_series = ts_slice[['year_period_ts', 'target']].set_index('year_period_ts', inplace=False)
                ts_slice = my_series.asfreq(freq = 'MS', fill_value = 0.).reset_index().rename(columns={'year_period_ts' : 'time'})
                ts_slice['year'] = pd.DatetimeIndex(ts_slice['time']).year
                ts_slice['period'] = pd.DatetimeIndex(ts_slice['time']).month
                ts_slice['month'] = ts_slice['period']
                ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _convert2date(row.year, row.period), axis=1)
        if periodicity == 52:
                #ts_slice = my_series.asfreq(freq = 'W', fill_value = 0.).reset_index()
                #ts_slice['year'] = pd.DatetimeIndex(ts_slice['time']).year
                ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _year_week(row.year, row.period), axis=1)
                ts_slice['month'] = [x.month for x in ts_slice['year_period_ts'].tolist()]

        ts_bag_region_only['-'.join(combi)] = ts_slice[['time', target_col, 'year', 'month','period']]

# Copy ALL series from all_combi for easier access: 
ts_bag_region_only['all'] = deepcopy(ts_bag['all'])

ts_bag_region_only['RUL-4-Cat']


#----------------
# 2.) LOB only:
#----------------
#if self.aggreg_level == 'lob_only':

gr0_lob = df.groupby(['Lob', 'LossCat', 'time'])
data_lob = gr0_lob.agg(target = (target_col, agg_func)).reset_index()
gr_lob = data_lob.groupby(['Lob', 'LossCat'])

sample_lob = gr_lob.agg(size=(target_col,'count'), q_50_target=(target_col, q_50)).reset_index()

if min_median_target is None:
   sample_lob['pooling'] = ((sample_lob['size'] < min_sample_size)*1).astype(object)
else:
   sample_lob['pooling'] = (((sample_lob['size'] < min_sample_size) | (sample_lob['q_50_target'] < min_median_target))*1).astype(object)


# LOB aggregation:
lookup_level_lob_singles = sample_lob.query('pooling == 0').drop(columns='pooling', inplace=False)
lookup_level_lob_agg = sample_lob.query('pooling == 1').drop(columns='pooling', inplace=False)
data_leftover_lob = lookup_level_lob_agg

ts_bag_lob_only = {}

# Append to already existing dictionary:
#----------------------------------------------------------
# for _, ts_info in lookup_level_lob_singles.iterrows():
#         combi = tuple(ts_info[:len(['Lob', 'LossCat'])])
#         ts_slice = gr_lob.get_group(combi).reset_index()

#         #ts_slice = gr_lob.get_group(ts_info).reset_index()
#         ts_slice['year'] = ts_slice['time'].apply(lambda x: x[:4]).astype(int)
#         ts_slice['period'] = ts_slice['time'].apply(lambda x: x[5:]).astype(int)

#         if periodicity == 52: 
#                 ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _year_week(row.year, row.period), axis=1)            
#         if periodicity == 12:
#                 ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _convert2date(row.year, row.period), axis=1)

#         ts_slice['month'] = [x.month for x in ts_slice['year_period_ts'].tolist()]
#         ts_bag_lob_only['-'.join(combi)] = ts_slice[['time', target_col, 'year', 'month','period']]

for _, ts_info in lookup_level_lob_singles.iterrows():
        combi = tuple(ts_info[:len(['Lob', 'LossCat'])])
        ts_slice = gr_lob.get_group(combi).reset_index()

        #ts_slice = gr_lob.get_group(ts_info).reset_index()
        ts_slice['year'] = ts_slice['time'].apply(lambda x: x[:4]).astype(int)
        ts_slice['period'] = ts_slice['time'].apply(lambda x: x[5:]).astype(int)

        # Adjust frequencies in case of irregular time series 
        # (i.e. missing timestamps, not full cycle)
        #------------------------------------------------------
        if periodicity == 12:
                ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _convert2date(row.year, row.period), axis=1)
                my_series = ts_slice[['year_period_ts', 'target']].set_index('year_period_ts', inplace=False)
                ts_slice = my_series.asfreq(freq = 'MS', fill_value = 0.).reset_index().rename(columns={'year_period_ts' : 'time'})
                ts_slice['year'] = pd.DatetimeIndex(ts_slice['time']).year
                ts_slice['period'] = pd.DatetimeIndex(ts_slice['time']).month
                ts_slice['month'] = ts_slice['period']
                ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _convert2date(row.year, row.period), axis=1)
        if periodicity == 52:
                #ts_slice = my_series.asfreq(freq = 'W', fill_value = 0.).reset_index()
                #ts_slice['year'] = pd.DatetimeIndex(ts_slice['time']).year
                ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _year_week(row.year, row.period), axis=1)
                ts_slice['month'] = [x.month for x in ts_slice['year_period_ts'].tolist()]

        ts_bag_lob_only['-'.join(combi)] = ts_slice[['time', target_col, 'year', 'month','period']]

# Copy ALL series from all_combi for easier access: 
ts_bag_lob_only['all'] = deepcopy(ts_bag['all'])

ts_bag_lob_only











