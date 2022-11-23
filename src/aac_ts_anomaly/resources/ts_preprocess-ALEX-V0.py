
import pandas as pd
import numpy as np
from copy import deepcopy
pd.set_option('display.max_rows', 10**5)
pd.set_option('display.max_columns', 10**5)
import os, sys
from importlib import reload
#from statsmodels.tsa.seasonal import seasonal_decompose
import adtk, os
from datetime import datetime

from aac_ts_anomaly.utils import tsa_utils as tsa
from aac_ts_anomaly.utils import utils_func as util
from aac_ts_anomaly.config import global_config as glob
from aac_ts_anomaly.services import file
from aac_ts_anomaly.resources import config

reload(file)
reload(config)
reload(glob)
reload(util)

periodicity = 52

if periodicity == 52:
    #config_input = config.in_out52['input']
    config_output = config.in_out52['output']
    config_detect = config.in_out52['detection']
if periodicity == 12:    
    #config_input = config.in_out12['input']
    config_output = config.in_out12['output']
    config_detect = config.in_out12['detection']

hyper_para = config_detect['training']['hyper_para']

aggreg_level, pre_filter, ignore_lag, min_sample_size, min_median_cnts = list(config_detect['preprocessing'].values())

print(pre_filter)
print(aggreg_level)
print(ignore_lag)
print(min_sample_size)
print(min_median_cnts)

def _year_week( y, w):
        return datetime.strptime(f'{y} {w} 1', '%G %V %u')  # %G, %V, %u are ISO equivalents of %Y, %W, %w

#---------------
# Import data:
#---------------
#helper = base.helpers(config_input, config_output)

#filename = list(config_input['service']['XLSXService'].values())[0]
if periodicity == 52:
    filename = util.get_newest_file(search_for = "agg_time_series_52")    # weekly
if periodicity == 12:
    filename = util.get_newest_file(search_for = "agg_time_series_12")       # monthly
filename

reload(file)

# Import data:
#---------------
csv = file.CSVService(path=filename, delimiter=',')

data_orig = csv.doRead() ; data_orig.shape
data_orig.head()

# Make dates as in the future...
#data_orig['claim_creation_week'] = data_orig['claim_creation_week'].apply(lambda x : x[:4]+'-'+x[4:])
data_orig.rename(columns={'time_index' : 'time', 'clm_cnt' : 'target' }, inplace=True)
data_orig.head()

df0 = deepcopy(data_orig)
df0.rename(columns={'lob': 'Lob', 'erartbez': 'event_descr'}, inplace=True)
df0.shape
df0.head()

if pre_filter is not None:
        try:
                df0 = df0.query(pre_filter)
                print('pre-filter applied.')
        except Exception as e:
                print(pre_filter,e)  

df = deepcopy(df0)
df.dropna(inplace=True)

df.head()

years = df['time'].apply(lambda x: x[:4]).astype(int)
periods = df['time'].apply(lambda x: x[5:]).astype(int)

periods

#df['claim_creation_week'] = df['claim_creation_week'].apply(lambda x: x[:4]+'-'+x[4:])
df.rename(columns={'lob': 'Lob', 'erartbez': 'event_descr'}, inplace=True)

df['year'] = df['time'].apply(lambda x: x[:4]).astype(int)
df['period'] = df['time'].apply(lambda x: x[5:]).astype(int)

#df['year_week_ts'] = df.apply(lambda row: _year_week(row.year, row.week), axis=1)

def _year_week( y, w):
        return datetime.strptime(f'{y} {w} 1', '%G %V %u')  # %G, %V, %u are ISO equivalents of %Y, %W, %w

# def _convert2date(y, t):
#     return datetime.strptime(f'{y} {t} 1', '%Y %m %d')  # %G, %V, %u are ISO equivalents of %Y, %W, %w

df.head()

df['year_period_ts'] = df.apply(lambda row: _year_week(row.year, row.period), axis=1)
#df['year_period_ts'] = df.apply(lambda row: _convert2date(row.year, row.period), axis=1)
df #.head()

#ignore_week_lag = 1
min_year = min(years)
max_year = max(years)
max_years = (years == max_year)
min_years = (years == min_year)

max_period = max(periods[max_years])
#max_week = max(weeks[max_years])
#min_week = min(weeks[min_years])
min_period = min(periods[min_years])


thresh_period = max_period - ignore_lag
thresh_period
thresh_year = max_year

min_year_period = str(int(min_year))+'-'+'{0:02d}'.format(int(min_period))
#min_calendar_week = str(int(min_year))+'-'+'{0:02d}'.format(int(min_week))
#max_calendar_week = str(int(max_year))+'-'+'{0:02d}'.format(int(max_week))
max_year_period = str(int(max_year))+'-'+'{0:02d}'.format(int(max_period))
print("Ignoring claims younger than:",str(int(thresh_year))+'-'+'{0:02d}'.format(int(thresh_period)))

time_filter = ~((periods > thresh_period) & (years == thresh_year))

#max_calendar_week in df['time'].tolist()
#df_orig = deepcopy(df)

# Apply time filter:
df = df[time_filter]

periods = periods[time_filter]
years = years[time_filter]

period_seq = np.arange(min(periods), max(periods)+1)
year_seq = np.arange(min(years), max(years)+1)

# For multivariate time series creation below:
time_index = []
for y in year_seq:
   for p in period_seq:
        time_index.append(str(int(y))+'-'+'{0:02d}'.format(int(p)))  # calendar weeks

max_period_index = np.where(max_year_period == np.array(time_index))[0][0]

#pd.date_range("20181101", "20181217", freq='W-SUN')

def q_50(x):
   return x.quantile(0.5)

#-------------------------------------Source---------------------------------------------------------------------------------

df.head()

# Deduplicate:
#----------------------
gr0 = df.groupby(['Lob','event_descr','time'])
data_lev1 = gr0.agg(target = ('target','sum')).reset_index()
#counts_lev1 = gr0.agg(counts = ('time','count')).reset_index()
#counts_lev1[counts_lev1.counts > 1]
data_lev1.shape

min_sample_size = 100
min_median_cnts = 30

ts_bag = {}

def q_50(x):
    return x.quantile(0.5)
def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)

###############################
# Level 1 (LOB - OE - Region)
###############################

gr = data_lev1.groupby(['Lob','event_descr'])
sample1 = gr.agg(size=('target','count'), q_50_cnt=('target',q_50)
                #, iqr_cnt=('clm_cnt',iqr)
                ).reset_index()
#gr = data_lev1.groupby(['Lob', 'OE', 'Region'])
#sample1 = gr.agg(size=('clm_cnt','count')).reset_index()
sample1.shape

sample1['pooling1'] = (((sample1['size'] < min_sample_size) | (sample1['q_50_cnt'] < min_median_cnts))*1).astype(object)

sample1.head()

lookup_level1_single = sample1.query('pooling1 == 0')      # singles, keep as they are
lookup_level1_singles = lookup_level1_single.drop(columns='pooling1', inplace=False)
lookup_level1_singles.head()

lookup_level1_ag = sample1.query('pooling1 == 1')        # aggregate further
lookup_level1_agg = lookup_level1_ag.drop(columns='pooling1', inplace=False)
lookup_level1_agg.head(10) 

#--------------------------------------------------------------------
for _, ts_info in lookup_level1_singles.iterrows():
        combi = tuple(ts_info[:3])
        print(combi)
        ts_slice = gr.get_group(combi).reset_index()
        # ts_slice['year'] = ts_slice['time'].apply(lambda x: x[:4]).astype(int)
        # ts_slice['period'] = ts_slice['time'].apply(lambda x: x[5:]).astype(int)

        # if periodicity == 52: 
        #         ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _year_week(row.year, row.period), axis=1)            
        # if periodicity == 12:
        #         ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _convert2date(row.year, row.period), axis=1)

        # ts_slice['month'] = [x.month for x in ts_slice['year_period_ts'].tolist()]
        # ts_bag['-'.join(combi)] = ts_slice[['time', 'target', 'year', 'month','period']]

print(len(ts_bag))
ts_bag

#data_lev1_augm = data_lev1.merge(sample1, how='right', left_on=['Lob','OE','Region'], right_on=['Lob','OE','Region'])

#data_lev1_augm.head()
#data_lev1_augm.shape

# Next take the left-overs you have to aggregate further:
#---------------------------------------------------------
#data_lev2 = data_lev1.merge(sample1, how='right', left_on=['Lob','OE','Region'], right_on=['Lob','OE','Region'])
data_lev2 = data_lev1.merge(lookup_level1_agg[['Lob','OE','Region']], how='right', left_on=['Lob','OE','Region'], right_on=['Lob','OE','Region'])

data_lev1.shape
data_lev2.shape

##########################
## Level 2 (OE - Region)
##########################
gr2 = data_lev2.groupby(['OE', 'Region','time'])
#gr2 = data_lev2.groupby(['OE', 'Region', 'Lob','time'])
data_lev2_new = gr2.agg(target = ('target','sum')).reset_index()     # aggregate by LoB (and actually source system, too)
#data_lev2_new = gr2.agg(clm_cnt = ('clm_cnt','sum')).reset_index()     # aggregate by LoB (and actually source system, too)
#counts_lev2 = gr2.agg(counts = ('time','count'))#.reset_index()

data_lev2_new.shape
data_lev2_new.head(10)

# Check cluster size again for filtering:
#-------------------------------------------
gr2b = data_lev2_new.groupby(['OE', 'Region'])
sample2 = gr2b.agg(size_lev2=('target','count'), q_50_cnt_lev2=('target',q_50)).reset_index()
sample2.shape

sample2['pooling2'] = (((sample2['size_lev2'] < min_sample_size) | (sample2['q_50_cnt_lev2'] < min_median_cnts))*1).astype(object)
sample2.head(10)
sample2.shape

lookup_level2_singles = sample2.query('pooling2 == 0')      # singles, keep as they are
lookup_level2_singles.drop(columns='pooling2', inplace=True)
lookup_level2_singles.head(10)

lookup_level2_agg = sample2.query('pooling2 == 1')        # aggregate, aggregate further
lookup_level2_agg.drop(columns='pooling2', inplace=True)

lookup_level2_agg.head()
lookup_level2_agg.shape

# Append to already existing dictionary:
#----------------------------------------------
level_wise_aggr = {}
for ts_index, ts_info in lookup_level2_singles.iterrows():
        combi = tuple(ts_info[:2])
        oe_region = list(combi) 
        ts_slice = gr2b.get_group(combi).reset_index()
        ts_slice['year'] = ts_slice['time'].apply(lambda x: x[:4]).astype(int)
        ts_slice['period'] = ts_slice['time'].apply(lambda x: x[5:]).astype(int)

        if periodicity == 52: 
                ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _year_week(row.year, row.period), axis=1)            
        if periodicity == 12:
                ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _convert2date(row.year, row.period), axis=1)

        ts_slice['month'] = [x.month for x in ts_slice['year_period_ts'].tolist()]

        ts_bag['-'.join(combi)] = ts_slice[['time', 'target', 'year', 'month','period']]
        # Get OE-Region_Lob combination that was aggregated:
        aggreg_lobs = lookup_level1_agg[(lookup_level1_agg['OE'] == oe_region[0]) & (lookup_level1_agg['Region'] == oe_region[1])]['Lob'] #.to_list()
        level_wise_aggr['-'.join(tuple(oe_region))] = tuple(aggreg_lobs)

#for i,k in level_wise_aggr.items(): print(i,k) 
print(len(ts_bag))

#data_lev3 = data_lev2_new.merge(lookup_level2_agg[['OE','Region']], how='right', left_on=['OE','Region'], right_on=['OE','Region'])

# Complementary set to loop above
# Which (Lob/OE/Region) combis remain after level2 (aggre. w.r.t.LoB), 
# thus has to be further aggreg. w.r.t. OE in level3:
#------------------------------------------------------------------------
lookup_agg_1_2 = lookup_level1_agg.merge(lookup_level2_agg, how='right', left_on=['OE','Region'], right_on=['OE','Region'])
lookup_agg_1_2.head()

#data_lev2.head()

# Next take the left-overs you have to aggregate:
#--------------------------------------------------
data_lev3 = data_lev2.merge(lookup_agg_1_2, how='right', left_on=['Lob','OE','Region'], right_on=['Lob','OE','Region']) 
data_lev3.head()

#data_lev2.shape
#data_lev3.shape     # decreasing

####################
# Level 3 (Region)
####################

gr3 = data_lev3.groupby(['Region', 'time'])
data_lev3_new = gr3.agg(target = ('target','sum')).reset_index()     # aggregate

# Check cluster size again for filtering:
#-------------------------------------------
gr3b = data_lev3_new.groupby(['Region'])
#sample3 = gr3b.agg(size=('Region','count')).reset_index()
sample3 = gr3b.agg(size_lev3=('target','count'), q_50_cnt_lev3=('target',q_50)).reset_index()
sample3.head()

#who_was_aggr_lev2 = lookup_level1_agg.groupby(['OE', 'Region', 'Lob'])
#ag = who_was_aggr_lev2.agg(lob_cnt = ('Lob','count')).reset_index()

sample3['pooling3'] = (((sample3['size_lev3'] < min_sample_size) | (sample3['q_50_cnt_lev3'] < min_median_cnts))*1).astype(object)
sample3.head(10)

lookup_level3_singles = sample3.query('pooling3 == 0')      # singles, keep
lookup_level3_singles.drop(columns='pooling3', inplace=True)
lookup_level3_singles#.head()

lookup_level3_agg = sample3.query('pooling3 == 1')        # aggregate further
lookup_level3_agg.drop(columns='pooling3', inplace=True)
lookup_level3_agg.head() #.shape

aggr_after_lev2 = data_lev3.merge(lookup_level3_singles, how='right', left_on=['Region'], right_on=['Region']) 
aggr_after_lev2.head()

# Get all corresponding LOb/OE/Region combinations to loop over next,
# Based on singles in level3
#-----------------------------------------------------------------------
gr4 = aggr_after_lev2.groupby(['Lob','OE','Region'])
data_lev4 = gr4.agg(size4 = ('target','count')).reset_index()     # aggregate

# Append to already existing dictionary:
#----------------------------------------------------------
for _, ts_info in lookup_level3_singles.iterrows():
        combi = tuple(ts_info[:1])                        # region string
        ts_slice = gr3b.get_group(combi[0]).reset_index()
        ts_slice['year'] = ts_slice['time'].apply(lambda x: x[:4]).astype(int)
        ts_slice['period'] = ts_slice['time'].apply(lambda x: x[5:]).astype(int)

        if periodicity == 52: 
                ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _year_week(row.year, row.period), axis=1)
        if periodicity == 12:
                ts_slice['year_period_ts'] = ts_slice.apply(lambda row: _convert2date(row.year, row.period), axis=1)
    
        ts_slice['month'] = [x.month for x in ts_slice['year_period_ts'].tolist()]
        #print(ts_slice.shape[0])
        ts_bag[combi[0]] = ts_slice[['time', 'target', 'year', 'month','period']]

        aggreg_oe_lob = data_lev4[(data_lev4['Region'] == combi[0])][['Lob','OE']]
        aggreg = aggreg_oe_lob.to_records(index=False)
        level_wise_aggr[combi[0]] = tuple(aggreg) if len(aggreg)>1 else tuple(aggreg)[0]

#for i,k in level_wise_aggr.items(): print(i,k) 

# Next take the left-overs you have to aggregate:
#--------------------------------------------------
data_leftover = data_lev3_new.merge(lookup_level3_agg['Region'], how='right', left_on=['Region'], right_on=['Region'])

data_leftover
data_leftover.shape

# Finally do a full aggregation over all dimensions:
#----------------------------------------------------
gr_full = df.groupby(['time'])
full_agg_series = gr_full.agg(target = ('target','sum')).reset_index()
full_agg_series['year'] = full_agg_series['time'].apply(lambda x: x[:4]).astype(int)
full_agg_series['period'] = full_agg_series['time'].apply(lambda x: x[5:]).astype(int)

if periodicity == 52: 
        full_agg_series['year_period_ts'] = full_agg_series.apply(lambda row: _year_week(row.year, row.period), axis=1)
if periodicity == 12:
        full_agg_series['year_period_ts'] = full_agg_series.apply(lambda row: _convert2date(row.year, row.period), axis=1)

#full_agg_series['year_week_ts'] = full_agg_series.apply(lambda row: _year_week(row.year, row.week), axis=1)
full_agg_series['month'] = [x.month for x in full_agg_series['year_period_ts'].tolist()]
ts_bag['all'] = full_agg_series[['time', 'target', 'year', 'month','period']]

ts_bag.get('all')
full_agg_series.tail(30)  # correct 


########## OLD STUFF - NOT ADAPTED to new world strcuture!!!!!!!!!!

def _correct_cweek_53(dat : pd.core.frame.DataFrame, time_col : str = 'year_week_ts', target_col : str = 'clm_cnt', verbose=False):
        """
        This aggregates non-unique dates which were generated by 
        transforming from calendar week to a representative day 
        of that week (using _year_week function)
        Example: 
        2019-53 : 2019-12-30 and 2020-01: 2019-12-30. 
        These two will be summed up to have unique date points 
        for time series analysis
        """
        my_ts = deepcopy(dat)
        ssa = my_ts.groupby([time_col]) 
        #time_duplicates = ssa.agg(size=(target_col,'count')).reset_index()
        deduplicated_ts = ssa.agg(clm_cnt = (target_col,'sum')).reset_index()
        my_ts.drop(columns=[target_col], inplace = True)
        my_ts_new = my_ts.merge(deduplicated_ts, how='left', left_on=[time_col], right_on=[time_col])
        my_ts_new.drop_duplicates(subset=[time_col, target_col], keep='last', inplace=True, ignore_index = True)
        if verbose:
                print('{} rows aggregated (calendar week 53?)'.format(my_ts.shape[0]-my_ts_new.shape[0]))
        return my_ts_new

#_correct_cweek_53(train.df)

time_col  = 'year_week_ts'
target_col = "clm_cnt"

my_ts = deepcopy(train.df)
my_ts#.head()

ssa = my_ts.groupby([time_col]) 
#time_duplicates = ssa.agg(size=(target_col,'count')).reset_index()
deduplicated_ts = ssa.agg(clm_cnt = (target_col,'sum')).reset_index()
deduplicated_ts


my_ts.drop(columns=[target_col], inplace = True)
my_ts_new = my_ts.merge(deduplicated_ts, how='left', left_on=[time_col], right_on=[time_col])
my_ts_new.drop_duplicates(subset=[time_col, target_col], keep='last', inplace=True, ignore_index = True)
my_ts_new.head()


#---------------
# Region only:
#---------------
gr0_region = df.groupby(['Region', 'time'])
data_region = gr0_region.agg(clm_cnt = ('clm_cnt','sum')).reset_index()

gr_region = data_region.groupby(['Region'])
sample_region = gr_region.agg(size=('clm_cnt','count'), q_50_cnt=('clm_cnt',q_50)).reset_index()

sample_region['pooling'] = (((sample_region['size'] < min_sample_size) | (sample_region['q_50_cnt'] < min_median_cnts))*1).astype(object)

sample_region.head()

# Region aggregation:
lookup_level_region_singles = sample_region.query('pooling == 0').drop(columns='pooling', inplace=False)
lookup_level_region_agg = sample_region.query('pooling == 1').drop(columns='pooling', inplace=False)

ts_bag_region_only = {}

# Append to already existing dictionary:
#----------------------------------------------------------
for ts_info in list(lookup_level_region_singles.Region):
        print(ts_info)
        ts_slice = gr_region.get_group(ts_info).reset_index()
        ts_slice['year'] = ts_slice['time'].apply(lambda x: x[:4]).astype(int)
        ts_slice['week'] = ts_slice['time'].apply(lambda x: x[5:]).astype(int)
        ts_slice['year_week_ts'] = ts_slice.apply(lambda row: _year_week(row.year, row.week), axis=1)
        ts_slice['month'] = [x.month for x in ts_slice['year_week_ts'].tolist()]
        ts_bag_region_only[ts_info] = ts_slice[['time', 'clm_cnt', 'year', 'month','week']]

        # Who was aggregated per Region -> for titles in plots!
        #aggreg_oe_lob = data_lev4[(data_lev4['Region'] == combi[0])][['Lob','OE']]
        #aggreg = aggreg_oe_lob.to_records(index=False)
        #self.level_wise_aggr[combi[0]] = tuple(aggreg) if len(aggreg)>1 else tuple(aggreg)[0]

len(ts_bag_region_only)

ts_bag_region_only['Asia Pacific']

#------------
# LOB only:
#-------------
gr0_lob = df.groupby(['Lob', 'time'])
data_lob = gr0_lob.agg(clm_cnt = ('clm_cnt','sum')).reset_index()

gr_lob = data_lob.groupby(['Lob'])
sample_lob = gr_lob.agg(size=('clm_cnt','count'), q_50_cnt=('clm_cnt',q_50)).reset_index()

sample_lob['pooling'] = (((sample_lob['size'] < min_sample_size) | (sample_lob['q_50_cnt'] < min_median_cnts))*1).astype(object)

sample_lob.head()

# LOB aggregation:
lookup_level_lob_singles = sample_lob.query('pooling == 0').drop(columns='pooling', inplace=False)
lookup_level_lob_agg = sample_lob.query('pooling == 1').drop(columns='pooling', inplace=False)

ts_bag_lob_only = {}

# Append to already existing dictionary:
#----------------------------------------------------------
for ts_info in list(lookup_level_lob_singles.Lob):
        print(ts_info)
        ts_slice = gr_lob.get_group(ts_info).reset_index()
        ts_slice['year'] = ts_slice['time'].apply(lambda x: x[:4]).astype(int)
        ts_slice['week'] = ts_slice['time'].apply(lambda x: x[5:]).astype(int)
        ts_slice['year_week_ts'] = ts_slice.apply(lambda row: _year_week(row.year, row.week), axis=1)
        ts_slice['month'] = [x.month for x in ts_slice['year_week_ts'].tolist()]
        ts_bag_lob_only[ts_info] = ts_slice[['time', 'clm_cnt', 'year', 'month','week']]

        # Who was aggregated per Region -> for titles in plots!
        #aggreg_oe_lob = data_lev4[(data_lev4['Region'] == combi[0])][['Lob','OE']]
        #aggreg = aggreg_oe_lob.to_records(index=False)
        #self.level_wise_aggr[combi[0]] = tuple(aggreg) if len(aggreg)>1 else tuple(aggreg)[0]

ts_bag_lob_only

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

ts_bag.get('ART-Belgium-MED')['time']
ts_bag.get('ART-Belgium-MED')['clm_cnt']


