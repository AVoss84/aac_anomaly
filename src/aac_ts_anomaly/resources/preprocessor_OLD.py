import pandas as pd
import numpy as np
#from scipy.special import loggamma #gamma as gamma_fct
#from math import cos, pi
#from numpy import log, sum, exp, prod
#from numpy.random import beta, binomial, normal, uniform, gamma, seed, rand, poisson
from copy import deepcopy
#import glob as gl
#import subprocess, os
from datetime import datetime
#import dateutil.parser as dateparser
#from textdistance import jaro_winkler
from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import ThresholdAD, InterQuartileRangeAD, GeneralizedESDTestAD, PersistAD, QuantileAD
from adtk.detector import LevelShiftAD, VolatilityShiftAD, SeasonalAD, AutoregressionAD
from adtk.transformer import DoubleRollingAggregate, RollingAggregate, Retrospect, ClassicSeasonalDecomposition
from adtk.pipe import Pipeline, Pipenet
from adtk.aggregator import AndAggregator, OrAggregator
#from adtk.data import split_train_test
from importlib import reload
from aac_ts_anomaly.config import global_config as glob
from aac_ts_anomaly.resources import config
from aac_ts_anomaly.resources.blueprints import AbstractPreprocessor
from aac_ts_anomaly.services import file
from aac_ts_anomaly.utils.utils_func import timer


class claims_reporting(AbstractPreprocessor):
    """
    Preprocessing class: Slices/aggregates the portfolio into univariate time series
    according to minimal sample size and median counts per series 
    """
    def __init__(self, ts_col='target', periodicity : int = 52):
        
        self.target_col = ts_col  #ts_col: column name of target time series in df
        self.periodicity = periodicity    # 12 or 52 : seasonal period in the data. Currently: monthly, weekly (i.e. calendar weeks)
        super().__init__()

        # Get parameters from I/O yaml
        if self.periodicity == 12 : 
            #self.config_input = config.in_out12['input']
            self.config_output = config.in_out12['output']
            self.config_detect = config.in_out12['detection']

        if self.periodicity == 52 : 
            #self.config_input = config.in_out52['input']
            self.config_output = config.in_out52['output']
            self.config_detect = config.in_out52['detection']
        #------------------------------------------------------------------
        self.hyper_para = self.config_detect['training']['hyper_para']           # model hyperparameter
        self.transformers = self.config_detect['training']['transformers']
        self.stat_transform = self.config_detect['training']['stat_transform']
        self.outlier_filter = self.config_detect['training']['outlier_filter']
        self.aggreg_level, self.pre_filter, self.ignore_lag, self.min_sample_size, self.min_median_target = list(self.config_detect['preprocessing'].values())
        self.tbl_name = self.config_output['database']['tbl_name']
        self.detect_thresh = self.config_detect['prediction']['detect_thresh']

    #@timer
    def process_data(self, data_orig : pd.core.frame.DataFrame, 
                    aggreg_level : str = None, agg_func : str = 'sum', aggreg_dimensions : list = ['Region', 'OE', 'Lob',  'LossCat'],
                    ignore_lag: int = None, min_sample_size: int = None, min_median_target: int = None, 
                    verbose = True):
        """
        Preprocess univariate time series and create generator to yield single time series.

        Arguments:
        ----------
        aggreg_level : string specifying the way the data should be aggregated.
                       Can be one of: all_combi, lob_only, region_only
        agg_func : string, the aggregation/pooling function used in the groupby statements
                   Mostly sum for money amounts or count e.g. for claim counts
        aggreg_dimensions : list of variable names which comprise the maximum number of dimensions
                            over which we will aggregate when using aggreg_level = all_combi
        ignore_lag : integer, number of time points to leave out of the preprocessing/training process
                     at the end of the time series. Mostly only the most recent time point is left out 
        min_sample_size : integer, minimum time series length contraint. If not fullfilled
                          the time series is not used for anomaly detection, but will be 
                          used in an additional coarser aggregation step (single vs. aggreg)
        min_median_target : float value (default: None), If specified is used as additional filter constraint to min_sample_size
                            Defines the lowest possible median target value (q50_target). If median is lower than that
                            the data points are further aggregated (see single vs. aggreg logic -> pooling == 1)                                                          
        """
        self.verbose = verbose
        self.agg_func = agg_func
        self.aggreg_dimensions = aggreg_dimensions
        if self.verbose: print('Periodicity: {}'.format(self.periodicity)) 

        # Overwrite I/O yaml spec. if user arguments are given:
        #--------------------------------------------------------
        if (ignore_lag is not None) & self.verbose:
            self.ignore_lag = ignore_lag
            print('Parameter: ignore_lag {}'.format(self.ignore_lag))
        if (min_sample_size is not None) & self.verbose:  
            self.min_sample_size = min_sample_size  
            print('Parameter: min_sample_size {}'.format(self.min_sample_size))
        if (min_median_target is not None) & self.verbose:  
            self.min_median_target = min_median_target         
            print('Parameter: min_median_target {}'.format(self.min_median_target))    
        if (aggreg_level is not None) & self.verbose:    
            self.aggreg_level = aggreg_level           
            print("Aggregation type: '{}'".format(self.aggreg_level)) 

        df0 = deepcopy(data_orig)
        df0.rename(columns={'Line of Business': 'Lob', 'Source System': 'source_sys', 'Sub Line of Business': 'SubLob', 'Loss Category': 'LossCat'}, inplace=True)
 
        # Apply a prefiltering step, 
        # most often to filter out Brazil, South America data
        if self.pre_filter is not None:
            try:
                df0 = df0.query(self.pre_filter)
                if self.verbose: print('pre-filter applied.')
            except Exception as e:
                if self.verbose: print(self.pre_filter,e)  

        df = deepcopy(df0)
        df.dropna(inplace=True)
        years = df['time'].apply(lambda x: x[:4]).astype(int)
        periods = df['time'].apply(lambda x: x[5:]).astype(int)   # months, wekks, days etc.
        self.max_year = max(years)
        self.min_year = min(years)
        max_years = (years == self.max_year)
        min_years = (years == self.min_year)
        self.max_period = max(periods[max_years])     # take max period of max year observations
        self.min_period = min(periods[min_years])
        thresh_period = self.max_period - self.ignore_lag
        thresh_year = self.max_year

        self.min_year_period = str(int(self.min_year))+'-'+'{0:02d}'.format(int(self.min_period))
        self.max_year_period = str(int(self.max_year))+'-'+'{0:02d}'.format(int(self.max_period))
        if self.verbose:
            print("Ignoring claims after than:",str(int(thresh_year))+'-'+'{0:02d}'.format(int(thresh_period)))
        time_filter = ~((periods > thresh_period) & (years == thresh_year))
        
        # Apply time filter to prefiltered data:
        df = df[time_filter]
        periods, years = periods[time_filter], years[time_filter]
        period_seq = np.arange(min(periods), max(periods)+1)
        year_seq = np.arange(min(years), max(years)+1)

        # For multivariate time series creation below:
        # create regular time range (i.e. without gaps) 
        #------------------------------------------------
        self.time_index = []
        for y in year_seq:
                for p in period_seq:
                   self.time_index.append(str(int(y))+'-'+'{0:02d}'.format(int(p)))             # calendar weeks, months
        self.max_period_index = np.where(self.max_year_period == np.array(self.time_index))[0][0]

        # Deduplicate:
        #---------------
        #gr0 = df.groupby(aggreg_dimensions + ['time'])
        gr0 = df.groupby(['Region', 'OE', 'Lob', 'LossCat'] + ['time'])
        data_lev1 = gr0.agg(target = (self.target_col, self.agg_func)).reset_index()

        ts_bag = {}

        ##################################################
        # Level 1 (LOB - OE - Region - LossCat)
        ##################################################

        #aggreg_dimensions1 = aggreg_dimensions.copy()
        #print("Aggregation level 1: {}".format(aggreg_dimensions1))
        #gr = data_lev1.groupby(aggreg_dimensions1)
        gr = data_lev1.groupby(['Region', 'OE', 'Lob', 'LossCat'])

        sample1 = gr.agg(size=(self.target_col, 'count'), q_50_target=(self.target_col, self.q_50)).reset_index()

        if self.min_median_target is None:
                sample1['pooling'] = ((sample1['size'] < self.min_sample_size)*1).astype(object)
        else:
            sample1['pooling'] = (((sample1['size'] < self.min_sample_size) | (sample1['q_50_target'] < self.min_median_target))*1).astype(object)

        lookup_level1_single = sample1.query('pooling == 0')      # singles
        lookup_level1_singles = lookup_level1_single.drop(columns='pooling', inplace=False)
        lookup_level1_ag = sample1.query('pooling == 1')        # aggregate
        lookup_level1_agg = lookup_level1_ag.drop(columns='pooling', inplace=False)
        
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
                if self.periodicity == 12:
                        ts_slice['year_period_ts'] = ts_slice.apply(lambda row: self._convert2date(row.year, row.period), axis=1)
                        my_series = ts_slice[['year_period_ts', 'target']].set_index('year_period_ts', inplace=False)
                        ts_slice = my_series.asfreq(freq = 'MS', fill_value = 0.).reset_index().rename(columns={'year_period_ts' : 'time'})
                        ts_slice['year'] = pd.DatetimeIndex(ts_slice['time']).year
                        ts_slice['period'] = pd.DatetimeIndex(ts_slice['time']).month
                        ts_slice['month'] = ts_slice['period']
                        ts_slice['year_period_ts'] = ts_slice.apply(lambda row: self._convert2date(row.year, row.period), axis=1)

                if self.periodicity == 52:
                        #ts_slice = my_series.asfreq(freq = 'W', fill_value = 0.).reset_index()
                        #ts_slice['year'] = pd.DatetimeIndex(ts_slice['time']).year
                        ts_slice['year_period_ts'] = ts_slice.apply(lambda row: self._year_week(row.year, row.period), axis=1)
                        ts_slice['month'] = [x.month for x in ts_slice['year_period_ts'].tolist()]

                ts_bag['-'.join(combi)] = ts_slice[['time', self.target_col, 'year', 'month','period']]

        # Next take the left-overs (having too short time series) 
        # and aggregate further -> level 2
        #----------------------------------------------------------
        data_lev2 = data_lev1.merge(lookup_level1_agg[['Region', 'OE', 'Lob', 'LossCat']], how='right', left_on=['Region', 'OE', 'Lob', 'LossCat'], right_on=['Region', 'OE', 'Lob', 'LossCat'])
        #data_lev2 = data_lev1.merge(lookup_level1_agg[aggreg_dimensions1], how='right', left_on=aggreg_dimensions1, right_on=aggreg_dimensions1)
        

        ##################################################
        # Level 2 (OE - Region - LossCat)
        ##################################################

        gr2 = data_lev2.groupby(['Region', 'OE', 'LossCat']+['time'])
        data_lev2_new = gr2.agg(target = (self.target_col, self.agg_func)).reset_index()     # aggregate

        # Check cluster size again for filtering:
        #-------------------------------------------
        #gr2b = data_lev2_new.groupby(aggreg_dimensions2)
        gr2b = data_lev2_new.groupby(['Region', 'OE', 'LossCat'])
        sample2 = gr2b.agg(size=(self.target_col,'count'), q_50_target=(self.target_col,self.q_50)).reset_index()

        if self.min_median_target is None:
                sample2['pooling'] = ((sample2['size'] < self.min_sample_size)*1).astype(object)
        else:
            sample2['pooling'] = (((sample2['size'] < self.min_sample_size) | (sample2['q_50_target'] < self.min_median_target))*1).astype(object)

        lookup_level2_single = sample2.query('pooling == 0')      # singles, constitute unpooled time series on that level
        lookup_level2_singles = lookup_level2_single.drop(columns='pooling', inplace=False)
        lookup_level2_ag = sample2.query('pooling == 1')        # aggregate further in next level!
        lookup_level2_agg = lookup_level2_ag.drop(columns='pooling', inplace=False)


        # Append to already existing dictionary:
        #-----------------------------------------
        self.level_wise_aggr = {}
        for _, ts_info in lookup_level2_singles.iterrows():
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
                if self.periodicity == 12:
                        ts_slice['year_period_ts'] = ts_slice.apply(lambda row: self._convert2date(row.year, row.period), axis=1)
                        my_series = ts_slice[['year_period_ts', 'target']].set_index('year_period_ts', inplace=False)
                        ts_slice = my_series.asfreq(freq = 'MS', fill_value = 0.).reset_index().rename(columns={'year_period_ts' : 'time'})
                        ts_slice['year'] = pd.DatetimeIndex(ts_slice['time']).year
                        ts_slice['period'] = pd.DatetimeIndex(ts_slice['time']).month
                        ts_slice['month'] = ts_slice['period']
                        ts_slice['year_period_ts'] = ts_slice.apply(lambda row: self._convert2date(row.year, row.period), axis=1)

                if self.periodicity == 52:
                        #ts_slice = my_series.asfreq(freq = 'W', fill_value = 0.).reset_index()
                        #ts_slice['year'] = pd.DatetimeIndex(ts_slice['time']).year
                        ts_slice['year_period_ts'] = ts_slice.apply(lambda row: self.transformers_year_week(row.year, row.period), axis=1)
                        ts_slice['month'] = [x.month for x in ts_slice['year_period_ts'].tolist()]

                ts_bag['-'.join(combi)] = ts_slice[['time', self.target_col, 'year', 'month','period']]

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
                self.level_wise_aggr['-'.join(combi)] = tuple(aggregation_categories)


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
        

        ##################################################
        # Level 3 (Region - LossCat)
        ##################################################

        gr3 = data_lev3.groupby(['Region','LossCat']+['time'])
        #gr3 = data_lev3.groupby(aggreg_dimensions3+['time'])
        data_lev3_new = gr3.agg(target = (self.target_col, self.agg_func)).reset_index()     # aggregate

        # Check cluster size again for filtering:
        #-------------------------------------------
        gr3b = data_lev3_new.groupby(['Region','LossCat'])
        #gr3b = data_lev3_new.groupby(aggreg_dimensions3)
        sample3 = gr3b.agg(size=(self.target_col,'count'), q_50_target=(self.target_col, self.q_50)).reset_index()

        if self.min_median_target is None:
            sample3['pooling'] = ((sample3['size'] < self.min_sample_size)*1).astype(object)
        else:
            sample3['pooling'] = (((sample3['size'] < self.min_sample_size) | (sample3['q_50_target'] < self.min_median_target))*1).astype(object)

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
        data_lev4 = gr4.agg(size4 = (self.target_col,'count')).reset_index()     # aggregate

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
                if self.periodicity == 12:
                        ts_slice['year_period_ts'] = ts_slice.apply(lambda row: self._convert2date(row.year, row.period), axis=1)
                        my_series = ts_slice[['year_period_ts', 'target']].set_index('year_period_ts', inplace=False)
                        ts_slice = my_series.asfreq(freq = 'MS', fill_value = 0.).reset_index().rename(columns={'year_period_ts' : 'time'})
                        ts_slice['year'] = pd.DatetimeIndex(ts_slice['time']).year
                        ts_slice['period'] = pd.DatetimeIndex(ts_slice['time']).month
                        ts_slice['month'] = ts_slice['period']
                        ts_slice['year_period_ts'] = ts_slice.apply(lambda row: self._convert2date(row.year, row.period), axis=1)
                if self.periodicity == 52:
                        #ts_slice = my_series.asfreq(freq = 'W', fill_value = 0.).reset_index()
                        #ts_slice['year'] = pd.DatetimeIndex(ts_slice['time']).year
                        ts_slice['year_period_ts'] = ts_slice.apply(lambda row: self._year_week(row.year, row.period), axis=1)
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
                self.level_wise_aggr['-'.join(combi)] = tuple(aggregation_levels)

        # Next take the left-overs you have to aggregate:
        #--------------------------------------------------
        self.data_leftover = data_lev3_new.merge(lookup_level3_agg[['Region','LossCat']], how='right', left_on=['Region','LossCat'], right_on=['Region','LossCat'])
        if self.verbose: print("Number of left over aggregations ('Level 3'):", lookup_level3_agg.shape[0])

        # Finally do a an aggregation over only LossCat:
        #-------------------------------------------------
        gr_full0 = df.groupby(['LossCat']+['time'])
        full_agg_series = gr_full0.agg(target = (self.target_col, self.agg_func)).reset_index()
        full_agg_series['year'] = full_agg_series['time'].apply(lambda x: x[:4]).astype(int)
        full_agg_series['period'] = full_agg_series['time'].apply(lambda x: x[5:]).astype(int)

        if self.periodicity == 52: 
                full_agg_series['year_period_ts'] = full_agg_series.apply(lambda row: self._year_week(row.year, row.period), axis=1)
                full_agg_series['month'] = [x.month for x in full_agg_series['year_period_ts'].tolist()]
                for lc in full_agg_series['LossCat'].unique():
                    ts_bag['all-'+str(lc)] = full_agg_series.loc[full_agg_series['LossCat'] == lc, ['time', 'target', 'year', 'month','period']]

        if self.periodicity == 12:
                full_agg_series['year_period_ts'] = full_agg_series.apply(lambda row: self._convert2date(row.year, row.period), axis=1)
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
        full_agg_series = gr_full.agg(target = (self.target_col, self.agg_func)).reset_index()
        full_agg_series['year'] = full_agg_series['time'].apply(lambda x: x[:4]).astype(int)
        full_agg_series['period'] = full_agg_series['time'].apply(lambda x: x[5:]).astype(int)

        if self.periodicity == 52: 
                full_agg_series['year_period_ts'] = full_agg_series.apply(lambda row: self._year_week(row.year, row.period), axis=1)
                full_agg_series['month'] = [x.month for x in full_agg_series['year_period_ts'].tolist()]

        if self.periodicity == 12:
                full_agg_series['year_period_ts'] = full_agg_series.apply(lambda row: self._convert2date(row.year, row.period), axis=1)
                ts_slice0 = full_agg_series[['year_period_ts', 'target']].set_index('year_period_ts', inplace=False)
                full_agg_series = ts_slice0.asfreq(freq = 'MS', fill_value = 0.).reset_index().rename(columns={'year_period_ts' : 'time'})
                full_agg_series['year'] = pd.DatetimeIndex(full_agg_series['time']).year
                full_agg_series['month'] = pd.DatetimeIndex(full_agg_series['time']).month
                full_agg_series['period'] = full_agg_series['month']

        ts_bag['all'] = full_agg_series[['time', self.target_col, 'year', 'month','period']]

        # Construct multivariate time series:
        #---------------------------------------
        tseries = deepcopy(ts_bag)

        # Make proper proper date index
        # in case you want to use multivariate ts methods:        
        #---------------------------------------------------
        # Format dates from 'YYYY-period' to proper date index 'YYYY-MM-01'
        # needed e.g. so seasonal decomp. can recognize it as date index:
        multi_ts_tmp = pd.DataFrame(self.time_index, columns = ['time'])
        multi_ts_tmp['year'] = multi_ts_tmp['time'].apply(lambda x: x[:4]).astype(int)
        multi_ts_tmp['period'] = multi_ts_tmp['time'].apply(lambda x: x[5:]).astype(int)

        if self.periodicity == 12:
                multi_ts_tmp['year_period_ts'] = multi_ts_tmp.apply(lambda row: self._convert2date(row.year, row.period), axis=1)
                multi_ts = multi_ts_tmp.set_index('year_period_ts', inplace=False).drop(columns=['time', 'year', 'period'])
        if self.periodicity == 52: 
                multi_ts_tmp['year_period_ts'] = multi_ts_tmp.apply(lambda row: self._year_week(row.year, row.period), axis=1)
                multi_ts = multi_ts_tmp.set_index('year_period_ts', inplace=False).drop(columns=['time', 'year', 'period']) 

        for name, ts in tseries.items():
            ts.index = ts['time']    
            multi_ts[name] = ts[self.target_col]   

        multi_ts = multi_ts.loc[multi_ts.index[:self.max_period_index+1],:]
        multi_ts.fillna(value=0, inplace=True)   # replace missings by zeros!
        all_time_series = multi_ts.columns

        #--------------------------------------------------
        # Next make aggregation over groups: Region, LOB:
        #--------------------------------------------------

        #--------------------
        # 1.) Region only:
        #--------------------
        if self.aggreg_level == 'region_only':
            # df has been time filtered already!!!
            gr0_region = df.groupby(['Region', 'LossCat', 'time'])

            data_region = gr0_region.agg(target = (self.target_col, self.agg_func)).reset_index()

            gr_region = data_region.groupby(['Region', 'LossCat'])
            sample_region = gr_region.agg(size=(self.target_col,'count'), q_50_target=(self.target_col, self.q_50)).reset_index()

            if self.min_median_target is None:
                sample_region['pooling'] = ((sample_region['size'] < self.min_sample_size)*1).astype(object)
            else:
                sample_region['pooling'] = (((sample_region['size'] < self.min_sample_size) | (sample_region['q_50_target'] < self.min_median_target))*1).astype(object)

            # Region aggregation:
            #-----------------------
            lookup_level_region_singles = sample_region.query('pooling == 0').drop(columns='pooling', inplace=False)
            lookup_level_region_agg = sample_region.query('pooling == 1').drop(columns='pooling', inplace=False)
            self.data_leftover_region = lookup_level_region_agg

            ts_bag_region_only = {}

            # Append to already existing dictionary:
            #----------------------------------------------------------
            for _, ts_info in lookup_level_region_singles.iterrows():
                    #ts_slice = gr_region.get_group(ts_info).reset_index()
                    combi = tuple(ts_info[:len(['Region', 'LossCat'])])
                    ts_slice = gr_region.get_group(combi).reset_index()
                    
                    ts_slice['year'] = ts_slice['time'].apply(lambda x: x[:4]).astype(int)
                    ts_slice['period'] = ts_slice['time'].apply(lambda x: x[5:]).astype(int)

                    # Adjust frequencies in case of irregular time series 
                    # (i.e. missing timestamps, not full cycle)
                    #------------------------------------------------------
                    if self.periodicity == 12:
                            ts_slice['year_period_ts'] = ts_slice.apply(lambda row: self._convert2date(row.year, row.period), axis=1)
                            my_series = ts_slice[['year_period_ts', 'target']].set_index('year_period_ts', inplace=False)
                            ts_slice = my_series.asfreq(freq = 'MS', fill_value = 0.).reset_index().rename(columns={'year_period_ts' : 'time'})
                            ts_slice['year'] = pd.DatetimeIndex(ts_slice['time']).year
                            ts_slice['period'] = pd.DatetimeIndex(ts_slice['time']).month
                            ts_slice['month'] = ts_slice['period']
                            ts_slice['year_period_ts'] = ts_slice.apply(lambda row: self._convert2date(row.year, row.period), axis=1)
                    if self.periodicity == 52:
                            #ts_slice = my_series.asfreq(freq = 'W', fill_value = 0.).reset_index()
                            #ts_slice['year'] = pd.DatetimeIndex(ts_slice['time']).year
                            ts_slice['year_period_ts'] = ts_slice.apply(lambda row: self._year_week(row.year, row.period), axis=1)
                            ts_slice['month'] = [x.month for x in ts_slice['year_period_ts'].tolist()]

                    ts_bag_region_only['-'.join(combi)] = ts_slice[['time', target_col, 'year', 'month','period']]
                    
            # Copy ALL series from all_combi for easier access: 
            ts_bag_region_only['all'] = deepcopy(ts_bag['all'])

        #----------------
        # 2.) LOB only:
        #----------------
        if self.aggreg_level == 'lob_only':

            gr0_lob = df.groupby(['Lob', 'LossCat', 'time'])
            data_lob = gr0_lob.agg(target = (self.target_col, self.agg_func)).reset_index()
            gr_lob = data_lob.groupby(['Lob', 'LossCat'])

            sample_lob = gr_lob.agg(size=(self.target_col,'count'), q_50_target=(self.target_col, self.q_50)).reset_index()

            if self.min_median_target is None:
                  sample_lob['pooling'] = ((sample_lob['size'] < self.min_sample_size)*1).astype(object)
            else:
                  sample_lob['pooling'] = (((sample_lob['size'] < self.min_sample_size) | (sample_lob['q_50_target'] < self.min_median_target))*1).astype(object)

            # LOB aggregation:
            lookup_level_lob_singles = sample_lob.query('pooling == 0').drop(columns='pooling', inplace=False)
            lookup_level_lob_agg = sample_lob.query('pooling == 1').drop(columns='pooling', inplace=False)
            self.data_leftover_lob = lookup_level_lob_agg

            ts_bag_lob_only = {}

            # Append to already existing dictionary:
            #------------------------------------------------------
            for _, ts_info in lookup_level_lob_singles.iterrows():
                    combi = tuple(ts_info[:len(['Lob', 'LossCat'])])
                    ts_slice = gr_lob.get_group(combi).reset_index()

                    #ts_slice = gr_lob.get_group(ts_info).reset_index()
                    ts_slice['year'] = ts_slice['time'].apply(lambda x: x[:4]).astype(int)
                    ts_slice['period'] = ts_slice['time'].apply(lambda x: x[5:]).astype(int)

                    if self.periodicity == 52: 
                            ts_slice['year_period_ts'] = ts_slice.apply(lambda row: self._year_week(row.year, row.period), axis=1)            
                    if self.periodicity == 12:
                            ts_slice['year_period_ts'] = ts_slice.apply(lambda row: self._convert2date(row.year, row.period), axis=1)

                    ts_slice['month'] = [x.month for x in ts_slice['year_period_ts'].tolist()]
                    ts_bag_lob_only['-'.join(combi)] = ts_slice[['time', self.target_col, 'year', 'month','period']]

            # Copy ALL series from all_combi for easier access: 
            ts_bag_lob_only['all'] = deepcopy(ts_bag['all'])


        #########################################
        # Finally define time series generators:
        #########################################

        # Ad1.) Region only!
        # Create univariate series generator:    
        #--------------------------------------
        if self.aggreg_level == 'region_only':
            for name, current_ts in ts_bag_region_only.items(): 
                yield name, current_ts 

        # Ad2.) LoB only!
        # Create univariate series generator:    
        #--------------------------------------
        if self.aggreg_level == 'lob_only':
            for name, current_ts in ts_bag_lob_only.items(): 
                yield name, current_ts 

        # Lob-Region-OE-ALL! -> includes ALL combinations
        # Create univariate series generator:    
        #-------------------------------------------------
        if self.aggreg_level == 'all_combi':
            for name, current_ts in ts_bag.items(): 
                yield name, current_ts         