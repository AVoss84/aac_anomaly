import pandas as pd
import numpy as np
from copy import deepcopy
import inspect
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from importlib import reload
from aac_ts_anomaly.config import global_config as glob
from aac_ts_anomaly.resources import config
from aac_ts_anomaly.resources.blueprints import AbstractPreprocessor
from aac_ts_anomaly.services import file
from aac_ts_anomaly.utils.utils_func import timer
from aac_ts_anomaly.utils import aggregation_functions


class claims_reporting(AbstractPreprocessor):
    """
    Preprocessing class: Slices/aggregates the portfolio into univariate time series
    according to minimal sample size and median counts per series 
    """
    def __init__(self, ts_col : str ='target', periodicity : int = 52):
        
        self.target_col = ts_col  #ts_col: column name of target time series in df
        self.periodicity = periodicity    # 12 or 52 : seasonal period in the data. Currently: monthly, weekly (i.e. calendar weeks)
        super().__init__()

        # Get parameters from I/O yaml
        if self.periodicity == 12 : 
            self.config_input = config.in_out12['input']
            self.config_output = config.in_out12['output']
            self.config_detect = config.in_out12['detection']

        if self.periodicity == 52 : 
            self.config_input = config.in_out52['input']
            self.config_output = config.in_out52['output']
            self.config_detect = config.in_out52['detection']
        #------------------------------------------------------------------
        self.hyper_para = self.config_detect['training']['hyper_para']           # model hyperparameter
        self.transformers = self.config_detect['training']['transformers']
        self.stat_transform = self.config_detect['training']['stat_transform']
        self.outlier_filter = self.config_detect['training']['outlier_filter']
        # set defaults:
        self.aggreg_level, self.pre_filter, self.ignore_lag, self.min_sample_size, self.min_median_target = list(self.config_detect['preprocessing'].values())
        self.tbl_name = self.config_output['database']['tbl_name']
        self.detect_thresh = self.config_detect['prediction']['detect_thresh']
        # Only show anomalies not older than 6 months in report: 
        age = 6
        if self.outlier_filter is None:
                six_months_ago = date.today() - relativedelta(months=age)
                self.outlier_filter = six_months_ago.strftime("%Y-%m")

    #@timer
    def process_data(self, data_orig : pd.core.frame.DataFrame, 
                    aggreg_level : str = None, agg_func = 'sum', aggreg_dimensions : list = ['Lob','Event_descr'],
                    ignore_lag: int = None, min_sample_size: int = None, min_median_target: int = None, 
                    verbose = False):
        """
        Preprocess univariate time series and create generator to yield single time series.

        Arguments:
        ----------
        aggreg_level : string specifying the way the data should be aggregated.
                       Can be one of: all_combi, lob_only, region_only
        agg_func : either string, the aggregation/pooling function used in the groupby statements, 
                   must match one of numpy's fct: np.mean, np.sum, etc., or name of custom aggregation function        aggreg_dimensions : list of variable names which comprise the maximum number of dimensions
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
        if hasattr(aggregation_functions, 'agg_func') and (self.agg_func is None):
                if self.verbose: print("Using custom agggregation function.\n\n", inspect.getsource(aggregation_functions.agg_func))
                self.agg_func = aggregation_functions.agg_func
        self.aggreg_dimensions = aggreg_dimensions

        # Overwrite I/O yaml spec. if user arguments are given:
        #--------------------------------------------------------
        if ignore_lag is not None:
            self.ignore_lag = ignore_lag
        if min_sample_size is not None:  
            self.min_sample_size = min_sample_size  
        if min_median_target is not None:  
            self.min_median_target = min_median_target         
        if aggreg_level is not None:    
            self.aggreg_level = aggreg_level           

        if self.verbose:
           print('Periodicity: {}'.format(self.periodicity)) 
           print("Aggregation type: '{}'".format(self.aggreg_level)) 
           print('Parameters: ignore_lag {}, min_sample_size {}, min_median_target {}'.format(self.ignore_lag, self.min_sample_size, self.min_median_target))

        df0 = deepcopy(data_orig)
        df0.rename(columns={'lob': 'Lob', 'erartbez': 'Event_descr', 'time_index': 'time', 'clm_cnt' : self.target_col}, inplace=True)
        df0 = df0[['time', self.target_col] + self.aggreg_dimensions]   # select only relevant columns

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
        #----------------------------------------------
        self.time_index = []
        for y in year_seq:
                for p in period_seq:
                   self.time_index.append(str(int(y))+'-'+'{0:02d}'.format(int(p)))             # calendar weeks, months
        self.max_period_index = np.where(self.max_year_period == np.array(self.time_index))[0][0]

        # Deduplicate:
        #---------------
        gr0 = df.groupby(aggreg_dimensions + ['time'])
        #gr0 = df.groupby(['Region', 'OE', 'Lob', 'Tier'] + ['time'])
        data_lev1 = gr0.agg(target = (self.target_col, self.agg_func)).reset_index()

        ts_bag = {}

        ##################################################
        # Level 1 (LOB - OE - Region - Tier)
        ##################################################

        aggreg_dimensions1 = aggreg_dimensions.copy()
        print("Aggregation level 1: {}".format(aggreg_dimensions1))
        gr = data_lev1.groupby(aggreg_dimensions1)
        #gr = data_lev1.groupby(['Region', 'OE', 'Lob', 'Tier'])

        sample1 = gr.agg(size=(self.target_col, 'count'), q_50_target=(self.target_col, self.q_50)).reset_index()

        if self.min_median_target is None:
                sample1['pooling'] = ((sample1['size'] < self.min_sample_size)*1).astype(object)
        else:
            sample1['pooling'] = (((sample1['size'] < self.min_sample_size) | (sample1['q_50_target'] < self.min_median_target))*1).astype(object)

        lookup_level1_single = sample1.query('pooling == 0')      # singles
        lookup_level1_singles = lookup_level1_single.drop(columns='pooling', inplace=False)
        lookup_level1_ag = sample1.query('pooling == 1')        # aggregate
        lookup_level1_agg = lookup_level1_ag.drop(columns='pooling', inplace=False)
        
        self.level_wise_aggr = {}    # only plots added here...
        
        for _, ts_info in lookup_level1_singles.iterrows():
                combi = tuple(ts_info[:len(aggreg_dimensions1)])
                #combi = tuple(ts_info[:len(['Region', 'OE', 'Lob', 'Tier'])])
                ts_slice = gr.get_group(combi).reset_index()
                ts_slice['year'] = ts_slice['time'].apply(lambda x: x[:4]).astype(int)
                ts_slice['period'] = ts_slice['time'].apply(lambda x: x[5:]).astype(int)

                if self.periodicity == 52: 
                        ts_slice['year_period_ts'] = ts_slice.apply(lambda row: self._year_week(row.year, row.period), axis=1)            
                if self.periodicity == 12:
                        ts_slice['year_period_ts'] = ts_slice.apply(lambda row: self._convert2date(row.year, row.period), axis=1)

                ts_slice['month'] = [x.month for x in ts_slice['year_period_ts'].tolist()]
                ts_bag['-'.join(combi)] = ts_slice[['time', self.target_col, 'year', 'month','period']]

        # Next take the left-overs (having too short time series) 
        # and aggregate further -> level 2
        #----------------------------------------------------------
        #data_lev2 = data_lev1.merge(lookup_level1_agg[['Region', 'OE', 'Lob', 'Tier']], how='right', left_on=['Region', 'OE', 'Lob', 'Tier'], right_on=['Region', 'OE', 'Lob', 'Tier'])
        data_lev2 = data_lev1.merge(lookup_level1_agg[aggreg_dimensions1], how='right', left_on=aggreg_dimensions1, right_on=aggreg_dimensions1)
        
        if self.verbose:  print("Number of left over aggregations ('Level 1'):", lookup_level1_agg.shape[0])

        # Finally do a an aggregation over only Tier:
        #-------------------------------------------------
        gr_full0 = df.groupby(['Event_descr']+['time'])
        full_agg_series = gr_full0.agg(target = (self.target_col, self.agg_func)).reset_index()
        full_agg_series['year'] = full_agg_series['time'].apply(lambda x: x[:4]).astype(int)
        full_agg_series['period'] = full_agg_series['time'].apply(lambda x: x[5:]).astype(int)

        if self.periodicity == 52: 
                full_agg_series['year_period_ts'] = full_agg_series.apply(lambda row: self._year_week(row.year, row.period), axis=1)
        if self.periodicity == 12:
                full_agg_series['year_period_ts'] = full_agg_series.apply(lambda row: self._convert2date(row.year, row.period), axis=1)

        full_agg_series['month'] = [x.month for x in full_agg_series['year_period_ts'].tolist()]

        for lc in full_agg_series['Event_descr'].unique():
            ts_bag['all-'+str(lc)] = full_agg_series.loc[full_agg_series['Event_descr'] == lc, ['time', 'target', 'year', 'month','period']]

        # Finally do a full aggregation over ALL dimensions:
        #----------------------------------------------------
        #gr_full = df0.groupby(['time'])     # unfiltered wrt to time filter and nans, reference for xls
        gr_full = df.groupby(['time'])     # filtered wrt to time filter and nans
        full_agg_series = gr_full.agg(target = (self.target_col, self.agg_func)).reset_index()
        full_agg_series['year'] = full_agg_series['time'].apply(lambda x: x[:4]).astype(int)
        full_agg_series['period'] = full_agg_series['time'].apply(lambda x: x[5:]).astype(int)

        if self.periodicity == 52: 
                full_agg_series['year_period_ts'] = full_agg_series.apply(lambda row: self._year_week(row.year, row.period), axis=1)
        if self.periodicity == 12:
                full_agg_series['year_period_ts'] = full_agg_series.apply(lambda row: self._convert2date(row.year, row.period), axis=1)

        full_agg_series['month'] = [x.month for x in full_agg_series['year_period_ts'].tolist()]

        ts_bag['all'] = full_agg_series[['time', 'target', 'year', 'month','period']]

        # Construct multivariate time series:
        #---------------------------------------
        tseries = deepcopy(ts_bag)
        self.multi_ts = pd.DataFrame(index = self.time_index, columns = list(ts_bag.keys()))
        for name, ts in tseries.items():
            ts.index = ts['time']    
            self.multi_ts[name] = ts[self.target_col]   

        self.multi_ts = self.multi_ts.loc[self.multi_ts.index[:self.max_period_index+1],:]
        self.multi_ts.fillna(value=0, inplace=True)   # replace missings by zeros!

        # Make proper proper date index
        # in case you want to use multivariate ts methods:        
        #---------------------------------------------------
        self.multi_ts = self.multi_ts.reset_index().rename(columns={'index': 'time'})
        self.multi_ts['year'] = self.multi_ts['time'].apply(lambda x: x[:4]).astype(int)
        self.multi_ts['period'] = self.multi_ts['time'].apply(lambda x: x[5:]).astype(int)

        if self.periodicity == 52: 
                self.multi_ts.index = self.multi_ts.apply(lambda row: self._year_week(row.year, row.period), axis=1)
        if self.periodicity == 12:
                self.multi_ts.index = self.multi_ts.apply(lambda row: self._convert2date(row.year, row.period), axis=1)

        self.multi_ts.drop(columns=['year','period'], inplace=True)
        all_time_series = self.multi_ts.columns

        #########################################
        # Finally define time series generators:
        #########################################

        # Lob-Region-OE-ALL! -> includes ALL combinations
        # Create univariate series generator:    
        #-------------------------------------------------
        if self.aggreg_level == 'all_combi':
            for name, current_ts in ts_bag.items(): 
                yield name, current_ts         
