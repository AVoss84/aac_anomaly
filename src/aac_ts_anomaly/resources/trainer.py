import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import datetime
from statsmodels.tsa import seasonal as sea

from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import ThresholdAD, InterQuartileRangeAD, GeneralizedESDTestAD, PersistAD, QuantileAD
from adtk.detector import LevelShiftAD, VolatilityShiftAD, SeasonalAD, AutoregressionAD
from adtk.transformer import DoubleRollingAggregate, RollingAggregate, Retrospect, ClassicSeasonalDecomposition
from adtk.pipe import Pipeline, Pipenet
#from adtk.aggregator import AndAggregator, OrAggregator
#from adtk.data import split_train_test
from typing import (Dict, List, Text, Optional, Any)
#from importlib import reload
from aac_ts_anomaly.config import global_config as glob
from aac_ts_anomaly.services import file
from aac_ts_anomaly.resources import config
from aac_ts_anomaly.resources.preprocessor import claims_reporting
from aac_ts_anomaly.utils.utils_func import timer, MAD, IQR

class trainer(claims_reporting):
    def __init__(self, hyper_para : dict = None, verbose : bool = False, **kwargs):
        """Training and prediction pipeline for anomaly detector ensemble

        Args:
            hyper_para (dict, optional): _description_. Defaults to None.
            verbose (bool, optional): _description_. Defaults to False.
        """
        super(trainer, self).__init__(**kwargs) 
        self.verbose_train = verbose
        if hyper_para is not None: self.hyper_para = hyper_para   # overwrite yaml parameter
        if self.verbose_train: print('Detect anomalies not older than {}.'.format(self.outlier_filter))

    #def __del__(self):
    #    class_name = self.__class__.__name__

    def fit(self, df: pd.DataFrame): 
        """
        Train model ensemble.

        Input:
        ------
        df: dataframe being the output of the above preprocessing class
            this contains a univariate time series with time stamps.
        """
        assert isinstance(self.stat_transform, list), 'transform must be list.'
        self.transform = self.stat_transform[0]
        if self.verbose_train : print(f"Using '{self.transform}' as time series preprocessing.")
        self.df = deepcopy(df)

        if self.periodicity == 52: 
                self.df['year_period_ts'] = self.df.apply(lambda row: self._year_week(row.year, row.period), axis=1)
                # Remove calendar week 53 if there! Frequ. = 52 from now on.
                self.df = self._correct_cweek_53(self.df, time_col = 'year_period_ts', verbose=self.verbose_train)
                self.df = self.upsample_time_series(self.df)

        if self.periodicity == 12:
                self.df['year_period_ts'] = self.df.apply(lambda row: self._convert2date(row.year, row.period), axis=1)
                raise NotImplementedError("Monthly frequency not fully implemented yet!")

        if self.transform in ['log', 'diff_log']:
            self.ts_index, self.ts_values = self.df['year_period_ts'], np.log(1 + self.df[self.target_col])   # log transform
        else:
            self.ts_index, self.ts_values = self.df['year_period_ts'], self.df[self.target_col]
        self.ts_values.index = pd.to_datetime(self.ts_index) 
        self.val_series = validate_series(self.ts_values)

        if self.transform in ['diff', 'diff_log']:
            y_lag = Retrospect(n_steps=2, step_size=1).transform(self.val_series)
            y_lag.dropna(inplace=True)              
            self.val_series = validate_series(y_lag["t-0"] - y_lag["t-1"])   # first differences
            self.df = self.df.iloc[1: , :]           # drop first row so dimension of orig. dataframe is up-to-date after first diff. 
            
        # No cweek 53 allowed in the following due to the following and other subsequent
        # specifications in time series methods! 
        self.s_deseasonal = deepcopy(self.val_series)    # instantiate
        
        # Have transfomations been specified?
        #--------------------------------------
        if self.transformers is not None:
            self.model_transf = list(self.transformers.keys())[0]         # take first, only one transformation allowed for now
            transf_hyper_para = self.transformers[self.model_transf]
            try:                
                result_mul = sea.seasonal_decompose(self.val_series, extrapolate_trend='freq', **self.transformers['ClassicSeasonalDecomposition'])
                self.s_deseasonal = result_mul.observed - result_mul.seasonal   # result_mul.resid

                # self.anomaly_transformer = eval(self.model_transf+"("+"**transf_hyper_para)")
                # self.s_deseasonal = self.anomaly_transformer.fit_transform(self.val_series)
            except Exception as e0:
                print(e0)
                if self.verbose_train: print("No seasonal adjustment used.")    
            
        anom_detector = None ; self.nof_models = len(self.hyper_para)
        if self.verbose_train: print('Using {} base outlier detectors...'.format(self.nof_models))
            
        # Loop over all base learners for building the ensemble learner    
        #---------------------------------------------------------------
        for z, model in enumerate(self.hyper_para.keys()):            
            anom_detector = eval(model+"("+"**self.hyper_para['"+model+"'])")               # evaluate estimator expressions
            model_abstr = [(model, anom_detector)]
            pipe = Pipeline(model_abstr)
            try:
                train_res = pipe.fit_detect(self.s_deseasonal).rename(model, inplace=False).to_frame()     # fit estimator and predict
            except Exception as e1:
                print(e1) ; train_res = None
            if z==0:
                anomalies = deepcopy(train_res)    # instantiate
            else:
                anomalies = pd.concat([anomalies,train_res], axis=1)
            self.anomalies = anomalies*1.    
        self.anomalies.fillna(0, inplace=True)         
        if 'anom_detector' in dir(): del anom_detector
        self.anomaly_counts = self.anomalies.sum(axis=1)    
        self.anomaly_proba = self.anomalies.mean(axis=1)
        self.anomalies_or = self.anomalies.max(axis=1)  # union/or operation
        del self.anomalies
        return self

    def predict(self, detect_thresh : float = None):
        """Output predicted anaomalies from ensemble
        Args:
            detect_thresh (float, optional): Decision threshold. Defaults to None.
        Returns:
            _type_: pointer
        """
        if detect_thresh is not None:
            self.detect_thresh = detect_thresh 
        df_out = deepcopy(self.df)    
        df_out.index = df_out['year_period_ts']
        df_out.drop(columns=['year_period_ts'], inplace = True)
        self.anomaly_proba.fillna(0, inplace=True)
        det_mask = self.anomaly_proba > self.detect_thresh
        self.anomalies = det_mask*1
        self.outliers = (self.anomalies == 1)
        #df_out['anomalies'] = det_mask*1
        #self.outliers = (df_out['anomalies'] == 1)
        if np.any(self.outliers):
            self.outlier_dates = df_out[self.outliers].time.tolist()
        else:
            self.outlier_dates = []

        self.nof_outliers = len(self.outlier_dates)
        #self.anomalies = df_out['anomalies']

        if self.verbose_train:
            print('\n->', self.nof_outliers,"outlier(s) detected!") 
            print("Occured at year-period(s):\n", self.outlier_dates)
        return self

    def run_all(self, write_table: bool = None, **prepro_para):
        """
        Run all steps from preprocessing to prediction
        for all time series and return all anomalies.
        Input:
        see arguments from process_data() method, including the input dataframe
        """
        #if 'verbose' in list(prepro_para.keys()):
        #   self.verbose_train = prepro_para['verbose']
        if write_table is None:
            write_table = self.config_output['database']['write_table']

        # Run preprocessing:
        #--------------------
        gen0 = self.process_data(**prepro_para)
        self.all_series = list(gen0)

        # Only look for outliers for y_{t}, t >= outlier_filter
        # i.e. only in more recent observations
        where = np.where(np.array(self.time_index) == self.outlier_filter)[0][0]
        self.outlier_search_list = self.time_index[where:]

        self.suspects, self.filt_suspects, self.filt_suspects_values = {}, {}, {}
        self.filt_suspects_plot, self.anomaly_info_all_series = {}, {}
        self.count_outliers = 0

        #-----------------------------------------------------
        # Loop over all univariate (aggregated) time series:
        #-----------------------------------------------------
        for i in range(len(self.all_series)):    
            label, sub_set = self.all_series[i]

            # In case of almost constant time series,
            # e.g. only zeros, check if distr. is close to a degenerate distr.
            # and then skip using this time series in training
            # as this might lead to a warning in the fit process 
            mad = MAD(sub_set['target'].values)
            iqr = IQR(sub_set['target'].values, prob_upper_lower = [90 ,10])
            nondegenerate = (mad > 0)*(iqr > 0)

            #print('{} {} N={} {} {}'.format(i, label, sub_set.shape[0], mad, iqr))

            if (label != 'all') and nondegenerate:        # all subseries except 'all'
                df = deepcopy(sub_set)
                fitted = self.fit(df = df)
                out = fitted.predict()
                self.anomaly_info_all_series[label] = {'df' : df, 'val_series': fitted.val_series, \
                                                       'anom_flag': fitted.anomalies, 'anom_evidence' : fitted.anomaly_proba} #save all anomaly predictions irrespectively if they are current or not, or have been found before. Such filters will be however applied below 

                self.count_outliers += out.nof_outliers

                # In case anomalies have been found:
                if out.nof_outliers > 0:
                    outlier_dates = out.outlier_dates
                    filt = [outl in self.outlier_search_list for outl in outlier_dates]    # filter out old anomalies, only keep new ones
                    filtered_outliers = np.array(outlier_dates)[filt].tolist()
                    self.suspects[label] = outlier_dates      # all

                    # In case they were new (according to time filter):
                    if len(filtered_outliers) > 0:
                        self.filt_suspects[label] = filtered_outliers
                        anom_val = [df[self.target_col].values[df['time'].values == fdates][0] for fdates in filtered_outliers]
                        self.filt_suspects_values[label] = {'anomaly_dates': filtered_outliers, 'anomaly_values': anom_val}
                        self.filt_suspects_plot[label] = {'df' : df, 'val_series': fitted.val_series, 
                                                          'anom_flag': fitted.anomalies, 'anom_evidence' : fitted.anomaly_proba}

        # Prepare for writing to postgres table.
        # row-wise append anomalies in dataframe:
        #------------------------------------------
        res_to_pg = pd.DataFrame(columns=['time_anomaly', 'time_series_name', 'clm_cnt'])  # old target name needed, do to scheam in PG
        for s_name, v in self.filt_suspects_values.items():
            dates_values = list(v.values())
            dates_list, values_list = dates_values[0], dates_values[1]
            assert len(dates_list) == len(values_list), 'Unequal number of anomaly dates and values!'
            tmp = pd.DataFrame(columns=['time_anomaly', 'time_series_name', 'clm_cnt'])   # old target name needed, do to scheam in PG

            # In case more than one outlier per series, loop over all outliers per series:
            for ii in range(len(dates_list)):
                combi = [dates_list[ii], s_name, values_list[ii]]
                tmp.loc[ii] = combi
            res_to_pg = res_to_pg.append(tmp, ignore_index=True)

        # Get historical anomalies from postgres:
        #-------------------------------------------
        #pg = file.PostgresService(verbose=False)
        pkl = file.PickleService(path = "anomaly_history.pkl", verbose=False)

        try :
            #self.anomaly_history = pg.doRead(qry="select * from "+self.tbl_name)
            self.anomaly_history = pkl.doRead()
        except Exception as e:
            if self.verbose_train:
                print(e)
                print("Create table", self.tbl_name, "first before you read from it!")
            self.anomaly_history = pd.DataFrame(columns=['time_anomaly', 'time_series_name', 'clm_cnt'])
            
        # Only return anomalies that have not been
        # reported before (according to postgres table):
        # If anomalies for same name and date are already in database
        # do not show them again (even if target value might differ)
        #---------------------------------------------------------------
        history = deepcopy(self.anomaly_history)
        history['flag'] = 1
        compared = history[['time_anomaly', 'time_series_name', 'flag']].merge(res_to_pg, how="right", left_on = ['time_anomaly', 'time_series_name'], right_on = ['time_anomaly', 'time_series_name'])
        res_to_pg_NEW = compared[compared['flag'].isna()].drop(columns=['flag'])

        if write_table:
            # write method checks for duplicates
            # anyway and appends only new records
            #pg.doWrite(res_to_pg_NEW, self.tbl_name, if_exists = "append")
            pkl.doWrite(res_to_pg_NEW)

        if self.verbose:
            print('Training finished. {} ({} new) outliers detected according to time filter.'.format(res_to_pg.shape[0], res_to_pg_NEW.shape[0]))   
        return res_to_pg, res_to_pg_NEW


    def print_anomalies(self, search_term : str):
        """
        Simple print function which by inputing a search term
        will output the corresponding 'Region/OE/LoB/Loss Cat' combination which
        had anomalies together with their anomalous values and dates 

        Args:
            search_term (str): [description]
        """
        c = 0
        for k,v in self.filt_suspects_values.items():
            if search_term in k:
                print('{}: {}\n'.format(k,v))
                c += 1
        print(f"-> {c} hits found for '{search_term}'.")  

