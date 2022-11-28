import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from scipy.special import loggamma #gamma as gamma_fct
from math import cos, pi
from numpy import log, sum, exp, prod
#from numpy.random import beta, binomial, normal, uniform, gamma, seed, rand, poisson
from copy import deepcopy
import glob as gl
import subprocess, os, time, logging
from datetime import datetime
import dateutil.parser as dateparser
from textdistance import jaro_winkler
from importlib import reload
from functools import wraps
from aac_ts_anomaly.resources import config
from aac_ts_anomaly.config import global_config as glob
from aac_ts_anomaly.services import file


def module_logger(mod_name : str, filename : str, **kwargs):
    """Create logger object for module specific logging"""
    logger = logging.getLogger(mod_name)
    #logging.basicConfig(filename=glob.UC_DATA_PKG_DIR+'session.log', level=logging.DEBUG, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
    logger.setLevel(logging.DEBUG)    # lowest severity level; higher will be shown too
    #handler = logging.StreamHandler()
    handler = logging.FileHandler(filename=filename, **kwargs)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger_utils = module_logger(__name__ + '.utils', os.path.join(glob.UC_DATA_PKG_DIR, 'utils.log'), mode='w')


def get_filename_NN(input_filename : str, lookup_dict : dict):

    """
    Get nearest neighbor file name among a set of reference file names
    This is used as a consistency check for the user (is file supported?) 

    Input:
    ------
    input_filename: filename from user interface (=original filename)
    lookup_dict: generic reference file names (without dates, versions etc.)
    """
    dist_prev = 1. ; match = ''; distances={}
    for ref_file_name in lookup_dict.keys():
        dist = jaro_winkler.normalized_distance(input_filename, ref_file_name)
        distances[ref_file_name] = dist
        if dist < dist_prev:   # accept candidate
            match = ref_file_name 
            dist_prev = dist
    assert len(match)>0, "File not known!!"       
    return distances, match    


def get_newest_file(search_for : str = "AGCS Global Claims PIC - Notification Counts", 
                            src_dir : str = glob.UC_DATA_DIR, verbose : bool = False):
        """
        Detect newest input file in directory and return its name
        Note: the function is fully agnostic of file formats, xls, pkl etc.
        """
        files_path = os.path.join(src_dir, '*')       # add wild card
        files = sorted(gl.iglob(files_path), key=os.path.getctime, reverse=True)   # sort hit list by creation date
        result = [ search_for in file for file in files ] 
        if verbose : print('src_dir: {}'.format(files_path)), print('sorted hitlist: {}'.format(files))
        try:
            hit = np.min(np.where(result))
            return os.path.basename(files[hit])                # take only name from file path name of youngest
        except Exception as ex:
            print('No input file found!\n',ex)   
            return None 
            
#---------------------------------------------------------------------------------------

# Run time decorator for any function func:
def timer(func):
    """Print the runtime of the decorated function"""
    @wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      
        run_time = end_time - start_time    
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

#------------------------------------------------------------------------------

def ts_plot(x, y: np.array, vertical : list = None, title : str ="", xlabel : str ='Time', ylabel : str ='target', dpi : int = None, **para):
        """Plot time series together with detected anomalies

        Args:
            x (_type_): _description_
            y (np.array): _description_
            vertical (list, optional): _description_. Defaults to None.
            title (str, optional): _description_. Defaults to "".
            xlabel (str, optional): _description_. Defaults to 'time'.
            ylabel (str, optional): _description_. Defaults to 'target'.
            dpi (int, optional): _description_. Defaults to None.

        Returns:
            plt.figure.Figure: matplotlib object
        """
        fig, ax = plt.subplots(figsize=(20, 4.5), dpi=dpi)
        ax.plot(x, y, color='tab:blue', linestyle='-', marker='o', markerfacecolor='orange', label='Observation', **para)
        if vertical : 
            plt.vlines(x=vertical, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='red', zorder=2, lw=.65, label='Anomaly',  linestyle='--')
        ax.set_title(title, fontdict={'fontsize': 16, 'fontweight': 'medium'})
        #ax.axhline(1, color="red", linestyle="--")
        # set monthly locator
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        # set formatter
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%W-%y'))
        # set font and rotation for date tick labels
        plt.gcf().autofmt_xdate()
        plt.xticks(rotation=70)
        plt.legend(loc = 'upper left')
        plt.tight_layout()
        plt.show() 
        return fig     


def anomaly_prob_plot(x, y: np.array, detect_thresh: float = 0.5, dpi : int = None, **para):
    """Plot anomaly probabilities for each date
    Args:
        x (_type_): _description_
        y (np.array): _description_
        detect_thresh (float, optional): _description_. Defaults to 0.5.
        dpi (int, optional): _description_. Defaults to None.

    Returns:
        plt.figure.Figure: matplotlib object
    """
    fig, ax = plt.subplots(figsize=(20, 4), dpi=dpi)
    pro = plt.plot(x, y, color='tab:blue',label="Anomaly probability", linestyle='--', marker='o', markerfacecolor='orange', linewidth=1)
    plt.plot(x, [detect_thresh]*len(x), label="Decision threshold",  linewidth=.4, color="red", linestyle='--')
    plt.gca().set(title="", xlabel="Time", ylabel="Probability", ylim = plt.ylim(-0.02, 1.05))   #plt.xlim(left=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%W-%y'))
    # set font and rotation for date tick labels
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation=70)
    plt.title(r'Anomaly probabilities $\pi_{t}, t=1,...,T$', fontdict = {'fontsize' : 14})
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()  
    return fig

#------------------------------------------------------------------------------
# Split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

#------------------------------------------------------------------------------
def difference(dataset, lag=1):
	diff = list()
	for i in range(lag, len(dataset)):
		value = dataset[i] - dataset[i - lag]
		diff.append(value)
	return pd.Series(diff)        

#------------------------------------------------------------------------------
def embed_ts(data, lags=1, dropnan=True):
  """
  Create lagged versions of univariate time series
  """  
  df = deepcopy(data) 
  assert isinstance(df, pd.core.frame.DataFrame), 'data must be a dataframe.'
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
def exact_post_cp(y : np.array, alpha : float = 2.2, beta : float = 1.1, gamma : float = 2.0, delta : float = 1.2):
    """
    Poisson change point model ('coal mining disaster') 
    Calculate the exact posterior of single change point.
    """
    n = len(y) ; s_n = np.sum(y) ; lprob = []
    for m in range(n):
    #for t in range(trim, n-2):
        s_m = np.sum(y[:m])
        #log_post_m = np.log(gamma_fct(alpha + s_m)) + np.log(gamma_fct(gamma + s_n - s_m)) - (alpha + s_m) * np.log(beta + m) - (gamma + s_n - s_m) * np.log(delta + n - m)
        log_post_m = loggamma(alpha + s_m) + loggamma(gamma + s_n - s_m) - (alpha + s_m) * np.log(beta + m) - (gamma + s_n - s_m) * np.log(delta + n - m)
        lprob.append(log_post_m)
    lprob = lprob - np.max(lprob)
    prob = np.exp(lprob)
    probm = prob/np.sum(prob)
    return probm

#-------------------------------------------------------------------------------------------------------------------

def MAD(x : np.array) -> float:
    """
    Calculate the median absolute deviation (MAD) as robust dispersion measure against outliers

    Args:
        x (np.array): time series observations
    Returns: MAD value
    """
    return np.nanmedian(abs(x - np.nanmedian(x)))  


def IQR(x: np.array, prob_upper_lower : list = [75 ,25]) -> float:
    """
    Calculate the inter quantile range (IQR) as robust dispersion measure against outliers.
    Ex. prob_upper_lower = [75, 25] is the Interquartile range, whereas [90, 10] yields the interdecile range

    Args:
        x (np.array): observations
        upper_lower (list, optional): list of upper lower percentiles (=prob.). Defaults to [75 ,25].

    Returns:
        float: IQR
    """
    q_up, q_low = np.nanpercentile(x, prob_upper_lower)
    return q_up - q_low
