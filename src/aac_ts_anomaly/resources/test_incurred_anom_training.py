
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
from claims_reporting.services import file
#from claims_reporting.services import base
from claims_reporting.resources import config

reload(tsa)
reload(file)
reload(config)
#reload(base)
reload(glob)


filename = util.get_newest_file(search_for = "AGCS CCO CRA - Monthly Incurred amounts",  src_dir=glob.UC_DATA_DIR)
xls = file.XLSXService(path=filename, root_path=glob.UC_DATA_DIR, dtype= {'time': str}, sheetname='data', index_col=None, header=0)
filename

data_orig = xls.doRead()

data_orig.shape

data_orig.head()

##############

from claims_reporting.resources import preprocessor_incurred as pre

reload(pre)

config_detect = config.in_out12['detection']
config_detect
outlier_filter = config_detect['training']['outlier_filter']
hyper_para = config_detect['training']['hyper_para']
stat_transform = config_detect['training']['stat_transform']


# Instantiate class:
#--------------------
claims = pre.claims_reporting(ts_col = 'target')

#aggreg_level, pre_filter, ignore_week_lag, min_sample_size, min_median_cnts = list(config_detect['preprocessing'].values())

gen = claims.process_data(data_orig, aggreg_level = 'all_combi', agg_func = 'sum', ignore_lag = 1)

all_series = dict(gen)

# Get next series
#-------------------
label, sub_set = next(gen)

print(label, sub_set.shape[0])
df = deepcopy(sub_set)
df.shape
df.head(20)