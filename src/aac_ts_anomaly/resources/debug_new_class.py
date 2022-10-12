
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

#os.getcwd()
#reload(util)

filename = util.get_newest_file(search_for = "AGCS CCO CRA - Monthly Incurred Movements",  src_dir=glob.UC_DATA_DIR)
filename

xls = file.XLSXService(path=filename, root_path=glob.UC_DATA_DIR, dtype= {'time': str}, sheetname='data', index_col=None, header=0)

data_orig = xls.doRead()

data_orig.shape

data_orig.head()

####################################################################################

from claims_reporting.resources import trainer

reload(trainer)

train = trainer.trainer(verbose=False)   # will call prerocessing...

results, results_new = train.run_all(data_orig = data_orig, write_table = False, verbose=True, aggreg_level = 'all_combi')

# 'Region'-'OE'- 'Lob' - 'LossCat'
new_anomalies = list(set(results_new['time_series_name']))
new_anomalies
     
train.print_anomalies(search_term = "France")


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

gen = claims.process_data(data_orig, aggreg_level = 'all_combi', ignore_lag = 1, min_sample_size = 20)

all_series = list(gen)

for i in range(len(all_series)):    
    label, sub_set = all_series[i]
    print(label)

# Get next series
#-------------------
label, sub_set = next(gen)

print(label, sub_set.shape[0])
df = deepcopy(sub_set)



############
# Training:
############

from claims_reporting.resources import trainer

reload(trainer)

train = trainer.trainer(verbose=False)   # will call prerocessing...
fitted = train.fit(df = df)

results, results_new = train.run_all(data_orig = data_orig, write_table = False, verbose=True, aggreg_level = 'all_combi')


train.periodicity

y = fitted.ts_values
out = fitted.predict(detect_thresh = None)
out

###########################################################################

from claims_reporting.resources import bocd

reload(bocd)

T = len(y)
hazard = .01  # Constant prior on changepoint probability.
mean0  = 0      # The prior mean on the mean parameter.
var0   = 2      # The prior variance for mean parameter.
varx   = 1      # The known variance of the data.
cps = None

model = bocd.GaussianUnknownMean(mean0, var0, varx)
bc = bocd.BayesOCPD(model, hazard, mini_run_length = 2)




















