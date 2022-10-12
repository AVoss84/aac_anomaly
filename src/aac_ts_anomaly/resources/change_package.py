import os, warnings
#warnings.filterwarnings("ignore")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.core.defchararray import rpartition
import seaborn as sns
import pandas as pd
#from statsmodels.tsa.stattools import adfuller, kpss, acf, grangercausalitytests
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, month_plot, quarter_plot
import numpy as np
from copy import deepcopy
import datetime
from importlib import reload
import adtk
#import pweave     # for markdown reports

#pd.set_option('display.max_rows', 10**5)
pd.set_option('display.max_columns', 10**5)

from aac_ts_anomaly.utils import tsa_utils as tsa
#from aac_ts_anomaly.utils import utils_func as util
from aac_ts_anomaly.config import global_config as glob
from aac_ts_anomaly.services import file
from aac_ts_anomaly.resources import config

reload(config)
reload(tsa)
reload(file)
reload(glob)

glob.UC_CODE_DIR
glob.UC_DATA_DIR


