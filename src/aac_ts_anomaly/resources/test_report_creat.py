
from importlib import reload
from aac_ts_anomaly.utils import utils_func as util
from aac_ts_anomaly.config import global_config as glob
from aac_ts_anomaly.resources import config, blueprints
from aac_ts_anomaly.resources import report_creator
from aac_ts_anomaly.services import file
from aac_ts_anomaly.resources import trainer

from aac_ts_anomaly.utils import tsa_utils as tsa
from aac_ts_anomaly.utils import utils_func as util
from aac_ts_anomaly.config import global_config as glob
from aac_ts_anomaly.services import file
from aac_ts_anomaly.resources import (config, preprocessor, trainer)

reload(glob)
reload(blueprints)
reload(report_creator)
reload(util)

glob.UC_DATA_PKG_DIR

print("Debugging!") 

fname = "agg_time_series_52.csv"

cr = report_creator.creator(output_to = ['local'], 
                            input_file = fname, verbose = True,
                            src_dir = glob.UC_DATA_DIR)
                    
out = cr.create(usecase = ['all'])     


########################################################################################
#- More precisely we use: $y_{t} = \frac{1}{n_{t}}\sum_{i=1}^{n_{t}} x_{i,t}$ with $x_{i,t}$ the incurred gross net value in Euro of claim $i$ at time $t$ and $n_{t}$ the number of claims at time $t$

"""
filename = util.get_newest_file(search_for = "AGCS CCO CRA - Monthly Incurred amounts",  src_dir=glob.UC_DATA_DIR)
xls = file.XLSXService(path=filename, root_path=glob.UC_DATA_DIR, dtype= {'time': str}, sheetname='data', index_col=None, header=0)
filename

data_orig = xls.doRead()

data_orig.shape

data_orig.head()

train = trainer.trainer(verbose=False)

results, results_new = train.run_all(data_orig = data_orig, aggreg_level = 'all_combi', write_table = False, verbose=True)

#results = deepcopy(results_new)
"""