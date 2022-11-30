import os
from .utils_func import module_logger
from aac_ts_anomaly.config import global_config as glob

# Set logger:
logger_utils = module_logger(__name__ + '.utils', os.path.join(glob.UC_DATA_PKG_DIR, 'utils.log'), mode='w')