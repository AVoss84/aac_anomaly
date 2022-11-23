import os
from aac_ts_anomaly.config import global_config as glob 
from aac_ts_anomaly.services.file import YAMLservice

# Input-Output yaml files for each usecase, i.e. weekly, monthly seasonality  
yaml_files = {'in_out12': 'io_monthly', 'in_out52': 'io_weekly'} 

# Read in yaml files for each subtable and assign keys as obj. names:
for conf, tbl in yaml_files.items():
    exec(conf+f"=YAMLservice(path='aac_ts_anomaly/resources/{tbl}.yaml').doRead()")

