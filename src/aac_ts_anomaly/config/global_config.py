import os, sys
from pathlib import Path

#-------------------------------
# Which dev environment to use?
#-------------------------------

#using = 'local'                # Jupyter server
using = 'vm'                   # own virtual machine
#using = 'docker'                # Docker container

## Check if required environment variables exist
## if not apply default paths from test environment:
#-----------------------------------------------------------
if using == 'vm':
    defaults = {
            "UC_CODE_DIR": (Path.home() / "Documents/GitHub/aac_anomaly/src").__str__(),       
            "UC_DATA_DIR": (Path.home() / "Documents/Arbeit/Allianz/AZVers/data").__str__(),                      
            "UC_DATA_PKG_DIR": (Path.home() / "Documents/GitHub/aac_anomaly/src/aac_ts_anomaly/data").__str__(),    # data folder within package
            #"UC_DB_CONNECTION": 'postgresql://postgres:kakYritiven@agcs-postgres-1-server.service.dsp.allianz/prod',
            "UC_SAVE_TO" : "local",  #"sftp"   local means save to pweave package folder  
            "UC_PORT": "5000", 
            "UC_APP_CONNECTION": "127.0.0.1"
    }
elif using == 'docker':
    defaults = {
            "UC_CODE_DIR": "/app/src/",                 
            "UC_DATA_DIR": "/app/data/",         
            "UC_DATA_PKG_DIR": "/app/src/aac_ts_anomaly/data/",    # data folder within package       
            #"UC_DB_CONNECTION": 'postgresql://postgres:kakYritiven@agcs-postgres-1-server.service.dsp.allianz/prod',
            "UC_SAVE_TO" : "local",     
            "UC_PORT": "5000",
            "UC_APP_CONNECTION": "0.0.0.0"}      # must be 0.0.0.0 for later deployment instead of 127.0.0.1 only works locally on the VM                
else:
    defaults = {"UC_CODE_DIR": (Path.home() / "aac_anomaly_detection/src/").__str__(),              # Jupyter
                "UC_DATA_DIR": (Path.home() / "data/").__str__(),     
                "UC_DATA_PKG_DIR": (Path.home() / "aac_anomaly_detection/src/aac_ts_anomaly/data").__str__(),      # data folder within package
                #"UC_DB_CONNECTION": 'postgresql://postgres:kakYritiven@agcs-postgres-1-server.service.dsp.allianz/prod',
                "UC_SAVE_TO" : "sftp",  #"local"   
                "UC_PORT":"5000", 
                "UC_APP_CONNECTION":"127.0.0.1"
        }
#-------------------------------------------------------------------------------------------------------------------------------

for env in list(defaults.keys()):
    if env not in os.environ:
        os.environ[env] = defaults[env]
        exec(f"{env} = os.environ['{env}']")
        print("Environment Variable " + str(env) + " has been set to default: " + str(os.environ[env]))
