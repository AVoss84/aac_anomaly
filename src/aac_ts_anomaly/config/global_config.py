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
            "UC_CODE_DIR": str(Path.home() / "Documents/GitHub/aac_anomaly/src"),       
            "UC_DATA_DIR": str(Path.home() / "Documents/Arbeit/Allianz/AZVers/data"),                      
            #"UC_REPORT_DIR": "/data/data/submission/report/",            # SFTP output 
            "UC_DATA_PKG_DIR": str(Path.home() / "Documents/GitHub/aac_anomaly/src/aac_ts_anomaly/data/"),    # data folder within package
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
            #"UC_REPORT_DIR": "/data/submission/report/",            # SFTP output 
            #"UC_REPORT_DIR": "/app/src/pweave/",           
            #"UC_DB_CONNECTION": 'postgresql://postgres:kakYritiven@agcs-postgres-1-server.service.dsp.allianz/prod',
            "UC_SAVE_TO" : "local",     
            "UC_PORT": "5000",
            "UC_APP_CONNECTION": "0.0.0.0"}      # must be 0.0.0.0 for later deployment instead of 127.0.0.1 only works locally on the VM                
else:
    defaults = {"UC_CODE_DIR": str(Path.home() / "aac_anomaly_detection/src/"),              # Jupyter
                "UC_DATA_DIR": str(Path.home() / "data/"),     # SFTP input
                #"UC_REPORT_DIR": "/data/submission/report/",            # SFTP output 
                "UC_DATA_PKG_DIR": str(Path.home() / "aac_anomaly_detection/src/aac_ts_anomaly/data"),      # data folder within package
                #"UC_DB_CONNECTION": 'postgresql://postgres:kakYritiven@agcs-postgres-1-server.service.dsp.allianz/prod',
                "UC_SAVE_TO" : "sftp",  #"local"   
                "UC_PORT":"5000", 
                "UC_APP_CONNECTION":"127.0.0.1"
        }
#-------------------------------------------------------------------------------------------------------------------------------

for env in list(defaults.keys()):
    if env not in os.environ:
        os.environ[env] = defaults[env]
        print("Environment Variable: " + str(env) + " has been set to default: " + str(os.environ[env]))

UC_SAVE_TO = os.environ['UC_SAVE_TO']
UC_PWEAVE_DIR = os.path.join(os.environ['UC_CODE_DIR'],"pweave/")
#UC_REPORT_DIR = os.environ['UC_REPORT_DIR'] 
UC_CODE_DIR = os.environ['UC_CODE_DIR']  
UC_DATA_DIR = os.environ['UC_DATA_DIR']                   # Data folder SFTP server
#UC_DATA_DIR_PKG = os.environ["UC_DATA_DIR_PKG"]            # Data folder claims-reporting package
UC_PORT = os.environ['UC_PORT']
#UC_DB_CONNECTION= os.environ['UC_DB_CONNECTION']
UC_APP_CONNECTION = os.environ['UC_APP_CONNECTION']
UC_DATA_PKG_DIR = os.environ['UC_DATA_PKG_DIR']

