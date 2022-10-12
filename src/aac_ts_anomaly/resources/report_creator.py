from copy import deepcopy
from locale import D_FMT
#import glob as gl
import subprocess, os, sys, stat, warnings
#from datetime import datetime
#import dateutil.parser as dateparser
#from textdistance import jaro_winkler
import pandas as pd
warnings.filterwarnings("ignore")
from datetime import date
from importlib import reload
from aac_ts_anomaly.utils import utils_func as util
from aac_ts_anomaly.config import global_config as glob
from aac_ts_anomaly.resources import config, blueprints
from aac_ts_anomaly.resources import (config, preprocessor, trainer)


class creator(blueprints.AbstractReportCreator):

    def __init__(self, *args, **kwargs):
        super(creator, self).__init__(*args, **kwargs)

        def _run_all(self, df : pd.DataFrame):
            """
            Run all 
            """
            train = trainer.trainer(verbose=False)
            res1, res2_new = train.run_all(data_orig = df, verbose=True, aggreg_level = 'all_combi')

            return res1, res2_new


    # def _run_all(self):
    #     """
    #     Creates the actual PDF report (for all LoB/Region/OE combinations) 
    #     """
    #     print('-> Creating report for all LoB/Region/OE combinations\n')
        
    #     if self.periodicity == 52:
    #         cmd_pweave = 'pweave --format=pandoc {}source_file_all_combi52.pmd '.format(glob.UC_PWEAVE_DIR) +' --output={}claims_anomaly_report.md'.format(self.output_files_dir) +' --figure-directory={}figures'.format(self.output_files_dir)

    #     if self.periodicity == 12:
    #         cmd_pweave = 'pweave --format=pandoc {}source_file_all_combi12_finance.pmd '.format(glob.UC_PWEAVE_DIR) +' --output={}claims_anomaly_report.md'.format(self.output_files_dir) +' --figure-directory={}figures'.format(self.output_files_dir)
    #     # Create pmd file via pweave:
    #     process = subprocess.Popen(cmd_pweave.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    #     self.stdout_pweave, self.stderr_pweave = process.communicate()

    #     try:
    #         cmd_chmod = 'chmod +rwx {}'.format(self.output_files_dir)
    #         process = subprocess.Popen(cmd_chmod.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    #         stdout_chmod, stderr_chmod = process.communicate()
            
    #         cmd_pandoc = '/usr/bin/pandoc -s -V geometry:margin=0.1in -o {} {}claims_anomaly_report.md'.format(self.output_files_dir + self.output_file_name_all_pdf, self.output_files_dir)
    #         process = subprocess.Popen(cmd_pandoc.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    #         self.stdout_pandoc, self.stderr_pandoc = process.communicate()
    #         print("PDF report: {} created.\n".format(self.output_files_dir + self.output_file_name_all_pdf))
            
    #         cmd_chmod2 = 'chmod 777 {}'.format(self.output_files_dir + self.output_file_name_all_pdf)
    #         process = subprocess.Popen(cmd_chmod2.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    #         stdout_chmod2, stderr_chmod2 = process.communicate()      
    #         check = "Job ALL successful!\n"     
    #     except Exception as e: 
    #         print(e);print("PDF report: {} could not be created.\n".format(self.output_files_dir + self.output_file_name_all_pdf)) 
    #         check = "Job ALL not successful!\n"
    #     return check

        
