
from importlib import reload
from claims_reporting.resources import report_creator
from claims_reporting.utils import utils_func as util
from claims_reporting.config import global_config as glob
from claims_reporting.services import file
from claims_reporting.resources import trainer

print("Debugging!") 

#reload(blueprints)
#reload(report_creator)

fname = "AGCS CCO CRA - Monthly Incurred amounts 18-11-2021.xlsx"

cr = report_creator.creator(output_to = ['local'], 
                            input_file = fname, verbose = True,
                            src_dir = glob.UC_DATA_PKG_DIR)

#out = cr.create(usecase = ['region'])                        
out = cr.create(usecase = ['all'])     


########################################################################################
#- More precisely we use: $y_{t} = \frac{1}{n_{t}}\sum_{i=1}^{n_{t}} x_{i,t}$ with $x_{i,t}$ the incurred gross net value in Euro of claim $i$ at time $t$ and $n_{t}$ the number of claims at time $t$


import os, warnings
import pandas as pd
import numpy as np
from copy import deepcopy
import glob as gl
import subprocess
#from collections import defaultdict 
#os.chdir("..")
#warnings.filterwarnings("ignore")
from importlib import reload
from claims_reporting.utils import utils_func as util
from claims_reporting.services import file, base
from claims_reporting.config import global_config as glob
from claims_reporting.resources import config

reload(file)
reload(config)
reload(base)
reload(glob)
reload(util)


input_file = "AGCS CCO CRA - Monthly Incurred amounts" 
#input_file = "AGCS CCO PIC - Payment Counts"

input_files_lookup = {"AGCS CCO PIC - Notification Counts" : 52, 
                    "AGCS CCO PIC - Payment Counts": 12,
                    "AGCS CCO CRA - Monthly Incurred amounts" : 12}
                            
assert input_file in input_files_lookup, "File not known!" 
periodicity = input_files_lookup[input_file]

print(periodicity)

#output_to = ['sftp', 'jupyter']
output_to = ['jupyter']
verbose = True
        
assert output_to[0] in ['sftp', 'jupyter'], 'arg output_to must be sftp or jupyter!'
if output_to[0] == 'sftp':
    output_files_dir = glob.UC_REPORT_DIR      # SFTP
else:
    output_files_dir = glob.UC_PWEAVE_DIR       # package folder / Jupyter server

output_files_dir 

#filename = list(config_input['service']['XLSXService'].values())[0]
filename = util.get_newest_file(search_for = input_file, verbose=False)
filename

filedate = filename[(len(filename)-15):(len(filename)-5)]
filedate = filedate.replace(" ", "_")
filedate

if verbose : 
    print('Input file: {} (Date: {})'.format(filename, filedate))
    print('Writing reports to: {}'.format(output_files_dir))

if periodicity == 52:
    #config_input = config.in_out52['input']
    config_output = config.in_out52['output']
if periodicity == 12:    
    #config_input = config.in_out12['input']
    config_output = config.in_out12['output']

filename = config_output['report_filename']         # without timestamp
append_this = ''
filename_new = filename+append_this

output_file_name_region_pdf = filename_new+'_region_only_'+filedate+'.pdf'
output_file_name_all_pdf = filename_new+'_all_combi_'+filedate+'.pdf'
output_file_name_lob_pdf = filename_new+'_lob_only_'+filedate+'.pdf'
print(" Used filenames:\n-----------------\n'{}'\n'{}'\n'{}".format(output_file_name_all_pdf, output_file_name_region_pdf, output_file_name_lob_pdf))
#print('-> Creating report for all LoB/Region/OE combinations\n')

##### RUN ALL
########################
#cmd_pweave = 'pweave --format=pandoc {}source_file_all_combi52.pmd '.format(glob.UC_PWEAVE_DIR) +' --output={}claims_anomaly_report.md'.format(output_files_dir) +' --figure-directory={}figures'.format(output_files_dir)

#if periodicity == 52:
#    cmd_pweave = 'pweave --format=pandoc {}source_file_all_combi52.pmd '.format(glob.UC_PWEAVE_DIR) +' --output={}claims_anomaly_report.md'.format(output_files_dir) +' --figure-directory={}figures'.format(output_files_dir)
#if periodicity == 12:
cmd_pweave = 'pweave --format=pandoc {}source_file_all_combi12_finance.pmd '.format(glob.UC_PWEAVE_DIR) +' --output={}claims_anomaly_report_fin.md'.format(output_files_dir) +' --figure-directory={}figures'.format(output_files_dir)

# Create pmd file via pweave:
process = subprocess.Popen(cmd_pweave.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
stdout_pweave, stderr_pweave = process.communicate()
print(stdout_pweave)

# Change rights on SFTP:
cmd_chmod = 'chmod +rwx {}'.format(output_files_dir)
process = subprocess.Popen(cmd_chmod.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
stdout_chmod, stderr_chmod = process.communicate()
print(stdout_chmod)

# Create PDF document via Pandoc:
cmd_pandoc = '/usr/bin/pandoc -s -V geometry:margin=0.1in -o {} {}claims_anomaly_report_fin.md'.format(output_files_dir + output_file_name_all_pdf, output_files_dir)
cmd_pandoc
process = subprocess.Popen(cmd_pandoc.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
stdout_pandoc, stderr_pandoc = process.communicate()
print(stdout_pandoc)

print("PDF report: {} created.\n".format(output_files_dir + output_file_name_all_pdf))



#pweave --format=pandoc source_file_all_combi52.pmd --output=claims_anomaly_report.md --figure-directory=figures
#pandoc -s -V geometry:margin=0.1in -o my_test.pdf claims_anomaly_report.md



#pweave --format=pandoc /app/src/pweave/source_file_all_combi12_finance.pmd  --output=/app/src/pweave/claims_anomaly_report_fin.md --figure-directory=/app/src/pweave/figures

#pweave --format=pandoc /app/src/pweave/test.pmd  --output=/app/src/pweave/test.md --figure-directory=/app/src/pweave/figures
