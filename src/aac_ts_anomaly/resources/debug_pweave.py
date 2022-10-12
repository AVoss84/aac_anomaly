
import os, sys, stat, warnings, emoji
warnings.filterwarnings("ignore")
from datetime import date
os.chdir("..")
from importlib import reload
from claims_reporting.config import global_config as glob
from claims_reporting.resources import config
from claims_reporting.utils import utils_func as util

filename = util.get_newest_file(search_for = "AGCS CCO CRA - Monthly Incurred Movements", src_dir=glob.UC_DATA_DIR)           # weekly, take newest input
#filename = util.get_newest_file(search_for = "AGCS Global Claims PIC - Notification Counts")   #"AGCS CCO PIC - Payment Counts"
#filedate = filename[(len(filename)-15):(len(filename)-5)]                 # get the date part for timestamp
#print('Recent input file: {}, using date {}.'.format(filename, filedate))
filename

periodicity = 12

if periodicity == 52:
    config_input = config.in_out52['input']
    config_output = config.in_out52['output']
    config_detect = config.in_out52['detection']
if periodicity == 12:    
    config_input = config.in_out12['input']
    config_output = config.in_out12['output']
    config_detect = config.in_out12['detection']


output_files_dir = glob.UC_PWEAVE_DIR       # package folder / Jupyter server
#output_files_dir = glob.UC_REPORT_DIR      # SFTP
print('Writing to: {}'.format(output_files_dir))

filename = config_output['report_filename']         # without timestamp
append_this = ''
filename_new = filename+append_this

output_file_name_all_pdf = filename_new+'_all_combi.pdf'
#output_file_name_all_pdf = filename_new+'_all_combi_'+filedate+'.pdf'
print("-----------------\n Used filenames:\n-----------------\n'{}'".format(output_file_name_all_pdf))


# First generate PANDOC markdown:
#---------------------------------
!pweave --format=pandoc {glob.UC_PWEAVE_DIR}source_file_all_combi12_finance.pmd --output={output_files_dir}claims_anomaly_report.md --figure-directory={output_files_dir}figures

# Then convert PANDOC to PDF:
#-----------------------------
try:
    !chmod +rwx {output_files_dir}
    !/usr/bin/pandoc -s -V geometry:margin=0.1in -o {output_files_dir + output_file_name_all_pdf} {output_files_dir}claims_anomaly_report.md
    print("PDF report: {} created.".format(output_files_dir + output_file_name_all_pdf))   
    !chmod 777 {output_files_dir + output_file_name_all_pdf}   
    print(emoji.emojize('Job successful! :thumbs_up:'))
except Exception as e: 
    print(e) ; print("PDF report: {} could not be created.".format(output_files_dir + output_file_name_all_pdf)) 
    print(emoji.emojize('Job not successful! :disappointed_relieved:', use_aliases=True))



