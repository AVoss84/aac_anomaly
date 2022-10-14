import streamlit as st
import pandas as pd
import numpy as np
import base64
import glob as gl
from zipfile import ZipFile
from os.path import basename
from PIL import Image
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from aac_ts_anomaly.utils import tsa_utils as tsa
from aac_ts_anomaly.utils import utils_func as util
from aac_ts_anomaly.config import global_config as glob
from aac_ts_anomaly.services import file
from aac_ts_anomaly.resources import (config, preprocessor, trainer)

import os, warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, month_plot, quarter_plot
import numpy as np
from copy import deepcopy
from importlib import reload
import adtk
from adtk.visualization import plot
import statsmodels.api as sm


# Set Page name and icon, Layout and sidebar expanded
img = Image.open(os.path.join(glob.UC_CODE_DIR,'templates','allianz_logo.jpg'))
st.set_page_config(page_title='Anomaly Report Creator', page_icon=img, layout="wide", initial_sidebar_state='expanded')



anomaly_history = pd.DataFrame(columns=['time_anomaly', 'time_series_name', 'clm_cnt'])

# Use pickle file as substitute for Postgres:
# pkl = file.PickleService(path = "anomaly_history.pkl")

# pkl.doWrite(anomaly_history)

# pkl.doRead()


# Methods
# def create_reports(df, filename, usecase):

#     # Set target location
#     target_location = glob.UC_SAVE_TO   
#     print(f'\nTarget location: {target_location}\n')

#     # Safe uploaded file to statics:
#     if df is not None:
#             try:
#                 out_path = glob.UC_DATA_PKG_DIR + filename
#                 df.to_excel(out_path, sheet_name='data', index=False)  
#                 print("Input file temporarily saved to: {}".format(out_path))
#             except ImportError as ex :
#                 print(ex)

#     #### Create Reports ####
#     # First save locally and then copy to SFTP in case needed
#     # is done to circumvent the issue that pdfs were not populated properly in this case
#     try:   
#         # Instantiate:    
#         cr = report_creator.creator(output_to = ['local'], 
#                         input_file = filename, verbose = True,
#                         src_dir = glob.UC_DATA_PKG_DIR)
        
#         # Run model and create reports
#         #----------------------------------
#         out = cr.create(usecase = usecase)  
#         print("stdout_pweave:\n",cr.stdout_pweave)

#     except Exception as ex1:
#         error = 'Report(s) could not be created!'
#         print(ex1); print(error)

#     finally:
#         # Drop tmp. uploaded file in case model run not worked:
#         os.remove(glob.UC_DATA_PKG_DIR + filename)
    

#     for ucs in usecase:
#         fname = cr.filename_lookup[ucs]
#         if target_location == 'sftp':                         
#                 try:
#                     #shutil.copy2(glob.UC_PWEAVE_DIR + fname, glob.UC_REPORT_DIR + fname)  # copy to SFTP
#                     os.remove(glob.UC_PWEAVE_DIR + fname)   # delete locally
#                 except Exception as ex3:
#                     print(ex3)
#                     print("Could not copy {} to SFTP!!".format(glob.UC_PWEAVE_DIR + fname, glob.UC_REPORT_DIR + fname))    
#                 print("Final PDF report saved to: {}\n".format(glob.UC_REPORT_DIR + cr.output_file_name_all_pdf))

#         if target_location == 'local':
#             if fname in os.listdir(glob.UC_PWEAVE_DIR):
#                 print("Final PDF report saved to: {}\n".format(glob.UC_PWEAVE_DIR + fname))
#             else:
#                 #print(os.listdir(glob.UC_REPORT_DIR))
#                 print("Final PDF report NOT saved to: {}\n".format(glob.UC_PWEAVE_DIR + fname))
        
#     # Delete contents if figure folder after having build the pdf file:
#     for f in gl.glob(glob.UC_PWEAVE_DIR+'figures/'+'*'): os.remove(f)
  
#     #Get all reportfiles
#     return cr.filename_lookup


######################################################################################
################################# Start App   ########################################
######################################################################################

def main():

    header = st.container()
    with header:
        # Title of app
        st.title("Anomaly Report Creator")
        st.markdown("***")
    with st.sidebar:
        st.image(os.path.join(glob.UC_CODE_DIR,'templates','agcs_banner.png'), use_column_width=True)
        # set up file upload:
        uploaded_file = st.file_uploader("Upload file:", type = ["csv"])    # returns byte object
        print(uploaded_file)

        # select box for usecase
        #usecases = ['All', 'Region', 'LoB']
        #selected_usecase =  st.multiselect("Use case:", usecases)

    plot_space = st.container()

        #st.info("Select reports and press the download button.")

    # report_explorer = st.container()
    # with report_explorer:
    #     #st.markdown("### Report Download Explorer")
    #     #st.info("Select reports and press the download button.")

    #     # Set target location
    #     target_location = glob.UC_SAVE_TO  

    #     # Get File List
    #     target_dir = glob.UC_REPORT_DIR if target_location == 'sftp' else glob.UC_PWEAVE_DIR
    #     file_list = sorted(gl.glob(target_dir+"*.pdf"), key=os.path.getctime, reverse=True)
    #     short_list = [filename.replace(target_dir+"claims_anomaly_report_incurred_", "").replace(".pdf", "") for filename in file_list]

    #     # Create Report Select Field
    #     selected_reports = st.multiselect("Files available for download:", short_list)
    #     st.markdown("***")

    #     #Create Download Button
    #     if len(selected_reports) == 1:
    #         index = short_list.index(selected_reports[0])
    #         with open(file_list[index], "rb") as file:
    #             btn = st.download_button(
    #                 label="File download",
    #                 data=file,
    #                 file_name=file.name.replace(target_dir, ""),
    #                 mime="application/pdf"
    #             )
    #     elif len(selected_reports) > 1:
    #         #Create Report zip
    #         zip_path = target_dir + "claims_anomaly_reports.zip"
    #         report_zip = ZipFile(zip_path, "w")
    #         for report in selected_reports:
    #             index = short_list.index(report)
    #             report_zip.write(file_list[index], basename(file_list[index]))
    #         report_zip.close()
    #         with open(zip_path, "rb") as file:
    #             btn = st.download_button(
    #                 label="Download Reports",
    #                 data=file,
    #                 file_name=file.name,
    #                 mime="application/zip"
    #             )

    data_orig = None
    report_lookup = {}

    if uploaded_file is not None:
        try:
            #data_orig = pd.read_excel(uploaded_file, sheet_name="data")
            data_orig = pd.read_csv(uploaded_file, delimiter=',')
            print(data_orig.shape)

            target_col='target'

            data_orig.rename(columns={'lob': 'Lob', 'erartbez': 'Event_descr', 'time_index': 'time', 'clm_cnt' : target_col}, inplace=True) 
            data_orig = data_orig[['time', 'Lob', 'Event_descr', target_col]]

            #st.markdown("### Report info")
            dataset = st.expander(label = "Display imported data")
            with dataset:
                #st.dataframe(data_orig.head(100))
                st.table(data_orig.head(100))
            st.success(uploaded_file.name + ' successfully uploaded!')
        except Exception as ex:
            st.error("Invalid File")
    
        # Only create button, if valid file is uploaded
        with st.sidebar:
           submitted = st.button('Run analysis')    # boolean 

        ################### ANOMALY DETECTION ####################################    
        periodicity = 52

        if periodicity == 52:
            config_output = config.in_out52['output']
            config_detect = config.in_out52['detection']
        if periodicity == 12:    
            config_output = config.in_out12['output']
            config_detect = config.in_out12['detection']


        hyper_para = config_detect['training']['hyper_para']
        transformers = config_detect['training']['transformers']
        stat_transform = config_detect['training']['stat_transform']
        outlier_filter = config_detect['training']['outlier_filter']
        aggreg_level, pre_filter, ignore_lag, min_sample_size, min_median_cnts = list(config_detect['preprocessing'].values())
        tbl_name = config_output['database']['tbl_name']
        detect_thresh = config_detect['prediction']['detect_thresh']


        #filename = list(config_input['service']['XLSXService'].values())[0]
        if periodicity == 52:
            filename = util.get_newest_file(search_for = "agg_time_series_52",  src_dir=glob.UC_DATA_DIR)    # weekly
        if periodicity == 12:
            filename = util.get_newest_file(search_for = "agg_time_series_12",  src_dir=glob.UC_DATA_DIR)       # monthly

        age = 6
        if outlier_filter is None:
                six_months_ago = date.today() - relativedelta(months=age)
                outlier_filter = six_months_ago.strftime("%Y-%m")

        #reload(util)
        #reload(config)

        #outlier_filter = config_detect['training']['outlier_filter']
        hyper_para = config_detect['training']['hyper_para']
        #print(hyper_para)
        stat_transform = config_detect['training']['stat_transform']

        # Instantiate class:
        #--------------------
        claims = preprocessor.claims_reporting(periodicity=periodicity)

        aggreg_level, pre_filter, ignore_lag, min_sample_size, min_median_cnts = list(config_detect['preprocessing'].values())

        gen = claims.process_data(data_orig, aggreg_level = 'all_combi')
        
        # Only execute if submit button has been clicked:
        if submitted:

            # Get next series
            #-------------------
            label, sub_set = next(gen)

            print('Claims from period {} to {}.'.format(claims.min_year_period, claims.max_year_period)) 

            print(label, sub_set.shape[0])
            df = deepcopy(sub_set)

            train = trainer.trainer(verbose=False)
            fitted = train.fit(df = df)

            y = fitted.ts_values
            #y = fitted.val_series
            out = fitted.predict(detect_thresh = None)

            where = np.where(np.array(claims.time_index) == outlier_filter)[0][0]
            outlier_search_list = claims.time_index[where:]

            filtered_outliers = []
            if out.nof_outliers > 0:
                outlier_dates = out.outlier_dates
                filt = [outl in outlier_search_list for outl in outlier_dates]
                filtered_outliers = np.array(outlier_dates)[filt].tolist()
                
                if len(filtered_outliers) > 0:
                    #print("\nSeries",i)
                    #print(label, sub_set.shape[0])
                    print("Anomaly found!")
                    print(filtered_outliers)
                
            #lag = 1
            #y_diff = util.difference(y, lag)
            # First diff.
            #util.ts_plot(x=x[lag:], y=y_diff, title='Weekly claim counts (First diff.): '+label) 

            # Detect anomalies:
            #----------------------
            inside = ''    
            if label in list(claims.level_wise_aggr.keys()):

                inside = claims.level_wise_aggr[label]       # then shows over which set it was aggregated    
                #new_inside = [str(i)+'\n' for i in inside] 
                
                #main = label +':\n\n '+ str(len(filtered_outliers)) + \
                #    ' outlier(s) detected!\n' + 'Occured at year-calendar week(s): '+ \
                #    ', '.join(filtered_outliers)+'\n'+'Aggregated over:'+str(new_inside)+'\n'
                
                main = label +':\n\n '+ str(len(filtered_outliers)) + \
                        ' outlier(s) detected!\n' + 'Occured at year-period(s): '+ \
                        ', '.join(filtered_outliers)+'\n'+'\nAggregated over: '
                for i in inside: main += str(i)+'\n'
                
            else:
                main = label +':\n\n '+ str(len(filtered_outliers)) + \
                    ' outlier(s) detected!\n' + 'Occured at year-period(s): '+ \
                    ', '.join(filtered_outliers)+'\n'

                
            pp = plot(fitted.val_series, anomaly_true = fitted.anomalies, ts_linewidth=1.2, ts_markersize=6, 
                at_markersize=5, at_color='red', freq_as_period=False, ts_alpha=0.8, at_alpha=0.5, title = main)


            # Anomaly probabilities:
            #-------------------------
            plt.figure(figsize=(12,4), dpi=100)
            pro = plt.plot(fitted.anomaly_proba.index, fitted.anomaly_proba, color='tab:blue',label="prob. of anomaly", linestyle='--', marker='o', markerfacecolor='orange', linewidth=1)
            plt.plot(fitted.anomaly_proba.index, [fitted.detect_thresh]*len(fitted.anomaly_proba.index), label="decision threshold",  linewidth=.5)
            plt.gca().set(title="", xlabel="time", ylabel="probability", ylim = plt.ylim(-0.02, 1.05))   #plt.xlim(left=0)
            locs, labels = plt.xticks()
            plt.title(r'Anomaly probabilities $\pi_{t}, t=1,...,T$', fontdict = {'fontsize' : 14})
            plt.legend(loc='upper left')
            plt.tight_layout()
            plt.show()  

            st.success('Reports created!')
            st.balloons()


            with plot_space:
                st.markdown("### Show plots")

                arr = np.random.normal(1, 1, size=10)
                fig, ax = plt.subplots()
                ax.hist(arr, bins=20)

                st.pyplot(fig)   # make above output fig

        ################### ANOMALY DETECTION ####################################  

        # if not selected_usecase:
        #     st.error('Select use case!')
        # else:
        #     st.info('Report is being created, please wait...')

            #Create Reports
            #report_lookup = create_reports(df, uploaded_file.name, map(str.lower, selected_usecase))

            # st.success('Reports successfully created!')
            # st.info('Reports are ready for download. Please refresh your browser page to see all available reports.')
            # st.balloons()
 
###########
# Run app:
###########
main()

