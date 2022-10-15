import streamlit as st
import pandas as pd
import numpy as np
import base64
import glob as gl
#from zipfile import ZipFile
#from os.path import basename
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
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, month_plot, quarter_plot
from copy import deepcopy
from importlib import reload
import adtk
from adtk.visualization import plot
import statsmodels.api as sm


# Set Page name and icon, Layout and sidebar expanded
img = Image.open(os.path.join(glob.UC_CODE_DIR,'templates','allianz_logo.jpg'))
st.set_page_config(page_title='Anomaly Report Creator', page_icon=img, layout="wide", initial_sidebar_state='expanded')


periodicity = 52
anomaly_history = pd.DataFrame(columns=['time_anomaly', 'time_series_name', 'clm_cnt'])
config_detect = config.in_out52['detection']
outlier_filter = config_detect['training']['outlier_filter']

age = 6
if outlier_filter is None:
        six_months_ago = date.today() - relativedelta(months=age)
        outlier_filter = six_months_ago.strftime("%Y-%m")


# Use pickle file as substitute for Postgres:
# pkl = file.PickleService(path = "anomaly_history.pkl")

# pkl.doWrite(anomaly_history)

# pkl.doRead()


######################################################################################
################################# Start App   ########################################
######################################################################################

def main():

    header = st.container()
    plot_space = st.container()
    
    with header:
        tabs = st.tabs(["Data", "Anomalies", "Seasonality"])
        tab_data = tabs[0]
        tab_plots = tabs[1]
        tab_plots_season = tabs[2]

        # Title of app
        #st.title("Anomaly Report Creator")
        #st.markdown("***")

    with st.sidebar:
        st.image(os.path.join(glob.UC_CODE_DIR,'templates','agcs_banner.png'), use_column_width=True)
        # set up file upload:
        uploaded_file = st.file_uploader("Upload file:", type = ["csv"])    # returns byte object
        #print(uploaded_file)

        # select box for usecase
        #usecases = ['All', 'Region', 'LoB']
        #selected_usecase =  st.multiselect("Use case:", usecases)

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

    if uploaded_file is not None:
        try:
            #data_orig = pd.read_excel(uploaded_file, sheet_name="data")
            data_orig = pd.read_csv(uploaded_file, delimiter=',')
            print(data_orig.shape)

            target_col='target'

            data_orig.rename(columns={'lob': 'Lob', 'erartbez': 'Event_descr', 'time_index': 'time', 'clm_cnt' : target_col}, inplace=True) 
            data_orig = data_orig[['time', 'Lob', 'Event_descr', target_col]]

            #st.markdown("### Report info")
            with tab_data:
                dataset = st.expander(label = "Display imported data")

                #st.dataframe(data_orig.head(100))
                with dataset:            
                    df0 = data_orig.rename(columns={'time': 'Time', target_col : 'Target'}, inplace=False)  # only for nicer displays
                    st.table(df0.head(100))
                    st.success(uploaded_file.name + ' successfully uploaded!')

            # Instantiate class:
            #--------------------
            #claims = preprocessor.claims_reporting(periodicity=periodicity)
            #gen = claims.process_data(data_orig, aggreg_level = 'all_combi')

            #get_all = dict(gen)
            #tseries_names = list(get_all.keys())
            #tseries_values = list(get_all.values())

            def widget_callback():
                """Callback function to retrieve API states from a running streamlit server"""
                st.session_state.indexer += 1
                #st.session_state.label = tseries_names[st.session_state.indexer]
                #st.session_state.sub_set = tseries_values[st.session_state.indexer]

        except Exception as ex:
            st.error("Invalid File")
    
        with st.sidebar:  
                
                if 'indexer' not in st.session_state:
                	st.session_state.indexer = 0 
                 
                # if 'label' not in st.session_state:
                # 	st.session_state.label = tseries_names[0]

                # if 'sub_set' not in st.session_state:
                # 	st.session_state.sub_set = tseries_values[0]
              
                # st.button('TEST', key='my_button', on_click = widget_callback)
                # #st.button('TEST', key='my_button', on_click=widget_callback, kwargs=dict(my_iter=my_test))

        # Only create button, if valid file is uploaded
        with st.sidebar:

            #submitted = st.button('Run analysis', key='my_button', on_click = widget_callback)    # boolean 
            submitted = st.button('Run analysis', key='train')    # no callback needed here
            
        if submitted:
            train0 = trainer.trainer(verbose=False)

            results_all, results_new = train0.run_all(data_orig = data_orig, verbose=False)   # write_table = False

            results_final = deepcopy(results_new)      # only show new outliers excluding ones shown before
            #results_final = deepcopy(results_all)      # show all detected outliers potentially including ones shown before

            results = deepcopy(results_final)
            results.rename(columns={'time_anomaly': 'Time', 'time_series_name': 'Time series', 'target': 'Claim counts'}, inplace=True)
            results.reset_index(inplace=True, drop=True)

            all_series = train0.all_series
            new_anomalies = list(set(results_final['time_series_name']))
            
            try:
                where = np.where(np.array(train0.time_index) == outlier_filter)[0][0]
                outlier_search_list = train0.time_index[where:]
            except Exception as ex:
                outlier_search_list = []

            with tab_data:
                st.success("Training done!")

            #st.info(f"Training done!")
            st.balloons()

            print(tuple(train0.filt_suspects_plot.keys()))

            with st.sidebar: 
                # Add callback dependency!!!
                option = st.selectbox('Select anomaly:', tuple(train0.filt_suspects_plot.keys()))

                st.write('You selected:', option)

            ################### ANOMALY DETECTION ####################################    
            
            # # Get next series
            # #-------------------
            # if submitted:
                
            #     st.write(f'Series = {st.session_state.indexer} (of {len(tseries_names)})')

            #     #label, sub_set = next(gen)
            #     label, sub_set = st.session_state.label, st.session_state.sub_set
            #     print('Claims from period {} to {}.'.format(claims.min_year_period, claims.max_year_period)) 

            #     print(label, sub_set.shape[0])
            #     df = deepcopy(sub_set)
                
            #     # Next do it like in the report
            #     # Run all and then just index over results -> faster!!!
            #     ##### !!!!!!!!!!!!!!!!!!!!!


            #     train = trainer.trainer(verbose=False)
            #     fitted = train.fit(df = df)

            #     y = fitted.ts_values
            #     #y = fitted.val_series
            #     out = fitted.predict(detect_thresh = None)

            #     where = np.where(np.array(claims.time_index) == outlier_filter)[0][0]
            #     outlier_search_list = claims.time_index[where:]

            #     filtered_outliers = []
            #     if out.nof_outliers > 0:
            #         outlier_dates = out.outlier_dates
            #         filt = [outl in outlier_search_list for outl in outlier_dates]
            #         filtered_outliers = np.array(outlier_dates)[filt].tolist()
                    
            #         if len(filtered_outliers) > 0:
            #             #print("\nSeries",i)
            #             #print(label, sub_set.shape[0])
            #             print("Anomaly found!")
            #             print(filtered_outliers)

            #     # Detect anomalies:
            #     #----------------------
            #     inside = ''    
            #     if label in list(claims.level_wise_aggr.keys()):

            #         inside = claims.level_wise_aggr[label]       # then shows over which set it was aggregated    
                    
            #         main = label +':\n\n '+ str(len(filtered_outliers)) + \
            #                 ' outlier(s) detected!\n' + 'Occured at year-period(s): '+ \
            #                 ', '.join(filtered_outliers)+'\n'+'\nAggregated over: '
            #         for i in inside: main += str(i)+'\n'
                    
            #     else:
            #         main = label +':\n\n '+ str(len(filtered_outliers)) + \
            #             ' outlier(s) detected!\n' + 'Occured at year-period(s): '+ \
            #             ', '.join(filtered_outliers)+'\n'

            #     pp = plot(fitted.val_series, anomaly_true = fitted.anomalies, ts_linewidth=1.2, ts_markersize=6, 
            #         at_markersize=5, at_color='red', freq_as_period=False, ts_alpha=0.8, at_alpha=0.5, title = main)

            #     #st.success('Reports created!')
            #     with tab_data: 
            #         st.info(f"Series: {label}  (T: {sub_set.shape[0]})")
            #         dataset_sub = st.expander(label = "Display training data")

            #         with dataset_sub:            
            #             df1 = df.rename(columns={'month': 'Month', 'time': 'Time', target_col : 'Target'}, inplace=False) 
            #             st.table(df1[['Time', 'Month','Target']])

            #     #st.balloons()

            #     with tab_plots:
            #             st.info(f"Label: {st.session_state.label}")
            #             st.pyplot(pp.figure)   # make above output fig

            #     with tab_plots_season:
            #         st.info(f"Label: {st.session_state.label}")

            #         # Draw Boxplot
            #         fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,6), dpi= 80)
    
            #         sns.boxplot(x='year', y='target', data=df, ax=axes[0])
            #         sns.boxplot(x='month', y='target', data=df, ax=axes[1]).set(ylabel="counts")
            #         if periodicity == 52 :
            #             sns.boxplot(x='period', y='target', data=df, ax=axes[2], orient='v').set(
            #             xlabel='week', ylabel="counts")
            #         #------------------------------------------------------------------------------------------
            #         # Set Titles
            #         axes[0].set_title('Yearly box plots\n(Trend)', fontsize=18) 
            #         axes[1].set_title('Monthly box plots\n(Seasonality)', fontsize=18)
            #         if periodicity == 52 :
            #             axes[2].set_title('Weekly box plots\n(Seasonality)', fontsize=18)
            #         #plt.yticks(rotation=15)
            #         plt.xticks(rotation=45)
            #         #plt.show()
            #         st.pyplot(fig)

 
###########
# Run app:
###########
main()

