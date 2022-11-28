import streamlit as st
import pandas as pd
import numpy as np
import base64
import glob as gl
from PIL import Image
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import os, warnings, adtk
warnings.filterwarnings("ignore")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import seaborn as sns
from copy import deepcopy
from importlib import reload
from adtk.visualization import plot
import statsmodels.api as sm

from aac_ts_anomaly.utils import tsa_utils as tsa
from aac_ts_anomaly.utils import utils_func as util
from aac_ts_anomaly.config import global_config as glob
from aac_ts_anomaly.services import file
from aac_ts_anomaly.resources import (config, preprocessor, trainer)

# Set Page name and icon, Layout and sidebar expanded
#--------------------------------------------------------
img = Image.open(os.path.join(glob.UC_CODE_DIR,'templates','allianz_logo.jpg'))
st.set_page_config(page_title='Anomaly Report Creator', page_icon=img, layout="wide", initial_sidebar_state='expanded')
#----------------------------------------------------------------------------------------------------------------------
periodicity = 52
anomaly_history = pd.DataFrame(columns=['time_anomaly', 'time_series_name', 'clm_cnt'])
config_detect = config.in_out52['detection']
outlier_filter = config_detect['training']['outlier_filter']
detect_thresh = config_detect['prediction']['detect_thresh']
#--------------------------------------------------------------------------------------------------------
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

    with st.sidebar:
        st.image(os.path.join(glob.UC_CODE_DIR,'templates','agcs_banner.png'), use_column_width=True)
        # set up file upload:
        uploaded_file = st.file_uploader("Upload file:", type = ["csv"])    # returns byte object


    data_orig, fig_anom, df, fig_anom_prob = None, None, None, None
    if uploaded_file is not None:
        try:
            #data_orig = pd.read_excel(uploaded_file, sheet_name="data")
            data_orig = pd.read_csv(uploaded_file, delimiter=',')

            target_col='target'
            data_orig.rename(columns={'lob': 'Lob', 'erartbez': 'Event_descr', 'time_index': 'time', 'clm_cnt' : target_col}, inplace=True) 
            data_orig = data_orig[['time', 'Lob', 'Event_descr', target_col]]

            with tab_data:
                dataset = st.expander(label = "Display full dataset")
                #st.dataframe(data_orig.head(100))
                with dataset:            
                    df0 = data_orig.rename(columns={'time': 'Time', target_col : 'Target'}, inplace=False)  # only for nicer displays
                    st.table(df0.head(100))
                    st.success(uploaded_file.name + ' successfully uploaded!')

            # def widget_callback():
            #     """Callback function to retrieve API states from a running streamlit server"""
            #     st.session_state.indexer += 1
            #     #st.session_state.label = tseries_names[st.session_state.indexer]
            #     #st.session_state.sub_set = tseries_values[st.session_state.indexer]

        except Exception as ex:
            st.error("Invalid File")
    
        with st.sidebar:  
                # Initialize states:
                if 'indexer' not in st.session_state:
                	st.session_state.indexer = 0 

                if 'ts_labels' not in st.session_state: 
                    st.session_state.ts_labels = ()

                if 'label' not in st.session_state: 
                    st.session_state.label = None

                if 'val' not in st.session_state: 
                    st.session_state.val = pd.DataFrame()

                if 'level_wise_aggr' not in st.session_state: 
                    st.session_state.level_wise_aggr = {}
                    
                if 'filt_suspects_plot' not in st.session_state: 
                    st.session_state.filt_suspects_plot = {}

                if 'new_anomalies' not in st.session_state: 
                    st.session_state.new_anomalies = []

                if 'filt_suspects_values' not in st.session_state: 
                    st.session_state.filt_suspects_values = {}

                # if 'label' not in st.session_state:
                # 	st.session_state.label = tseries_names[0]

                # if 'sub_set' not in st.session_state:
                # 	st.session_state.sub_set = tseries_values[0]
              
                # st.button('TEST', key='my_button', on_click = widget_callback)
                # #st.button('TEST', key='my_button', on_click=widget_callback, kwargs=dict(my_iter=my_test))

        # Only create button, if valid file is uploaded
        with st.sidebar:
            #submitted = st.button('Run analysis', key='my_button', on_click = widget_callback)    # boolean 
            if st.button('Run analysis', key='train'):    # no callback needed here
                
                st.text(" ")
                with st.spinner('Wait for it...'):

                    ################### ANOMALY DETECTION #################################### 
                    train0 = trainer.trainer(verbose=False)

                    results_all, results_new = train0.run_all(data_orig = data_orig, verbose=False)   # write_table = False

                    results_final = deepcopy(results_new)      # only show new outliers excluding ones shown before
                    #results_final = deepcopy(results_all)      # show all detected outliers potentially including ones shown before
                    results = deepcopy(results_final)
                    results.rename(columns={'time_anomaly': 'Time', 'time_series_name': 'Time series', 'target': 'Claim counts'}, inplace=True)
                    results.reset_index(inplace=True, drop=True)

                    all_series = train0.all_series
                    new_anomalies = list(set(results_final['time_series_name']))
                    st.session_state.new_anomalies = new_anomalies

                    st.session_state.filt_suspects_values = train0.filt_suspects_values
                    st.session_state.filt_suspects_plot = train0.filt_suspects_plot 
                    st.session_state.ts_labels = tuple(train0.filt_suspects_plot.keys())
                    st.session_state.level_wise_aggr = train0.level_wise_aggr
                    
                    try:
                        where = np.where(np.array(train0.time_index) == outlier_filter)[0][0]
                        outlier_search_list = train0.time_index[where:]
                    except Exception as ex:
                        outlier_search_list = []

                    st.success("Training done!")
                    st.info(f"{len(all_series)} time series analyzed")
                    #st.balloons()
            
            #---------------------------------------------------------------------------------------
            #st.markdown("***")
            st.text(" ")

            label = st.selectbox('Select anomaly:', st.session_state.ts_labels)
            #st.write('You selected:', label)
            st.session_state.label = label
            
            if st.session_state.label is not None:
                st.session_state.val = st.session_state.filt_suspects_plot[label]

            with tab_data: 
                dataset_sub = st.expander(label = "Display selected data")

            if label in st.session_state.new_anomalies:
                    
                    fitted_val_series = st.session_state.val['val_series']
                    y = fitted_val_series
                    fitted_anomalies = st.session_state.val['anom_flag']
                    fitted_anomaly_proba = st.session_state.val['anom_evidence']           # anomaly probabilities

                    filtered_outliers = st.session_state.filt_suspects_values[label]['anomaly_dates']
                    sub_set = st.session_state.filt_suspects_plot[label]['df']
                    df = deepcopy(sub_set)
                    #-----------------------------------------------------------------------------
                    inside = ''    
                    if label in list(st.session_state.level_wise_aggr.keys()):
                        inside = st.session_state.level_wise_aggr[label]       # then shows over which set it was aggregated    

                        main = label +':\n\n '+ str(len(filtered_outliers)) + \
                                ' outlier(s) detected!\n' + 'Occured at year-period(s): '+ \
                                ', '.join(filtered_outliers)+'\n'+'\nAggregated over: '
                        for i in inside: main += str(i)+'\n'
                    else:
                        main = label +'\n\n '+ str(len(filtered_outliers)) + \
                            ' outlier(s) detected!\n' + 'Occured at year-period(s): '+ \
                            ', '.join(filtered_outliers)+'\n'
                    
                    # Plot time series with anomalies: 
                    #--------------------------------------------------------------------------------------
                    where = np.where(fitted_anomalies)[0] 
                    # Transformed
                    #fig_anom = util.ts_plot(fitted_val_series.index, fitted_val_series.values, vertical=fitted_anomalies[where].index.strftime("%Y-%m-%d").tolist(), title=main, xlabel='')
                    # Original series:
                    fig_anom = util.ts_plot(df['year_period_ts'].values, df['target'].values, vertical=fitted_anomalies[where].index.strftime("%Y-%m-%d").tolist(), title=main, dpi=100)
                    
                    # Plot anomaly probabilities:
                    #-----------------------------
                    fig_anom_prob = util.anomaly_prob_plot(x = fitted_anomaly_proba.index, y = fitted_anomaly_proba, detect_thresh = detect_thresh, dpi=100)
                    #-------------------------------------------------------------------------------------

                    with tab_data: 
                        #st.info(f"Selected series (T = {sub_set.shape[0]}): {label}")
                        dataset_sub = st.expander(label = "Display selected data")

                        with dataset_sub:            
                            df1 = df.rename(columns={'month': 'Month', 'time': 'Time', target_col : 'Target'}, inplace=False) 
                            st.table(df1[['Time', 'Month','Target']])

            #---------------------------------------------------------------------------------------------------
            with tab_plots:
                    #st.info(f"Series: {st.session_state.label}")
                    #if pp is not None: st.pyplot(pp.figure)   # make above output fig
                    if fig_anom is not None: st.pyplot(fig_anom)  
                    st.text(" ")
                    if fig_anom_prob: st.pyplot(fig_anom_prob)  

            with tab_plots_season:
                st.info(f"Series: {st.session_state.label}")

                # Draw Boxplot
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18,6))   # , dpi= 60
                if df is not None:
                    sns.boxplot(x='year', y='target', data=df, ax=axes[0])
                    sns.boxplot(x='month', y='target', data=df, ax=axes[1]).set(ylabel="")
                    if periodicity == 52 :
                        sns.boxplot(x='period', y='target', data=df, ax=axes[2], orient='v').set(
                        xlabel='week', ylabel="")
                #------------------------------------------------------------------------------------------
                # Set Titles
                fontsize=12
                axes[0].set_title('Yearly box plots\n(Trend)', fontsize=fontsize) 
                axes[1].set_title('Monthly box plots\n(Seasonality)', fontsize=fontsize)
                if periodicity == 52 : axes[2].set_title('Weekly box plots\n(Seasonality)', fontsize=fontsize)
                #plt.yticks(rotation=15)
                plt.xticks(rotation=45)
                #plt.show()
                st.pyplot(fig)   

###########
# Run app:
###########
main()

# Next ToDos: deploy to AWS
