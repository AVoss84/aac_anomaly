import streamlit as st
import pandas as pd
import numpy as np
import base64
import glob as gl
import os
from zipfile import ZipFile
from os.path import basename
from PIL import Image
from aac_ts_anomaly.resources import config
from aac_ts_anomaly.config import global_config as glob
from aac_ts_anomaly.resources import report_creator
from aac_ts_anomaly.services import file

# Set Page name and icon, Layout and sidebar expanded
img = Image.open(os.path.join(glob.UC_CODE_DIR,'templates','allianz_logo.jpg'))
st.set_page_config(page_title='Anomaly Report Creator', page_icon=img, layout="wide", initial_sidebar_state='expanded')


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
        #st.image(os.path.join(glob.UC_CODE_DIR,'templates','agcs_banner.png'), use_column_width=True)
        # set up file upload:
        uploaded_file = st.file_uploader("Upload file:", type = ["csv"])    # returns byte object
        print(uploaded_file)

        # select box for usecase
        usecases = ['All', 'Region', 'LoB']
        selected_usecase =  st.multiselect("Use case:", usecases)

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

    df = None
    report_lookup = {}

    if uploaded_file is not None:
        try:
            #df = pd.read_excel(uploaded_file, sheet_name="data")
            df = pd.read_csv(uploaded_file, delimiter=',')
            print(df.shape)
            #st.markdown("### Report info")
            dataset = st.expander(label = "Display imported data")
            with dataset:
                #st.dataframe(df.head(100))
                st.table(df.head(100))
            st.success(uploaded_file.name + ' successfully uploaded!')
        except Exception as ex:
            st.error("Invalid File")
    
        # Only create button, if valid file is uploaded
        with st.sidebar:
           submitted = st.button('Run analysis')    # boolean 
        
        # Only execute if submit button has been clicked:
        if submitted:

            if not selected_usecase:
                st.error('Select use case!')
            else:
                st.info('Report is being created, please wait...')

                #Create Reports
                #report_lookup = create_reports(df, uploaded_file.name, map(str.lower, selected_usecase))

                st.success('Reports successfully created!')
                st.info('Reports are ready for download. Please refresh your browser page to see all available reports.')
                st.balloons()
 
###########
# Run app:
###########
main()

