"""
Services for reading and writing from and to AWS S3 of various file formats
"""
import pandas as pd
from aac_ts_anomaly.config import global_config as glob
from aac_ts_anomaly.config.aws_config import bucket_obj as bucket 
from aac_ts_anomaly.resources.blueprints import AbstractServices
from imp import reload
import os, yaml, toml, json
from typing import (Dict, List, Text, Optional, Any, Callable, Union)
from botocore.exceptions import ClientError

class CSVService(AbstractServices):

    def __init__(self, path : Optional[str] = "", delimiter : str = "\t", encoding : str = "UTF-8", schema_map : Optional[dict] = None, 
                 root_path : str = glob.UC_DATA_DIR, verbose : bool = False):
        """Read/write service instance for CSV files
        Args:
            path (str, optional): Filename. Defaults to "".
            delimiter (str, optional): see pd.read_csv. Defaults to "\t".
            encoding (str, optional): see pd.read_csv. Defaults to "UTF-8".
            schema_map (Optional[dict], optional): mapping scheme for renaming of columns, see pandas rename. Defaults to None.
            root_path (str, optional): root path where file is located. Defaults to glob.UC_DATA_DIR.
            verbose (bool, optional): should user information be displayed? Defaults to False.
        """
        super().__init__()
        self.path = os.path.join(root_path, path)
        self.delimiter = delimiter
        self.verbose = verbose
        self.encoding = encoding
        self.schema_map = schema_map
        self.s3obj = bucket.Object(self.path).get()
    
    def doRead(self, **kwargs)-> pd.DataFrame:
        """Read data from CSV
        Returns:
            pd.DataFrame: data converted to dataframe
        """
        try:
            df = pd.read_csv(self.s3obj['Body'], index_col=0, encoding=self.encoding, delimiter=self.delimiter, **kwargs)
            if self.verbose: print(f"CSV Service Read from File: {str(self.path)}")
            if self.schema_map:
                df.rename(columns=self.schema_map, inplace=True)
            return df
        except Exception as ex:
            print(ex)

    def doWrite(self, X : pd.DataFrame, **kwargs):
        """Write X to CSV file
        Args:
            X (pd.DataFrame): input data
        """
        X.to_csv(path_or_buf=self.path, encoding=self.encoding, sep=self.delimiter, **kwargs)
        if self.verbose: print(f"CSV Service Output to File: {str(self.path)}")

