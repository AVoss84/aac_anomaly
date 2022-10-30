import pandas as pd
from aac_ts_anomaly.config import global_config as glob
#import sqlalchemy as sql
from imp import reload
import os, yaml
from typing import (Dict, List, Text, Optional, Any, Callable)

class CSVService:
    def __init__(self, path : Optional[str] = "", delimiter : str = "\t", encoding : str = "UTF-8", schema_map : Optional[dict] = None, 
                 root_path : str = glob.UC_DATA_DIR, verbose : bool = False):
        """Generic read/write service for CSV files

        Args:
            path (str, optional): _description_. Defaults to "".
            delimiter (str, optional): _description_. Defaults to "\t".
            encoding (str, optional): _description_. Defaults to "UTF-8".
            schema_map (Optional[dict], optional): _description_. Defaults to None.
            root_path (str, optional): _description_. Defaults to glob.UC_DATA_DIR.
            verbose (bool, optional): _description_. Defaults to False.
        """
        self.path = os.path.join(root_path, path)
        self.delimiter = delimiter
        self.verbose=verbose
        self.encoding = encoding
        self.schema_map = schema_map

    def doRead(self, **kwargs) -> pd.DataFrame:
        """Read data from CSV

        Returns:
            pd.DataFrame: _description_
        """
        df = pd.read_csv(filepath_or_buffer=self.path, encoding=self.encoding, delimiter=self.delimiter, **kwargs)
        if self.verbose: print(f"CSV Service Read from File: {str(self.path)}")
        if self.schema_map:
            df.rename(columns=self.schema_map, inplace=True)
        return df

    def doWrite(self, X : pd.DataFrame, **kwargs):
        """Write X to CSV file

        Args:
            X (pd.DataFrame): _description_
        """
        X.to_csv(path_or_buf=self.path, encoding=self.encoding, sep=self.delimiter, **kwargs)
        if self.verbose: print(f"CSV Service Output to File: {str(self.path)}")


class XLSXService:
    def __init__(self, path : Optional[str] = "", sheetname : str = "Sheet1", root_path : str = glob.UC_DATA_DIR, schema_map : Optional[dict] = None, verbose : bool = False):
        """Generic read/write service for XLS files

        Args:
            path (str, optional): _description_. Defaults to "".
            sheetname (str, optional): _description_. Defaults to "Sheet1".
            root_path (str, optional): _description_. Defaults to glob.UC_DATA_DIR.
            schema_map (Optional[dict], optional): _description_. Defaults to None.
            verbose (bool, optional): _description_. Defaults to False.
        """
        self.path = os.path.join(root_path, path)
        self.writer = pd.ExcelWriter(self.path)
        self.sheetname = sheetname
        self.verbose=verbose
        self.schema_map = schema_map
        
    def doRead(self, **kwargs) -> pd.DataFrame:
        """Read from XLS

        Returns:
            pd.DataFrame: _description_
        """
        df = pd.read_excel(self.path, self.sheetname, **kwargs)
        if self.verbose: print(f"XLS Service Read from File: {str(self.path)}")
        if self.schema_map:
            df.rename(columns=self.schema_map, inplace=True)
        return df    
        
    def doWrite(self, X : pd.DataFrame, **kwargs):
        """Write to XLS

        Args:
            X (pd.DataFrame): _description_
        """
        X.to_excel(self.writer, self.sheetname, **kwargs)
        self.writer.save()
        if self.verbose: print(f"XLSX Service Output to File: {str(self.path)}")


class PickleService:
    def __init__(self, path : Optional[str] = "", root_path : str = glob.UC_DATA_DIR, schema_map : Optional[dict] = None, verbose : bool = False):
        """Generic read/write service for Pkl files

        Args:
            path (str, optional): _description_. Defaults to "".
            root_path (str, optional): _description_. Defaults to glob.UC_DATA_DIR.
            schema_map (Optional[dict], optional): _description_. Defaults to None.
            verbose (bool, optional): _description_. Defaults to False.
        """
        self.path = os.path.join(root_path, path)
        self.schema_map = schema_map
        self.verbose=verbose

    def doRead(self, **kwargs)-> pd.DataFrame:
        """Read pkl files

        Returns:
            pd.DataFrame: _description_
        """
        df = pd.read_pickle(self.path, **kwargs)
        if self.verbose : print(f"Pickle Service Read from file: {str(self.path)}")
        if self.schema_map: df.rename(columns = self.schema_map, inplace = True)
        return df

    def doWrite(self, X: pd.DataFrame, **kwargs)-> bool:
        """Write to pkl file

        Args:
            X (pd.DataFrame): _description_

        Returns:
            bool: _description_
        """
        try:
            X.to_pickle(path = self.path, compression = None)    
            if self.verbose : print(f"Pickle Service Output to File: {str(self.path)}")
            return True
        except Exception as e0:
            print(e0); return False        


class YAMLservice:
        def __init__(self, path : Optional[str] = "", root_path : str = glob.UC_CODE_DIR, 
                     verbose : bool = False):
            """Generic read/write service for YAML files

            Args:
                path (str, optional): _description_. Defaults to "".
                root_path (str, optional): _description_. Defaults to glob.UC_CODE_DIR.
                verbose (bool, optional): _description_. Defaults to False.
            """
            self.root_path = root_path
            self.path = path
            self.verbose = verbose 
        
        def doRead(self, filename : str = None, **kwargs)-> Any:  
            """Read from yaml file

            Args:
                filename (str, optional): _description_. Defaults to None.

            Returns:
                Any: _description_
            """
            with open(os.path.join(self.root_path, self.path, filename), 'r') as stream:
                try:
                    my_yaml_load = yaml.load(stream, Loader=yaml.FullLoader, **kwargs)   
                    if self.verbose: print(f"Read: {self.root_path+self.path+filename}")
                except yaml.YAMLError as exc:
                    print(exc) 
            return my_yaml_load
        
        def doWrite(self, X: pd.DataFrame, filename : str = None, **kwargs)-> bool:
            """Write dictionary X to YAMl file

            Args:
                X (pd.DataFrame): _description_
                filename (str, optional): _description_. Defaults to None.

            Returns:
                bool: _description_
            """
            with open(os.path.join(self.root_path, self.path, filename), 'w') as outfile:
                try:
                    yaml.dump(X, outfile, default_flow_style=False)
                    if self.verbose: print(f"Write to: {self.root_path+self.path+filename}")
                    return True
                except yaml.YAMLError as exc:
                    print(exc); return False


class TXTService:
    def __init__(self, path : Optional[str] = "", encoding="utf-8", root_path : str = glob.UC_CODE_DIR, verbose : bool = True):
        """Generic read/write service for TXT files

        Args:
            path (Optional[str], optional): _description_. Defaults to "".
            encoding (str, optional): _description_. Defaults to "utf-8".
            root_path (str, optional): _description_. Defaults to glob.UC_CODE_DIR.
            verbose (bool, optional): _description_. Defaults to True.
        """
        self.myfile = path        # filename
        self.myfolder = root_path          # folder/subfolder
        self.path = os.path.join(root_path, path)
        self.encoding = encoding
        self.verbose = verbose

    def doRead(self, **kwargs) -> list:
        """Read TXT files

        Returns:
            list: _description_
        """
        try:
            download = self.file_client.download_file()
            download_bytes = download.readall()
            assert self.file_client.get_file_properties().size > 0, 'File has size zero.'
            df = download_bytes.decode(self.encoding).splitlines() 
            if self.verbose : print(f"TXT Service read from file: {str(self.path)}")    
        except Exception as e0:
            print(e0); df = None
        finally: 
            return df
        
    def doWrite(self, X : list)-> bool:
        """Write TXT files

        Args:
            X (list): _description_

        Returns:
            bool: _description_
        """
        try:
            data = '\n'.join(X).encode(self.encoding)
            self.file_client.create_file()
            self.file_client.append_data(data, offset=0, length=len(data))
            self.file_client.flush_data(len(data))
            if self.verbose : print(f"TXT Service output to file: {str(self.path)}")  
            return True
        except Exception as e0:
            print(e0); return False
