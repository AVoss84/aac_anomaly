import pandas as pd
from aac_ts_anomaly.config import global_config as glob
#import sqlalchemy as sql
from imp import reload
import os, yaml
from typing import (Dict, List, Text, Optional, Any, Callable)


class CSVService:
    def __init__(self, path : str = "", delimiter : str = "\t", encoding : str = "UTF-8", schema_map : Optional[dict] = None, 
                 root_path : str = glob.UC_DATA_DIR, verbose : bool = False,
                 **kwargs):
        # self.__dict__.update(kwargs)

        self.path = os.path.join(root_path, path)
        self.delimiter = delimiter
        self.verbose=verbose
        self.encoding = encoding
        self.schema_map = schema_map
        self.kwargs = kwargs

    def doRead(self, **kwargs) -> pd.DataFrame:
        df = pd.read_csv(filepath_or_buffer=self.path, encoding=self.encoding, delimiter=self.delimiter, **kwargs)
        if self.verbose:
            print("CSV Service Read from File: " + str(self.path))
        if self.schema_map:
            df.rename(columns=self.schema_map, inplace=True)
        return df

    def doWrite(self, X : pd.DataFrame):
        X.to_csv(path_or_buf=self.path, encoding=self.encoding, sep=self.delimiter, **self.kwargs)
        if self.verbose:
            print("CSV Service Output to File: " + str(self.path))


class XLSXService:
    def __init__(self, path : str = "", sheetname : str = "Sheet1", root_path : str = glob.UC_DATA_DIR, schema_map : Optional[dict] = None, verbose : bool = False, **kwargs):
        self.path = os.path.join(root_path, path)
        self.writer = pd.ExcelWriter(self.path)
        self.sheetname = sheetname
        self.verbose=verbose
        self.schema_map = schema_map
        self.kwargs = kwargs
        
    def doRead(self, **kwargs) -> pd.DataFrame:
        df = pd.read_excel(self.path, self.sheetname, **self.kwargs)
        if self.verbose: print("XLS Service Read from File: " + str(self.path))
        if self.schema_map:
            df.rename(columns=self.schema_map, inplace=True)
        return df    
        
    def doWrite(self, X, sheetname="Sheet1"):
        X.to_excel(self.writer, self.sheetname, **self.kwargs)
        self.writer.save()
        if self.verbose: 
            print("XLSX Service Output to File: " + str(self.path))


class PickleService:
    def __init__(self, path : str = "", root_path : str = glob.UC_DATA_DIR, schema_map : Optional[dict] = None, verbose : bool = False, **kwargs):
        self.path = os.path.join(root_path, path)
        self.schema_map = schema_map
        self.kwargs = kwargs
        self.verbose=verbose

    def doRead(self, **kwargs)-> pd.DataFrame:
        df = pd.read_pickle(self.path, **kwargs)
        if self.verbose : print("Pickle Service Read from File: " + str(self.path))
        if self.schema_map: df.rename(columns = self.schema_map, inplace = True)
        return df

    def doWrite(self, X: pd.DataFrame):
        X.to_pickle(path = self.path, compression = None)        # "gzip"
        if self.verbose : print("Pickle Service Output to File: "+ str(self.path))


# class PostgresService:
#     def __init__(self,
#                  connection_string : str = glob.UC_DB_CONNECTION,   # ToDo: save in .pgpass file!
#                 verbose : bool = False):
#         self.connection_string = connection_string
#         self.verbose = verbose
#         self.conn = self._create_connection(self.connection_string)
        
#     def __del__(self):
#         """
#         Destructor: destroy existing class instance
#         """    
#         try:
#             self.conn.dispose()
#             if self.verbose: print("connection disposed in destructor")
#         except Exception as e:
#             if self.verbose:
#                 print(e)        

#     def _create_connection(self, connection_string: str):
#         try:
#             engine = sql.create_engine(self.connection_string)
#             if self.verbose: print("Connection engine created")
#         except Exception as e:
#             engine = None
#             print(e) ; print("Connection could not be established")
#         return engine


#     def doRead(self, qry: str, **other):
#         """Send (any) query - except of insert - to PG server"""
#         self.qry = qry ;
#         if self.conn is None:
#             print("Error - No Connection available")
#             return False
#         else:
#             try:
#                self.selected_tables = pd.read_sql_query(self.qry, self.conn, **other)
#                if self.verbose: print("Query successful.")
#                return self.selected_tables
#             except IOError as e:
#                print(e) ; print("Query not successful!!")
#                return False

#     def doWrite(self, X : pd.DataFrame, output_tbl : str = 'my_table', **others):
#         """Write to PG server"""
#         self.output_tbl = output_tbl
#         dat = X.copy()
#         try:
#             dat.to_sql(self.output_tbl, con=self.conn, index=False, **others)
#             if self.verbose: print("Table",self.output_tbl,"successfully written to database.")
#         except IOError as e:
#             print(e) ; print("Could not write to database!")   
                    

class YAMLservice:
        def __init__(self, path : str = "", 
                     root_path : str = glob.UC_CODE_DIR, 
                     verbose : bool = False, **kwargs):
            
            self.root_path = root_path
            self.path = path
            self.verbose = verbose 
        
        def doRead(self, filename : str = None, **kwargs):  
            """Read in YAMl file from specified path"""
            with open(os.path.join(self.root_path, self.path, filename) , 'r') as stream:
                try:
                    my_yaml_load = yaml.load(stream, Loader=yaml.FullLoader, **kwargs)   
                    if self.verbose:
                        print("Read: "+self.root_path+self.path+filename)
                except yaml.YAMLError as exc:
                    print(exc) 
            return my_yaml_load
        
        def doWrite(self, X: pd.DataFrame, filename : str = None):
            """Write dictionary X to YAMl file"""
            with open(os.path.join(self.root_path, self.path, filename), 'w') as outfile:
                try:
                    yaml.dump(X, outfile, default_flow_style=False)
                    if self.verbose:
                        print("Write to: "+self.root_path+self.path+filename)
                except yaml.YAMLError as exc:
                    print(exc); return False 


