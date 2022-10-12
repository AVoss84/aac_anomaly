import pandas as pd
from aac_ts_anomaly.config import global_config as glob 
import sqlalchemy as sql
from imp import reload
import os, yaml, abc

# Using abstract base class
# to define a class blueprint (API)
# with later implemented concrete methods 
# for reading and writing data:

class services(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def doRead(self, **others):
       return 

    def doWrite(self, X, **others):
       return 


class XLSXService(services):

    def __init__(self, path="", sheetname="Sheet1", 
                       root_path=glob.UC_DATA_DIR, 
                       schema_map=None, verbose=False, **kwargs):

        self.path = os.path.join(root_path, path)
        self.writer = pd.ExcelWriter(self.path)
        self.sheetname = sheetname
        self.verbose=verbose
        self.schema_map = schema_map
        self.kwargs = kwargs

    @property    
    def doRead(self, **others):
        df = pd.read_excel(self.path, self.sheetname, **self.kwargs)
        if self.verbose:
            print("XLS Service Read from File: " + str(self.path))
        if self.schema_map != None:
            df.rename(columns=self.schema_map, inplace=True)   
        return df    
           
    def doWrite(self, X, **others):
        X.to_excel(self.writer, self.sheetname, **self.kwargs)
        self.writer.save()
        if self.verbose:
            print("XLSX Service Output to File: " + str(self.path))



class PostgresService(services):

    def __init__(self,
                 connection_string = glob.UC_DB_CONNECTION,   # ToDo: save in .pgpass file!
                verbose = False):

        self.connection_string = connection_string
        self.verbose = verbose
        self.conn = self._create_connection(self.connection_string)
        
    def __del__(self):
        """
        Destructor: destroy existing class instance
        """    
        try:
            self.conn.dispose()
            if self.verbose:
                print("connection disposed in destructor")
        except Exception as e:
            if self.verbose:
                print(e)        

    def _create_connection(self, connection_string):
        try:
            engine = sql.create_engine(self.connection_string)
            if self.verbose:
                print("Connection engine created")
        except Exception as e:
            engine = None
            print(e)
            print("Connection could not be established")
        return engine


    def doRead(self, **others):
        """
        Send (any) query - except of insert - to PG server
        """
        self.others = others        
        self.others.setdefault('sql', 'select * from default_table')
        if self.conn is None:
            print("Error - No Connection available")
            return False
        else:
            try:
               self.selected_tables = pd.read_sql_query(con=self.conn, **self.others)
               if self.verbose:
                   print("Query successful.")
               return self.selected_tables
            except IOError as e:
               print(e)
               print("Query not successful!!")
               return None

    def doWrite(self, X, **others):
        """
        Write to Postgres server
        """
        self.others = others
        self.others.setdefault('name', 'my_pgtable_name')
        dat = X.copy()
        try:
            dat.to_sql(con=self.conn, index=False, **self.others)
            if self.verbose:
                print("Table",self.others['name'],"successfully written to database.")
        except IOError as e:
            print(e)
            print("Could not write to database!")   
                    

class YAMLservice(services):

        def __init__(self, source_system = "", path="", 
                     root_path = glob.UC_CODE_DIR, verbose = False):
            
            self.source_system = source_system
            self.root_path = root_path
            self.path = path
            self.verbose = verbose 
        
        def doRead(self, **others):  
            """
            Read in YAMl file from specified path
            """
            others.setdefault('filename', 'my_yml_name')
            self.filename = others['filename']
            with open(os.path.join(self.root_path, self.path, self.source_system +self.filename) , 'r') as stream:
                try:
                    my_yaml_load = yaml.load(stream, Loader=yaml.FullLoader)
                    if self.verbose:
                        print("Read: "+self.root_path+self.path+self.source_system+self.filename)
                except yaml.YAMLError as exc:
                    print(exc) 
            return my_yaml_load
        
        def doWrite(self, X, **others):
            """
            Write dictionary X to YAMl file
            """
            others.setdefault('filename', 'my_yml_name')
            self.filename = others['filename']
            with open(os.path.join(self.root_path, self.path, self.source_system +self.filename), 'w') as outfile:
                try:
                    yaml.dump(X, outfile, default_flow_style=False)
                    if self.verbose:
                        print("Write to: "+self.root_path+self.path+self.source_system+self.filename)
                except yaml.YAMLError as exc:
                    print(exc) 

        
        
     
        
        
        
        