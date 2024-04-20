import os 
import sys 
import pickle 
import warnings
from sqlalchemy import create_engine
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.exception import CustomException
from sklearn.metrics import r2_score
from src.logger import logging


@dataclass
class ConnectDBConfig(): 
        host = 'localhost'
        user = 'root'
        password = 'Sagnik123#'
        database = 'nyc_taxi_data'
        table_name = 'nyc_taxi_trips_data'
        dataset_path:str = os.path.join('dataset', 'nyc_taxi_data.csv')

class ConnectDB():     
    def __init__(self):
         self.connect_db_config = ConnectDBConfig()  
      
    def retrieve_data(self):
        try:
            logging.info('Initiating Database Connection')
            engine = create_engine(f'mysql+mysqlconnector://{self.connect_db_config.user}:{self.connect_db_config.password}@{self.connect_db_config.host}/{self.connect_db_config.database}')
            query = f"SELECT * From {self.connect_db_config.table_name}"
            df = pd.read_sql(query, engine)
            os.makedirs(os.path.dirname(self.connect_db_config.dataset_path),exist_ok=True)
            df.to_csv(self.connect_db_config.dataset_path,index=False)
            logging.info('Copy of Dataset stored in dataset folder as a csv file')
        except Exception as e: 
            raise CustomException(e,sys)
        finally:
                engine.dispose()
                logging.info('Database connection closed')
        
def save_function(file_path, obj): 
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)
        with open (file_path, "wb") as file_obj: 
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)         

def model_performance(X_train, y_train, X_test, y_test, models): 
    try: 
        report = {}
        for i in range(len(models)): 
            model = list(models.values())[i]
# Train models
            model.fit(X_train, y_train)
# Test data
            y_test_pred = model.predict(X_test)
            #R2 Score 
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e: 
        raise CustomException(e,sys)

# Function to load a particular object 
def load_obj(file_path):
    try: 
        with open(file_path, 'rb') as file_obj: 
            return pickle.load(file_obj)
    except Exception as e: 
        logging.info("Error in load_object fuction in utils")
        raise CustomException(e,sys)

def calculate_haversine_distance(lat1, lng1, lat2, lng2):

    AVG_EARTH_RADIUS = 6371  # Average radius of the earth in km
    
    # Converting latitude and longitude from degrees to radians
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    # Computing the differences in coordinates
    dlat = lat2 - lat1
    dlng = lng2 - lng1

    # Haversine formula used to calculate the distance between the coordinates
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = AVG_EARTH_RADIUS * c

    return distance

def remove_outliers(df, columns, threshold):
   
    #Making a copy of the dataframe first
    df_cleaned = df.copy()
    
    for col in columns:
        # Calculating the quartiles
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        
        # Calculating the IQR
        IQR = Q3 - Q1
        upper_bound = Q3 + threshold * IQR
        

        lower_bound = 0         #Kept the lower bound as 0 because neither distance nor trip duration can be negative   
        upper_bound = Q3 + threshold * IQR
        

        df_cleaned = df_cleaned[(df_cleaned[col] > lower_bound) & (df_cleaned[col] <= upper_bound)]
    
    return df_cleaned

def data_transform(df): 
             try:
                  logging.info("Data transformation started")
                  df['distance'] = calculate_haversine_distance(df['pickup_latitude'],
                                           df['pickup_longitude'],
                                           df['dropoff_latitude'],
                                           df['dropoff_longitude'])

                  #Converting pickup_datetime and dropoff_datetime to datetime type 
                  df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime']) 
                  df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime']) 

                  #day(0 = Monday, 1 = Tuesday..)
                  df['day_of_the_week'] = df['pickup_datetime'].dt.day_of_week.astype(object)  

                  #month (1 = January...6 = June)
                  df['month'] = df['pickup_datetime'].dt.month.astype(object) 

                  #hour(0 = 12am, 1 = 1am ... 23 = 11pm)                 
                  df['hour'] = df['pickup_datetime'].dt.hour.astype(object)                          

                  # Calculating duration of trip in hours
                  df['calculated_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 3600
                  
                  #New feature speed, engineered from distance and calculated_duration
                  df['speed'] = df['distance'] / df['calculated_duration']                  
                  
                  logging.info("Dataframe transformation complete")
                  return df
             
             except Exception as e:
                  logging.info("Error occured in data_transform function in prediction_pipeline")
                  raise CustomException(e,sys)