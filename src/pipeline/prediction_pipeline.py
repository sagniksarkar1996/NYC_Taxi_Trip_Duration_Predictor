import sys 
import os 
from src.exception import CustomException 
from src.logger import logging 
from src.utils import load_obj, calculate_haversine_distance
import pandas as pd

class PredictPipeline: 
    def __init__(self) -> None:
        pass

    def predict(self, features): 
        try: 
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', "model.pkl")

            preprocessor = load_obj(preprocessor_path)
            model = load_obj(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e: 
            logging.info("Error occured in predict function in prediction_pipeline location")
            raise CustomException(e,sys)
        
class CustomData: 
        def __init__(self, vendor_id:int,
                     pickup_datetime:str, 
                     dropoff_datetime:str, 
                     passenger_count:int, 
                     pickup_longitude:float, 
                     pickup_latitude:float, 
                     dropoff_longitude:float, 
                     dropoff_latitude:float): 
             self.vendor_id = vendor_id
             self.pickup_datetime = pickup_datetime
             self.dropoff_datetime = dropoff_datetime
             self.passenger_count = passenger_count 
             self.pickup_longitude = pickup_longitude
             self.pickup_latitude = pickup_latitude
             self.dropoff_longitude = dropoff_longitude 
             self.dropoff_latitude = dropoff_latitude 

        def get_data_as_dataframe(self): 
             try: 
                  custom_data_input_dict = { 
                       'vendor_id': [self.vendor_id], 
                       'pickup_datetime': [self.pickup_datetime], 
                       'dropoff_datetime': [self.dropoff_datetime],
                       'passenger_count':[self.passenger_count],
                       'pickup_longitude':[self.pickup_longitude], 
                       'pickup_latitude': [self.pickup_latitude], 
                       'dropoff_longitude': [self.dropoff_longitude], 
                       'dropoff_latitude': [self.dropoff_latitude]
                  }
                  df = pd.DataFrame(custom_data_input_dict)
                  logging.info("Dataframe created")

                  return df
             
             except Exception as e:
                  logging.info("Error occured in get function in prediction_pipeline")
                  raise CustomException(e,sys)       