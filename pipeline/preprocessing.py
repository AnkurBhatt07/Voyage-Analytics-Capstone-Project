# Load preprocessing artifacts 
# Take raw user input (dict)
# Convert it into a final_matrix row

"""This script will do this preprocessing at inference time:

Raw columns
↓
from, to           → OneHotEncoder
flightType         → OrdinalEncoder → StandardScaler
time, distance     → PowerTransformer
agency              → OneHotEncoder
date                → year, month, dow, dom → StandardScaler
↓
Concatenate all into final_matrix

"""


import pickle 
import numpy as np 
import pandas as pd 

class FlightPreprocessor:
    def __init__(self , artifacts_path = "artifacts/"):
        # load all saved preprocessing objects

        self.from_to_encoder = pickle.load(
            open(f"{artifacts_path}/from_to_encoder.pkl" , 'rb')
        )

        self.flightType_encoder = pickle.load(open(f"{artifacts_path}/flightType_encoder.pkl" , 'rb'))

        self.flightType_scaler = pickle.load(open(f"{artifacts_path}/flightType_scaler.pkl" , 'rb'))

        self.time_dis_pt = pickle.load(open(f"{artifacts_path}/time_dis_pt.pkl" , 'rb'))

        self.agency_ohe = pickle.load(
            open(f"{artifacts_path}/agency_ohe.pkl", "rb")
        )

        self.date_scaler = pickle.load(
            open(f"{artifacts_path}/date_scaler.pkl", "rb")
        )
        self.date_cols = pickle.load(
            open(f"{artifacts_path}/date_cols.pkl", "rb")
        )

        self.city_pair_distance = pickle.load(
            open(f"{artifacts_path}/city_pair_distance.pkl", "rb")
        )

        self.city_pair_time = pickle.load(
            open(f"{artifacts_path}/city_pair_time.pkl", "rb")
        )

    def transform(self , input_dict:dict) -> np.ndarray:
        """
        input_dict : raw user input (from Flask / API)
        returns : numpy array of shape (1, n_features)
        """

        # convert input to Dataframe

        df = pd.DataFrame([input_dict])

        from_to_encoded = self.from_to_encoder.transform(df[['from','to']])

        flight_type_encoded = self.flightType_encoder.transform(df[['flightType']])

        flight_type_scaled = self.flightType_scaler.transform(flight_type_encoded)

        try:
            df['distance'] = self.city_pair_distance[(df['from'].values[0], df['to'].values[0])]
            df['time'] = self.city_pair_time[(df['from'].values[0], df['to'].values[0])]
        except KeyError:
            raise ValueError("City pair not found in training data.")
        


        time_distance_transformed = self.time_dis_pt.transform(df[['time','distance']])

        agency_encoded = self.agency_ohe.transform(df[['agency']])

        # Date preprocessing

        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day

        date_features_scaled = self.date_scaler.transform(df[self.date_cols])

        # Final Concatenation

        final_matrix = np.concatenate(
            [
                from_to_encoded,
                flight_type_scaled,
                time_distance_transformed,
                agency_encoded,
                date_features_scaled
            ],
            axis = 1
            )
        return final_matrix
    

    