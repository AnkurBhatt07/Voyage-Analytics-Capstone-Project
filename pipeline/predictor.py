"""
The Script will load trained model , call preprocessing.py and transform the raw user input , return predicted price using the model
"""

import pickle 
from pipeline.preprocessing import FlightPreprocessor 


class FlightPricePredictor:
    def __init__(self):
        self.model = pickle.load(
            open("artifacts/grad_boost_best.pkl",'rb')
        )

        self.preprocessor = FlightPreprocessor()

    def predict(self , input_dict):
        X = self.preprocessor.transform(input_dict)
        prediction = self.model.predict(X)[0]
        return float(prediction)
    
