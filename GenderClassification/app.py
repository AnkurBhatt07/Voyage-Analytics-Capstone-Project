from flask import Flask , request , jsonify 

import pickle 
import numpy as np 
import pandas as pd

app = Flask(__name__)


with open("artifacts1/preprocessor.pkl","rb") as f:
    preprocessor = pickle.load(f)

with open("artifacts1/best_model.pkl","rb") as f:
    model = pickle.load(f)


try :
    with open("artifacts1/pre_feature_names.pkl","rb") as f:
        feature_names = pickle.load(f)
except:
    feature_names = None 


@app.route('/health' , methods = ["GET"])
def health_check():
    return jsonify({"status":"ok"})


@app.route('/predict' , methods = ['POST'])
def predict():
    data = request.get_json()

    if data is None:
        return jsonify({"error":"No input data provided"}), 400
    
    try:
        if feature_names:
            X = pd.DataFrame([data])

        else:
            X = np.array([list(data.values())])

        X_p = preprocessor.transform(X)
        pred = model.predict(X_p)

        pred_label = pred[0]

        return jsonify({'predicted':str(pred_label)})
    

    except Exception as e:
        return jsonify({"error":str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug = True)