from flask import Flask , request , render_template
from pipeline.predictor import FlightPricePredictor
import pickle 

app = Flask(__name__)

# Load predictor once when app starts
predictor = FlightPricePredictor()

# Loading UI related artifacts
from_cities = pickle.load(open("artifacts/from_cities_list.pkl" , 'rb'))
agency_list = pickle.load(open("artifacts/agency_list.pkl" , 'rb'))
flight_types = pickle.load(open("artifacts/flight_type_list.pkl" , 'rb'))
date_range = pickle.load(open("artifacts/date_range.pkl" , 'rb'))


@app.route('/' , methods = ['GET'])
def home():
    return render_template('index.html' , 
                           cities = from_cities,
                           agencies = agency_list,
                           flight_types = flight_types,
                           min_date = date_range['min_date'].date(),
                           max_date = date_range['max_date'].date()
                           )

@app.route('/predict' , methods = ['POST'])
def predict():
    # Collecting form data
    input_data = {
        "from": request.form['from'],
        'to' : request.form['to'],
        'flightType' : request.form['flightType'],
        'agency' : request.form['agency'],
        'date' : request.form['date']
    }

    # Getting prediction
    try:
        predicted_price = predictor.predict(input_data)
    except Exception as e:
        return render_template('predict.html',error = str(e))
    

    return render_template('predict.html',
                           prediction = round(predicted_price,2),
                           data = input_data)

if __name__ == "__main__":
    app.run(debug = True)