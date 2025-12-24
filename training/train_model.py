import os 
import pickle 

import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

import mlflow 
import mlflow.sklearn 

artifacts_dir = "artifacts"
data_path = "data/flights.csv"
model_path = os.path.join(artifacts_dir, "grad_boost_best_airflow.pkl")


# best params taken from the mlflow logs for gradient boosting model
best_params = { "n_estimators" : 200 ,
               "max_depth" : 8,
               "learning_rate" : 0.1 ,
               "random_state" : 42
               }


def load_data(path : str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def preprocess_data(df : pd.DataFrame , artifacts_path=artifacts_dir):
    """Training-time preprocessing using saved artifacts."""
    # Load artifacts (same as inference time )

    from_to_encoder = pickle.load(open(os.path.join(artifacts_path, "from_to_encoder.pkl"), "rb"))
    flightType_encoder = pickle.load(open(os.path.join(artifacts_path, "flightType_encoder.pkl"), "rb"))
    flightType_scaler = pickle.load(open(os.path.join(artifacts_path, "flightType_scaler.pkl"), "rb"))
    time_dis_pt = pickle.load(open(os.path.join(artifacts_path, "time_dis_pt.pkl"), "rb"))
    agency_ohe = pickle.load(open(os.path.join(artifacts_path, "agency_ohe.pkl"), "rb"))
    date_scaler = pickle.load(open(os.path.join(artifacts_path, "date_scaler.pkl"), "rb"))
    date_cols = pickle.load(open(os.path.join(artifacts_path, "date_cols.pkl"), "rb"))

    # categorical encoding 

    from_to_encoded = from_to_encoder.transform(df[['from','to']])
    flightType_encoded = flightType_encoder.transform(df[['flightType']])
    flightType_scaled = flightType_scaler.transform(flightType_encoded)

    agency_encoded = agency_ohe.transform(df[['agency']])


    # Numerical transformations

    time_distance_transformed = time_dis_pt.transform(df[['time' , 'distance']])

    # date features 

    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day

    date_features_scaled = date_scaler.transform(df[date_cols])

    # Combine all features
    final_matrix = np.concatenate(
        [
            from_to_encoded,
            flightType_scaled,
            time_distance_transformed,
            agency_encoded,
            date_features_scaled
        ],
        axis=1
    )

    target = df['price'].values

    return final_matrix , target 

def train_model(X_train , y_train , params = best_params ):
    model = GradientBoostingRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        random_state=params['random_state']
    )

    model.fit(X_train , y_train)

    return model


def evaluate_model(model, X_val , y_val):
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val , preds)
    rmse = mean_squared_error(y_val , preds , squared=False)
    r2 = r2_score(y_val , preds)
    return mae , rmse , r2


def save_model(model , path : str):
    with open(path , 'wb') as f:
        pickle.dump(model , f)


def train_and_save_model():
    """Entry point for DAG"""

    os.makedirs(artifacts_dir , exist_ok=True)
    df = load_data(data_path)
    X , y = preprocess_data(df , artifacts_path=artifacts_dir)
    X_train , X_val , y_train , y_val = train_test_split(X , y , test_size=0.2 , random_state=best_params['random_state'])


    mlflow.set_tracking_uri("file:/opt/airflow/mlruns")
    mlflow.set_experiment("flight_price_baseline")


    with mlflow.start_run(run_name = 'gradient_boost_airflow_retraining'):
        model = train_model(X_train , y_train)
        mae , rmse , r2 = evaluate_model(model , X_val , y_val)

        mlflow.log_params(best_params)
        mlflow.log_metric("mae" , mae)
        mlflow.log_metric("rmse" , rmse)
        mlflow.log_metric("r2" , r2)

        save_model(model , model_path)
        mlflow.sklearn.log_model(model , "gradient_boosting_model_airflow")

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }

if __name__ == "__main__":
    train_and_save_model()  








