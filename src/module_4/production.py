from train import (
    get_relevant_orders_df,
    get_features,
    BASE_DIR,
    MODEL_FOLDER,
    COLS_TO_USE,
)
import os
import joblib
import logging
import pandas as pd
import datetime
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_FOLDER = os.path.join(BASE_DIR, "data", "predictions")
MODEL_NAME = "20241113-155202_rf.pkl"
THRESHOLD_PREDICTION = 0.05

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_model(model_path:str)->Pipeline: 
    """Load the model from the model_path"""
    logger.info(f"Model loaded from {model_path}")
    return joblib.load(model_path)

def save_predictions(predictions_df:pd.DataFrame)->None:
    """Save the predictions in a csv file. If the folder does not exist, it will be created"""
    os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)
    predictions_path = os.path.join(PREDICTIONS_FOLDER, f'predictions_rf_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.csv')
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f'Predictions saved in {predictions_path}')
    


def main():
    model_path = os.path.join(MODEL_FOLDER, MODEL_NAME)
    model = load_model(model_path)
    
    data = get_relevant_orders_df()
    data_to_predict = get_features(data, COLS_TO_USE) 
    
    predictions = model.predict_proba(data_to_predict)[:, 1]
    predictions = (predictions > THRESHOLD_PREDICTION).astype(int)
    predictions_df = data[["user_id","variant_id"]].copy()
    predictions_df["prediction"] = predictions
    
    save_predictions(predictions_df)
    return predictions_df
    

if __name__ == "__main__":
    main()
