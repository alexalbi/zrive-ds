import pandas as pd
import os
import joblib
from typing import Tuple
import logging
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline 
from sklearn.metrics import auc, precision_recall_curve

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "data", "feature_frame.csv")
MODEL_FOLDER = os.path.join(BASE_DIR, "data", "models")


COLS_TO_USE = ['user_order_seq',   
            'ordered_before',
            'abandoned_before',
            'active_snoozed',
            'set_as_regular',
            'normalised_price',
            'discount_pct',
            'global_popularity',
            'days_since_purchase_variant_id',
            'avg_days_to_buy_variant_id',
            'std_days_to_buy_variant_id',
            'days_since_purchase_product_type',
            'avg_days_to_buy_product_type',
            'std_days_to_buy_product_type']

TARGET_COL = "outcome"
TRAIN_SIZE = 0.7

PARAMS_MODEL = {'max_depth': 13, 'min_samples_leaf': 9, 
               'min_samples_split': 6, 'n_estimators': 149}

NUM_ORDERS = 5


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_data() -> pd.DataFrame:
    return pd.read_csv(FILE_PATH)


def get_relevant_orders_df() -> pd.DataFrame:
    """We get the orders that have more or equal than num_orders products bought"""
    data = load_data()
    bought_products = data.query("outcome == 1")
    orders_relevants = (
        bought_products.groupby("order_id")["variant_id"]
        .agg(lambda x: len(list(x)))
        .reset_index()
        .query(f"variant_id >= {NUM_ORDERS}")
    )
    return data.merge(orders_relevants[["order_id"]], on="order_id")

def get_features(data: pd.DataFrame,feature_cols:list[str]) -> pd.DataFrame:
    return data.loc[:, feature_cols]

def get_target(data: pd.DataFrame, target_col: str) -> pd.Series:
    return data.loc[:, target_col]

def split_feature_target(
    data: pd.DataFrame, feature_cols: list[str], target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    return get_features(data,feature_cols), get_target(data,target_col)


def train_test_split_data(
    data: pd.DataFrame, train_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """We split the data into feature and tar atrain and test"""
    X, y = split_feature_target(data, COLS_TO_USE, TARGET_COL)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE)
    return X_train, X_test, y_train, y_test


def model_evaluation_pr_auc(model:Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    y_pred = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)
    return pr_auc


def save_model(model:Pipeline, model_path: str) -> None:
    """We save the model to the model_path. If the folder does not exist, we create it"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Saving model to {model_path}")


def model_fit(data:pd.DataFrame) -> None:
    """We train RF model and save it"""
    X_train, X_test, y_train, y_test = train_test_split_data(data, TRAIN_SIZE)

    rf = Pipeline([('scaler', StandardScaler()),
                    ('estimator', RandomForestClassifier(**PARAMS_MODEL))]) 
    

    rf.fit(X_train, y_train)

    rf_pr_auc = model_evaluation_pr_auc(rf, X_test, y_test)

    logger.info(
        f'Model: Random Forest  ; PR AUC: {rf_pr_auc}'
    )
    model_path = os.path.join(MODEL_FOLDER, f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_rf.pkl')
    save_model(rf, model_path)


def main():
    df = get_relevant_orders_df()
    model_fit(df)


if __name__ == "__main__":
    main()
