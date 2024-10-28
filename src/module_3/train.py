import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve
from typing import Tuple
import logging
import datetime

FILE_PATH = "./data/feature_frame.csv"
MODEL_FOLDER = "./data/models/"
COLS_TO_USE = [
    "global_popularity",
    "ordered_before",
    "abandoned_before",
    "normalised_price",
]
TARGET_COL = "outcome"
TRAIN_SIZE = 0.7

PARAMS_MODEL = {"C": 0.001, "solver": "liblinear"}


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_data() -> pd.DataFrame:
    return pd.read_csv(FILE_PATH)


def get_relevant_orders_df(num_orders: int) -> pd.DataFrame:
    """We get the orders that have more or equal than num_orders products bought"""
    data = load_data()
    bought_products = data.query("outcome == 1")
    orders_relevants = (
        bought_products.groupby("order_id")["variant_id"]
        .agg(lambda x: len(list(x)))
        .reset_index()
        .query(f"variant_id >= {num_orders}")
    )
    return data.merge(orders_relevants[["order_id"]], on="order_id")


def split_data(
    data: pd.DataFrame, feature_cols: list, target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    return data[feature_cols], data[target_col]


def train_test_split_scaled(
    data: pd.DataFrame, train_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """We split the data into train and test and scale the features"""
    X, y = split_data(data, COLS_TO_USE, TARGET_COL)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(
        scaler,
        f'{MODEL_FOLDER}{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_scaler.pkl',
    )
    logger.info(
        f'Saving scaler to {MODEL_FOLDER}{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_scaler.pkl'
    )
    return X_train, X_test, y_train, y_test


def model_evaluation_pr_auc(model, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    y_pred = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)
    return pr_auc


def save_model(model, model_path: str) -> None:
    joblib.dump(model, model_path)
    logger.info(f"Saving model to {model_path}")


def model_selection(data) -> None:
    """We train two models with different penalties and select the best one based on PR AUC"""

    X_train, X_test, y_train, y_test = train_test_split_scaled(data, TRAIN_SIZE)

    model_l1 = LogisticRegression(**PARAMS_MODEL, penalty="l1")
    model_l2 = LogisticRegression(**PARAMS_MODEL, penalty="l2")

    model_l1.fit(X_train, y_train)
    model_l2.fit(X_train, y_train)

    model_l1_pr_auc = model_evaluation_pr_auc(model_l1, X_test, y_test)
    model_l2_pr_auc = model_evaluation_pr_auc(model_l2, X_test, y_test)

    best_model = model_l1 if model_l1_pr_auc > model_l2_pr_auc else model_l2
    penalty_type = "Lasso" if model_l1_pr_auc > model_l2_pr_auc else "Ridge"

    logger.info(
        f'Model: {penalty_type} C: {PARAMS_MODEL["C"]} ; PR AUC: {model_l1_pr_auc if model_l1_pr_auc > model_l2_pr_auc else model_l2_pr_auc}'
    )
    model_path = f'{MODEL_FOLDER}{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{penalty_type}_{PARAMS_MODEL["C"]}.pkl'
    save_model(best_model, model_path)


def main():
    df = get_relevant_orders_df(5)
    model_selection(df)


if __name__ == "__main__":
    main()
