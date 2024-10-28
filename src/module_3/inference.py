from train import (
    get_relevant_orders_df,
    model_evaluation_pr_auc,
    split_data,
    MODEL_FOLDER,
    COLS_TO_USE,
    TARGET_COL,
)
import joblib
import logging


MODEL_NAME = "20241028-145946_Lasso_0.001.pkl"
SCALER_NAME = "20241028-145943_scaler.pkl"

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    model = joblib.load(f"{MODEL_FOLDER}{MODEL_NAME}")
    scaler = joblib.load(f"{MODEL_FOLDER}{SCALER_NAME}")
    df = get_relevant_orders_df(5)
    X, y = split_data(df, COLS_TO_USE, TARGET_COL)
    X = scaler.transform(X)
    pr_auc = model_evaluation_pr_auc(model, X, y)
    logging.info(f"Inference PR AUC : {pr_auc}")


if __name__ == "__main__":
    main()
