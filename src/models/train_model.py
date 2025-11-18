from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.models.classifier import SklearnClassifier
from src.utils.config import load_config
from src.utils.guardrails import validate_evaluation_metrics
from src.utils.store import AssignmentStore


@validate_evaluation_metrics
def main():
    store = AssignmentStore()
    config = load_config()

    # Load processed dataset
    df = store.get_processed("transformed_dataset.csv")
    df_train, df_test = train_test_split(df, test_size=config["test_size"], random_state=33)

 
    xgb_estimator = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=4,   
        random_state=33
    )

    # Wrap inside your custom classifier
    model = SklearnClassifier(xgb_estimator, config["features"], config["target"])
    
    # Train
    model.train(df_train)

    # Evaluate
    metrics = model.evaluate(df_test)

    # Save model + metrics
    store.put_model("saved_model.pkl", model)
    store.put_metrics("metrics.json", metrics)


if __name__ == "__main__":
    main()
