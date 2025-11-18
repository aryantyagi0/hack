
import numpy as np
import pandas as pd

from src.features.build_features import apply_feature_engineering
from src.utils.guardrails import validate_prediction_results
from src.utils.store import AssignmentStore


@validate_prediction_results
def main():
    store = AssignmentStore()

   
    df_test = store.get_raw("test_data.csv")
    df_test["is_test"] = 1

    
    df_hist = store.get_raw("participant_log.csv")
    df_hist["is_test"] = 0

    
    df_hist = df_hist.rename(columns={"event_timestamp": "event_timestamp"})
    df_test = df_test.rename(columns={"event_timestamp": "event_timestamp"})

    
    df_full = pd.concat([df_hist, df_test], ignore_index=True)

    
    df_full = apply_feature_engineering(df_full)

    
    df_test_final = df_full[df_full["is_test"] == 1].copy()

    
    model = store.get_model("saved_model.pkl")

    
    df_test_final["score"] = model.predict(df_test_final)

    
    selected = choose_best_driver(df_test_final)

    store.put_predictions("results.csv", selected)


def choose_best_driver(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby("order_id").agg({"driver_id": list, "score": list}).reset_index()
    df["best_driver"] = df.apply(
        lambda r: r["driver_id"][np.argmax(r["score"])], axis=1
    )
    df = df.drop(["driver_id", "score"], axis=1)
    df = df.rename(columns={"best_driver": "driver_id"})
    return df


if __name__ == "__main__":
    main()
