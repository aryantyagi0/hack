import pandas as pd
from haversine import haversine

from src.utils.time import robust_hour_of_iso_date


def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_distance"] = df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ),
        axis=1,
    )
    return df


def hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    df["event_hour"] = df["event_timestamp"].apply(robust_hour_of_iso_date)
    return df


def driver_historical_completed_bookings(df: pd.DataFrame) -> pd.DataFrame:
    # i m Sorting by event time so past bookings come first
    # df = df.sort_values("event_timestamp")

    # Completed = participant_status == "ACCEPTED"
    df["completed_flag"] = (df["participant_status"] == "ACCEPTED").astype(int)

    # Cumulative completed bookings per driver BEFORE the current booking
    df["driver_completed_history"] = df.groupby("driver_id")["completed_flag"].cumsum() - df["completed_flag"]

    return df

def driver_accept_rate(df: pd.DataFrame) -> pd.DataFrame:
    df["accepted_flag"] = (df["participant_status"] == "ACCEPTED").astype(int)
    df["driver_total_requests"] = df.groupby("driver_id")["accepted_flag"].cumcount() + 1
    df["driver_accepts_till_now"] = df.groupby("driver_id")["accepted_flag"].cumsum()
    df["driver_accept_rate"] = (
        df["driver_accepts_till_now"] / df["driver_total_requests"]
    )
    return df
