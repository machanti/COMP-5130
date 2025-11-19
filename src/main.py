import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from pmdarima import auto_arima

def preproc_csgo(df: pd.DataFrame) -> pd.Series:
    timestamps = pd.to_datetime(df["DateTime"])

    s = pd.Series(list(df["Players"]), index=list(timestamps))

    # Resample to hourly and interpolate
    hourly = s.resample("H").interpolate()

    # Convert to integers (you cannot have 0.5 players in a game)
    return pd.Series(hourly).astype(int)

DATASETS = [
    ("csgo", preproc_csgo)
]
SPLIT_PERCENT = 0.1 # Remove last 10% of dataset, and try and predict it

def main():
    warnings.filterwarnings("ignore")

    data_dir = Path("./data")
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, preproc_func in DATASETS:
        print(f"Analyzing dataset {name}")
        print("\tLoading dataset")
        df = pd.read_csv(data_dir / f"{name}.csv")
        print("\tPreprocessing dataset")
        data = preproc_func(df)

        print("\tSplitting dataset")
        split_index = int(len(data) * (1 - SPLIT_PERCENT))
        train_set = data[:split_index]
        test_set = data[split_index:]

        print("\tTraining ARIMA")
        arima = auto_arima(train_set, seasonal=False)
        print(f"\t\tFitted order: {arima.order}")
        print(f"\t\tPhi = {", ".join(f"{x:.3f}" for x in arima.arparams())}")
        print(f"\t\tTheta = {", ".join(str(f"{x:.3f}") for x in arima.maparams())}")

        print("\tTraining Prophet")
        proph = Prophet()
        proph.fit(train_set.to_frame(name="y").rename_axis("ds").reset_index())

        print("\tPredicting")
        arima_prediction = arima.predict(len(test_set))
        proph_prediction = proph.predict(proph.make_future_dataframe(periods=len(test_set)))[["yhat"]]
        print(arima_prediction)
        print(proph_prediction)


if __name__ == "__main__":
    main()
