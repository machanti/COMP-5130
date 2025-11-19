import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from pmdarima import auto_arima

def mae(actual: pd.Series, expected: pd.Series) -> float:
    return np.mean(np.abs(actual - expected))

def mse(actual: pd.Series, expected: pd.Series) -> float:
    return np.mean(np.pow(actual - expected, 2))

def preproc_csgo(df: pd.DataFrame, freq: str) -> pd.Series:
    timestamps = pd.to_datetime(df["DateTime"])

    # Interpolate
    df["Players"] = df["Players"].interpolate()

    s = pd.Series(list(df["Players"]), index=list(timestamps))

    # Resample and interpolate
    resampled = s.resample(freq).interpolate()

    # Remove outliers
    Q1 = resampled.quantile(0.25)
    Q3 = resampled.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    resampled.loc[(resampled < lower_bound) | (resampled > upper_bound)] = np.nan

    # Interpolate between removed outliers
    resampled = resampled.interpolate()

    # Convert to integers (you cannot have 0.5 players in a game)
    return pd.Series(resampled).astype(int)

DATASETS = [
    ("csgo", preproc_csgo, "2D")
]
SPLIT_PERCENT = 0.1 # Remove last 10% of dataset, and try and predict it

def main():
    warnings.filterwarnings("ignore")

    data_dir = Path("./data")
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, preproc_func, frequency in DATASETS:
        print(f"Analyzing dataset {name}")
        print("\tLoading dataset")
        df = pd.read_csv(data_dir / f"{name}.csv")
        print("\tPreprocessing dataset")
        data = preproc_func(df, frequency)
        print(f"\t\tProcessed dataset size: {len(data)}")

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
        arima_prediction = arima.predict(len(test_set)).astype(int)
        print(f"\t\tARIMA{arima.order} MAE: {mae(arima_prediction, test_set)}")
        print(f"\t\tARIMA{arima.order} MSE: {mse(arima_prediction, test_set)}")

        future = proph.make_future_dataframe(periods=len(test_set), freq=frequency)
        proph_prediction = proph.predict(future)[["ds", "yhat"]][split_index:].set_index("ds")["yhat"].astype(int)
        print(f"\t\tProphet MAE: {mae(proph_prediction, test_set)}")
        print(f"\t\tProphet MSE: {mse(proph_prediction, test_set)}")

        print("\tCreating plots")

        print("\t\tPrediction plot")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        ax1.set_title(f"Time Series with ARIMA{arima.order} and Prophet Predictions ({name})")
        ax1.plot(data.index, data, label="Actual Data", color="black")
        ax1.plot(test_set.index, arima_prediction, label=f"ARIMA{arima.order} Prediction", color="blue")
        ax1.plot(test_set.index, proph_prediction, label="Prophet Prediction", color="red")
        ax1.legend()

        ax2.plot(test_set.index, test_set, label="Actual Data", color="black")
        ax2.plot(test_set.index, arima_prediction, label=f"ARIMA{arima.order} Prediction", color="blue")
        ax2.plot(test_set.index, proph_prediction, label="Prophet Prediction", color="red")

        for ax in [ax1, ax2]:
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            ax.grid(alpha=0.3)

        fig.savefig(output_dir / f"{name}_prediction.png")



if __name__ == "__main__":
    main()
