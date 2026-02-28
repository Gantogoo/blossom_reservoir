import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from comp_data import load_sakura_data
from sklearn import preprocessing, linear_model
from sklearn.metrics import mean_squared_error

import torch
from datetime import datetime, timedelta

from comp_data import create_sakura_data

noaa_stations = {
    "Vancouver": "CA001108395",
    "Washington": "USW00013743",
    "New York": "USW00094728",   # Central Park
    "Liestal": "SZ000001940",
    "Kyoto": "JA000047759",
}

# Additional bloom history CSVs from the competition repo
external_bloom_files = {
    "New York": "data/nyc.csv",
    "Washington": "data/washingtondc.csv",
    "Vancouver": "data/vancouver.csv",
    "Liestal": "data/liestal.csv",
}

create_sakura_data(
    first_season=1956,
    last_season=2025,               # uses Oct 2025–Feb 2026 temps
    prediction_seasons=[2026],      # adds unlabeled 2026 rows
    save_data=True,
    noaa_stations=noaa_stations,
    noaa_cache_dir="data/noaa_cache",
    divide_by_10=True,
    external_bloom_files=external_bloom_files,
)

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def get_date_from_doy(year: int, doy: float) -> str:
    start_date = datetime(year, 1, 1)
    d = int(round(float(doy)))
    d = max(1, min(366, d))
    target_date = start_date + timedelta(days=d - 1)
    return target_date.strftime("%B %d, %Y")


def conformal_quantile(abs_residuals: np.ndarray, alpha: float) -> float:
    r = np.asarray(abs_residuals).ravel()
    r = r[~np.isnan(r)]
    n = r.size
    if n == 0:
        return float("nan")

    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(max(level, 0.0), 1.0)

    try:
        return float(np.quantile(r, level, method="higher"))
    except TypeError:
        return float(np.quantile(r, level, interpolation="higher"))


# ESN reservoir
class TorchESNReservoir:
    def __init__(
        self,
        N: int,
        input_dim: int,
        con_prob: float = 0.2,
        tau: float = 10.0,
        g: float = 1.5,
        noise: float = 0.0,
        dt: float = 1.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
        zero_diagonal: bool = False,
    ):
        self.N = N
        self.input_dim = input_dim
        self.con_prob = con_prob
        self.tau = tau
        self.g = g
        self.noise = noise
        self.dt = dt
        self.alpha = dt / tau
        self.device = torch.device(device)
        self.dtype = dtype

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        self.Win = (2.0 * torch.rand((input_dim, N), generator=gen) - 1.0).to(self.device, self.dtype)

        sigma = math.sqrt(1.0 / (con_prob * N))
        mask = (torch.rand((N, N), generator=gen) < con_prob)

        W = torch.zeros((N, N), dtype=self.dtype)
        W[mask] = torch.randn((mask.sum().item(),), generator=gen, dtype=self.dtype) * sigma

        if zero_diagonal:
            W.fill_diagonal_(0.0)

        self.Wrec = W.to(self.device)

    @torch.no_grad()
    def run(self, U: np.ndarray, sim_time: int = 100, reset_each_sample: bool = True, batch_size: int = 256) -> np.ndarray:
        U_t = torch.as_tensor(U, device=self.device, dtype=self.dtype)

        states = []
        n = U_t.shape[0]

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            u_batch = U_t[start:end]
            B = u_batch.shape[0]
            x = torch.zeros((B, self.N), device=self.device, dtype=self.dtype)

            for _ in range(sim_time):
                r = torch.tanh(x)
                inp = u_batch @ self.Win
                rec = r @ self.Wrec.T

                if self.noise > 0.0:
                    nterm = self.noise * (2.0 * torch.rand_like(x) - 1.0)
                else:
                    nterm = 0.0

                x = (1.0 - self.alpha) * x + self.alpha * (inp + self.g * rec + nterm)

            states.append(torch.tanh(x).cpu())

        return torch.cat(states, dim=0).numpy()


def main():
    ensure_dir("figures")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load full data WITHOUT global dropna()
    df = load_sakura_data()

    # Use Oct-Mar monthly features + coordinates
    feature_cols = [
        "temp_Oct", "temp_Nov", "temp_Dec",
        "temp_Jan", "temp_Feb", 
        "Lat", "Lng"
    ]

    target_col = "Blossom"
    target_season = 2026

    # Historical labeled data
    hist_df = df[df[target_col].notna()].copy()

    # Training rows must have all model inputs
    hist_df = hist_df.dropna(subset=feature_cols + [target_col]).copy()

    # 2026 prediction rows: Blossom can be NaN, features must exist
    pred_df_raw = df[df["Season"] == target_season].copy()

    if pred_df_raw.empty:
        raise ValueError(
            f"No prediction rows found for Season={target_season}. "
            f"Need to generate 2026 rows with temp_Oct...temp_Feb, Lat, Lng."
        )

    # Fill missing monthly temps with historical medians to avoid losing rows
    temp_cols = ["temp_Oct", "temp_Nov", "temp_Dec", "temp_Jan", "temp_Feb"]
    temp_medians = hist_df[temp_cols].median(numeric_only=True)
    pred_df_raw[temp_cols] = pred_df_raw[temp_cols].fillna(temp_medians)

    pred_df = pred_df_raw.dropna(subset=feature_cols).copy()

    if pred_df.empty:
        missing_cities = pred_df_raw[pred_df_raw[feature_cols].isna().any(axis=1)]["City"].unique().tolist()
        raise ValueError(
            f"Prediction rows for Season={target_season} exist but still have missing features "
            f"for cities: {missing_cities}. Ensure temp_Oct...temp_Feb and Lat/Lng are present."
        )

    # train on seasons before 2026
    hist_df = hist_df[hist_df["Season"] < target_season].copy()

    # Season-based fit/cal split, not random row split
    seasons = sorted(hist_df["Season"].unique())
    n_cal_seasons = max(3, int(np.ceil(0.2 * len(seasons))))
    cal_seasons = seasons[-n_cal_seasons:]
    fit_seasons = seasons[:-n_cal_seasons]

    fit_df = hist_df[hist_df["Season"].isin(fit_seasons)].copy()
    cal_df = hist_df[hist_df["Season"].isin(cal_seasons)].copy()

    if fit_df.empty or cal_df.empty:
        raise ValueError("Fit/calibration split failed. Need more historical seasons.")

    X_fit_raw = fit_df[feature_cols].values
    y_fit_raw = fit_df[[target_col]].values

    X_cal_raw = cal_df[feature_cols].values
    y_cal_raw = cal_df[[target_col]].values

    X_pred_raw = pred_df[feature_cols].values

    # Scale using minmax
    x_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    y_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

    X_fit = x_scaler.fit_transform(X_fit_raw)
    y_fit = y_scaler.fit_transform(y_fit_raw)

    X_cal = x_scaler.transform(X_cal_raw)
    y_cal = y_scaler.transform(y_cal_raw)

    X_pred = x_scaler.transform(X_pred_raw)

    alpha = 0.10
    con_prob = 0.2
    sim_time = 100
    reset_each_sample = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # City-specific interval scaling around the conformal q, as some cities have more variable bloom dates and may benefit from wider intervals, while others are more consistent.
    city_interval_scales = {
        "Kyoto": 0.25,
        "Liestal": 0.35,
        "Washington": 0.35,
        "New York": 0.5,
        "Vancouver": 0.5,
    }

    # N_values = [400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    N_values = [700]

    best = {"N": None, "mse_day": float("inf")}
    predictions_2026 = []
    intervals_2026 = []

    for N in N_values:
        reservoir = TorchESNReservoir(
            N=N,
            input_dim=len(feature_cols),
            con_prob=con_prob,
            tau=10.0,
            g=1.5,
            noise=0.0,
            dt=1.0,
            device=device,
            seed=42 + N,
            zero_diagonal=False,
        )

        # Fit on older seasons
        Xr_fit = reservoir.run(X_fit, sim_time=sim_time, reset_each_sample=reset_each_sample)
        lasso = linear_model.Lasso(alpha=0.001, max_iter=10000)
        lasso.fit(Xr_fit, y_fit.ravel())

        # Calibrate on latest known seasons
        Xr_cal = reservoir.run(X_cal, sim_time=sim_time, reset_each_sample=reset_each_sample)
        y_cal_pred_scaled = lasso.predict(Xr_cal)

        cal_pred_days = y_scaler.inverse_transform(y_cal_pred_scaled.reshape(-1, 1)).ravel()
        cal_true_days = y_scaler.inverse_transform(y_cal).ravel()

        abs_res = np.abs(cal_true_days - cal_pred_days)
        q = conformal_quantile(abs_res, alpha=alpha)

        mse_day = mean_squared_error(cal_true_days, cal_pred_days)
        print(f"N={N} | calibration MSE (DOY)={mse_day:.4f} | 90% PI half-width q={q:.2f} days")

        if mse_day < best["mse_day"]:
            best = {"N": N, "mse_day": mse_day}

        # Predict 2026
        Xr_pred = reservoir.run(X_pred, sim_time=sim_time, reset_each_sample=True)
        y_pred_scaled = lasso.predict(Xr_pred)
        pred_days = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # New York and Kyoto historically bloom together (from previous observations)
        # therefore, I aligned them to their mean prediction.
        city_idx = {city: idx for idx, city in enumerate(pred_df["City"].tolist())}
        if "Kyoto" in city_idx and "New York" in city_idx:
            kyoto_idx = city_idx["Kyoto"]
            ny_idx = city_idx["New York"]
            mean_doy = float(np.mean([pred_days[kyoto_idx], pred_days[ny_idx]]))
            pred_days[kyoto_idx] = mean_doy
            pred_days[ny_idx] = mean_doy

        for i, (_, row) in enumerate(pred_df.iterrows()):
            city = row["City"]
            pred_doy = pred_days[i]
            city_q = q * city_interval_scales.get(city, 1.0)
            lo = pred_doy - city_q
            up = pred_doy + city_q

            predictions_2026.append({
                "City": city,
                "Season": target_season,
                "Predicted_DOY": pred_doy,
                "Predicted_Date": get_date_from_doy(target_season, pred_doy),
            })

            intervals_2026.append({
                "City": city,
                "Season": target_season,
                "Lower_DOY": lo,
                "Pred_DOY": pred_doy,
                "Upper_DOY": up,
                "Lower_Date": get_date_from_doy(target_season, lo),
                "Pred_Date": get_date_from_doy(target_season, pred_doy),
                "Upper_Date": get_date_from_doy(target_season, up),
            })

            print(
                f"{city}: DOY {pred_doy:.2f} [{lo:.2f}, {up:.2f}] -> "
                f"{get_date_from_doy(target_season, pred_doy)} "
                f"[{get_date_from_doy(target_season, lo)}, {get_date_from_doy(target_season, up)}]"
            )

    print(f"\nBest N by calibration DOY MSE: N={best['N']} with MSE={best['mse_day']:.6f}")

    pd.DataFrame(predictions_2026).to_csv("pred_2026.csv", index=False)
    pd.DataFrame(intervals_2026).to_csv("pred_2026_intervals.csv", index=False)


if __name__ == "__main__":
    main()