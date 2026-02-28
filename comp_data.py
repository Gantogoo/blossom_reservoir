import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
from urllib.request import urlretrieve


NOAA_BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/"


def get_ghcn_station(station_id: str, cache_dir: str = "data/noaa_cache") -> pd.DataFrame:
    """
    Download one NOAA GHCN-Daily station file (.csv.gz), cache it locally,
    and return a daily dataframe.

    Output columns:
        Date, Year, TMIN, TMAX, temp, PRCP
    where:
        temp = (TMAX + TMIN) / 2   # still in NOAA raw units unless divided later
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    local_file = cache_path / f"{station_id}.csv.gz"
    if not local_file.exists():
        url = f"{NOAA_BASE_URL}{station_id}.csv.gz"
        urlretrieve(url, local_file)

    cols = ["STATION", "YYYYMMDD", "ELEMENT", "VALUE", "MFLAG", "QFLAG", "SFLAG", "OBS_TIME"]
    data = pd.read_csv(local_file, header=None, names=cols, compression="gzip")

    data["Date"] = pd.to_datetime(data["YYYYMMDD"].astype(str), format="%Y%m%d", errors="coerce")
    data["VALUE"] = pd.to_numeric(data["VALUE"], errors="coerce")

    # NOAA missing sentinel
    data.loc[data["VALUE"] == -9999, "VALUE"] = np.nan

    # Keep only what we need before pivoting
    data = data.loc[data["ELEMENT"].isin(["TMIN", "TMAX", "PRCP"]), ["Date", "ELEMENT", "VALUE"]]
    data = data.dropna(subset=["Date"])

    data_wide = (
        data.pivot_table(index="Date", columns="ELEMENT", values="VALUE", aggfunc="first")
        .reset_index()
    )

    # Ensure expected columns exist
    for col in ["TMIN", "TMAX", "PRCP"]:
        if col not in data_wide.columns:
            data_wide[col] = np.nan

    # compute temp from TMIN/TMAX
    data_wide = data_wide.dropna(subset=["TMIN", "TMAX"]).copy()
    data_wide["temp"] = (data_wide["TMAX"] + data_wide["TMIN"]) / 2.0
    data_wide["Year"] = data_wide["Date"].dt.year

    return data_wide[["Date", "Year", "TMIN", "TMAX", "temp", "PRCP"]]


def build_oct_mar_monthly_features_from_noaa(
    noaa_stations: dict,
    cache_dir: str = "data/noaa_cache",
    divide_by_10: bool = True,
) -> pd.DataFrame:
    """
    Download NOAA daily station data for multiple cities and convert it into one row
    per (City, Season) with monthly average temperatures from October to March.

    Season definition:
      - Oct/Nov/Dec of year Y belong to blossom season Y+1
      - Jan/Feb/Mar of year Y belong to blossom season Y

    Parameters
    ----------
    noaa_stations : dict
        Mapping from blossom city name -> NOAA station ID
        Example: {"Kyoto": "JA000047759"}

    divide_by_10 : bool
        True for real °C / mm instead of NOAA raw units.

    Returns
    -------
    DataFrame with columns:
        City, Season, temp_Oct, temp_Nov, temp_Dec, temp_Jan, temp_Feb
    """
    frames = []

    for city, station_id in noaa_stations.items():
        station_df = get_ghcn_station(station_id, cache_dir=cache_dir).copy()
        station_df["City"] = city
        frames.append(station_df)

    if not frames:
        return pd.DataFrame(columns=[
            "City", "Season",
            "temp_Oct", "temp_Nov", "temp_Dec",
            "temp_Jan", "temp_Feb"
        ])

    weather = pd.concat(frames, ignore_index=True)

    # Convert NOAA raw units to real units
    if divide_by_10:
        for col in ["TMIN", "TMAX", "temp", "PRCP"]:
            if col in weather.columns:
                weather[col] = weather[col] / 10.0

    weather["Month"] = weather["Date"].dt.month
    weather["Year"] = weather["Date"].dt.year

    # Keep only Oct-Mar
    weather = weather[weather["Month"].isin([10, 11, 12, 1, 2, 3])].copy()

    # Assign bloom season
    weather["Season"] = weather["Year"]
    weather.loc[weather["Month"] >= 10, "Season"] += 1

    monthly = (
        weather.groupby(["City", "Season", "Month"], as_index=False)["temp"]
        .mean()
        .pivot(index=["City", "Season"], columns="Month", values="temp")
        .reset_index()
    )

    monthly = monthly.rename(columns={
        10: "temp_Oct",
        11: "temp_Nov",
        12: "temp_Dec",
        1:  "temp_Jan",
        2:  "temp_Feb",
        3:  "temp_Mar",
    })

    return monthly


def create_sakura_data(
    first_season: int = 1956,
    last_season: int = 2020,
    prediction_seasons: list[int] | None = None,
    save_data: bool = True,
    noaa_stations: dict | None = None,
    noaa_cache_dir: str = "data/noaa_cache",
    divide_by_10: bool = True,
    external_bloom_files: dict | None = None,
):
    blossom_df = pd.read_csv("./data/sakura_full_bloom_dates.csv")
    temps_df = pd.read_csv("./data/Japanese_City_Temps.csv")
    city_df = pd.read_csv("./data/worldcities.csv")

    temps_df["Date"] = pd.to_datetime(temps_df["Date"])

    blossom_cities = set(blossom_df["Site Name"])
    temp_cities = set(temps_df.columns) - {"Date"}
    common_cities = list(blossom_cities.intersection(temp_cities))

    def get_temps(city: str, start_date: pd.Timestamp, end_date: pd.Timestamp):
        if pd.isnull(start_date):
            start_date = end_date - timedelta(days=365)
        if pd.isnull(end_date):
            end_date = start_date + timedelta(days=365)

        mask = (temps_df["Date"] >= start_date) & (temps_df["Date"] < end_date)
        return temps_df.loc[mask, city].tolist()

    result_rows = []

    # historical labeled rows
    for city in common_cities:
        city_data = blossom_df[blossom_df["Site Name"] == city]
        if city_data.empty:
            continue

        row = city_data.iloc[0]

        for year in range(first_season, last_season + 1):
            prev_col = str(year - 1)
            curr_col = str(year)

            if prev_col not in row.index or curr_col not in row.index:
                continue

            start_date = pd.to_datetime(row[prev_col], errors="coerce")
            end_date = pd.to_datetime(row[curr_col], errors="coerce")

            if pd.isnull(end_date):
                continue

            temps = get_temps(city, start_date, end_date)

            if len(temps) > 0:
                result_rows.append({
                    "City": city,
                    "Season": year,
                    "Blossom": end_date.dayofyear,
                    "Mean_Temp": float(np.mean(temps)),
                    "Temps": temps,
                })

    result_df = pd.DataFrame(result_rows)

    # Ensure coordinate columns exist even if result_rows is empty
    for coord in ["Lat", "Lng"]:
        if coord not in result_df.columns:
            result_df[coord] = np.nan

    # append additional blooming datasets (e.g., Washington DC, New York, Vancouver, Liestal)
    # Expected columns in each CSV (case-insensitive): year, bloom_doy; optional lat, long/lng
    if external_bloom_files:
        ext_rows = []
        for city_name, file_path in external_bloom_files.items():
            ext_df = pd.read_csv(file_path)
            cols = {c.lower(): c for c in ext_df.columns}

            if "year" not in cols or "bloom_doy" not in cols:
                raise ValueError(f"External bloom file {file_path} must contain 'year' and 'bloom_doy' columns")

            # Optional city-specific filters
            if city_name.lower() == "washington":
                ext_df = ext_df[pd.to_numeric(ext_df[cols["year"]], errors="coerce") >= 2000]

            years = pd.to_numeric(ext_df[cols["year"]], errors="coerce")
            doy = pd.to_numeric(ext_df[cols["bloom_doy"]], errors="coerce")
            lat = pd.to_numeric(ext_df[cols.get("lat", cols.get("latitude", ""))], errors="coerce") if ("lat" in cols or "latitude" in cols) else np.nan
            lng_col = cols.get("long", cols.get("lng", cols.get("longitude", "")))
            lng = pd.to_numeric(ext_df[lng_col], errors="coerce") if lng_col else np.nan

            for y, d, la, ln in zip(years, doy, np.broadcast_to(lat, len(years)), np.broadcast_to(lng, len(years))):
                if pd.isna(y) or pd.isna(d):
                    continue
                ext_rows.append({
                    "City": city_name,
                    "Season": int(y),
                    "Blossom": float(d),
                    "Mean_Temp": np.nan,
                    "Temps": [],
                    "Lat": la if not pd.isna(la) else np.nan,
                    "Lng": ln if not pd.isna(ln) else np.nan,
                })

        if ext_rows:
            result_df = pd.concat([result_df, pd.DataFrame(ext_rows)], ignore_index=True)

    # geo info (used to fill missing Lat/Lng). Keep one row per city to avoid cartesian blowup.
    city_geo = city_df[["city_ascii", "lat", "lng"]].rename(
        columns={"city_ascii": "City", "lat": "Lat", "lng": "Lng"}
    ).drop_duplicates(subset=["City"], keep="first")

    # Only merge geo for rows missing coordinates to avoid duplicating cities with provided lat/lng
    needs_geo = result_df[result_df[["Lat", "Lng"]].isna().any(axis=1)].copy()
    has_geo = result_df[result_df[["Lat", "Lng"]].notna().all(axis=1)].copy()

    if not needs_geo.empty:
        needs_geo = needs_geo.merge(city_geo, how="left", on="City", suffixes=("", "_geo"))
        for coord in ["Lat", "Lng"]:
            geo_col = f"{coord}_geo"
            if geo_col in needs_geo.columns:
                needs_geo[coord] = needs_geo[coord].fillna(needs_geo[geo_col])
                needs_geo = needs_geo.drop(columns=[geo_col])

    result_df = pd.concat([has_geo, needs_geo], ignore_index=True)

    monthly_features = None
    if noaa_stations is not None:
        monthly_features = build_oct_mar_monthly_features_from_noaa(
            noaa_stations=noaa_stations,
            cache_dir=noaa_cache_dir,
            divide_by_10=divide_by_10,
        )

        # If a requested prediction season is missing for a city (e.g., NOAA data not yet
        # updated for the latest winter), copy the most recent available monthly row for
        # that city so downstream code always has a feature row to work with.
        if prediction_seasons is not None and not monthly_features.empty:
            carry_rows = []

            for city, group in monthly_features.groupby("City"):
                if group.empty:
                    continue

                latest_row = group.sort_values("Season").iloc[-1]
                seen_seasons = set(group["Season"].tolist())

                for season in prediction_seasons:
                    if season not in seen_seasons:
                        row_copy = latest_row.copy()
                        row_copy["Season"] = season
                        carry_rows.append(row_copy)

            if carry_rows:
                monthly_features = pd.concat([monthly_features, pd.DataFrame(carry_rows)], ignore_index=True)

        # merge monthly features onto historical rows
        result_df = result_df.merge(
            monthly_features,
            how="left",
            on=["City", "Season"]
        )

    # add explicit future prediction rows
    if prediction_seasons is not None and monthly_features is not None:
        pred_rows = monthly_features[
            monthly_features["Season"].isin(prediction_seasons)
        ].copy()

        if not pred_rows.empty:
            pred_rows = pred_rows.merge(city_geo, how="left", on="City")

            pred_rows["Blossom"] = np.nan
            pred_rows["Mean_Temp"] = np.nan
            pred_rows["Temps"] = [[] for _ in range(len(pred_rows))]

            # keep same column order
            wanted_cols = [
                "City", "Season", "Blossom", "Mean_Temp", "Temps", "Lat", "Lng",
                "temp_Oct", "temp_Nov", "temp_Dec", "temp_Jan", "temp_Feb"
            ]

            # add temp_Mar too if it exists
            if "temp_Mar" in pred_rows.columns:
                wanted_cols.append("temp_Mar")

            pred_rows = pred_rows[wanted_cols]

            # remove duplicates
            if not result_df.empty:
                existing_idx = result_df.set_index(["City", "Season"]).index
                pred_rows = pred_rows[
                    ~pred_rows.set_index(["City", "Season"]).index.isin(existing_idx)
                ]

            result_df = pd.concat([result_df, pred_rows], ignore_index=True)

    if save_data:
        result_df.to_csv("data/training_data.csv", index=False, sep=";")

    return result_df


def load_sakura_data(file_path: str = "data/training_data.csv") -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=";")

    # Restore Temps from string to np.ndarray
    def regex_magic(x):
        if isinstance(x, str):
            return np.fromstring(x.replace("[", "").replace("]", ""), sep=", ", dtype=float)
        return np.array([], dtype=float)

    df["Temps"] = df["Temps"].apply(regex_magic)
    return df


if __name__ == "__main__":
    noaa_stations = {
        "Vancouver": "CA001108395",
        "Washington": "USW00013743",
        "New York": "USW00094728",   # Central Park
        "Liestal": "SZ000001940",
        "Kyoto": "JA000047759",
    }

    df = create_sakura_data(
        save_data=True,
        noaa_stations=noaa_stations,
        noaa_cache_dir="data/noaa_cache",
        divide_by_10=True,
    )

    print(df.head())
    print(df.columns.tolist())