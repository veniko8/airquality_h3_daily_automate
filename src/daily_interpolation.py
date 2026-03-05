"""
Wind-aware daily interpolation to H3 (res=9) for Zagreb.

This is a standalone script that:
1. Fetches AQI data from Supabase
2. Aggregates hourly data to daily
3. Performs wind-aware IDW interpolation to H3 grid
4. Saves results to CSV

All dependencies included in one file.

Dependencies:
  pip install pandas numpy h3 shapely pyproj python-dotenv supabase requests
"""

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import h3
from pyproj import Geod
from shapely.geometry import shape
from dotenv import load_dotenv


# ==============================
# SUPABASE CLIENT SETUP
# ==============================
load_dotenv()


def get_supabase_client() -> Any:
	"""Create and return a Supabase client. Raises a clear error if env vars are missing.

	Import `supabase` is safe — the real client is created lazily when used.
	"""
	url = os.getenv("SUPABASE_URL")
	key = os.getenv("SUPABASE_KEY")
	if not url or not key:
		raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in the environment or .env file")

	# Import here to avoid raising ImportError at module import time if package missing
	from supabase import create_client

	return create_client(url, key)


class _LazySupabase:
	"""Proxy that creates the real Supabase client on first attribute access."""

	def __init__(self):
		self._client = None

	def _ensure(self):
		if self._client is None:
			self._client = get_supabase_client()

	def __getattr__(self, name):
		self._ensure()
		return getattr(self._client, name)


# Lazy Supabase client initialization
supabase = _LazySupabase()


# ==============================
# DATA FETCHING
# ==============================
def fetch_aqi_data_gold(limit=200000, order_by: str | None = None, desc: bool = False):
    """Fetch rows from the `aqi_weather_hourly` table.

    - `limit`: int or None. If None, all matching rows are paginated.
    - `order_by`: optional column name to order by.
    - `desc`: whether to order descending.

    This variant restricts data to *yesterday's* UTC calendar day by applying
    the filter on the server; the result should include every row from that
    single day regardless of the table size.
    """
    # determine the UTC date range for yesterday
    now = pd.Timestamp.utcnow()
    yesterday_start = (now.floor("D") - pd.Timedelta(days=1))
    yesterday_end = yesterday_start + pd.Timedelta(days=1)
    print(f"[debug] applying server-side date filter {yesterday_start} <= ts_utc < {yesterday_end}")

    base_query = (
        supabase.table("aqi_weather_hourly")
        .select("*")
        .gte("ts_utc", yesterday_start.isoformat())
        .lt("ts_utc", yesterday_end.isoformat())
    )

    if order_by:
        base_query = base_query.order(order_by, desc=desc)

    # helper executor with debug
    def _exec(q):
        resp = q.execute()
        print(f"[debug] response status_code=", getattr(resp, "status_code", None))
        if hasattr(resp, "data"):
            print(f"[debug] rows returned=", 0 if resp.data is None else len(resp.data))
        return resp

    if limit is None:
        # paginate until no more rows
        page_size = 10000
        rows = []
        offset = 0
        while True:
            q = base_query.limit(page_size).offset(offset)
            resp = _exec(q)
            if not resp.data:
                break
            rows.extend(resp.data)
            if len(resp.data) < page_size:
                break
            offset += page_size
        df = pd.DataFrame(rows)
        return df
    else:
        q = base_query.limit(limit)
        resp = _exec(q)
        if resp.data is None:
            raise ValueError("No data returned from Supabase")
        return pd.DataFrame(resp.data)


# ==============================
# GEODESY HELPERS
# ==============================
GEOD = Geod(ellps="WGS84")


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Geodesic distance in meters."""
    _, _, dist = GEOD.inv(lon1, lat1, lon2, lat2)
    return float(dist)


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Bearing FROM point1 TO point2 in degrees (0=N, 90=E).
    """
    fwd_az, _, _ = GEOD.inv(lon1, lat1, lon2, lat2)
    return float((fwd_az + 360.0) % 360.0)


def angdiff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees."""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


# ==============================
# WIND-AWARE WEIGHTING
# ==============================
def wind_multiplier(
    station_to_cell_bearing: float,
    wind_dir_from_deg: float,
    wind_speed: float,
    min_mult: float = 0.30,
    max_mult: float = 2.00,
) -> float:
    """
    Assumes meteorological convention: wind_direction is the direction wind is COMING FROM.
    Downwind direction = wind_from + 180.
    """
    downwind = (wind_dir_from_deg + 180.0) % 360.0
    diff = angdiff_deg(station_to_cell_bearing, downwind)
    diff_rad = math.radians(diff)

    # Wind effect strength (0 at calm, ~1 at 5 m/s+)
    k = min(1.0, max(0.0, wind_speed / 5.0))

    # cos: +1 = downwind, -1 = upwind
    mult = 1.0 + k * math.cos(diff_rad)

    # keep stable
    return float(max(min_mult, min(max_mult, mult)))


def idw_wind_aware_cell(
    cell_lat: float,
    cell_lon: float,
    stations: pd.DataFrame,
    power: float = 2.0,
    eps_m: float = 50.0,
) -> float:
    """
    stations columns required:
      lat, lon, aqi_value, wind_speed, wind_direction

    Each station uses its own wind values (station-specific wind).
    """
    weights = []
    values = []

    for r in stations.itertuples(index=False):
        s_lat = float(r.lat)
        s_lon = float(r.lon)
        v = float(r.aqi_value)

        ws = float(r.wind_speed)
        wd = float(r.wind_direction)

        d = distance_m(s_lat, s_lon, cell_lat, cell_lon)
        d_eff = max(d, eps_m)

        b = bearing_deg(s_lat, s_lon, cell_lat, cell_lon)
        wm = wind_multiplier(b, wd, ws)

        w = wm * (1.0 / (d_eff ** power))
        weights.append(w)
        values.append(v)

    wsum = float(np.sum(weights))
    if wsum == 0.0:
        return float("nan")

    return float(np.dot(weights, values) / wsum)


# ==============================
# H3 GRID CREATION FROM GEOJSON
# ==============================
def polyfill_h3_from_geojson(geojson_obj: dict, h3_res: int = 9) -> list[str]:
    """
    geojson_obj can be:
      - a GeoJSON FeatureCollection
      - a GeoJSON Feature
      - a GeoJSON Geometry (Polygon/MultiPolygon)

    Returns list of H3 cell IDs at resolution h3_res.
    """
    if geojson_obj.get("type") == "FeatureCollection":
        # union all features
        cells = set()
        for feat in geojson_obj["features"]:
            cells |= set(polyfill_h3_from_geojson(feat, h3_res=h3_res))
        return sorted(cells)

    geom = geojson_obj.get("geometry", geojson_obj)
    shp = shape(geom)

    cells = set()

    def fill_polygon(poly):
        # shapely coords are (lon, lat)
        lonlat = list(poly.exterior.coords)
        latlon = [(lat, lon) for lon, lat in lonlat]
        poly_h3 = h3.LatLngPoly(latlon)
        return h3.polygon_to_cells(poly_h3, h3_res)

    if shp.geom_type == "Polygon":
        cells |= set(fill_polygon(shp))
    elif shp.geom_type == "MultiPolygon":
        for poly in shp.geoms:
            cells |= set(fill_polygon(poly))
    else:
        raise ValueError(f"Unsupported geometry type: {shp.geom_type}")

    return sorted(cells)


def h3_centroids(cells: list[str]) -> pd.DataFrame:
    """Return DataFrame: h3, lat, lon for each cell center."""
    rows = []
    for c in cells:
        lat, lon = h3.cell_to_latlng(c)
        rows.append((c, lat, lon))
    return pd.DataFrame(rows, columns=["h3", "lat", "lon"])


# ==============================
# DATA PREP
# ==============================
def normalize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures types + required columns.
    """
    required = ["ts_utc", "station", "pollutant", "aqi_value", "wind_speed", "wind_direction", "lat", "lon"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = df.copy()

    d["ts_utc"] = pd.to_datetime(d["ts_utc"], utc=True, errors="coerce")

    for c in ["aqi_value", "wind_speed", "wind_direction", "lat", "lon"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=required)

    # Optional: enforce hourly alignment (if your ts_utc is already hourly, this changes nothing)
    d["ts_utc"] = d["ts_utc"].dt.floor("h")

    return d


def circular_mean_deg(series: pd.Series) -> float:
    """Compute circular mean of angles in degrees. Returns NaN for empty/na-only series."""
    vals = pd.to_numeric(series.dropna(), errors="coerce")
    if vals.empty:
        return float("nan")
    rad = np.deg2rad(vals.values.astype(float))
    s = np.sin(rad).mean()
    c = np.cos(rad).mean()
    ang = math.degrees(math.atan2(s, c)) % 360.0
    return float(ang)


# ==============================
# INTERPOLATION
# ==============================
def build_surface_h3(
    df: pd.DataFrame,
    grid: pd.DataFrame,
    power: float = 2.0,
    min_stations: int = 2,
) -> pd.DataFrame:
    """
    df: normalized station-hour rows
    grid: h3 centroids with columns h3, lat, lon

    Returns: ts_utc, pollutant, h3, aqi_est
    """
    outputs = []
    skipped_na = 0
    skipped_min_stations = 0
    processed = 0

    for (t, pol), g in df.groupby(["ts_utc", "pollutant"], dropna=False):
        if pd.isna(t) or pd.isna(pol):
            print(f"  ⚠️  Skipping: ts_utc or pollutant is NaN")
            skipped_na += 1
            continue

        n_unique = g["station"].nunique()
        if n_unique < min_stations:
            print(f"  ⚠️  Skipping: ts={t}, pollutant={pol} has only {n_unique} unique stations (need {min_stations})")
            skipped_min_stations += 1
            continue

        print(f"  ✓ Processing: ts={t}, pollutant={pol} with {n_unique} stations")
        processed += 1

        stations = g[["lat", "lon", "aqi_value", "wind_speed", "wind_direction"]].copy()

        est = []
        for cell in grid.itertuples(index=False):
            aqi = idw_wind_aware_cell(
                cell_lat=float(cell.lat),
                cell_lon=float(cell.lon),
                stations=stations,
                power=power,
            )
            est.append(aqi)

        outputs.append(
            pd.DataFrame(
                {
                    "ts_utc": t,
                    "pollutant": pol,
                    "h3": grid["h3"].values,
                    "aqi_est": est,
                }
            )
        )

    print(f"\n=== build_surface_h3 Summary ===")
    print(f"Groups skipped (NaN): {skipped_na}")
    print(f"Groups skipped (< {min_stations} stations): {skipped_min_stations}")
    print(f"Groups processed: {processed}")
    print(f"Output rows: {sum(len(o) for o in outputs) if outputs else 0}\n")

    if not outputs:
        return pd.DataFrame(columns=["ts_utc", "pollutant", "h3", "aqi_est"])

    return pd.concat(outputs, ignore_index=True)


# ==============================
# OPTIONAL: REST UPSERT TO SUPABASE
# ==============================
def supabase_rest_upsert(
    surface_df: pd.DataFrame,
    table: str,
    supabase_url: str,
    supabase_service_key: str,
    chunk_size: int = 5000,
):
    """
    Optional helper to upsert via PostgREST.
    Requires the target table to have a UNIQUE constraint on (ts_utc, pollutant, h3).

    If you already have a Supabase client doing upserts, ignore this function and use yours.
    """
    import requests

    if surface_df.empty:
        return

    endpoint = f"{supabase_url}/rest/v1/{table}?on_conflict=ts_utc,pollutant,h3"
    headers = {
        "apikey": supabase_service_key,
        "Authorization": f"Bearer {supabase_service_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }

    # Convert timestamps to ISO strings
    payload_df = surface_df.copy()
    payload_df["ts_utc"] = pd.to_datetime(payload_df["ts_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    records = payload_df.to_dict(orient="records")

    for i in range(0, len(records), chunk_size):
        chunk = records[i : i + chunk_size]
        r = requests.post(endpoint, headers=headers, data=json.dumps(chunk), timeout=120)
        if not (200 <= r.status_code < 300):
            raise RuntimeError(f"Supabase upsert failed: {r.status_code} {r.text}")


# ==============================
# MAIN PIPELINE
# ==============================
def main():
    """Run the daily interpolation pipeline."""
    print("=" * 70)
    print("Wind-aware Daily H3 Interpolation for Zagreb")
    print("=" * 70 + "\n")

    # 1) Fetch data from Supabase
    print("📊 Step 1: Fetching AQI data from Supabase...")
    df = fetch_aqi_data_gold()

    # optionally restrict to yesterday's date (calendar day) rather than a rolling 24h window
    # if not df.empty and "ts_utc" in df.columns:
    #     df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    #     max_ts = df["ts_utc"].max()
    #     if pd.notna(max_ts):
    #         # take the day before the most recent timestamp; floor to midnight
    #         yesterday = (max_ts - pd.Timedelta(days=1)).floor("D")
    #         df = df[df["ts_utc"].dt.floor("D") == yesterday]
    #         print(f"  → filtered to yesterday ({yesterday.date()}), {len(df)} rows remaining")

    print("\n=== DEBUG: Initial df ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Nulls:\n{df.isnull().sum()}")
    print(f"Sample:\n{df.head()}\n")

    # 2) Load Zagreb boundary GeoJSON
    print("📍 Step 2: Loading Zagreb boundary GeoJSON...")
    ZAGREB_GEOJSON_PATH = Path("data/zagreb_boundary.geojson")
    if not ZAGREB_GEOJSON_PATH.exists():
        raise FileNotFoundError(
            f"Missing {ZAGREB_GEOJSON_PATH}. Export Zagreb boundary as GeoJSON and save it here."
        )

    zagreb_geojson = json.loads(ZAGREB_GEOJSON_PATH.read_text(encoding="utf-8"))

    # 3) Build H3 grid
    print("🔷 Step 3: Building H3 grid (res=9)...")
    H3_RES = 9
    cells = polyfill_h3_from_geojson(zagreb_geojson, h3_res=H3_RES)
    grid = h3_centroids(cells)
    print(f"✓ H3 cells created: {len(grid)}\n")

    # 4) Normalize and aggregate data
    print("🔄 Step 4: Normalizing data...")
    df_norm = normalize_input_df(df)
    
    print("=== DEBUG: After normalize_input_df ===")
    print(f"Shape: {df_norm.shape}")
    print(f"Nulls:\n{df_norm.isnull().sum()}")
    print(f"Unique timestamps: {df_norm['ts_utc'].nunique()}")
    print(f"Unique pollutants: {df_norm['pollutant'].unique()}")
    print(f"Unique stations: {df_norm['station'].nunique()}")
    print(f"Sample:\n{df_norm.head()}\n")

    print("📅 Step 5: Aggregating hourly to daily...")
    df_norm["ts_date"] = df_norm["ts_utc"].dt.floor("D")

    # Custom aggregation functions
    def agg_aqi(series):
        pol = series.name[2]  # pollutant is the third level in the MultiIndex
        if pol in ["PM2.5", "PM10"]:
            return series.mean()
        elif pol == "NO2":
            return float(np.nanpercentile(series.values.astype(float), 90))
        else:
            return series.mean()

    def agg_wind_speed(series):
        return float(series.mean()) if not series.dropna().empty else float("nan")

    def agg_wind_direction(series):
        return circular_mean_deg(series) if not series.dropna().empty else float("nan")

    # Group and aggregate
    grouped = df_norm.groupby(["ts_date", "station", "pollutant"], dropna=False).agg({
        "aqi_value": agg_aqi,
        "wind_speed": agg_wind_speed,
        "wind_direction": agg_wind_direction,
        "lat": "first",
        "lon": "first"
    }).reset_index()

    # Rename columns and set ts_utc
    grouped = grouped.rename(columns={"ts_date": "ts_utc"})
    grouped["ts_utc"] = pd.to_datetime(grouped["ts_utc"], utc=True)
    df_daily = grouped[["ts_utc", "station", "pollutant", "aqi_value", "wind_speed", "wind_direction", "lat", "lon"]].copy()

    print("=== DEBUG: After daily aggregation ===")
    print(f"Shape: {df_daily.shape}")
    print(f"Unique dates: {df_daily['ts_utc'].nunique()}")
    print("Sample:")
    print(df_daily.head())
    print()

    print("=== DEBUG: Before build_surface_h3 (daily) ===")
    for (t, pol), g in df_daily.groupby(["ts_utc", "pollutant"], dropna=False):
        print(f"  date={t}, pollutant={pol}: {len(g)} station-rows, {g['station'].nunique()} unique")

    # 5) Interpolate
    print("\n🌬️  Step 6: Running wind-aware interpolation...")
    surface_df_daily = build_surface_h3(
        df=df_daily,
        grid=grid,
        power=2.0,
        min_stations=2,
    )

    print("\n=== DEBUG: After build_surface_h3 ===")
    print(f"Shape: {surface_df_daily.shape}")
    print(f"Empty? {surface_df_daily.empty}")
    if not surface_df_daily.empty:
        print(f"Nulls:\n{surface_df_daily.isnull().sum()}")
        print(f"Sample:\n{surface_df_daily.head()}\n")
    else:
        print("⚠️  surface_df is EMPTY! Check the debug output above.\n")

    print(surface_df_daily.head())
    print("Surface rows:", len(surface_df_daily))

    # 6) Upsert to Supabase
    print("\n💾 Step 7: Upserting results to Supabase...")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")
    if not supabase_url or not supabase_service_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in the environment or .env file")
    supabase_rest_upsert(surface_df_daily, "h3_aqi_surface_daily", supabase_url, supabase_service_key)
    print("✓ Upserted to h3_aqi_surface_daily")

    print("\n" + "=" * 70)
    print("✅ Pipeline complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
