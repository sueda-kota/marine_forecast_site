from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import shutil

import copernicusmarine
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


# ============================================================
# Copernicus Marine SST forecast settings
# ============================================================

# Product:
#   GLOBAL_ANALYSISFORECAST_PHY_001_024
#
# Dataset:
#   Temperature, 6-hourly
#   cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i
#
# Variable:
#   thetao = sea water potential temperature [degC]
#
# Region:
#   latitude  : 20N to 50N
#   longitude : 120E to 170E
#
# Depth:
#   top layer around 0.49 m
#
# Grid thinning:
#   original grid: about 0.083 deg
#   keep every 4th grid point: 0.083 * 4 = about 0.33 deg
#   data amount becomes roughly 1/16 for horizontal dimensions
#   because both latitude and longitude are thinned by 4.
#
# Note:
#   Copernicus Marine subset can subset by variable, time, depth,
#   longitude and latitude. Grid stride is applied after download
#   using xarray.


DATASET_ID = "cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i"
VARIABLE = "thetao"

LON_MIN = 120
LON_MAX = 170
LAT_MIN = 20
LAT_MAX = 50

# Request a small depth range around the first layer.
# The first depth is usually around 0.49 m, but exact value may be 0.494...
DEPTH_MIN = 0
DEPTH_MAX = 1
TARGET_DEPTH = 0.49

# Forecast range
# 6-hourly data, 10 days.
FORECAST_DAYS = 10

# Keep every 4th grid point.
# This means 3 grid points are skipped between kept points.
SPATIAL_STRIDE = 4

# Output directories
TMP_DIR = Path("tmp")
PUBLIC_DIR = Path("public")

TMP_DIR.mkdir(exist_ok=True)
PUBLIC_DIR.mkdir(exist_ok=True)

RAW_NC = TMP_DIR / "sst_raw.nc"
LIGHT_NC = PUBLIC_DIR / "latest_sst_light.nc"
PNG_FILE = PUBLIC_DIR / "sst_map.png"
HTML_FILE = PUBLIC_DIR / "index.html"
META_FILE = PUBLIC_DIR / "metadata.json"


# ============================================================
# Time range
# ============================================================

now_utc = datetime.now(timezone.utc)

# Use today's 00 UTC as start.
# If the dataset has not been updated yet, this may fail.
# In that case, change start_date to now_utc.date() - timedelta(days=1).
start_date = now_utc.date()
end_date = start_date + timedelta(days=FORECAST_DAYS)

start_datetime = f"{start_date}T00:00:00"
end_datetime = f"{end_date}T00:00:00"


# ============================================================
# Download from Copernicus Marine
# ============================================================

print("Downloading Copernicus Marine SST forecast...")
print(f"dataset_id     : {DATASET_ID}")
print(f"variable       : {VARIABLE}")
print(f"longitude      : {LON_MIN} to {LON_MAX}")
print(f"latitude       : {LAT_MIN} to {LAT_MAX}")
print(f"depth          : {DEPTH_MIN} to {DEPTH_MAX} m")
print(f"time           : {start_datetime} to {end_datetime}")
print(f"raw output     : {RAW_NC}")

copernicusmarine.subset(
    dataset_id=DATASET_ID,
    variables=[VARIABLE],
    minimum_longitude=LON_MIN,
    maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN,
    maximum_latitude=LAT_MAX,
    minimum_depth=DEPTH_MIN,
    maximum_depth=DEPTH_MAX,
    start_datetime=start_datetime,
    end_datetime=end_datetime,
    output_filename=str(RAW_NC),
    file_format="netcdf",
    overwrite=True,
    netcdf_compression_level=1,
)

print("Download complete.")


# ============================================================
# Read NetCDF and select the top layer
# ============================================================

print("Opening NetCDF with xarray...")
ds = xr.open_dataset(RAW_NC)

print(ds)

# Select nearest top layer around 0.49 m.
# This is safer than assuming the exact depth coordinate is exactly 0.49.
sst = ds[VARIABLE].sel(depth=TARGET_DEPTH, method="nearest")

selected_depth = float(sst["depth"].values)
print(f"Selected depth: {selected_depth:.3f} m")


# ============================================================
# Thin grid resolution
# ============================================================

# Keep every 4th point in latitude and longitude.
# Original 0.083 deg grid becomes about 0.33 deg.
sst_light = sst.isel(
    latitude=slice(None, None, SPATIAL_STRIDE),
    longitude=slice(None, None, SPATIAL_STRIDE),
)

# Convert DataArray to Dataset for NetCDF output.
out_ds = sst_light.to_dataset(name=VARIABLE)

# Add useful metadata.
out_ds.attrs["source_product"] = "GLOBAL_ANALYSISFORECAST_PHY_001_024"
out_ds.attrs["source_dataset"] = DATASET_ID
out_ds.attrs["variable"] = VARIABLE
out_ds.attrs["region"] = f"{LAT_MIN}-{LAT_MAX}N, {LON_MIN}-{LON_MAX}E"
out_ds.attrs["selected_depth_m"] = selected_depth
out_ds.attrs["spatial_stride"] = SPATIAL_STRIDE
out_ds.attrs["generated_at_utc"] = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

# Save lightweight NetCDF for the website.
out_ds.to_netcdf(LIGHT_NC)
print(f"Saved lightweight NetCDF: {LIGHT_NC}")


# ============================================================
# Interactive Plotly animation for all time steps
# ============================================================

print("Creating interactive Plotly animation...")

# Plotly用の座標・値を取り出す
lons = sst_light["longitude"].values
lats = sst_light["latitude"].values
times = sst_light["time"].values
z_all = sst_light.values  # shape: time, latitude, longitude

# 色スケールを全時刻で固定する
# 極端値の影響を少し避けるため、2〜98パーセンタイルを使用
zmin = float(np.nanpercentile(z_all, 2))
zmax = float(np.nanpercentile(z_all, 98))

time_labels = [str(t)[:19].replace("T", " ") for t in times]

# 初期表示
fig = go.Figure(
    data=[
        go.Heatmap(
            z=z_all[0],
            x=lons,
            y=lats,
            zmin=zmin,
            zmax=zmax,
            colorscale="Turbo",
            colorbar=dict(title="thetao (°C)"),
        )
    ]
)

# 各時刻のフレームを作成
fig.frames = [
    go.Frame(
        data=[
            go.Heatmap(
                z=z_all[i],
                x=lons,
                y=lats,
                zmin=zmin,
                zmax=zmax,
                colorscale="Turbo",
                colorbar=dict(title="thetao (°C)"),
            )
        ],
        name=time_labels[i],
    )
    for i in range(len(time_labels))
]

# スライダー
slider_steps = [
    {
        "method": "animate",
        "label": time_labels[i],
        "args": [
            [time_labels[i]],
            {
                "mode": "immediate",
                "frame": {"duration": 0, "redraw": True},
                "transition": {"duration": 0},
            },
        ],
    }
    for i in range(len(time_labels))
]

fig.update_layout(
    title=(
        f"Northwest Pacific SST Forecast<br>"
        f"thetao at {selected_depth:.3f} m, "
        f"{LAT_MIN}–{LAT_MAX}°N, {LON_MIN}–{LON_MAX}°E"
    ),
    xaxis_title="Longitude",
    yaxis_title="Latitude",
    width=1000,
    height=650,
    sliders=[
        {
            "active": 0,
            "currentvalue": {"prefix": "Time: "},
            "pad": {"t": 50},
            "steps": slider_steps,
        }
    ],
    updatemenus=[
        {
            "type": "buttons",
            "direction": "left",
            "x": 0.1,
            "y": -0.12,
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": 500, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                        },
                    ],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                },
            ],
        }
    ],
)

# HTMLとして保存
# include_plotlyjs="cdn" にするとHTMLが軽くなります。
# 完全オフライン対応にしたい場合は include_plotlyjs=True にします。
fig.write_html(
    HTML_FILE,
    include_plotlyjs="cdn",
    full_html=True,
)

print(f"Saved interactive HTML: {HTML_FILE}")