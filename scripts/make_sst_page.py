from pathlib import Path
from datetime import datetime, timedelta, timezone
from urllib.request import urlopen
import json

import numpy as np
import xarray as xr
import plotly.graph_objects as go
import plotly.io as pio
import copernicusmarine


# ============================================================
# Settings
# ============================================================

# Product:
#   GLOBAL_ANALYSISFORECAST_PHY_001_024
#
# Dataset:
#   Temperature, 6-hourly
#   cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i
#
# Variable:
#   thetao = sea water potential temperature
#
# Region:
#   latitude  : 20N to 50N
#   longitude : 120E to 170E
#
# Depth:
#   top layer around 0.49 m
#
# Grid thinning:
#   keep every 4th grid point
#   original 0.083 degree grid -> about 0.33 degree grid

PRODUCT_ID = "GLOBAL_ANALYSISFORECAST_PHY_001_024"
DATASET_ID = "cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i"
VARIABLE = "thetao"

LAT_MIN = 20
LAT_MAX = 50
LON_MIN = 120
LON_MAX = 170

DEPTH_MIN = 0
DEPTH_MAX = 1
TARGET_DEPTH = 0.49

FORECAST_DAYS = 10
SPATIAL_STRIDE = 4

TMP_DIR = Path("tmp")
PUBLIC_DIR = Path("public")

TMP_DIR.mkdir(exist_ok=True)
PUBLIC_DIR.mkdir(exist_ok=True)

RAW_NC = TMP_DIR / "sst_raw.nc"
LIGHT_NC = PUBLIC_DIR / "latest_sst_light.nc"
HTML_FILE = PUBLIC_DIR / "index.html"
META_FILE = PUBLIC_DIR / "metadata.json"


# ============================================================
# Time range
# ============================================================

now_utc = datetime.now(timezone.utc)

# Copernicus Marineの更新が遅れる場合があるため、
# 今日で失敗したら昨日、さらに一昨日で再試行します。
download_success = False
last_error = None
used_start_date = None
used_end_date = None


# ============================================================
# Download from Copernicus Marine
# ============================================================

for start_offset_days in [0, 1, 2]:
    start_date = now_utc.date() - timedelta(days=start_offset_days)
    end_date = start_date + timedelta(days=FORECAST_DAYS)

    start_datetime = f"{start_date}T00:00:00"
    end_datetime = f"{end_date}T00:00:00"

    print("Trying Copernicus Marine download")
    print(f"dataset_id     : {DATASET_ID}")
    print(f"variable       : {VARIABLE}")
    print(f"longitude      : {LON_MIN} to {LON_MAX}")
    print(f"latitude       : {LAT_MIN} to {LAT_MAX}")
    print(f"depth          : {DEPTH_MIN} to {DEPTH_MAX} m")
    print(f"time           : {start_datetime} to {end_datetime}")
    print(f"raw output     : {RAW_NC}")

    try:
        if RAW_NC.exists():
            RAW_NC.unlink()

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
            force_download=True,
        )

        download_success = True
        used_start_date = start_date
        used_end_date = end_date
        break

    except Exception as e:
        last_error = e
        print(f"Download failed for start date {start_date}: {e}")
        print("Trying older start date...")


if not download_success:
    raise RuntimeError(f"All download attempts failed. Last error: {last_error}")

print("Download complete.")


# ============================================================
# Read NetCDF
# ============================================================

print("Opening NetCDF with xarray...")
ds = xr.open_dataset(RAW_NC)
print(ds)

if VARIABLE not in ds:
    raise KeyError(f"Variable '{VARIABLE}' was not found in the dataset.")

# Select nearest top layer around 0.49 m.
sst = ds[VARIABLE].sel(depth=TARGET_DEPTH, method="nearest")
selected_depth = float(sst["depth"].values)

print(f"Selected depth: {selected_depth:.3f} m")


# ============================================================
# Thin grid
# ============================================================

# 0.083度格子から「1点残して3点飛ばす」
# つまり4点に1点を残す。
sst_light = sst.isel(
    latitude=slice(None, None, SPATIAL_STRIDE),
    longitude=slice(None, None, SPATIAL_STRIDE),
)

out_ds = sst_light.to_dataset(name=VARIABLE)

out_ds.attrs["source_product"] = PRODUCT_ID
out_ds.attrs["source_dataset"] = DATASET_ID
out_ds.attrs["variable"] = VARIABLE
out_ds.attrs["region"] = f"{LAT_MIN}-{LAT_MAX}N, {LON_MIN}-{LON_MAX}E"
out_ds.attrs["selected_depth_m"] = selected_depth
out_ds.attrs["spatial_stride"] = SPATIAL_STRIDE
out_ds.attrs["generated_at_utc"] = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

out_ds.to_netcdf(LIGHT_NC)

print(f"Saved lightweight NetCDF: {LIGHT_NC}")


# ============================================================
# Prepare data for Plotly
# ============================================================

print("Preparing Plotly animation...")

lons = sst_light["longitude"].values
lats = sst_light["latitude"].values
times = sst_light["time"].values
z_all = sst_light.values

# 配列の型を軽くする
z_all = z_all.astype("float32")

# ============================================================
# Time labels in JST
# ============================================================

JST = timezone(timedelta(hours=9))


def numpy_datetime64_to_datetime_utc(t):
    """
    numpy.datetime64 を Python datetime UTC に変換する。
    """
    seconds = t.astype("datetime64[s]").astype("int64")
    return datetime.fromtimestamp(int(seconds), tz=timezone.utc)


time_dt_utc = [numpy_datetime64_to_datetime_utc(t) for t in times]
time_dt_jst = [t.astimezone(JST) for t in time_dt_utc]

# 年はスライダーに毎回出さず、タイトルや説明欄で1回だけ表示する
years_jst = sorted({t.year for t in time_dt_jst})
if len(years_jst) == 1:
    year_label_jst = str(years_jst[0])
else:
    year_label_jst = f"{years_jst[0]}–{years_jst[-1]}"

# スライダーに表示する短い時刻ラベル
# 例：06/26 09:00
time_labels = [t.strftime("%m/%d %H:%M") for t in time_dt_jst]

# metadata用にはJSTで少し詳しく残す
time_labels_jst_full = [t.strftime("%Y-%m-%d %H:%M JST") for t in time_dt_jst]

# 全時刻で色スケールを固定
zmin = float(np.nanpercentile(z_all, 2))
zmax = float(np.nanpercentile(z_all, 98))

print(f"Number of time steps: {len(time_labels)}")
print(f"Longitude points: {len(lons)}")
print(f"Latitude points : {len(lats)}")
print(f"Color range: {zmin:.2f} to {zmax:.2f}")


# ============================================================
# Land polygons from Natural Earth
# ============================================================

print("Preparing land polygons...")

LAND_GEOJSON_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_10m_land.geojson"
)


def ring_intersects_domain(ring, lon_min, lon_max, lat_min, lat_max):
    """
    GeoJSONの1つのringが表示領域と交差するか簡易判定する。
    """
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]

    if max(xs) < lon_min:
        return False
    if min(xs) > lon_max:
        return False
    if max(ys) < lat_min:
        return False
    if min(ys) > lat_max:
        return False

    return True


def make_land_traces():
    """
    Natural Earthの陸域GeoJSONを取得し、
    PlotlyのScatter traceとして返す。

    ここではcartopy/geopandasを使わず、
    GitHub Actionsで軽く動くように標準ライブラリでGeoJSONを読む。
    """
    land_traces = []

    try:
        with urlopen(LAND_GEOJSON_URL, timeout=60) as response:
            land_geojson = json.load(response)

        for feature in land_geojson["features"]:
            geom = feature["geometry"]
            geom_type = geom["type"]
            coords = geom["coordinates"]

            if geom_type == "Polygon":
                polygons = [coords]
            elif geom_type == "MultiPolygon":
                polygons = coords
            else:
                continue

            for polygon in polygons:
                # polygon[0] は外側の境界線。
                # 穴はここでは省略する。
                outer_ring = polygon[0]

                if not ring_intersects_domain(
                    outer_ring, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX
                ):
                    continue

                x = [p[0] for p in outer_ring]
                y = [p[1] for p in outer_ring]

                land_traces.append(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        fill="toself",
                        fillcolor="rgba(238, 232, 218, 0.95)",
                        line=dict(color="rgba(80, 80, 80, 0.9)", width=0.8),
                        hoverinfo="skip",
                        showlegend=False,
                        name="Land",
                    )
                )

        print(f"Land polygon traces: {len(land_traces)}")

    except Exception as e:
        # 陸域GeoJSON取得に失敗しても、SSTページ自体は作成する
        print(f"Warning: failed to load land polygons: {e}")
        land_traces = []

    return land_traces


land_traces = make_land_traces()


# ============================================================
# Helper traces
# ============================================================

def make_heatmap_trace(z):
    """
    SST塗りつぶし用Heatmap。
    """
    return go.Heatmap(
        z=z,
        x=lons,
        y=lats,
        zmin=zmin,
        zmax=zmax,
        colorscale="Turbo",
        colorbar=dict(title="thetao (°C)"),
        hovertemplate=(
            "Lon: %{x:.2f}<br>"
            "Lat: %{y:.2f}<br>"
            "SST: %{z:.2f} °C<extra></extra>"
        ),
        name="SST",
    )


def make_18c_contour_trace(z):
    """
    18℃だけの等温線。
    """
    return go.Contour(
        z=z,
        x=lons,
        y=lats,
        contours=dict(
            start=18,
            end=18,
            size=1,
            coloring="lines",
            showlabels=True,
            labelfont=dict(size=11, color="black"),
        ),
        line=dict(color="black", width=2),
        showscale=False,
        hoverinfo="skip",
        name="18°C isotherm",
        connectgaps=False,
    )


# ============================================================
# Create Plotly figure
# ============================================================

# traceの順番：
# 0: SST heatmap
# 1: 18℃等温線
# 2以降: 陸域ポリゴン
#
# frameでは0番と1番だけを更新し、陸域は固定背景として残す。
fig = go.Figure(
    data=[
        make_heatmap_trace(z_all[0]),
        make_18c_contour_trace(z_all[0]),
        *land_traces,
    ]
)

fig.frames = [
    go.Frame(
        data=[
            make_heatmap_trace(z_all[i]),
            make_18c_contour_trace(z_all[i]),
        ],
        traces=[0, 1],
        name=time_labels[i],
    )
    for i in range(len(time_labels))
]

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
        f"Northwest Pacific SST Forecast ({year_label_jst} JST)<br>"
        f"{VARIABLE} at {selected_depth:.3f} m, "
        f"{LAT_MIN}–{LAT_MAX}°N, {LON_MIN}–{LON_MAX}°E"
    ),
    xaxis_title="Longitude",
    yaxis_title="Latitude",
    width=1050,
    height=720,
    margin=dict(l=60, r=40, t=95, b=145),
    xaxis=dict(
        range=[LON_MIN, LON_MAX],
        constrain="domain",
    ),
    yaxis=dict(
        range=[LAT_MIN, LAT_MAX],
        scaleanchor="x",
        scaleratio=1,
    ),
    sliders=[
        {
            "active": 0,
            "currentvalue": {
                "prefix": f"Time ({year_label_jst} JST): ",
                "font": {"size": 14},
            },
            "pad": {"t": 50},
            "steps": slider_steps,
        }
    ],
    updatemenus=[
        {
            "type": "buttons",
            "direction": "left",
            "x": 0.05,
            "y": -0.13,
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

plot_html = pio.to_html(
    fig,
    include_plotlyjs="cdn",
    full_html=False,
)


# ============================================================
# metadata.json
# ============================================================

metadata = {
    "generated_at_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
    "generated_at_jst": now_utc.astimezone(JST).strftime("%Y-%m-%d %H:%M:%S JST"),
    "product_id": PRODUCT_ID,
    "product_name": "Global Ocean Physics Analysis and Forecast",
    "dataset_id": DATASET_ID,
    "dataset_name": "Temperature, 6-hourly",
    "variable": VARIABLE,
    "variable_description": "sea water potential temperature",
    "longitude_min": LON_MIN,
    "longitude_max": LON_MAX,
    "latitude_min": LAT_MIN,
    "latitude_max": LAT_MAX,
    "requested_depth_min_m": DEPTH_MIN,
    "requested_depth_max_m": DEPTH_MAX,
    "target_depth_m": TARGET_DEPTH,
    "selected_depth_m": selected_depth,
    "forecast_days": FORECAST_DAYS,
    "start_datetime_utc": f"{used_start_date}T00:00:00",
    "end_datetime_utc": f"{used_end_date}T00:00:00",
    "time_steps_jst": time_labels_jst_full,
    "time_year_label_jst": year_label_jst,
    "number_of_time_steps": len(time_labels),
    "original_grid_degree": 0.083,
    "spatial_stride": SPATIAL_STRIDE,
    "approx_output_grid_degree": 0.083 * SPATIAL_STRIDE,
    "longitude_points_after_thinning": int(len(lons)),
    "latitude_points_after_thinning": int(len(lats)),
    "color_scale_min": zmin,
    "color_scale_max": zmax,
    "contour_levels_celsius": [18],
    "land_polygon_source": LAND_GEOJSON_URL,
    "netcdf_file": LIGHT_NC.name,
    "html_file": HTML_FILE.name,
    "note": (
        "Latitude and longitude are thinned by keeping every 4th grid point. "
        "The original 0.083 degree grid becomes approximately 0.33 degree. "
        "Time labels are shown in JST. The 18°C isotherm is drawn as a black contour."
    ),
}

META_FILE.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"Saved metadata JSON: {META_FILE}")


# ============================================================
# index.html
# ============================================================

html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>Northwest Pacific SST Forecast</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 2rem;
      line-height: 1.6;
      color: #222;
    }}
    h1 {{
      margin-bottom: 0.3rem;
    }}
    .meta {{
      background: #f6f8fa;
      border: 1px solid #d0d7de;
      border-radius: 8px;
      padding: 1rem;
      margin: 1rem 0 1.5rem 0;
    }}
    code {{
      background: #f2f2f2;
      padding: 0.1rem 0.3rem;
      border-radius: 4px;
    }}
    a {{
      color: #0969da;
    }}
  </style>
</head>
<body>
  <h1>Northwest Pacific SST Forecast</h1>

  <div class="meta">
    <p>
      Data source:
      Copernicus Marine
      <code>{PRODUCT_ID}</code>
    </p>
    <ul>
      <li>Dataset: <code>{DATASET_ID}</code></li>
      <li>Variable: <code>{VARIABLE}</code></li>
      <li>Region: {LAT_MIN}–{LAT_MAX}°N, {LON_MIN}–{LON_MAX}°E</li>
      <li>Selected depth: {selected_depth:.3f} m</li>
      <li>Time range: {used_start_date} to {used_end_date} UTC</li>
      <li>Display time zone: JST</li>
      <li>Grid thinning: keep every {SPATIAL_STRIDE}th point</li>
      <li>18°C isotherm: black contour</li>
      <li>Generated at: {now_utc.astimezone(JST).strftime("%Y-%m-%d %H:%M:%S JST")}</li>
    </ul>
    <p>
      <a href="latest_sst_light.nc">Download lightweight NetCDF</a>
      /
      <a href="metadata.json">View metadata JSON</a>
    </p>
  </div>

  <h2>Interactive SST animation ({year_label_jst} JST)</h2>
  <p>
    Use the slider or Play/Pause buttons to view changes through time.
    Time labels are shown in JST. The black line indicates the 18°C isotherm.
  </p>

  {plot_html}

</body>
</html>
"""

HTML_FILE.write_text(html, encoding="utf-8")

print(f"Saved HTML: {HTML_FILE}")