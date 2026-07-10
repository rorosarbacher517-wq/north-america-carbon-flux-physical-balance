import cdsapi
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize CDS API client
c = cdsapi.Client()

# Read site info
sites_df = pd.read_csv("./input_data/All_sites_with_three_dataset.csv")

# Variables to download
variables = [
    "2m_dewpoint_temperature",
    "2m_temperature",
    "surface_solar_radiation_downwards",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "surface_pressure",
    "total_precipitation",
    # "evaporation_from_vegetation_transpiration",
    # "evaporation_from_bare_soil",
    # "total_evaporation",
    # "potential_evaporation",
]

# 24 hours
hours = [f"{h:02d}:00" for h in range(24)]

# Output directory
base_out_dir = r"E:/HLS carbon flux/data/ERA5_Land_hourly/"
os.makedirs(base_out_dir, exist_ok=True)

# Valid year range
valid_start_year = 2013
valid_end_year = 2025

# ===== 可调参数：每组月份数（3 表示季度，4 表示四个月） =====
group_size = 12  # 改成 4 就是每四个月一次

def make_month_groups(group_size):
    """把 1-12 月分组"""
    return [list(range(i, min(i + group_size, 13))) for i in range(1, 13, group_size)]

month_groups = make_month_groups(group_size)

def download_variable_period(site_id, lat, lon, year, months, variable, out_dir):
    """Download one variable for a site for several months"""
    out_file = os.path.join(
        out_dir,
        f"{year}{months[0]:02d}-{months[-1]:02d}_{variable}.zip"
    )
    if os.path.exists(out_file):
        return f"{out_file} already exists, skip."

    print(f"# Downloading {site_id} {year} months {months} {variable} ...")

    request = {
        "variable": [variable],
        "year": str(year),
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in range(1, 32)],  # 31天都写，CDS自动忽略无效日期
        "time": hours,
        "data_format": "grib",
        "download_format": "zip",
        "area": [lat, lon, lat, lon]
    }

    try:
        c.retrieve("reanalysis-era5-land", request).download(out_file)
        return f"Finished: {site_id} {year} months {months} {variable}"
    except Exception as e:
        return f"Failed: {site_id} {year} months {months} {variable}, Error: {e}"


# Main loop: site -> year -> month group
for idx in range(70, 80):  # 控制测试范围
    row = sites_df.iloc[idx]
    site_id = row['SITE_ID']
    lat = row['Latitude']
    lon = row['Longitude']

    start_year = max(int(row['StartYear']), valid_start_year)
    end_year = min(int(row['EndYear']), valid_end_year)
    if start_year > end_year:
        print(f"Skip {site_id}, no data in {valid_start_year}-{valid_end_year} range.")
        continue

    site_dir = os.path.join(base_out_dir, site_id)
    os.makedirs(site_dir, exist_ok=True)

    for year in range(start_year, end_year + 1):
        year_dir = os.path.join(site_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)

        for months in month_groups:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(download_variable_period, site_id, lat, lon, year, months, variable, year_dir): variable
                    for variable in variables
                }
                for future in as_completed(futures):
                    print(future.result())
