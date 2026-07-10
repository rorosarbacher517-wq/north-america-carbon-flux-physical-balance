import os
import cdsapi
import pandas as pd
import numpy as np
import xarray as xr
from zipfile import ZipFile
import urllib3

# Disable warnings for data download via API
urllib3.disable_warnings()

# API Configuration
URL = 'https://ads.atmosphere.copernicus.eu/api'
KEY = '2eb3201e-9a07-47d6-8ed6-cd4b696590f9'
DATADIR = 'E:/HLS carbon flux/data/Sites data/Ameriflux_sites_carbon_dioxide/'
input_path = 'E:/HLS carbon flux/data/Sites data/Ameriflux_site_csv_shp/Ameriflux_site_list.csv'

# Load the site data
site_df = pd.read_csv(input_path)

# Create a CDS API client
client = cdsapi.Client(url=URL, key=KEY)

# Iterate through each site
for index, row in site_df.iterrows():
    site_id = row['Site ID']
    site_name = row['Name']
    lat = row['Lat']
    lon = row['Long']
    start_year = int(row['AmeriFlux BASE Start'])
    end_year = int(row['AmeriFlux BASE End'])

    # Create directory for the site if it doesn't exist
    site_dir = os.path.join(DATADIR, site_name.replace(" ", "_"))
    os.makedirs(site_dir, exist_ok=True)

    # Retrieve data for each year
    for year in range(start_year, end_year + 1):
        # Create the request according to the new structure using satellite observation
        request = {
            "variable": "carbon_dioxide",
            "input_observations": "satellite",
            "time_aggregation": "instantaneous",
            "version": "latest",
            "year": [str(year)],
            "month": [f"{m:02d}" for m in range(1, 13)],  # All months
        }

        try:
            # Make the request
            print(f"Downloading data for {site_name} in {year} ...")
            response = client.retrieve("cams-global-greenhouse-gas-inversion", request)
            zip_filename = os.path.join(site_dir, f'{year}_CO2_data.zip')
            response.download(zip_filename)

            # Extract the ZIP file
            with ZipFile(zip_filename, 'r') as zipObj:
                zipObj.extractall(path=site_dir)

            # Load the NetCDF file from the extracted folder
            nc_filename = os.path.join(site_dir, f'{year}_CO2_data.nc')  # Replace with correct file naming
            ds = xr.open_dataset(nc_filename)

            # Extract and process variable
            co2_data = ds['carbon_dioxide']  # Assume 'carbon_dioxide' is the variable name
            co2_df = co2_data.to_dataframe().reset_index()  # Convert to DataFrame

            # Add site details to DataFrame
            co2_df['Site ID'] = site_id
            co2_df['Site Name'] = site_name
            co2_df['Latitude'] = lat
            co2_df['Longitude'] = lon

            # Save DataFrame to CSV
            csv_filename = os.path.join(site_dir, f'{year}_CO2_data.csv')
            co2_df.to_csv(csv_filename, index=False)
            print(f"Saved data for {site_name} to {csv_filename}")

        except Exception as e:
            print(f"Error retrieving data for {site_name} in {year}: {e}")

print("All data processing is completed!")
