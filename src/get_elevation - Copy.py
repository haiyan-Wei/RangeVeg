import ee
import os
import requests
import rasterio
from rasterio.transform import from_origin
import numpy as np
from utils import get_coordinates, create_polygon
from config import PROJECT_ROOT, FULL_OUTPUT_DIR
import geopandas as gpd

def process_and_export_elevation_data_for_polygons():
    collect0 = ee.Image('WWF/HydroSHEDS/03CONDEM')
    buffs = gpd.read_file(os.path.join(PROJECT_ROOT, 'inputs', 'Vectors', 'PRISM_EE.shp')).loc[:,['PK_EE', 'geometry']]
    buffs['PrimaryKey'] = buffs['PK_EE']
    proj = get_coordinates()

    for i, point in buffs.iterrows():
        folder_name = str(point['PrimaryKey'])
        folder_path_out = os.path.join(FULL_OUTPUT_DIR, 'elevation', folder_name)
        os.makedirs(folder_path_out, exist_ok=True)

        polygon = create_polygon(point, proj)

        print(f'Downloading elevation data for polygon {folder_name}')

        try:
            band1 = collect0.select(['b1'])
            url = band1.getDownloadURL({
                'scale': collect0.select('b1').projection().nominalScale().getInfo(),
                'crs': 'EPSG:4326',
                'region': polygon
            })

            response = requests.get(url)
            if response.status_code == 200:
                # Save the GeoTIFF file
                output_file = os.path.join(folder_path_out, 'elevation.tif')
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                print(f"Successfully downloaded elevation data for polygon {folder_name}")

                # Read the downloaded GeoTIFF and create a new one with correct metadata
                with rasterio.open(output_file) as src:
                    elevation_data = src.read(1)
                    transform = from_origin(src.bounds.left, src.bounds.top, src.res[0], src.res[1])

                    new_dataset = rasterio.open(
                        output_file,
                        'w',
                        driver='GTiff',
                        height=elevation_data.shape[0],
                        width=elevation_data.shape[1],
                        count=1,
                        dtype=elevation_data.dtype,
                        crs='+proj=longlat +datum=WGS84 +no_defs',
                        transform=transform,
                    )

                    new_dataset.write(elevation_data, 1)
                    new_dataset.close()

                print(f"Elevation data saved as GeoTIFF for polygon {folder_name}")
            else:
                print(f"Failed to download elevation data for polygon {folder_name}. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error processing elevation data for polygon {folder_name}: {str(e)}")
