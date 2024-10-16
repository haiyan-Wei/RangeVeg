import ee
import os
import requests
import rasterio
from rasterio.transform import from_origin
import numpy as np
from step0_utils import get_coordinates, create_polygon
from config import PROJECT_ROOT, FULL_OUTPUT_DIR
import geopandas as gpd


def get_elevation():
    collect0 = ee.Image('WWF/HydroSHEDS/03CONDEM')
    buffs = gpd.read_file(os.path.join(PROJECT_ROOT, 'inputs', 'Vectors', 'PRISM_EE.shp')).loc[:,['PK_EE', 'geometry']]
    buffs['PrimaryKey'] = buffs['PK_EE']
    proj = get_coordinates()

    for i in range(len(buffs.index.tolist())):
        point = buffs.iloc[[i]]
        folder_name = str(point['PrimaryKey'].values[0])
        folder_path_out = os.path.join(FULL_OUTPUT_DIR, 'elevation', folder_name)
        os.makedirs(folder_path_out, exist_ok=True)

        polygon = create_polygon(point.iloc[0], proj)

        print(f'Downloading elevation data for polygon {folder_name}')

        try:
            band1 = collect0.select(['b1'])
            url = band1.getDownloadURL({
                'scale': collect0.select('b1').projection().nominalScale().getInfo(),
                'crs': proj.getInfo()['wkt'],
                'region': polygon,
                'format': 'GEO_TIFF'
            })

            response = requests.get(url)
            if response.status_code == 200:
                output_file = os.path.join(folder_path_out, 'elevation.tif')
                with open(output_file, 'wb') as f:
                    f.write(response.content)

                print(f"Successfully downloaded elevation data for polygon {folder_name}")

                # Verify the downloaded file
                try:
                    with rasterio.open(output_file) as src:
                        print(f"Elevation data saved as GeoTIFF for polygon {folder_name}")
                        print(f"File info: {src.meta}")
                except rasterio.errors.RasterioIOError as e:
                    print(f"Error opening the downloaded file: {str(e)}")
                    print("The file might not be a valid GeoTIFF. Saving raw content for inspection.")
                    with open(os.path.join(folder_path_out, 'elevation_raw.bin'), 'wb') as f:
                        f.write(response.content)

            else:
                print(f"Failed to download elevation data for polygon {folder_name}. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error processing elevation data for polygon {folder_name}: {str(e)}")
            import traceback
            print(traceback.format_exc())

        print('Request completed for', folder_name)



