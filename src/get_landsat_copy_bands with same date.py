import rasterio
import pandas as pd
import os
import ee
import requests
import zipfile
import io
from utils import create_polygon, get_coordinates
import geopandas as gpd
from config import PROJECT_ROOT, START_DATE, END_DATE, FULL_OUTPUT_DIR


def process_and_export_landsat_data_for_polygons():
    """ Process and export Landsat data for polygons 
    final output is a folder with the name of the polygon and a folder with the name of the polygon and a tif file for each band"""
    

    buffs = gpd.read_file(os.path.join(PROJECT_ROOT, 'inputs', 'Vectors', 'Area.shp')).loc[:,['PK_EE', 'geometry']]
    buffs['PrimaryKey'] = buffs['PK_EE']

    collect0 = get_landsat_collection()
    proj = get_coordinates()

    listOfBands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa']


    for i, point in buffs.iterrows():
        folder_name = str(point['PrimaryKey'])
        raw_folder_path = os.path.join(FULL_OUTPUT_DIR, 'landsat', 'raw', folder_name)
        folder_path_out = os.path.join(FULL_OUTPUT_DIR, 'landsat', folder_name)
        os.makedirs(raw_folder_path, exist_ok=True)
        os.makedirs(folder_path_out, exist_ok=True)

        polygon = create_polygon(point, proj)
        collectSR = filter_collection(collect0, polygon, START_DATE, END_DATE)

        print(f'There are {collectSR.size().getInfo()} Landsat images between {START_DATE} and {END_DATE} around polygon {folder_name}')

        if collectSR.size().getInfo() != 0:
            print(f'Downloading images for polygon {folder_name}')
            save_metadata(collectSR, folder_path_out)
            download_images(collectSR, polygon, raw_folder_path, listOfBands)
            # print(f'done')
        else:
            print(f'No images found for polygon {folder_name}')

    create_multiband_tifs(buffs, FULL_OUTPUT_DIR, listOfBands)


def filter_collection(collect0, polygon, start_date, end_date):
    return collect0.filterBounds(polygon).filterDate(start_date, end_date).sort('system:time_start')

def save_metadata(collectSR, folder_path_out):
    listOfDate = collectSR.aggregate_array('system:time_start').getInfo()
    listOfDate = [ee.Date(date).format('Y-MM-dd').getInfo() for date in listOfDate]
    listOfID = collectSR.aggregate_array('system:id').getInfo()
    listOfSAT = collectSR.aggregate_array('SPACECRAFT_ID').getInfo()

    df = pd.DataFrame({'date': listOfDate, 'Image_ID': listOfID, 'Satellite': listOfSAT})
    df.to_csv(os.path.join(folder_path_out, 'dates_SR.csv'), index=False)

def download_images(collectSR, polygon, raw_folder_path, listOfBands):
    image_list = collectSR.toList(collectSR.size())
    for j in range(collectSR.size().getInfo()):
        try:
            image = ee.Image(image_list.get(j))
            image_id = image.get('system:id').getInfo()
            image_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            
            # print(f"Downloading image {image_id} from {image_date}")
            
            image_folder = os.path.join(raw_folder_path, f"{image_date}_{image_id}")
            os.makedirs(image_folder, exist_ok=True)
            
            download_bands(image, polygon, image_folder, listOfBands, image_id)
        except Exception as e:
            print(f"Error processing image at index {j}: {str(e)}")

def download_bands(image, polygon, image_folder, listOfBands, image_id):
    for band in listOfBands:
        try:
            url = image.select(band).getDownloadURL({
                'scale': 30,
                'crs': 'EPSG:4269',
                'region': polygon
            })
            
            response = requests.get(url)
            if response.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(response.content))
                z.extractall(image_folder)
                print(f"Successfully downloaded and extracted {band} for image {image_id}")
            else:
                print(f"Failed to download {band} for image {image_id}. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error processing band {band} for image {image_id}: {str(e)}")

def create_multiband_tifs(buffs, output_dir, listOfBands):
    for i, point in buffs.iterrows():
        folder_name = str(point['PrimaryKey'])
        raw_folder_path = os.path.join(output_dir, 'landsat', 'raw', folder_name)
        folder_path_out = os.path.join(output_dir, 'landsat', folder_name)
        os.makedirs(folder_path_out, exist_ok=True)
        for band in listOfBands:
            print(f'creating multiband landsat tif for polygon {folder_name} and band {band}')            
            output_file_out = os.path.join(folder_path_out, f'{band}.tif')
            create_multiband_tif_rasterio(raw_folder_path, output_file_out, band)
            # print(f'done')


def create_multiband_tif_rasterio(input_dir, output_file, band):        
     
    all_bands = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(f"{band}.tif"):
                all_bands.append(os.path.join(root, file))
    
    all_bands.sort()  # Sort to ensure consistent date order

    # Read metadata of first file
    with rasterio.open(all_bands[0]) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    meta.update(count = len(all_bands))

    # Read each layer and write it to stack
    with rasterio.open(output_file, 'w', **meta) as dst:
        for id, layer in enumerate(all_bands, start=1):
            with rasterio.open(layer) as src1:
                dst.write_band(id, src1.read(1))           
                dst.set_band_description(id, os.path.basename(layer).split('_')[2])
    print(f'Multiband tif created for {output_file}, number of bands: {len(all_bands)}')

def get_landsat_collection():

    listOfSensors = [
        "LANDSAT/LT04/C02/T1_L2",  # Landsat 4
        "LANDSAT/LT05/C02/T1_L2",  # Landsat 5
        "LANDSAT/LE07/C02/T1_L2",  # Landsat 7
        "LANDSAT/LC08/C02/T1_L2",  # Landsat 8
        "LANDSAT/LC09/C02/T1_L2"   # Landsat 9 (if needed)
    ]

    collectAll = []
    for i in range(len(listOfSensors)):
        collect0 = ee.ImageCollection(listOfSensors[i])
        if i >= 3:  # For Landsat 8 and 9
            qual = 'QA_PIXEL'
            collect0 = collect0.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', qual], 
                                       ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa'])
        else:  # For Landsat 4-7
            qual = 'QA_PIXEL'
            collect0 = collect0.select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', qual], 
                                       ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa'])
        collectAll.append(collect0)

    collect0 = collectAll[0].merge(collectAll[1]).merge(collectAll[2]).merge(collectAll[3])
    return collect0