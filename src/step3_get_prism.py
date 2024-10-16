import ee
import os
import io
import zipfile
import requests
import pandas as pd
import geopandas as gpd
from step0_utils import get_coordinates, create_polygon
from config import PROJECT_ROOT, START_DATE, END_DATE, FULL_OUTPUT_DIR

def download_prism_data():

    collect0 = ee.ImageCollection('OREGONSTATE/PRISM/AN81m')
    df_aoi_buffs = gpd.read_file(os.path.join(PROJECT_ROOT, 'inputs', 'Vectors', 'PRISM_EE.shp'))
    df_aoi_buffs = df_aoi_buffs.loc[:,['PK_EE', 'geometry']]
    df_aoi_buffs['PrimaryKey'] = df_aoi_buffs['PK_EE']
    proj = get_coordinates()

    listOfBands = ['ppt', 'tmean']

    for i, point in df_aoi_buffs.iterrows():
        folder_name = str(point['PK_EE'])
        raw_folder_path = os.path.join(FULL_OUTPUT_DIR, 'prism', 'raw', folder_name)
        folder_path_out = os.path.join(FULL_OUTPUT_DIR, 'prism', folder_name)
        os.makedirs(raw_folder_path, exist_ok=True)
        os.makedirs(folder_path_out, exist_ok=True)

        polygon = create_polygon(point, proj)
        collectSR = filter_collection(collect0, polygon, START_DATE, END_DATE)

        print(f'There are {collectSR.size().getInfo()} PRISM images between {START_DATE} and {END_DATE} around polygon {folder_name}')

        if collectSR.size().getInfo() != 0:
            print(f'Downloading PRISM data for polygon {folder_name}')
            save_metadata_csv(collectSR, folder_path_out)
            download_images(collectSR, polygon, raw_folder_path, listOfBands, proj, collect0)
            # print(f'Done downloading PRISM data for polygon {folder_name}')
        else:
            print(f'No PRISM data found for polygon {folder_name}')

    create_multiband_tifs(df_aoi_buffs, FULL_OUTPUT_DIR, listOfBands, proj)

def filter_collection(collect0, polygon, start_date, end_date):
    return collect0.filterBounds(polygon).filterDate(start_date, end_date).sort('system:time_start')

def save_metadata_csv(collectSR, folder_path_out):
    listOfDate = collectSR.aggregate_array('system:time_start').getInfo()
    listOfDate = [ee.Date(date).format('Y-MM-dd').getInfo() for date in listOfDate]
    listOfID = collectSR.aggregate_array('system:id').getInfo()

    df = pd.DataFrame({'date': listOfDate, 'Image_ID': listOfID})
    df.to_csv(os.path.join(folder_path_out, 'dates_PRISM.csv'), index=False)

def download_images(collectSR, polygon, raw_folder_path, listOfBands, proj, collect0):
    image_list = collectSR.toList(collectSR.size())
    for j in range(collectSR.size().getInfo()):
        try:
            image = ee.Image(image_list.get(j))
            image_id = image.get('system:id').getInfo()
            image_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            
            image_folder = os.path.join(raw_folder_path, f"{image_date}_{image_id}")
            os.makedirs(image_folder, exist_ok=True)
            
            download_bands(image, polygon, image_folder, listOfBands, image_id, image_date, proj, collect0)
        except Exception as e:
            print(f"Error processing PRISM image at index {j}: {str(e)}")

def download_bands(image, polygon, image_folder, listOfBands, image_id, image_date, proj, collect0):
    for band in listOfBands:
        try:
            url = image.select(band).getDownloadURL({
                'scale': collect0.first().select('ppt').projection().nominalScale().getInfo(),
                'crs': proj,  # Use the provided projection
                'region': polygon
            })
            
            response = requests.get(url)
            if response.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(response.content))
                z.extractall(image_folder)
                # Rename the extracted file to include the date
                old_filename = [name for name in os.listdir(image_folder) if name.endswith(f"{band}.tif")][0]
                new_filename = f"{image_date[:7]}_{band}.tif"  # YYYYMM_bandname.tif
                os.rename(os.path.join(image_folder, old_filename), os.path.join(image_folder, new_filename))
                print(f"   Successfully downloaded and extracted {new_filename} for PRISM image {image_id}")
            else:
                print(f"   Failed to download {band} for PRISM image {image_id}. Status code: {response.status_code}")
        except Exception as e:
            print(f"   Error processing band {band} for PRISM image {image_id}: {str(e)}")

def create_multiband_tifs(df_aoi_buffs, output_dir, listOfBands, proj):
    for i, point in df_aoi_buffs.iterrows():
        folder_name = str(point['PK_EE'])
        raw_folder_path = os.path.join(output_dir, 'prism', 'raw', folder_name)
        folder_path_out = os.path.join(output_dir, 'prism', folder_name)
        os.makedirs(folder_path_out, exist_ok=True)
        for band in listOfBands:
            print(f'Creating multiband tif for polygon {folder_name} and PRISM band {band}')            
            output_file_out = os.path.join(folder_path_out, f'{band}.tif')
            create_multiband_tif_rasterio(raw_folder_path, output_file_out, band, proj)
            # print(f'Done creating multiband tif for PRISM band {band}')

def create_multiband_tif_rasterio(input_dir, output_file, band, proj):
    import rasterio
    from rasterio.crs import CRS
    
    all_bands = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(f"_{band}.tif"):
                all_bands.append(os.path.join(root, file))
    
    all_bands.sort()  # Sort to ensure consistent date order

    # Read metadata of first file
    with rasterio.open(all_bands[0]) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers and the new projection
    meta.update(count=len(all_bands))
    
    # Convert the Earth Engine projection to a CRS object
    ee_proj = proj.getInfo()
    crs = CRS.from_wkt(ee_proj['wkt'])
    meta.update(crs=crs)

    # Read each layer and write it to stack
    with rasterio.open(output_file, 'w', **meta) as dst:
        for id, layer in enumerate(all_bands, start=1):
            with rasterio.open(layer) as src1:
                dst.write_band(id, src1.read(1))           
                # Use the filename (without extension) as the band description
                band_description = os.path.splitext(os.path.basename(layer))[0]
                dst.set_band_description(id, band_description)
    
    print(f'Multiband tif created for {output_file}, number of bands: {len(all_bands)}, with projection: {crs.to_string()}')
