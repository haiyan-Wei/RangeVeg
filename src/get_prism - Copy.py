import ee
from utils import get_buffs, get_coordinates
from config import PROJECT_ROOT, START_DATE, END_DATE
import geopandas as gpd
import os
def process_and_export_prism_data_for_polygons():
    
    collect0 = ee.ImageCollection('OREGONSTATE/PRISM/AN81m')
    proj = get_coordinates()


    buffs = gpd.read_file(os.path.join(PROJECT_ROOT, 'inputs', 'Vectors', 'PRISM_EE.shp')).loc[:,['PK_EE', 'geometry']]
    buffs['PrimaryKey'] = buffs['PK_EE']


    for i in range(len(buffs.index.tolist())):
        
        # select the polygon
        point = buffs.iloc[[i]]
        folder_name = str(point['PrimaryKey'].values[0])

        # filter collection by the polygons's bounds
        polygonBounds = point.total_bounds.tolist()

        # filter the image collection by the polygons's bounds
        polygon = ee.Geometry.Rectangle([polygonBounds[0],polygonBounds[1],polygonBounds[2],
                                            polygonBounds[3]], proj=proj, evenOdd=False)
        collectSR = collect0.filterBounds(polygon)

        # filter collection by the time range of interest. Here we would like to get 20 seasons before Summer 2020
        collectSR = collectSR.filterDate(START_DATE, END_DATE).sort('system:time_start')

        if collectSR.size().getInfo() != 0:

            # list of PRISM bands to iterate
            listOfBands = ['ppt', 'tmean']

            # iterate over the bands and export to GDrive folder
            for bands in listOfBands:
                band1 = collectSR.select([bands]).toBands()
                taskToexport = ee.batch.Export.image.toDrive(
                    image = band1,
                    crs = proj.getInfo()['wkt'],             
                    region = polygon,
                    description = bands,
                    scale = collect0.first().select('ppt').projection().nominalScale().getInfo(), # we use the original pixel size of the PRISM data which is about 4638 m
                    folder = folder_name)

                taskToexport.start()
        
                print('Request sent for', folder_name)
        
        else:
            print('No image exists for {}. Please check the region and date range'.format(folder_name))
