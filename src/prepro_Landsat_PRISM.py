import rasterio
import rasterio.mask
from rasterio.vrt import WarpedVRT
import numpy as np
import pandas as pd
import shutil
import os
import geopandas as gpd
import multiprocessing as mp
from scipy.stats import linregress


# here we declare the directory to required data
 
# 1- the empty grid polygons you wish to make a map for
# modify the directory to the shapefile of the empty grid polygons (current directory is just an example)
# Please note that each polygon of the grid must have two attributes: 
# - PrimaryKey: a unique ID for the polygon
# - PK_EE: the name of the Landsat data that the polygon is within that
extent_all = gpd.read_file('Vectors/Polygons_withEEid.shp')

# 2- the table in csv format: 
# modify the directory to the csv file (current directory is just an example)
# each row is related to a polygon with a unique PrimaryKey. The following
# attributes must be captured for each polygon:
# - CLIGEN station ID
# - slope and slope shape
# - soil texture
# - G, DIST, POR, SMAX, FRAC1-5 from soil texture table
# - Latitude and Longitude of its center in NAD83
dataset = pd.read_csv('data/tables/dataset.csv')

# 3- the precipitation(mm) and temperature mean(celsius) tif raster files from PRISM
# modify the directories to the tif files (current directories are just examples)
ppt = rasterio.open('PRISM/ppt.tif')
tmean = rasterio.open('PRISM/tmean.tif')

# 4- the central points of the grid polygons with the PrimaryKey attribute assigned
# modify the directory to the shapefile of the points (current directory is just an example)
points_all = gpd.read_file('Vectors/Points.shp')


# here we declare the year and month of the final map
year_last='2020'
month_last='02'

# here we declare the directories to the output preprocessed data
# 1- a directory to save the temporary files during the preprocessing steps
directory_temp = 'prepro/'

# 2- a directory to save the final preprocessed Landsat and PRISM data in npz
# format. This directory must have two subdirectories:
# - Landsat
# - PRISM
directory_final = 'Final_data/'


# here we declare the number of cpu cores to work in parallel for data preprocessing
# modify the this number (current number is just an example)
number_of_cpu_cores = 15







# these are the regression parameters of the 6 bands for harmonizing
# Landsat 8 with PLI with the previous ETM/TM (see table 2 of 
# https://doi.org/10.1016/j.rse.2015.12.024)

intrcp = [0.0183, 0.0123, 0.0123, 0.0448, 0.0306, 0.0116]
intrcp = np.multiply(intrcp,10000)

slopes = [0.8850, 0.9317, 0.9372, 0.8339, 0.8639, 0.9165]

bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7']

# This dictionary maps the data quality codes of OLI to the data quality codes of Landsat ETM/TM
OLItoETM = {0:0, 1:1, 1346:1, 1348:1, 1350:1, 1352:1, 322:66, 386:130, 324:68, 388:132, 328:72, 392:136, 
            336:80, 368:112, 400:144, 432:176, 352:96, 416:160, 480:224, 834:224, 898:224, 836:224, 
            900:224, 840:224, 904:224, 848:224, 880:224, 912:224, 944:224, 864:224, 880:224, 928:224, 
            944:224, 992:224}

# clear pixel codes from Landsat ETM/TM documentation
clearCodes = [66, 68, 72, 80, 96, 112]


def prePro1(rowGdrive, year_last=year_last, month_last=month_last):
    if not os.path.exists(directory_temp + str(rowGdrive.PrimaryKey)):
        os.makedirs(directory_temp + str(rowGdrive.PrimaryKey))
        
    dir_EE = 'Landsat/{}/'.format(rowGdrive.PK_EE)

    dates = pd.read_csv(dir_EE + 'dates_SR.csv')
    
    with WarpedVRT(rasterio.open(dir_EE+"pixel_qa.tif"), dtype='float32') as src:
        ImageOLI_qa, out_transform_qa = rasterio.mask.mask(src, [rowGdrive.geometry], 
                                                        crop=True, nodata=-999, all_touched=True)
        
        ras_meta_qa = src.meta
    ### remove extra -999 rows and cols
    if ImageOLI_qa.shape[2] > 3:
        mask2 = ~(ImageOLI_qa[0,...]==-999).all(axis=0)
        ImageOLI_qa = ImageOLI_qa[...,mask2]
    if ImageOLI_qa.shape[1] > 3:
        mask1 = ~(ImageOLI_qa[0,...]==-999).all(axis=1)
        ImageOLI_qa = ImageOLI_qa[:,mask1,:]
    
    ### remove extra non -999 row and cols
    if ImageOLI_qa.shape[2] > 3:
        diff2 = ImageOLI_qa.shape[2] - 3
        ImageOLI_qa = ImageOLI_qa[...,diff2:]
    if ImageOLI_qa.shape[1] > 3:
        diff1 = ImageOLI_qa.shape[1] - 3
        ImageOLI_qa = ImageOLI_qa[:,diff1:,:]
        
    ### add extra -999 cols and/or cols if they are less than 3
    if ImageOLI_qa.shape[1] < 3:
        diff2_1 = 3 - ImageOLI_qa.shape[1]
        ImageOLI_qa = np.concatenate([ImageOLI_qa, np.full((ImageOLI_qa.shape[0], diff2_1, ImageOLI_qa.shape[2]), -999)], axis=1)
    if ImageOLI_qa.shape[2] < 3:
        diff1_1 = 3 - ImageOLI_qa.shape[2]
        ImageOLI_qa = np.concatenate([ImageOLI_qa, np.full((ImageOLI_qa.shape[0], ImageOLI_qa.shape[1], diff1_1), -999)], axis=2)
    
    for index, band in enumerate(bands):
        dates_table1 = dates.copy()

        with rasterio.open(dir_EE+'/{}.tif'.format(band)) as src:
            ImageOLI, out_transform = rasterio.mask.mask(src, [rowGdrive.geometry], 
                                                          crop=True, nodata=-999, all_touched=True)
            ImageOLI = ImageOLI.astype('float32')
            ras_meta = src.meta
        
        ### remove extra -999 rows and cols
        if ImageOLI.shape[2] > 3:
            ImageOLI = ImageOLI[...,mask2]
        if ImageOLI.shape[1] > 3:
            ImageOLI = ImageOLI[:,mask1,:]

        ### remove extra non -999 row and cols
        if ImageOLI.shape[2] > 3:
            ImageOLI = ImageOLI[...,diff2:]
        if ImageOLI.shape[1] > 3:
            ImageOLI = ImageOLI[:,diff1:,:]

        ### add extra -999 cols and/or cols if they are less than 3
        if ImageOLI.shape[1] < 3:
            ImageOLI = np.concatenate([ImageOLI, np.full((ImageOLI.shape[0], diff2_1, ImageOLI.shape[2]), -999)], axis=1)
        if ImageOLI.shape[2] < 3:
            ImageOLI = np.concatenate([ImageOLI, np.full((ImageOLI.shape[0], ImageOLI.shape[1], diff1_1), -999)], axis=2)
        
        if 'LANDSAT_8' in dates['Satellite'].values:

            OLI_index = dates[dates.Satellite == 'LANDSAT_8'].index.values.tolist()

            formulaIndex = index
            ImageOLI_to_ETM = ImageOLI.copy()

            ImageOLI_to_ETM[OLI_index,:,:] = np.add(intrcp[formulaIndex], np.multiply(slopes[formulaIndex], ImageOLI_to_ETM[OLI_index,:,:]))
            ImageOLI_to_ETM = np.where((ImageOLI <= 0) | (ImageOLI > 10000), -999 , ImageOLI_to_ETM) ### masking the out of valid range pixels, including saturated pixels
            ImageOLI_to_ETM = np.round(ImageOLI_to_ETM)
            ImageOLI = ImageOLI_to_ETM.copy()

            if index == 0:
                try:
                    ImageOLI_qa[OLI_index,:,:] = np.vectorize(OLItoETM.get)(ImageOLI_qa[OLI_index,:,:])
                except:
                    ImageOLI_qa[OLI_index,:,:] = np.where(np.isin(ImageOLI_qa[OLI_index,:,:], list(OLItoETM.keys())), ImageOLI_qa[OLI_index,:,:], 0) # sometimes the pixel values of pixel_qa.tif are not valid.
                    ImageOLI_qa[OLI_index,:,:] = np.vectorize(OLItoETM.get)(ImageOLI_qa[OLI_index,:,:])

        ImageOLI = np.where((ImageOLI <= 0) | (ImageOLI > 10000), -999 , ImageOLI) ### masking the out of valid range pixels, including saturated pixels
        ImageOLI = np.where(np.isinf(ImageOLI), -999 , ImageOLI) ### masking the out Inf

        duplicateRowsDF = dates[dates.duplicated(['date'])]


          
        ImageQA_mask = np.copy(ImageOLI_qa)
        dates_table1 = dates.copy()
        for index1, row1 in duplicateRowsDF.iterrows():
            dupIndex = dates_table1[dates_table1['date'] == row1.date].index.tolist()
            ImageQA_dup = ImageOLI_qa[dupIndex, :, :]
            ImageETM_dup = ImageOLI[dupIndex, :, :]
            mask = np.where((np.isin(ImageQA_dup, clearCodes)) & (ImageETM_dup != -999), 1, 0)
            sumOfClearPix = np.sum(mask, axis=(1,2))
            maxIndex = np.argmax(sumOfClearPix)
            dates_table1.drop(np.delete(dupIndex, maxIndex).tolist(), inplace=True)
              
        ImageOLI = ImageOLI[dates_table1.index.tolist(),:,:]
        ImageQA_mask = ImageQA_mask[dates_table1.index.tolist(),:,:]
        ImageOLI = np.where(np.isin(ImageQA_mask, clearCodes), ImageOLI, -999)

        ras_meta.update({"driver": "GTiff",
                        "compress":'lzw',
                        "count":len(dates_table1.index.tolist()),
                        "width": 3,
                         "height": 3,
                        'dtype': "float32",
                        "transform": out_transform})

        with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/{}.tif'.format(band), 
                           'w', **ras_meta) as dst:
            dst.write(ImageOLI)
            
    ras_meta_qa.update({"driver": "GTiff",
                        "compress":'lzw',
                        "count":len(dates_table1.index.tolist()),
                       "width": 3,
                        "height": 3,
                       "transform": out_transform_qa})
    ras_meta_qa['dtype'] = "float32"

    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/{}'.format('pixel_qa.tif'), 
                       'w', **ras_meta_qa) as dst:
        dst.write(ImageQA_mask)

    dates_table1.reset_index(drop = True, inplace=True)
    dates_table1['band'] = dates_table1.index + 1
    dates_table1.to_csv(directory_temp + str(rowGdrive.PrimaryKey) + '/dates_SR.csv', index=False)


    ############################################################

    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/pixel_qa.tif') as src:
        ImageOLI_qa = src.read().astype(float)
        ras_meta_qa = src.profile

    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/B1.tif') as src:
        band = src.read()
        band = np.expand_dims(band, axis=-1)
        ras_meta = src.profile
    for b in bands[1:]:
        with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/' + b + '.tif') as src:
            band1 = src.read()
            band1 = np.expand_dims(band1, axis=-1)
            band = np.concatenate((band, band1), axis=-1)

    dates = pd.read_csv(directory_temp + str(rowGdrive.PrimaryKey) + '/dates_SR.csv')

    listOfDel = []
    for timestep in range(band.shape[0]):
        img = band[timestep,:,:,:]

        if np.all(img == -999):
            listOfDel.append(timestep)

    if len(listOfDel) != 0:
        band = np.delete(band, listOfDel, 0)
        ImageOLI_qa = np.delete(ImageOLI_qa, listOfDel, 0)

        dates.drop(listOfDel, inplace=True)
        dates.reset_index(inplace=True, drop=True)

        dates['band'] = dates.index + 1

        ras_meta.update({"driver": "GTiff",
                          "compress":'lzw',
                          "count":len(dates.index.tolist())})
        for index4, b in enumerate(bands):
            with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/' + b + '.tif', 'w', **ras_meta) as dst:
                dst.write(band[:, :, :, index4])

        ras_meta_qa.update({"driver": "GTiff",
                            "compress":'lzw',
                            "count":len(dates.index.tolist())})
        with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/pixel_qa.tif', 'w', **ras_meta_qa) as dst:
            dst.write(ImageOLI_qa.astype('uint16'))

    dates['year'] = dates.date.astype('str').str[:4].astype(int)
    dates['month'] = dates.date.astype('str').str[5:7]
    dates.loc[np.isin(dates.month, ['01','02', '12']), 'meteorological_season'] = '04'
    dates.loc[np.isin(dates.month, ['03','04', '05']), 'meteorological_season'] = '01'
    dates.loc[np.isin(dates.month, ['06','07', '08']), 'meteorological_season'] = '02'
    dates.loc[np.isin(dates.month, ['09','10', '11']), 'meteorological_season'] = '03'
    dates.loc[dates.month == '12', 'meteorological_season'] = (1+dates.year).astype('str')+'_'+dates.meteorological_season
    dates.loc[dates.month != '12', 'meteorological_season'] = dates.year.astype('str')+'_'+dates.meteorological_season

    dates.to_csv(directory_temp + str(rowGdrive.PrimaryKey) + '/dates_SR.csv', index=False)

#     #########
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/B1.tif') as src:
        B1 = src.read().astype("float32")
        B1 = np.where(B1==-999, np.nan, B1)/10000
        B1_meta = src.profile
    B1_meta['dtype'] = "float32"
    B1_meta.update({"driver": "GTiff",
              "compress":'lzw'})

    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/B2.tif') as src:
        B2 = src.read().astype("float32")
        B2 = np.where(B2==-999, np.nan, B2)/10000
        B2_meta = src.profile
    B2_meta['dtype'] = "float32"
    B2_meta.update({"driver": "GTiff",
              "compress":'lzw'})

    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/B3.tif') as src:
        B3 = src.read().astype("float32")
        B3 = np.where(B3==-999, np.nan, B3)/10000
        B3_meta = src.profile
    B3_meta['dtype'] = "float32"
    B3_meta.update({"driver": "GTiff",
                  "compress":'lzw'})

    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/B4.tif') as src:
        B4 = src.read().astype("float32")
        B4 = np.where(B4==-999, np.nan, B4)/10000
        B4_meta = src.profile
    B4_meta['dtype'] = "float32"
    B4_meta.update({"driver": "GTiff",
              "compress":'lzw'})

    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/B5.tif') as src:
        B5 = src.read().astype("float32")
        B5 = np.where(B5==-999, np.nan, B5)/10000
        B5_meta = src.profile
    B5_meta['dtype'] = "float32"
    B5_meta.update({"driver": "GTiff",
              "compress":'lzw'})

    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/B7.tif') as src:
        B7 = src.read().astype("float32")
        B7 = np.where(B7==-999, np.nan, B7)/10000
        B7_meta = src.profile
    B7_meta['dtype'] = "float32"
    B7_meta.update({"driver": "GTiff",
              "compress":'lzw'})


# #       ###################################
    NDVI = (B4 - B3)/(B4 + B3)
    NDVI = np.where((NDVI > 1) | (NDVI < -1), -999, NDVI) #### put -999 for out of (-1,1) range
    NDVI = np.where(np.isnan(NDVI), -999, NDVI) #### put -999 for NaN
    NDVI = np.where(np.isinf(NDVI), -999, NDVI) #### put -999 for Inf
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/NDVI.tif', 'w', **B3_meta) as dst:
        dst.write(NDVI)

    EVI = 2.5 * ((B4 - B3) / (B4 + 6 * B3 - 7.5 * B1 + 1))
    EVI = np.where((EVI > 1) | (EVI < -1), -999, EVI)
    EVI = np.where(np.isnan(EVI), -999, EVI)
    EVI = np.where(np.isinf(EVI), -999, EVI)
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/EVI.tif', 'w', **B3_meta) as dst:
        dst.write(EVI)

    NDSI = (B2 - B5) / (B2 + B5)
    NDSI = np.where((NDSI > 1) | (NDSI < -1), -999, NDSI)
    NDSI = np.where(np.isnan(NDSI), -999, NDSI)
    NDSI = np.where(np.isinf(NDSI), -999, NDSI)
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/NDSI.tif', 'w', **B2_meta) as dst:
        dst.write(NDSI)

    NDMI = (B4 - B5) / (B4 + B5)
    NDMI = np.where((NDMI > 1) | (NDMI < -1), -999, NDMI)
    NDMI = np.where(np.isnan(NDMI), -999, NDMI)
    NDMI = np.where(np.isinf(NDMI), -999, NDMI)
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/NDMI.tif', 'w', **B4_meta) as dst:
        dst.write(NDMI)

    SAVI = ((B4 - B3) / (B4 + B3 + 0.5)) * (1.5)
    SAVI = np.where((SAVI > 1) | (SAVI < -1), -999, SAVI)
    SAVI = np.where(np.isnan(SAVI), -999, SAVI)
    SAVI = np.where(np.isinf(SAVI), -999, SAVI)
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/SAVI.tif', 'w', **B4_meta) as dst:
        dst.write(SAVI)

    SATVI = ((B5 - B3)/(B5 + B3 + 0.5)) * (1.5) - (B7 / 2)
    SATVI = np.where((SATVI > 1) | (SATVI < -1), -999, SATVI)
    SATVI = np.where(np.isnan(SATVI), -999, SATVI)
    SATVI = np.where(np.isinf(SATVI), -999, SATVI)
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/SATVI.tif', 'w', **B3_meta) as dst:
        dst.write(SATVI)

    MSAVI = (2 * B4 + 1 - np.sqrt((2 * B4 + 1) ** 2 - 8 * (B4 - B3))) / 2
    MSAVI = np.where((MSAVI > 1) | (MSAVI < -1), -999, MSAVI)
    MSAVI = np.where(np.isnan(MSAVI), -999, MSAVI)
    MSAVI = np.where(np.isinf(MSAVI), -999, MSAVI)
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/MSAVI.tif', 'w', **B3_meta) as dst:
        dst.write(MSAVI)

    NBR = (B4 - B7) / (B4 + B7)
    NBR = np.where((NBR > 1) | (NBR < -1), -999, NBR)
    NBR = np.where(np.isnan(NBR), -999, NBR)
    NBR = np.where(np.isinf(NBR), -999, NBR)
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/NBR.tif', 'w', **B4_meta) as dst:
        dst.write(NBR)
        

    BSCI = 1-(2*(np.abs(B3 - B2)))/((B2+B3+B4)/3)
    BSCI = np.where((BSCI > 1) | (BSCI < -1), -999, BSCI)
    BSCI = np.where(np.isnan(BSCI), -999, BSCI)
    BSCI = np.where(np.isinf(BSCI), -999, BSCI)
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/BSCI.tif', 'w', **B4_meta) as dst:
        dst.write(BSCI)

    NBR2 = (B5 - B7) / (B5 + B7)
    NBR2 = np.where((NBR2 > 1) | (NBR2 < -1), -999, NBR2)
    NBR2 = np.where(np.isnan(NBR2), -999, NBR2)
    NBR2 = np.where(np.isinf(NBR2), -999, NBR2)
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/NBR2.tif', 'w', **B4_meta) as dst:
        dst.write(NBR2)

    NDRI = (B5 - B4)/(np.sqrt(B5 + B4) * 20)
    NDRI = np.where((NDRI > 1) | (NDRI < -1), -999, NDRI)
    NDRI = np.where(np.isnan(NDRI), -999, NDRI)
    NDRI = np.where(np.isinf(NDRI), -999, NDRI)
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/NDRI.tif', 'w', **B4_meta) as dst:
        dst.write(NDRI)


    TCb = B1 * 0.2043 + B2*0.4158 + B3*0.5524 + B4*0.5741 + B5*0.3124 + B7*0.2303
    TCb = np.where(np.isnan(TCb), -999, TCb)
    TCb = np.where(np.isinf(TCb), -999, TCb)
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/TCb.tif', 'w', **B4_meta) as dst:
        dst.write(TCb)

    TCg = B1 * -0.1603 + B2*0.2819 + B3*-0.4934 + B4*0.7940 + B5*-0.0002 + B7*-0.1446
    TCg = np.where(np.isnan(TCg), -999, TCg)
    TCg = np.where(np.isinf(TCg), -999, TCg)
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/TCg.tif', 'w', **B4_meta) as dst:
        dst.write(TCg)

    TCw = B1 * 0.0315 + B2*0.2021 + B3*0.3102 + B4*0.1594 + B5*-0.6808 + B7*-0.6109
    TCw = np.where(np.isnan(TCw), -999, TCw)
    TCw = np.where(np.isinf(TCw), -999, TCw)
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/TCw.tif', 'w', **B4_meta) as dst:
        dst.write(TCw)


    red = B3[0,...].flatten()
    niRed = B4[0,...].flatten()
    mask = ~np.isnan(red) & ~np.isnan(niRed)
    try:
        alpha = linregress(red[mask], niRed[mask])[0]
        beta = linregress(red[mask], niRed[mask])[1]
    except:
        alpha = np.nan
        beta = np.nan
    pvi = (B4[0:1,...] - alpha*B3[0:1,...] - beta)/np.sqrt(1+alpha**2)

    for i in range(1,B3.shape[0]):
        red = B3[i,...].flatten()
        niRed = B4[i,...].flatten()
        mask = ~np.isnan(red) & ~np.isnan(niRed)
        try:
            alpha = np.pi/2 - np.arctan(linregress(red[mask], niRed[mask])[0])
        except:
            alpha = np.nan
        pvi_temp = (np.sin(alpha)*B4[i:i+1,...]) - (np.cos(alpha)*B3[i:i+1,...])

        pvi = np.concatenate([pvi, pvi_temp], axis=0)

    pvi = np.where(np.isnan(pvi), -999, pvi)
    pvi = np.where(np.isinf(pvi), -999, pvi)
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/pvi.tif', 'w', **B4_meta) as dst:
        dst.write(pvi)


    CI = 1- (B3 - B1) / (B3 + B1)
    CI = np.where(np.isnan(CI), -999, CI)
    CI = np.where(np.isinf(CI), -999, CI)
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/CI.tif', 'w', **B4_meta) as dst:
        dst.write(CI)

    DFI = (1- B7/B5) * (B3 / B4)
    DFI = np.where(np.isnan(DFI), -999, DFI)
    DFI = np.where(np.isinf(DFI), -999, DFI)
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/DFI.tif', 'w', **B4_meta) as dst:
        dst.write(DFI)
        
    B1 = np.where(np.isnan(B1), -999., B1)
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/B1.tif', 'w', **B1_meta) as dst:
        dst.write(B1)
        
    B2 = np.where(np.isnan(B2), -999., B2)
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/B2.tif', 'w', **B2_meta) as dst:
        dst.write(B2)
        
    B3 = np.where(np.isnan(B3), -999., B3)
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/B3.tif', 'w', **B3_meta) as dst:
        dst.write(B3)
        
    B4 = np.where(np.isnan(B4), -999., B4)
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/B4.tif', 'w', **B4_meta) as dst:
        dst.write(B4)
        
    B5 = np.where(np.isnan(B5), -999., B5)
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/B5.tif', 'w', **B5_meta) as dst:
        dst.write(B5)
        
    B7 = np.where(np.isnan(B7), -999., B7)
    with rasterio.open(directory_temp + str(rowGdrive.PrimaryKey) + '/B7.tif', 'w', **B7_meta) as dst:
        dst.write(B7)
        
###############################################
    
    band = np.expand_dims(B1, axis=-1)
    for file in [B2, B3, B4, B5, B7, EVI, MSAVI, NBR, NDMI, NDSI, NDVI, SATVI, SAVI, BSCI, NBR2, NDRI, TCb, TCg, TCw, pvi, CI, DFI]:
        band1 = np.expand_dims(file, axis=-1)
        band = np.concatenate((band, band1), axis=-1)
    
    band = np.where(band==-999, np.nan, band)
    
    cycle_list = [year_last+'_'+month_last]
    for i in range(19):
        if month_last == '01':
            month_last = '04'
            cycle_list.append(year_last+'_'+month_last)
        elif month_last == '02':
            month_last = '01'
            cycle_list.append(year_last+'_'+month_last)
        elif month_last == '03':
            month_last = '02'
            cycle_list.append(year_last+'_'+month_last)
        else:
            month_last = '03'
            year_last = str(int(year_last)-1)
            cycle_list.append(year_last+'_'+month_last)

    cycle_list = list(reversed(cycle_list))
    index = dates[dates['meteorological_season'] == cycle_list[0]].index.values.tolist()
    if len(index) != 0:
        arr2 = np.nanmean(band[index,:,:,:], axis=0, keepdims=True)
    else:
        arr2 = np.full((1, band.shape[1], band.shape[2], band.shape[3]), -999.)
        
    for i in cycle_list[1:]:
        index = dates[dates['meteorological_season'] == i].index.values.tolist()
        if len(index) != 0:
            arr_temp = np.nanmean(band[index,:,:,:], axis=0, keepdims=True)
        else:
            arr_temp = np.full((1, band.shape[1], band.shape[2], band.shape[3]), -999.)
        arr2 = np.concatenate([arr2, arr_temp], axis=0)
    
    arr2 = np.where(np.isnan(arr2), -999., arr2)
    arr2 = np.where(np.isinf(arr2), -999., arr2)
######################################################
    
    prism_point = points_all[points_all.PrimaryKey == rowGdrive.PrimaryKey]
    coord_list = [(x,y) for x,y in zip(prism_point['geometry'].x , prism_point['geometry'].y)]
    prism_ppt = [x for x in ppt.sample(coord_list)][0]
    prism_tmean = [x for x in tmean.sample(coord_list)][0]
    
    prism_ppt = np.expand_dims(prism_ppt, -1)
    prism_tmean = np.expand_dims(prism_tmean, -1)
    prism = np.concatenate([prism_ppt, prism_tmean], -1)
    prism = (1/3)*(prism[0::3] + prism[1::3] + prism[2::3])
    
    prism[:,0] = (prism[:,0] - 0) / (731.663 - 0) ## max perc: 731.663mm, min perc: 0 in the US contiguous states, based on PRISM data of ee
    prism[:,1] = (prism[:,1] - (-40.009)) / (49.048 - (-40.009)) ## max temp: 49.048, min temp: -40.009 in the US contiguous states, based on PRISM data of ee
    
    prism = np.round(prism, 5)
    
    np.savez_compressed(directory_final + 'Landsat/{}.npz'.format(rowGdrive.PrimaryKey), arr2)
    np.savez_compressed(directory_final + 'PRISM/{}.npz'.format(rowGdrive.PrimaryKey), prism)
    
    shutil.rmtree(directory_temp + str(rowGdrive.PrimaryKey))



if __name__ == '__main__':
    pool = mp.Pool(number_of_cpu_cores)

    extent_all = extent_all[extent_all.PrimaryKey.isin(dataset.PrimaryKey)].sort_values(by='PrimaryKey')
    for indexGdrive, rowGdrive in extent_all.iterrows():

        pool.map_async(prePro1, (rowGdrive,))
        

    pool.close()
    pool.join()


