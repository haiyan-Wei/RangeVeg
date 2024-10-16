import os
import arcpy
import dbfread
import pandas as pd
arcpy.env.overwriteOutput = True
from config import FULL_OUTPUT_DIR

# question: try using zonal?

def get_slope_for_landsat_cells():
    """If profile curvature is calculated using Curvature tool
    curvature < 0 is convex
    curvature > 0 is concave
    curvature = 0 is uniform"""

    # inputs
    dem = os.path.join(FULL_OUTPUT_DIR, "elevation", "1", "elevation.tif")
    landsat_fishnet_dir = os.path.join(FULL_OUTPUT_DIR, "fishnet_shapefiles")   
    landsat_cells = os.path.join(landsat_fishnet_dir, "landsat_cells.shp")  
    landsat_centers = os.path.join(landsat_fishnet_dir, "landsat_cell_centers.shp")

    # outputs
    output_dir = os.path.join(FULL_OUTPUT_DIR, "slope_curvature_centerpoints")
    os.makedirs(output_dir, exist_ok=True)
    slope_tif = os.path.join(output_dir, "slope_percent.tif")
    curvature_tif = os.path.join(output_dir, "curvature_profile.tif")
    sample_table = os.path.join(output_dir, "sample_slope_curvature.dbf")
    sample_csv = os.path.join(output_dir, "landsat_cells_slope_curvature.csv")

    zonal_stats_table = os.path.join(output_dir, "zonal_stats_slope_curvature.dbf")

    get_slope_curvature(dem, slope_tif, curvature_tif)
    sample_at_centers(landsat_centers, slope_tif, curvature_tif, sample_table, sample_csv)
    
    # calculate_zonal_statistics(landsat_cells, slope_tif, curvature_tif, zonal_stats_table)
    
    print("Done")

def get_slope_curvature(dem, slope_output, curvature_output):
    slope_raster = arcpy.sa.Slope(dem, "PERCENT_RISE")
    slope_raster.save(slope_output)
    arcpy.sa.Curvature(dem, z_factor=1, 
                       out_profile_curve_raster=curvature_output, out_plan_curve_raster=None)


def calculate_zonal_statistics(landsat_cells, slope_tif, curvature_tif, output_table):
    arcpy.sa.ZonalStatisticsAsTable(landsat_cells, "PrimaryKey", slope_tif, 
                                    output_table, "DATA", "MEAN")
    arcpy.sa.ZonalStatisticsAsTable(landsat_cells, "PrimaryKey", curvature_tif, 
                                    output_table, "DATA", "MEAN", "APPEND")

def sample_at_centers(landsat_centers, slope_tif, curvature_tif, output_table, output_csv):
    sample_inputs = "{0};{1}".format(slope_tif, curvature_tif)
    arcpy.sa.Sample(sample_inputs, landsat_centers, output_table, unique_id_field="PrimaryKey")
    table = dbfread.DBF(output_table)
    df_sample = pd.DataFrame(iter(table))
    for col in df_sample.columns:
        if col.startswith("landsat_"):
            df_sample.rename(columns={col: "PrimaryKey"}, inplace=True)
        if col.startswith("slope"):
            df_sample.rename(columns={col: "slope"}, inplace=True)
        if col.startswith("curvature"):
            df_sample.rename(columns={col: "curvature_value"}, inplace=True)

    # Calculate curvature classification based on curvature_value
    df_sample['slope_shape'] = df_sample['curvature_value'].apply(lambda x: 
        'convex' if x < 0 else
        'concave' if x > 0 else
        'uniform' if x == 0 else
        'unknown'
    )
    df_sample.to_csv(output_csv, index=False)
    return df_sample
        

