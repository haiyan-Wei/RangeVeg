import os
import pandas as pd
from config import FULL_OUTPUT_DIR, PROJECT_ROOT
import arcpy
arcpy.env.overwriteOutput = True

def get_cligen_stations():

    """Cligen input - either shapefile from Kia's example, stations from WEPP\cligen website, or ArcGIS Online Feature Service that RHEM uses
    Alternate intput from AGOL that RHEM uses
    cligen_name = "cligen_rhem"
    cligen = arcpy.management.MakeFeatureLayer("https://services1.arcgis.com/SyUSN23vOoYdfLC8/arcgis/rest/services/cligen_stations_in_rhem_conus_and_hi/FeatureServer/0", cligen_name)
    """
    
    # inputs
    landsat_fishnet_dir = os.path.join(FULL_OUTPUT_DIR, "fishnet_shapefiles")   
    landsat_centers = os.path.join(landsat_fishnet_dir, "landsat_cell_centers.shp")
    cligen = os.path.join(PROJECT_ROOT, "inputs", "cligen", "CLIGEN_points.shp")

    os.makedirs(FULL_OUTPUT_DIR, exist_ok=True)

    # outputs below
    output_dir = os.path.join(FULL_OUTPUT_DIR, "cligen")    
    os.makedirs(output_dir, exist_ok=True)
    spatial_join_output = os.path.join(output_dir, "landsat_cligen")
    spatial_join_csv_output = os.path.join(output_dir,  "landsat_cligen.csv")

    print("Spatial joining landsat cells with CLIGEN stations.")
    arcpy.analysis.SpatialJoin(
        target_features=landsat_centers,
        join_features=cligen,
        out_feature_class=spatial_join_output,
        join_operation="JOIN_ONE_TO_ONE",
        join_type="KEEP_ALL",
        match_option="CLOSEST",
        search_radius=None)
    
    print("Exporting to CSV.")
    arcpy.conversion.ExportTable(
        in_table=spatial_join_output,
        out_table=spatial_join_csv_output )

    # print("Removing unneeded columns")
    df = pd.read_csv(spatial_join_csv_output)
    df = df[["PrimaryKey", "PK_EE", "prcp", "dur", "tp", "ip", "Lat", "Long", "cligen_id"]]
    df.to_csv(spatial_join_csv_output, index=False)

    print(f"number of cells with CLIGEN stations: {len(df)}\n"
          f"unique CLIGEN stations: {df.cligen_id.nunique()}\n")


    print("Done")


get_cligen_stations()