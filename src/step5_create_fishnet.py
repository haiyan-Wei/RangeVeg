import os
import arcpy
import pandas as pd
from datetime import datetime
from config import PROJECT_ROOT, FULL_OUTPUT_DIR
arcpy.env.overwriteOutput = True


# use cell size of 3000m for testing

def create_fishnet_with_arcpy():
    """ 
    Create a fishnet (grid) of polygons.
    """

    t0 = datetime.now()

    # inputs
    area_shp = os.path.join(PROJECT_ROOT, 'inputs', 'Vectors', 'Area.shp')
    landsat_dir = os.path.join(PROJECT_ROOT, 'inputs', 'Landsat')
    
    # outputs
    output_dir = os.path.join(FULL_OUTPUT_DIR, 'fishnet_shapefiles')
    os.makedirs(output_dir, exist_ok=True)
    fishnet_polygon = os.path.join(output_dir, 'fishnet_polygon.shp')
    fishnet_points = os.path.join(output_dir, 'fishnet_points.shp')
    cell_shpfile = os.path.join(output_dir, 'landsat_cells.shp')
    cell_center_shpfile = os.path.join(output_dir, 'landsat_cell_centers.shp')

    # start processing
    x_min, y_min, x_max, y_max, cell_width, cell_height = get_landsat_dimensions(landsat_dir)
    # cell_height, cell_width = 3000, 3000
    create_fishnet(x_min, y_min, x_max, y_max, cell_width, cell_height, fishnet_polygon, fishnet_points)
    join_fishnet_with_area(fishnet_polygon, fishnet_points, area_shp, cell_shpfile, cell_center_shpfile)
    add_lat_lon_to_centroids(cell_shpfile, "cell")
    add_lat_lon_to_centroids(cell_center_shpfile, "point")

    t1 = datetime.now()
    print(f"Done. Time taken: {t1 - t0}")


def get_landsat_dimensions(landsat_dir):
    """Get the dimensions of the landsat images to create a fishnet"""
    print("Getting landsat dimensions...")
    first_pass = True
    for subdir, _, files in os.walk(landsat_dir):    
        for file in files:
            if file.lower() == "B1.tif".lower():
                landsat = arcpy.Raster(os.path.join(subdir, file))                
                extent = landsat.extent
                if first_pass:
                    sr_landsat = landsat.spatialReference
                    arcpy.env.outputCoordinateSystem = sr_landsat
                    cell_size = landsat.getRasterInfo().getCellSize()
                    cell_width = cell_size[0]
                    cell_height = cell_size[1]
                    x_min = extent.XMin
                    x_max = extent.XMax
                    y_min = extent.YMin
                    y_max = extent.YMax
                    
                    first_pass = False
                else:
                    x_min = min(x_min, extent.XMin)
                    x_max = max(x_max, extent.XMax)
                    y_min = min(y_min, extent.YMin)
                    y_max = max(y_max, extent.YMax)
    return x_min, y_min, x_max, y_max, cell_width, cell_height


def create_fishnet(x_min, y_min, x_max, y_max, cell_width, cell_height, cell_shpfile, fishnet_points):
    """Create a fishnet (grid) of polygons."""
    
    
    print("Creating fishnet...")
    origin_coord = str(x_min) + " " + str(y_min)
    y_axis_coord = str(x_min) + " " + str(y_min+10)
    corner_coord = str(x_max) + " " + str(y_max)
    arcpy.management.CreateFishnet(
        out_feature_class=cell_shpfile,
        origin_coord=origin_coord,
        y_axis_coord=y_axis_coord,
        cell_width=cell_width,
        cell_height=cell_height,
        number_rows=None,
        number_columns=None,
        corner_coord=corner_coord,
        labels="LABELS",
        template=None,
        geometry_type="POLYGON")
    print("Creating fishnet points...")
    arcpy.management.CopyFeatures(cell_shpfile.replace('.shp', "_label.shp"), fishnet_points)
    # arcpy.management.DeleteFeatures(cell_shpfile.replace('.shp', "_label.shp"))
    print("Number of points in fishnet: ", arcpy.management.GetCount(fishnet_points))


def join_fishnet_with_area(fishnet_polygon, fishnet_points, area_shp, cell_shpfile, center_shpfile):
    """Join the fishnet with the area shapefile."""
    print("Joining fishnet with area...")

    # Add PrimaryKey field to fishnet polygon and points
    arcpy.management.AddField(fishnet_polygon, "PrimaryKey", "LONG")
    arcpy.management.AddField(fishnet_points, "PrimaryKey", "LONG")

    # Calculate PrimaryKey for fishnet polygon and points
    fishnet_polygon_oid = arcpy.Describe(fishnet_polygon).OIDFieldName
    fishnet_points_oid = arcpy.Describe(fishnet_points).OIDFieldName
    arcpy.management.CalculateField(fishnet_polygon, "PrimaryKey", f"!{fishnet_polygon_oid}!", "PYTHON3")
    arcpy.management.CalculateField(fishnet_points, "PrimaryKey", f"!{fishnet_points_oid}!", "PYTHON3")

    # Create a FieldMappings object
    field_mappings = arcpy.FieldMappings()

    # Add all fields from the area shapefile
    field_mappings.addTable(area_shp)

    # Add the PrimaryKey field from the fishnet polygon
    pk_field = arcpy.FieldMap()
    pk_field.addInputField(fishnet_polygon, "PrimaryKey")
    field_mappings.addFieldMap(pk_field)

    # Perform the spatial join with the field mapping for polygons
    arcpy.SpatialJoin_analysis(fishnet_polygon, area_shp, cell_shpfile, 
                               "JOIN_ONE_TO_ONE", "KEEP_ALL", 
                               field_mapping=field_mappings,
                               match_option="HAVE_THEIR_CENTER_IN")
    
    # Perform spatial join for points
    arcpy.SpatialJoin_analysis(fishnet_points, area_shp, center_shpfile, 
                               "JOIN_ONE_TO_ONE", "KEEP_ALL", 
                               match_option="INTERSECT")

    # Add PrimaryKey field to both output shapefiles
    arcpy.management.AddField(cell_shpfile, "PrimaryKey", "LONG")
    arcpy.management.AddField(center_shpfile, "PrimaryKey", "LONG")

    # Assign PrimaryKey field using the OID of the polygon
    cell_oid = arcpy.Describe(cell_shpfile).OIDFieldName
    center_oid = arcpy.Describe(center_shpfile).OIDFieldName
    
    arcpy.management.CalculateField(cell_shpfile, "PrimaryKey", f"!{cell_oid}!", "PYTHON3", '', "TEXT", "NO_ENFORCE_DOMAINS")
    arcpy.management.CalculateField(center_shpfile, "PrimaryKey", f"!{center_oid}!", "PYTHON3", '', "TEXT", "NO_ENFORCE_DOMAINS")

    print("Fishnet joined with area and PrimaryKey fields added")


def add_lat_lon_to_centroids(center_shpfile, type):
    """Add Latitude and Longitude fields to the centroids or points."""
    
    arcpy.management.AddField(center_shpfile, "Lat_NAD83", "DOUBLE")
    arcpy.management.AddField(center_shpfile, "Lon_NAD83", "DOUBLE")
    input_sr = arcpy.Describe(center_shpfile).spatialReference
    
    out_sr = arcpy.SpatialReference(4269)  # 4269 is the EPSG code for NAD83 geographic

    print(f"Input spatial reference: {input_sr.name}")
    print(f"Output spatial reference: {out_sr.name}")

    # Get available transformations
    transformations = arcpy.ListTransformations(input_sr, out_sr)
    print(f"Available transformations: {transformations}")

    if not transformations:
        print("No transformations available. Checking if spatial references are the same...")
        if input_sr.name == out_sr.name:
            print("Input and output spatial references are the same. No transformation needed.")
        else:
            print("Spatial references are different, but no transformation available.")
            print(f"Input SR factory code: {input_sr.factoryCode}")
            print(f"Output SR factory code: {out_sr.factoryCode}")

    # Calculate Latitude and Longitude
    with arcpy.da.UpdateCursor(center_shpfile, ["SHAPE@", "Lat_NAD83", "Lon_NAD83"]) as cursor:
        for row in cursor:
            if type == "cell":
                geometry = row[0].centroid
            elif type == "point":
                geometry = row[0].firstPoint
            else:
                raise ValueError("Invalid type. Must be 'cell' or 'point'.")

            point_geometry = arcpy.PointGeometry(geometry, input_sr)
            
            if transformations:
                transformed_point = point_geometry.projectAs(out_sr, transformation_name=transformations[0])
            else:
                transformed_point = point_geometry.projectAs(out_sr)
            
            row[1] = transformed_point.firstPoint.Y
            row[2] = transformed_point.firstPoint.X
            cursor.updateRow(row)

    print(f"Latitude and Longitude fields added and calculated for {type} shapefile")


create_fishnet_with_arcpy()
