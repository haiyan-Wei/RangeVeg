import os
import arcpy
import numpy as np
import pandas as pd
import arcpy.analysis
from datetime import datetime
from config import FULL_OUTPUT_DIR, PROJECT_ROOT

t0 = datetime.now()

def get_soil_texture_for_lansat_cells():

    landsat_cells = os.path.join(FULL_OUTPUT_DIR, "fishnet_shapefiles", "landsat_cells.shp")
    soil_layer_path = r"E:\gis_data_us\gSSURGO_CONUS\gSSURGO_CONUS.gdb\MUPOLYGON"
    rhem_lookup_table = os.path.join(PROJECT_ROOT, "inputs", "Soil_Hydraulic_Properties.csv")
    max_thickness = 4
    max_horizons = 2
    
    # output_dir = os.path.join(FULL_OUTPUT_DIR, "soil_texture")
    output_dir = os.path.join(FULL_OUTPUT_DIR, "soil_texture_with_RHEM_lut")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "df_landsat_cells_texture_with_RHEM_lut.csv")

    workspace = os.path.join(output_dir, "soil_texture.gdb")
    if not arcpy.Exists(workspace):
        arcpy.management.CreateFileGDB(os.path.dirname(workspace), os.path.basename(workspace))
    arcpy.env.workspace = workspace
    arcpy.env.overwriteOutput = True
    

    print("Intersecting soils with hillslopes.")
    intersect_feature_class = os.path.join(workspace, f"landsat_soils_intersect")    
    if arcpy.Exists(intersect_feature_class):
        arcpy.Delete_management(intersect_feature_class)
        pass
    else:
        arcpy.analysis.PairwiseIntersect(f"'{soil_layer_path}'; '{landsat_cells}'", 
                                        intersect_feature_class, "ALL", None, "INPUT")
        
    print("Loading tables.")
    soil_gdb = os.path.split(soil_layer_path)[0]
    (df_mapunit, df_component, df_horizon, df_texture_group, df_texture) = load_tables(soil_gdb, 
        intersect_feature_class, rhem_lookup_table)    

    print("Querying soil parameters.")
    query_soil_parameters(df_mapunit, df_component, df_horizon, df_texture_group, df_texture, max_thickness, max_horizons)
    
    print("Calculating dominant soil texture for each mukey.")
    df_dominant_component = get_dominant_texture_for_mukey()

    print("Getting dominant soil texture for PrimaryKey")
    get_dominant_texture_for_primarykey(df_dominant_component, landsat_cells, intersect_feature_class, output_csv)

    get_RHEM_parameters(output_dir, rhem_lookup_table, output_csv)

    t1 = datetime.now()
    print(f"Done. Time taken: {t1 - t0}")       


def get_RHEM_parameters(output_dir, rhem_lookup_table, output_csv):

    df_texture = pd.read_csv(os.path.join(output_dir, output_csv))
    df_rhem_lut = pd.read_csv(rhem_lookup_table)
    
    df_texture['TextureClass_texcl'] = df_texture['TextureClass_texcl'].str.lower()

    replacements = {
        'coarse sand': 'sand', 'fine sand': 'sand', 'very fine sand': 'sand',
        'coarse sandy loam': 'sandy loam', 'fine sandy loam': 'sandy loam', 'very fine sandy loam': 'sandy loam',
        'loamy coarse sand': 'loamy sand', 'loamy fine sand': 'loamy sand', 'loamy very fine sand': 'loamy sand'}
    
    df_texture['TextureClass_texcl'] = df_texture['TextureClass_texcl'].replace(replacements)
    df_texture = pd.merge(df_texture, df_rhem_lut, how="left", left_on="TextureClass_texcl", right_on="Soil_Texture_Class")

    print(f"Number of textures not in the RHEM lookup table: {sum(df_texture['Soil_Texture_Class'].isna())}"
          f"\nTextures not in the RHEM lookup table: {df_texture[df_texture['Soil_Texture_Class'].isna()].Soil_Texture_Class.unique()}"
          f"\nComponent names with textures not in the RHEM lookup table: {df_texture[df_texture['Soil_Texture_Class'].isna()].compname.unique()}")
    
    df_texture.to_csv(output_csv, index=False)


def query_soil_parameters(df_mapunit, df_component, df_horizon, df_texture_group, df_texture, max_thickness, max_horizons):
    """Query soil parameters. Called in intersect_soils_get_texture function."""

    start_time = datetime.now()
    df_mapunit['mukey'] = df_mapunit['mukey'].astype(str)
    df_component['mukey'] = df_component['mukey'].astype(str)

    # Initialize an empty list to store horizon parameters
    horizon_parameters_list = []

    mukeys = df_mapunit["mukey"].unique()
    for i, mukey in enumerate(mukeys):
        print(f"processing mukey: {mukey}, {i+1} of {len(mukeys)}")
        df_component_filtered = df_component[df_component["mukey"] == mukey]
        
        for _, row in df_component_filtered.iterrows():
            cokey = row.cokey
            comppct_r = row.comppct_r
            compname = row.compname

            df_horizon_filtered = df_horizon[
                (df_horizon["cokey"] == cokey) & 
                (df_horizon["hzdept_r"] < max_thickness)].sort_values(by='hzdept_r', ascending=True)

            if df_horizon_filtered.empty:
                horizon_parameters = {"mukey": mukey, "cokey": cokey, "compname": compname, "TextureClass_texcl": "NODATA"}
                horizon_parameters_list.append(horizon_parameters)
            else:                    
                df_horizon_filtered = df_horizon_filtered.head(max_horizons)

                for horizon_count, (_, row) in enumerate(df_horizon_filtered.iterrows(), start=1):
                    horizon_id = row.chkey
                    horizon_name = row.hzname
                    top_depth = row.hzdept_r
                    bottom_depth = row.hzdepb_r                
                    horizon_thickness = bottom_depth - top_depth

                    df_texture_group_filtered = df_texture_group[df_texture_group["chkey"] == horizon_id]
                    rv_yes_row = df_texture_group_filtered[df_texture_group_filtered["rvindicator"].str.lower() == "yes"]
                    if not rv_yes_row.empty:
                        rv_yes_row = rv_yes_row.iloc[0]
                        chtgkey_RV_Yes = rv_yes_row['chtgkey']
                        texture_desc_in_group = rv_yes_row.texdesc
                    else:
                        df_texture_group_filtered = df_texture_group_filtered.iloc[0]
                        chtgkey_RV_Yes = df_texture_group_filtered['chtgkey']
                        texture_desc_in_group = df_texture_group_filtered['texdesc']
                    
                    texture = df_texture[df_texture["chtgkey"] == chtgkey_RV_Yes].texcl.values[0]
                    if texture == "None":
                        texture = df_texture[df_texture["chtgkey"] == chtgkey_RV_Yes].lieutex.values[0]
                    
                    horizon_parameters = {
                        "mukey": mukey,
                        "cokey": cokey,
                        "compname": compname,
                        "comppct_r": comppct_r,
                        "HorizonChkey": horizon_id,
                        "HorizonNumber": horizon_count,
                        "HorizonName": horizon_name,
                        "HorizonTopDepth": top_depth,
                        "HorizonBottomDepth": bottom_depth,
                        "HorizonThickness": horizon_thickness,
                        "TextureGroupChtgkey": chtgkey_RV_Yes,
                        "TextureDesc": texture_desc_in_group,
                        "TextureClass_texcl": texture
                    }
                    horizon_parameters_list.append(horizon_parameters)

    # Create DataFrame from the list of dictionaries
    df_horizon_parameters_all = pd.DataFrame(horizon_parameters_list)
    
    print("Saving soil parameters to csv.")
    df_horizon_parameters_all.to_csv(os.path.join(FULL_OUTPUT_DIR, "df_horizon_parameters.csv"), index=False)
    end_time = datetime.now()
    print(f"Time taken for query_soil_parameters: {end_time - start_time}")
    return df_horizon_parameters_all


def get_dominant_texture_for_primarykey(df_dominant_component, landsat_cells, intersect_feature_class, output_csv):
    """Get the dominant texture for each PrimaryKey.
    Called in intersect_soils_get_texture function."""
    
    # Read the intersect feature class into a DataFrame
    df_intersect = pd.DataFrame(arcpy.da.TableToNumPyArray(intersect_feature_class, 
                                                           ["PrimaryKey", "MUKEY", "Shape_Area"]))    
    df_intersect['MUKEY'] = df_intersect['MUKEY'].astype(str)    
    df_dominant_mukey = df_intersect.loc[df_intersect.groupby('PrimaryKey')['Shape_Area'].idxmax()]    
    df_dominant_component['mukey'] = df_dominant_component['mukey'].astype(str)

    df_mukey_texture = pd.merge(df_dominant_mukey[['PrimaryKey', 'MUKEY']], 
                         df_dominant_component, 
                         left_on='MUKEY', 
                         right_on='mukey', 
                         how='left')

    # Merge df_mukey_texture with df_landsat_cells
    df_landsat_cells = pd.DataFrame(arcpy.da.TableToNumPyArray(landsat_cells, ["PrimaryKey"]))
    df_landsat_cells.PrimaryKey = df_landsat_cells.PrimaryKey.astype("int64")
    df_landsat_cells_texture = pd.merge(df_landsat_cells, df_mukey_texture, on="PrimaryKey", how="left")

    # Save df_mukey_texture and df_landsat_cells_texture to csv
    df_mukey_texture.to_csv(os.path.join(FULL_OUTPUT_DIR, "df_mukey_texture.csv"), index=False)
    df_landsat_cells_texture.to_csv(output_csv, index=False)    
    
    if True:
        # Create a new feature class with parameters associated with each PrimaryKey
        print(f"Creating feature class with soil parameters.")
        output_fc = os.path.join(arcpy.env.workspace, "landsat_cells_with_soil_texture")
        if arcpy.Exists(output_fc):
            arcpy.Delete_management(output_fc)

        # Copy the original landsat_cells feature class
        arcpy.CopyFeatures_management(landsat_cells, output_fc)

        # Add new fields for soil parameters
        soil_fields = ['TextureClass_texcl', 'comppct_r', 'cokey', 'HorizonChkey', 'HorizonThickness', 'TextureGroupChtgkey']
        for field in soil_fields:
            arcpy.AddField_management(output_fc, field, "TEXT" if field == 'TextureClass_texcl' else "DOUBLE")

        # Update the feature class with soil parameters
        with arcpy.da.UpdateCursor(output_fc, ['PrimaryKey'] + soil_fields) as cursor:
            for row in cursor:
                primary_key = row[0]
                soil_data = df_mukey_texture[df_mukey_texture['PrimaryKey'] == primary_key]
                if not soil_data.empty:
                    row[1] = soil_data['TextureClass_texcl'].values[0]
                    row[2] = soil_data['comppct_r'].values[0]
                    row[3] = soil_data['cokey'].values[0]
                    row[4] = soil_data['HorizonChkey'].values[0]
                    row[5] = soil_data['HorizonThickness'].values[0]
                    row[6] = soil_data['TextureGroupChtgkey'].values[0]
                    cursor.updateRow(row)

    return


def get_dominant_texture_for_mukey():

    # filter out horizons with non-positive thickness and components with non-positive percentage
    df = pd.read_csv(os.path.join(FULL_OUTPUT_DIR, "df_horizon_parameters.csv"))
    def print_warning(count, message):
        if count > 0:
            print(f"Warning: {count} {message}")
    print_warning(sum(df.HorizonThickness <= 0), "horizons have non-positive thickness. These will be ignored.")
    print_warning(sum(df.comppct_r <= 0), "components have non-positive percentage. These will be ignored.")

    df = df[(df.HorizonThickness > 0) & (df.comppct_r > 0)]
    
    # get the maximum thickness for each unique combination of mukey, cokey, and HorizonChkey
    max_thickness = df.groupby(['mukey', 'cokey', 'HorizonChkey'])['HorizonThickness'].transform('max')
    df_dominant_horizon = df[df['HorizonThickness'] == max_thickness]
    df_dominant_horizon = df_dominant_horizon.reset_index(drop=True)

    # get the maximum percentage for each mukey
    idx = df_dominant_horizon.groupby('mukey')['comppct_r'].idxmax()
    df_dominant_component = df_dominant_horizon.loc[idx]
    df_dominant_component = df_dominant_component.reset_index(drop=True)

    # save to csv
    df_dominant_horizon.to_csv(os.path.join(FULL_OUTPUT_DIR, "df_dominant_horizon.csv"), index=False)
    df_dominant_component.to_csv(os.path.join(FULL_OUTPUT_DIR, "df_dominant_component.csv"), index=False)

    return df_dominant_component


def query_soil_parameters_new(df_mapunit, df_component, df_horizon, df_texture_group, df_texture, max_thickness, max_horizons):
    """Query soil parameters. Called in intersect_soils_get_texture function."""

    # This approach may save time by avoiding the nested loops. But results need to be verified.

    start_time = datetime.now()


    df_mapunit['mukey'] = df_mapunit['mukey'].astype(str)
    df_component['mukey'] = df_component['mukey'].astype(str)
    df_component['cokey'] = df_component['cokey'].astype(str)
    df_horizon['cokey'] = df_horizon['cokey'].astype(str)

    df_horizon_parameters_all = pd.DataFrame()

    n_cokeys = len(df_component.cokey.unique())
    for i, cokey in enumerate(df_component.cokey.unique()):
        print(f"processing cokey: {cokey}, {i} of {n_cokeys}")
        df_horizon_filtered = df_horizon[df_horizon['cokey'] == cokey]

        df_horizon_filtered = df_horizon[
                (df_horizon["cokey"] == cokey) & 
                (df_horizon["hzdept_r"] < max_thickness)].sort_values(by='hzdept_r', ascending=True)

        if df_horizon_filtered.empty:
            continue        
        df_horizon_filtered = df_horizon_filtered.head(max_horizons)

        # Step 3: Loop 3, process each horizon
        for horizon_count, (_, row) in enumerate(df_horizon_filtered.iterrows(), start=1):
            horizon_id = row.chkey
            horizon_name = row.hzname
            top_depth = row.hzdept_r
            bottom_depth = row.hzdepb_r                
            horizon_thickness = bottom_depth - top_depth

            # Step 4: process texture group                
            df_texture_group_filtered = df_texture_group[df_texture_group["chkey"] == horizon_id]
            rv_yes_row = df_texture_group_filtered[df_texture_group_filtered["rvindicator"].str.lower() == "yes"]
            if not rv_yes_row.empty:
                rv_yes_row = rv_yes_row.iloc[0]
                chtgkey_RV_Yes = rv_yes_row['chtgkey']
                texture_desc_in_group = rv_yes_row.texdesc
            else:
                df_texture_group_filtered = df_texture_group_filtered.iloc[0]
                chtgkey_RV_Yes = df_texture_group_filtered['chtgkey']
                texture_desc_in_group = df_texture_group_filtered['texdesc']
            
            # Step 5: get texture from texture table - this is be
            texture = df_texture[df_texture["chtgkey"] == chtgkey_RV_Yes].texcl.values[0]
            if texture == "None":
                texture = df_texture[df_texture["chtgkey"] == chtgkey_RV_Yes].lieutex.values[0]
            
            # Step 6: Query parameters from horizon table and the kin_lut table     
            horizon_parameters = {
                "cokey": cokey,
                "HorizonChkey": horizon_id,
                "HorizonNumber": horizon_count,
                "HorizonName": horizon_name,
                "HorizonTopDepth": top_depth,
                "HorizonBottomDepth": bottom_depth,
                "HorizonThickness": horizon_thickness,
                "TextureGroupChtgkey": chtgkey_RV_Yes,
                "TextureDesc": texture_desc_in_group,
                "TextureClass_texcl": texture}

            df_horizon_parameters_all = pd.concat([df_horizon_parameters_all, 
                    pd.DataFrame([horizon_parameters])], axis=0, ignore_index=True)
            
    df_mukey_parameters = pd.merge(df_component, df_horizon_parameters_all, on='cokey')
    df_mukey_parameters.to_csv(os.path.join(FULL_OUTPUT_DIR, "df_mukey_parameters.csv"), index=False)

    end_time = datetime.now()
    print(f"Time taken: {end_time - start_time}")

    return df_mukey_parameters


def load_tables(soil_gdb, intersect_feature_class, rhem_lookup_table):
    """Load tables from gSSURGO database and AGWA lookup table."""

    def safe_convert(value):
        if value is None:
            return np.nan
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    # reading tables from AGWA directory and gSSURGO database
    component_table = os.path.join(soil_gdb, "component")
    horizon_table = os.path.join(soil_gdb, "chorizon")
    texture_table = os.path.join(soil_gdb, "chtexture")
    texture_group_table = os.path.join(soil_gdb, "chtexturegrp")
    
    # define fields needed and read tables into dataframes
    df_mapunit = pd.DataFrame(arcpy.da.TableToNumPyArray(intersect_feature_class, ["mukey"]))
    df_mapunit.mukey = df_mapunit.mukey.astype(str)
    print(f'number of mapunits: {len(df_mapunit)}')
    print(f"number of unique mapunits: {len(df_mapunit.mukey.unique())}")

    data = [[safe_convert(value) for value in row] for row in arcpy.da.SearchCursor(
        component_table, ["cokey", "comppct_r", "mukey", "compname"])]
    df_component = pd.DataFrame(data, columns=["cokey", "comppct_r", "mukey", "compname"])
    df_component.mukey = df_component.mukey.astype(str)
    df_component.cokey = df_component.cokey.astype(str) 
    df_component = df_component[df_component.mukey.isin(df_mapunit.mukey)]
    print(f'number of recrods in df_components: {len(df_component)}')

    df_horizon = pd.DataFrame(arcpy.da.TableToNumPyArray(horizon_table, 
        ["cokey", "chkey", "hzname", "hzdept_r", "hzdepb_r"]))
    df_horizon.cokey = df_horizon.cokey.astype(str)
    df_horizon = df_horizon[df_horizon.cokey.isin(df_component.cokey)]
    print(f'number of recrods in df_horizon: {len(df_horizon)}')

    df_texture_group = pd.DataFrame(arcpy.da.TableToNumPyArray(texture_group_table, "*"))
    df_texture_group.chkey = df_texture_group.chkey.astype(str)
    df_texture_group.chtgkey = df_texture_group.chtgkey.astype(str)
    df_texture_group = df_texture_group[df_texture_group.chkey.isin(df_horizon.chkey)]
    print(f'number of recrods in df_texture_group: {len(df_texture_group)}')

    df_texture = pd.DataFrame(arcpy.da.TableToNumPyArray(texture_table, "*"))
    df_texture.chtgkey = df_texture.chtgkey.astype(str)
    df_texture = df_texture[df_texture.chtgkey.isin(df_texture_group.chtgkey)]
    print(f'number of recrods in df_texture: {len(df_texture)}')



    return df_mapunit, df_component, df_horizon, df_texture_group, df_texture


get_soil_texture_for_lansat_cells() 