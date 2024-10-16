import os
import pandas as pd
from config import FULL_OUTPUT_DIR


fishnet_path = os.path.join(FULL_OUTPUT_DIR, "fishnet", "landsat_cells.csv")
cligen_path = os.path.join(FULL_OUTPUT_DIR, "cligen", "landsat_cligen.csv")
slope_path = os.path.join(FULL_OUTPUT_DIR, "slope_curvature_centerpoints", "landsat_cells_slope_curvature.csv")
soil_texture_path = os.path.join(FULL_OUTPUT_DIR, "soil_texture_with_RHEM_lut", "df_landsat_cells_texture_with_RHEM_lut.csv")

df_cligen = pd.read_csv(cligen_path)
df_slope = pd.read_csv(slope_path)
df_soil_texture = pd.read_csv(soil_texture_path)
df_fishnet = pd.read_csv(fishnet_path)

n_fishnet = len(df_fishnet)
n_cligen = len(df_cligen)
n_slope = len(df_slope)
n_soil_texture = len(df_soil_texture)

print(f"Number of records in fishnet: {n_fishnet}")
print(f"Number of records in cligen: {n_cligen}")
print(f"Number of records in slope: {n_slope}")
print(f"Number of records in soil texture: {n_soil_texture}")
if n_fishnet != n_cligen or n_fishnet != n_slope or n_fishnet != n_soil_texture:
    print("Number of records in cligen, slope, and soil texture are not the same. Exiting.")
    exit()


df_cligen = df_cligen[["PrimaryKey", "cligen_id"]]
df_slope = df_slope[["PrimaryKey", "slope", "slope_shape"]]
df_soil_texture = df_soil_texture[["PrimaryKey", "Soil_Texture_Class", "G", "DIST", "POR", "SMAX", "FRAC1", "FRAC2", "FRAC3", "FRAC4", "FRAC5"]]
df_merge = pd.merge(df_cligen, df_slope, how="left", on=["PrimaryKey"])
df_merge = pd.merge(df_merge, df_soil_texture, how="left", on=["PrimaryKey"])

df_merge.to_csv(os.path.join(FULL_OUTPUT_DIR, "df_merged_tables.csv"), index=False)
