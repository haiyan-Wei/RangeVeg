import os
import numpy as np
import pandas as pd
import glob
from config import FULL_OUTPUT_DIR

FULL_OUTPUT_DIR = r'E:\range_vegcover\original_notebooks_outputs\outputs_original'


# landsat_dir = os.path.join(FULL_OUTPUT_DIR, "Final_dataset", "Landsat")
# prism_dir = os.path.join(FULL_OUTPUT_DIR, "Final_dataset", "PRISM")

# landsat_data_summary = pd.DataFrame(columns=["PrimaryKey", "max", "min", "shape"])
# npz_files = glob.glob(os.path.join(landsat_dir, "*.npz"))
# print(f"Number of Landsat npz files: {len(npz_files)}")
# for npz_file in npz_files:
#     data = np.load(npz_file)
#     data_max = data["arr_0"].max()
#     data_min = data["arr_0"].min()
#     data_shape = data["arr_0"].shape
#     landsat_data_summary = pd.concat([landsat_data_summary, pd.DataFrame({
#         "PrimaryKey": [npz_file.split(".npz")[0].split('Landsat\\')[-1]],
#         "max": [data_max],
#         "min": [data_min],
#         "shape": [data_shape]
#     })], ignore_index=True)

# landsat_data_summary.to_csv(os.path.join(FULL_OUTPUT_DIR, "Final_dataset", "landsat_data_summary.csv"), index=False)

# prism_data_summary = pd.DataFrame(columns=["PrimaryKey", "max", "min",  "shape"])
# npz_files = glob.glob(os.path.join(prism_dir, "*.npz"))
# print(f"Number of PRISM npz files: {len(npz_files)}")
# for npz_file in npz_files:
#     data = np.load(npz_file)
#     data_max = data["arr_0"].max()
#     data_min = data["arr_0"].min()
#     data_shape = data["arr_0"].shape
#     prism_data_summary = pd.concat([prism_data_summary, pd.DataFrame({
#         "PrimaryKey": [npz_file.split(".npz")[0].split('PRISM\\')[-1]],
#         "max": [data_max],
#         "min": [data_min],
#         "shape": [data_shape]
#     })], ignore_index=True)

# prism_data_summary.to_csv(os.path.join(FULL_OUTPUT_DIR, "Final_dataset", "prism_data_summary.csv"), index=False)

# print("Done!")


#
import os
from datetime import datetime
results_dir = r'E:\CEAP\Kiastool\prepro\Final_data\Landsat\9.npz'
creation_time = os.path.getctime(results_dir)
creation_datetime = datetime.fromtimestamp(creation_time)

# Print formatted datetime
print(f"File creation datetime: {creation_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

npz1 = np.load(results_dir)
for var in npz1:
    # print('test', var, len(npz1[var]), npz1[var][0][0])
    d1 = npz1[var][0][0][0]

results_dir = r'E:\range_vegcover\pyfiles\outputs\original_prepro_output\Final_data\Landsat\9.npz'
creation_time = os.path.getctime(results_dir)
creation_datetime = datetime.fromtimestamp(creation_time)
print(f"File creation datetime: {creation_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

npz1 = np.load(results_dir)
for var in npz1:
    # print('test', var, len(npz1[var]), npz1[var][0][0])
    d2 = npz1[var][0][0][0]

df = pd.DataFrame({"d1": d1, "d2": d2})
# df.to_csv(os.path.join(FULL_OUTPUT_DIR, "Final_dataset", "landsat_compare.csv"), index=False)
print(df)