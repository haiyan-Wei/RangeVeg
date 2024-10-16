import ee
import sys
# from step1_google_auth import authenticate_google_drive
# from step2_get_landsat import download_landsat_data
# from step3_get_prism import download_prism_data
# from step4_get_elevation import get_elevation
# from step5_create_fishnet import create_fishnet_with_arcpy
# from step6_get_slope import get_slope_for_landsat_cells
from step7_get_cligen import get_cligen_stations
# from step8_get_soil_texture import get_soil_texture_for_lansat_cells
# from step9_merge_csv_tables import merge_csv_tables



def main():

    if False:
        drive_service = authenticate_google_drive()
        if not drive_service:
            print("Google Drive authentication failed.")
            sys.exit(1)
        print("Google Drive authentication successful.")
        try:

            ee.Initialize(project='rangeveg')
            print("Earth Engine initialized successfully.")
        except Exception as e:
            print(f"Error initializing Earth Engine: {str(e)}")
            print("Please make sure you have set up Earth Engine authentication correctly.")
            print("Visit https://developers.google.com/earth-engine/guides/python_install for more information.")
            sys.exit(1)

    if True:
        
        # download_landsat_data()

        # download_prism_data()
        # get_elevation()
        # create_fishnet_with_arcpy()
        # get_slope_for_landsat_cells()
        get_cligen_stations()


    print('Done.')

if __name__ == "__main__":
    main()