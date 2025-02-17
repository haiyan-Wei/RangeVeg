import ee


def create_polygon(point, proj):
    polygonBounds = point.geometry.bounds
    return ee.Geometry.Rectangle(polygonBounds, proj=proj, evenOdd=False)


def get_coordinates():

    proj = ee.Projection('\
                PROJCS["USA_Contiguous_Lambert_Conformal_Conic", \
                GEOGCS["GCS_North_American_1983",\
                    DATUM["North_American_Datum_1983",\
                        SPHEROID["GRS_1980",6378137,298.257222101]],\
                    PRIMEM["Greenwich",0],\
                    UNIT["Degree",0.017453292519943295]],\
                PROJECTION["Lambert_Conformal_Conic_2SP"],\
                PARAMETER["False_Easting",0],\
                PARAMETER["False_Northing",0],\
                PARAMETER["Central_Meridian",-96],\
                PARAMETER["Standard_Parallel_1",33],\
                PARAMETER["Standard_Parallel_2",45],\
                PARAMETER["Latitude_Of_Origin",39],\
                UNIT["Meter",1],\
                AUTHORITY["EPSG","102004"]]')
    return proj

