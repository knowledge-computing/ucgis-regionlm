import geohash
import h3
from shapely import geometry
from shapely.geometry import Polygon

def geohash_to_polygon(geo):
    """
    :param geo: String that represents the geohash.
    :return: Returns a Shapely's Polygon instance that represents the geohash.
    """
    lat_centroid, lng_centroid, lat_offset, lng_offset = geohash.decode_exactly(geo)
    
    corner_1 = (lat_centroid - lat_offset, lng_centroid - lng_offset)[::-1]
    corner_2 = (lat_centroid - lat_offset, lng_centroid + lng_offset)[::-1]
    corner_3 = (lat_centroid + lat_offset, lng_centroid + lng_offset)[::-1]
    corner_4 = (lat_centroid + lat_offset, lng_centroid - lng_offset)[::-1]

    return geometry.Polygon([corner_1, corner_2, corner_3, corner_4, corner_1])

def cell_to_shapely(cell):
    coords = h3.h3_to_geo_boundary(cell)
    flipped = tuple(coord[::-1] for coord in coords)
    return Polygon(flipped)