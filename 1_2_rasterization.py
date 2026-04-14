import logging
import os, sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely import Point, Polygon
from shapely import wkt, wkb

import geohash
from polygeohasher import polygeohasher
from polygon_geohasher.polygon_geohasher import geohashes_to_polygon

import h3.api.basic_str as h3
import h3pandas

from utils import const


def generate_aoi_region(in_csv_file_path, out_csv_file_path, context_field_names, region_type, region_level):
        try:
            df = pd.read_csv(in_csv_file_path)
            df[const.regioncontext_geometry_field_name] = df[const.regioncontext_geometry_field_name].apply(wkt.loads)
            
            df[const.aggregated_field_name] = ''
            for field_name in context_field_names:
                df[field_name] = df[field_name].astype(str).apply(lambda x: x.lower().encode('ascii', 'ignore').strip().decode('ascii') if x != 'nan' else '')
                df[const.aggregated_field_name] = df[const.aggregated_field_name] + df[field_name] + ':'
                
            df = df[df[const.aggregated_field_name].notnull()]
            df[const.aggregated_field_name] = df[const.aggregated_field_name].apply(lambda x: x.strip(':'))
            df = df[df[const.aggregated_field_name] != '']
            
            df = df.reset_index(drop=True)

            gdf = gpd.GeoDataFrame(df, geometry=const.regioncontext_geometry_field_name, crs="EPSG:4326")

            initial_df = pd.DataFrame()
            
            if region_type == 'geohash':
                pgh = polygeohasher.Polygeohasher(gdf)
                initial_df = pgh.create_geohash_list(region_level, inner=False)
                initial_df = initial_df.explode('geohash_list')

                initial_df[const.regioncontext_geometry_field_name] = initial_df['geohash_list'].apply(lambda x: geohashes_to_polygon(x))
                
                def geohash_to_point(gh):
                    lat, lon = geohash.decode(gh)
                    return Point(lon, lat)  # (lon, lat) order
                
                initial_df[const.regioncontext_geometry_field_name] = initial_df['geohash_list'].astype(str).apply(geohash_to_point)

            elif region_type == 'h3':
                gdf = gdf.explode(index_parts=False)
                gdf = gdf.reset_index(drop=True)
                initial_df = gdf.h3.polyfill(region_level, explode=True)
                initial_df = initial_df.rename(columns={'h3_polyfill': 'h3_list'})
                initial_df = initial_df.dropna(subset=['h3_list'], ignore_index=True)
                initial_df[const.regioncontext_geometry_field_name] = initial_df['h3_list'].apply(lambda x: Point(*h3.h3_to_geo(x)[::-1]))

            initial_df[const.poi_aoi_field_name] = const.aoi_field_value
            initial_df.to_csv(out_csv_file_path, index=False)
            
        except Exception as e:
            logging.error(f"Error generating grid data: {e}")
        return None
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv_path', type=str, required=True)
    parser.add_argument('--output_csv_path', type=str, required=True)
    parser.add_argument('--regioncontext_region_type', type=str, default='h3')
    parser.add_argument('--regioncontext_region_level', type=int, default=11)
    args = parser.parse_args()

    if 'buildings' in args.input_csv_path:
        context_field_names = ['type', 'name']
    else:
        context_field_names = ['fclass', 'name']
    
    generate_aoi_region(
        in_csv_file_path=args.input_csv_path, 
        out_csv_file_path=args.output_csv_path, 
        context_field_names = context_field_names, 
        region_type=args.regioncontext_region_type, 
        region_level=args.regioncontext_region_level
    )


if __name__ == "__main__":
    main()