import logging
import os, sys
import ast

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely import Point, wkt
import geohash
import h3 

from model_trainer._utils.helpers import geohash_to_polygon, cell_to_shapely
from utils import const

import argparse

class GenerateRegionEmbCSV:
    def __init__(self):
        pass

    def fit_tranform(self, in_csv_file_path, out_csv_file_path, region_type='h3', region_level=10):
        # Add your code here to generate the JSON
        try:
            self.in_csv_file_path = in_csv_file_path
            self.out_csv_file_path = out_csv_file_path
            self.region_type = region_type
            self.region_level = region_level
            self.grouped_df = None

            df = pd.read_csv(self.in_csv_file_path)
            df[const.regioncontext_geometry_field_name] = df[const.regioncontext_geometry_field_name].apply(wkt.loads)

            self.gdf = gpd.GeoDataFrame(df, geometry=const.regioncontext_geometry_field_name, crs="EPSG:4326")
            self.gdf[const.spabert_emb_field_name] = self.gdf[const.spabert_emb_field_name].apply(lambda x: np.array(ast.literal_eval(x)))

            if self.region_type == 'geohash':
                self.gdf[self.region_type] = self.gdf[const.regioncontext_geometry_field_name].apply(lambda x: geohash.encode(x.y, x.x, precision=self.region_level))
                self.grouped_df = self.gdf.groupby(self.region_type)[const.spabert_emb_field_name].apply(lambda x: np.mean(np.vstack(x), axis=0)).reset_index()
                self.grouped_df[const.regioncontext_geometry_field_name] = self.grouped_df[self.region_type].apply(lambda x: geohash_to_polygon(x)) 
                
            elif self.region_type == 'h3' :
                self.gdf[self.region_type] = self.gdf[const.regioncontext_geometry_field_name].apply(lambda x: h3.geo_to_h3(x.y, x.x, self.region_level))
                self.grouped_df = self.gdf.groupby(self.region_type)[const.spabert_emb_field_name].apply(lambda x: np.mean(np.vstack(x), axis=0)).reset_index()
                self.grouped_df[const.regioncontext_geometry_field_name] = self.grouped_df[self.region_type].apply(lambda x: cell_to_shapely(x)) 

            self.grouped_df[const.spabert_emb_field_name] = self.grouped_df[const.spabert_emb_field_name].apply(lambda x: x.tolist())
            self.grouped_df.to_csv(self.out_csv_file_path, index=False)
            return self.grouped_df

        except Exception as e:
            logging.error(f"Error predicting data: {e}")
        return None
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_csv_file_path', type=str, required=True, default = 'data/nyc_osm_pois/pois/pois_pseudo_sentence_v2_nn100_sdm100_spabert_embedding.csv')
    parser.add_argument('--out_csv_file_path', type=str, required=True, default = 'data/nyc_osm_pois/pois/pois_pseudo_sentence_v2_nn100_sdm100_region_embedding.csv')

    args = parser.parse_args()


    # Generate the region embeddings
    generate_region_emb_csv = GenerateRegionEmbCSV()
    generate_region_emb_csv.fit_tranform(in_csv_file_path=args.in_csv_file_path,
                                            out_csv_file_path=args.out_csv_file_path,
                                            region_type=const.regioncontext_region_type, region_level=const.regioncontext_region_level)


