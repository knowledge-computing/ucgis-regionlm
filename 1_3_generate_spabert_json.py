import os, sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import json
import scipy.spatial as scp

from shapely import wkt

from utils import const

class GenerateSpaBERTJSON:
    def __init__(
            self, 
            verbose: bool = False,):
        self.verbose = verbose
        self.__df = None

    def fit_transform(self, csv_file_path, context_field_names, geometry_field_name,
            num_neighbors, 
            search_radius_meters, pseudo_sentence_json_file_path, processed_aoi_csv_file_path):
        self.generate_json(csv_file_path, context_field_names, geometry_field_name,
            num_neighbors, 
            search_radius_meters, processed_aoi_csv_file_path)
        self.save_json(pseudo_sentence_json_file_path)

    def generate_json(self, csv_file_path, context_field_names, geometry_field_name,
            num_neighbors=100, 
            search_radius_meters=100, processed_aoi_csv_file_path=None) -> pd.DataFrame:
        
        self.csv_file_path = csv_file_path
        self.context_field_names = context_field_names
        self.geometry_field_name = geometry_field_name
        self.num_neighbors = num_neighbors
        self.search_radius_meters = search_radius_meters
        self.processed_aoi_csv_file_path = processed_aoi_csv_file_path

        df = pd.read_csv(self.csv_file_path)
        df[const.aggregated_field_name]=''
        for field_name in self.context_field_names:
            df[field_name] = df[field_name].astype(str).apply(lambda x: x.lower().encode('ascii', 'ignore').strip().decode('ascii') if x != 'nan' else '')
            df[const.aggregated_field_name] = df[const.aggregated_field_name] + df[field_name] + ':'

        df = df[df[const.aggregated_field_name].notnull()]
        df[const.aggregated_field_name] = df[const.aggregated_field_name].apply(lambda x: x.strip(':'))
        df = df[df[const.aggregated_field_name] != '']
        
        df = df[df[self.geometry_field_name].notnull()]
        df = df.reset_index(drop=True)

        if self.processed_aoi_csv_file_path is not None:
            building_df = pd.read_csv(self.processed_aoi_csv_file_path)
            df = pd.concat([df[[const.aggregated_field_name, self.geometry_field_name]], building_df[[const.aggregated_field_name, self.geometry_field_name, const.poi_aoi_field_name]]], axis=0, ignore_index=True, sort=False)

        df[self.geometry_field_name] = df[self.geometry_field_name].apply(wkt.loads)
        df[const.x_y_field_name] = df[self.geometry_field_name].apply(lambda x: [x.y, x.x])
        df[const.psuedo_sentence_field_name] =''
        ordered_neighbor_coordinate_list = scp.KDTree(df[const.x_y_field_name].values.tolist())

        for index, row in df.iterrows():
            if const.poi_aoi_field_name in row:
                if row[const.poi_aoi_field_name] == const.aoi_field_value:
                    continue
            nearest_dist, nearest_neighbors_idx = ordered_neighbor_coordinate_list.query([row[const.x_y_field_name]], k=self.num_neighbors+1)
            # Loop through nearest_neighbors_idx and remove elements based on some condition
            filtered_neighbors_idx = []
            filtered_nearest_dist = []
            idx = 0
            for i in nearest_dist[0]:
                if float(i) < float(self.search_radius_meters/const.wgs842meters):   #100 meters
                    filtered_neighbors_idx.append(
                        nearest_neighbors_idx[0][idx])
                    filtered_nearest_dist.append(i)
                idx += 1

            nearest_neighbors_context = []
            nearest_neighbors_coords = []
            nearest_neighbors_dist = []
            idx = 0

            for i in filtered_neighbors_idx:
                neighbor_context = df[const.aggregated_field_name][i]
                neighbor_coords = df[const.x_y_field_name][i]
                nearest_neighbors_context.append(neighbor_context)
                nearest_neighbors_coords.append(
                    {const.neighbor_coordinates: neighbor_coords})
                nearest_neighbors_dist.append(
                    {const.neighbor_distance: float(filtered_nearest_dist[idx])*const.wgs842meters})
                idx += 1
            if self.verbose:
                print(f"Processing row {index+1}/{len(df)}")
            neighbor_info = {const.neighbor_context_list: nearest_neighbors_context,
                            const.neighbor_geometry_list: nearest_neighbors_coords, const.neighbor_distance: nearest_neighbors_dist}
            ps = {const.psuedo_sentence_info: {const.pivot_id: index+1, const.pivot_context: row[const.aggregated_field_name], const.pivot_geometry: {
                    const.pivot_coordinates: row[const.x_y_field_name]}}, const.neighbor_info: neighbor_info}
            df.at[index, const.psuedo_sentence_field_name] = ps
        if self.verbose:
            print(df.head(2))
        self.__df = df[df[const.psuedo_sentence_field_name]!=''][[const.psuedo_sentence_field_name]]
        return self.__df
    
    def save_json(self, pseudo_sentence_json_file_path):
        with open(pseudo_sentence_json_file_path, 'w') as out_f:
            for index, row in self.__df.iterrows():
                if const.poi_aoi_field_name in row:
                    if row[const.poi_aoi_field_name] == const.aoi_field_value:
                        continue
                out_f.write(json.dumps(row[const.psuedo_sentence_field_name]))
                out_f.write('\n')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file_path', type=str, required=True)
    parser.add_argument('--num_neighbors', type=int, default=100)
    parser.add_argument('--search_radius_meters', type=int, default=100)
    parser.add_argument('--processed_aoi_csv_file_path', type=str, default='')
    args = parser.parse_args()

    parent_dir = Path(args.csv_file_path).resolve().parent
    file_name = Path(args.csv_file_path).resolve().stem
    data_dir = Path(parent_dir) / file_name
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    pseudo_sentence_json_file_path = data_dir / \
            f"{file_name}_pseudo_sentence_v2_nn{args.num_neighbors}_sdm{args.search_radius_meters}.json"
        
    generate_spabert_json = GenerateSpaBERTJSON()
    generate_spabert_json.fit_transform(csv_file_path=args.csv_file_path,
                                        context_field_names=['fclass', 'name'],
                                        geometry_field_name=const.regioncontext_geometry_field_name,
                                        num_neighbors=args.num_neighbors,
                                        search_radius_meters=args.search_radius_meters,
                                        pseudo_sentence_json_file_path=pseudo_sentence_json_file_path,
                                        processed_aoi_csv_file_path=args.processed_aoi_csv_file_path)


if __name__ == "__main__":
    main()









