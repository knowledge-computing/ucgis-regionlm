import logging
import os, sys

import numpy as np
import pandas as pd
import ast

from sklearn.cluster import KMeans
from kneed import KneeLocator    

from clustering._base import ClusteringBase
from utils import const

class KMeansClustering(ClusteringBase):
    def __init__(self, verbose=False):
        super().__init__(verbose)

    def fit_predict(self, input_csv_file_path, output_csv_file_path, min_component, max_component, clustering_by_group=False, min_required_data_points = 10, verbose=False):
        try:
            self.input_csv_file_path = input_csv_file_path
            self.output_csv_file_path = output_csv_file_path
            # Delete the output file if it exists
            if os.path.exists(output_csv_file_path):
                os.remove(output_csv_file_path)
            self.min_component = min_component
            self.max_component = max_component
            self.clustering_by_group = clustering_by_group
            self.min_required_data_points = min_required_data_points
            self.verbose = verbose

            # Load the dataset
            df = pd.read_csv(self.input_csv_file_path)
            # Drop the fourth column
            #df = df.drop(columns=[const.spabert_emb_field_name])
            # Loop through all rows and group by the name column
            df[const.regioncontext_type_field_name] = ''
            # Add a placeholder column for consistency
            if self.clustering_by_group and const.regioncontext_context_field_name in df.columns:
                grouped_data = df.groupby(const.regioncontext_context_field_name)
            else:
                grouped_data = df.groupby(const.regioncontext_type_field_name)
            if verbose:
                print(f"Number of Groups: {len(grouped_data)}")
            count = 0
            header = pd.DataFrame(columns=df.columns)
            header.to_csv(output_csv_file_path, index=False)

            for name, group in grouped_data:
                count = count +1
                if verbose:
                    print(f"Group Name: {name}, Number of Rows: {len(group)}, Count: {count}/{len(grouped_data)}")
                group[const.regioncontext_type_field_name] =''
                    
                if len(group) < self.min_required_data_points:
                        continue            
                emb = group[const.spabert_emb_enc_field_name].apply(lambda x: ast.literal_eval(x))
                emb_array = np.array(emb.tolist())

                ssd = []
                K = range(self.min_component,self.max_component)
                for k in K:
                    kmeans = KMeans(n_clusters=k)
                    kmeans = kmeans.fit(emb_array)
                    ssd.append(kmeans.inertia_)

                # Use KneeLocator to find the elbow point
                kl = KneeLocator(K, ssd, curve="convex", direction="decreasing")
                optimal_k = kl.elbow
                if optimal_k is None:
                    optimal_k=self.min_component
                # Create a KMeans object
                kmeans = KMeans(n_clusters=optimal_k)
                
                #Fit the data
                kmeans.fit(emb_array)
                
                group[const.regioncontext_type_field_name]  = kmeans.predict(emb_array)
                group.to_csv(output_csv_file_path, mode='a', header=False, index=False)
                
                if self.verbose:
                    logging.info("KMeans clustering performed successfully.")
                
        except Exception as e:
            logging.error(f"Error performing KMeans clustering: {e}")
        return None
