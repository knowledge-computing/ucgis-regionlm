from dimension_reducer.autoencoder import AutoencoderReducer 
from clustering.kmeans_clustering import KMeansClustering
from utils import const
import argparse 


def dimension_reduction(args):
    # Dimension reduction
    autoencoder_reducer = AutoencoderReducer()
    autoencoder_reducer.fit_transform(in_csv_file_path=args.input_region_embedding_csv,
                                        out_csv_file_path=args.output_dimension_reduce_csv,
                                        epoch=300)

def clustering(args):
    # Clustering
    min_component = 3  # 10 - 50
    max_component = 4

    kmeans_clustering = KMeansClustering()
    kmeans_clustering.fit_predict(input_csv_file_path=args.output_dimension_reduce_csv,
                                    output_csv_file_path=args.output_cluster_csv,
                                    min_component=min_component, max_component=max_component, clustering_by_group=False, min_required_data_points=10, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_region_embedding_csv', type=str, required=True, default = 'data/nyc_osm_pois/pois/pois_pseudo_sentence_v2_nn100_sdm100_region_embedding.csv')
    parser.add_argument('--output_dimension_reduce_csv', type=str, required=True, default = 'data/nyc_osm_pois/pois/pois_pseudo_sentence_v2_nn100_sdm100_dimension_reduce.csv')
    parser.add_argument('--output_cluster_csv', type=str, required=True, default = 'data/nyc_osm_pois/pois/pois_pseudo_sentence_v2_nn100_sdm100_cluster.csv')
    

    args = parser.parse_args()


    # dimension_reduction(args)
    clustering(args)



    