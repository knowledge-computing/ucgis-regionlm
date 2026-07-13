import pandas as pd
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt
import contextily as ctx
import argparse 




def visualize(args):
    # -----------------------------------------------------
    # Load CSV
    # -----------------------------------------------------

    # Read CSV
    df = pd.read_csv(args.input_cluster_csv)

    # Convert WKT string to shapely Polygon
    df["geometry"] = df["geometry"].apply(wkt.loads)

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry="geometry",
        crs="EPSG:4326"
    )

    num_clusters = gdf["regioncontext_type"].nunique()
    print(f"Number of unique clusters: {num_clusters}")

    # Convert to Web Mercator for plotting with contextily
    gdf = gdf.to_crs(epsg=3857)

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------

    fig, ax = plt.subplots(figsize=(14, 14))

    gdf.plot(
        ax=ax,
        column="regioncontext_type",
        cmap="tab20",
        edgecolor="black",
        linewidth=0.2,
        alpha=0.8,
        legend=True,
    )

    # Zoom-in to Downtown Manhattan bounding box (Web Mercator)
    ax.set_xlim(-8250000, -8225000)
    ax.set_ylim(4965000, 4990000)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=300, bbox_inches="tight")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_cluster_csv', type=str, required=True, default = 'data/nyc_osm_pois/pois/pois_pseudo_sentence_v2_nn100_sdm100_cluster.csv')
    # parser.add_argument('--output_png', type=str, required=True, default = 'data/nyc_osm_pois/h3_clusters.png')
    parser.add_argument('--input_cluster_csv', type=str,  default = 'data/nyc_osm_pois/pois/pois_pseudo_sentence_v2_nn100_sdm100_cluster.csv')
    parser.add_argument('--output_png', type=str,  default = 'data/nyc_osm_pois/h3_clusters.png')
    

    args = parser.parse_args()

    visualize(args)
