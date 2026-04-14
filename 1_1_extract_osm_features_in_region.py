import argparse
import geopandas as gpd
from pathlib import Path


def intersect_region_and_osm_features(
    region_shp: str,
    osm_feature_shp: str,
    output_shp: str,
    predicate: str = "intersects",
    region_attr_cols: list = None,
):
    """
    Extract OSM features within/intersecting a target region and preserve
    both OSM attributes and selected region attributes.

    Parameters
    ----------
    region_shp : str
        Path to the region shapefile.
    osm_feature_shp : str
        Path to the OSM feature shapefile.
    output_shp : str
        Path to the output shapefile.
    predicate : str, optional
        Spatial predicate for the join. Common options:
        - "intersects"
        - "within"
        - "contains"
        - "touches"
        - "crosses"
        - "overlaps"
        Default is "intersects".
    region_attr_cols : list, optional
        Region columns to preserve in the output.
        Default is ["BoroName", "NTAName"].

    Returns
    -------
    geopandas.GeoDataFrame
        Output GeoDataFrame containing:
        - all columns from OSM features
        - selected columns from region shapefile
        - geometry from OSM features
    """

    if region_attr_cols is None:
        region_attr_cols = ["BoroName", "NTAName"]

    region_path = Path(region_shp)
    osm_path = Path(osm_feature_shp)
    output_path = Path(output_shp)

    if not region_path.exists():
        raise FileNotFoundError(f"Region shapefile not found: {region_shp}")
    if not osm_path.exists():
        raise FileNotFoundError(f"OSM feature shapefile not found: {osm_feature_shp}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read shapefiles
    region_gdf = gpd.read_file(region_shp)
    osm_gdf = gpd.read_file(osm_feature_shp)

    if region_gdf.empty:
        raise ValueError("Region shapefile is empty.")
    if osm_gdf.empty:
        raise ValueError("OSM feature shapefile is empty.")

    if region_gdf.crs is None:
        raise ValueError("Region shapefile has no CRS defined.")
    if osm_gdf.crs is None:
        raise ValueError("OSM feature shapefile has no CRS defined.")

    # Reproject OSM to region CRS if needed
    if osm_gdf.crs != region_gdf.crs:
        osm_gdf = osm_gdf.to_crs(region_gdf.crs)

    # Check region columns exist
    missing_cols = [col for col in region_attr_cols if col not in region_gdf.columns]
    if missing_cols:
        raise ValueError(f"These region columns are missing: {missing_cols}")

    # Keep only requested region attributes + geometry
    region_subset = region_gdf[region_attr_cols + ["geometry"]].copy()

    # Optional: fix invalid geometries
    region_subset = region_subset[region_subset.geometry.notnull()].copy()
    osm_gdf = osm_gdf[osm_gdf.geometry.notnull()].copy()

    # Spatial join
    joined = gpd.sjoin(
        osm_gdf,
        region_subset,
        how="inner",
        predicate=predicate
    )

    # Remove helper column from sjoin
    joined = joined.drop(columns=["index_right"], errors="ignore")

    # Save output
    joined.to_file(output_shp)

    print(f"Predicate: {predicate}")
    print(f"Selected {len(joined)} features")
    print(f"Saved to: {output_shp}")

    return joined


def read_shapefile_to_csv(input_shapefile_path, columns_to_keep, output_csv_path):
    try:
        # Check if the shapefile exists
        input_shapefile_path = Path(input_shapefile_path)
        if not input_shapefile_path.exists():
            raise FileNotFoundError(f"Shapefile not found at {input_shapefile_path}")
        
        # Read the shapefile using geopandas
        shapefile = gpd.read_file(input_shapefile_path)
        print(shapefile.head(2))
        # Convert the geometry to WGS84
        shapefile = shapefile.to_crs("EPSG:4326")

        #shapefile['geometry'] = shapefile['geometry'].apply(lambda geom: geom.wkt)
        # Loop through the columns in the shapefile
        for column in shapefile.columns:
            # Check if the column is in the dictionary
            if column not in columns_to_keep and column != 'geometry':
                # Delete the column from the shapefile
                shapefile.drop(column, axis=1, inplace=True)
        # Save the shapefile to CSV
        shapefile.to_csv(output_csv_path, index=False)
        return shapefile
    except FileNotFoundError as e:
        print(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--region_shp', type=str, required=True)
    parser.add_argument('--osm_feature_shp', type=str, required=True)
    parser.add_argument('--output_shp', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--predicate', type=str, default='within')
    
    args = parser.parse_args()
    
    result = intersect_region_and_osm_features(
        region_shp=args.region_shp,
        osm_feature_shp=args.osm_feature_shp,
        output_shp=args.output_shp,
        predicate=args.predicate
    )

    if args.output_csv:
        columns_to_keep = ['osm_id', 'fclass', 'name', 'type', 'BoroName', 'NTAName']
        read_shapefile_to_csv(args.output_shp, columns_to_keep, args.output_csv)


if __name__ == "__main__":
    main()


# python 1_1_extract_osm_features.py --region_shp "data/nyc_regions/region.shp" --osm_feature_shp "data/gis_osm_pois_free_1/gis_osm_pois_free_1.shp" --output_shp "data/nyc_osm_pois/pois.shp"