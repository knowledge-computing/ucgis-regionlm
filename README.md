# RegionLM

RegionLM is a geospatial representation learning pipeline built around SpaBERT-style contextual embeddings for OpenStreetMap (OSM) features. It extracts features inside target regions, converts nearby spatial context into pseudo-sentences, trains or applies a spatial BERT model, aggregates feature embeddings into region embeddings, and clusters the resulting regions.

The repository is currently organized as a script-driven research workflow. The notebook [`1_0_preprocess_OSM_data.ipynb`](https://github.com/knowledge-computing/ucgis-regionlm/blob/main/1_0_preprocess_OSM_data.ipynb) shows the intended end-to-end sequence, while the numbered Python scripts provide CLI entrypoints for each stage.

## Pipeline Overview

1. Extract OSM features that intersect a target region.
2. Rasterize polygons such as buildings or land use into H3 or geohash area-of-interest (AOI) points.
3. Build SpaBERT pseudo-sentence JSON from POIs plus optional AOI context.
4. Train SpaBERT or load existing model weights to generate POI embeddings.
5. Aggregate POI embeddings into region-level embeddings.
6. Optionally reduce dimensions and cluster the resulting regions.

## Repository Layout

- [`1_0_preprocess_OSM_data.ipynb`](https://github.com/knowledge-computing/ucgis-regionlm/blob/main/1_0_preprocess_OSM_data.ipynb): notebook walkthrough for preprocessing and pipeline execution
- [`1_1_extract_osm_features_in_region.py`](https://github.com/knowledge-computing/ucgis-regionlm/blob/main/1_1_extract_osm_features_in_region.py): clip OSM shapefiles to the study region and export CSV
- [`1_2_rasterization.py`](https://github.com/knowledge-computing/ucgis-regionlm/blob/main/1_2_rasterization.py): convert polygon features to H3/geohash AOI points
- [`1_3_generate_spabert_json.py`](https://github.com/knowledge-computing/ucgis-regionlm/blob/main/1_3_generate_spabert_json.py): create pseudo-sentence JSONL for SpaBERT
- [`2_1_train_predict_spabert.py`](https://github.com/knowledge-computing/ucgis-regionlm/blob/main/2_1_train_predict_spabert.py): train SpaBERT or generate feature embeddings
- [`2_2_region_embedding.py`](https://github.com/knowledge-computing/ucgis-regionlm/blob/main/2_2_region_embedding.py): aggregate feature embeddings into region embeddings
- [`2_3_dimension_reduction_clustering.py`](https://github.com/knowledge-computing/ucgis-regionlm/blob/main/2_3_dimension_reduction_clustering.py): dimension reduction and KMeans clustering
- [`spabert/`](https://github.com/knowledge-computing/ucgis-regionlm/tree/main/spabert): SpaBERT model and dataset utilities
- [`utils/const.py`](https://github.com/knowledge-computing/ucgis-regionlm/blob/main/utils/const.py): shared field names and default region settings

## Requirements

This project targets Python 3.10+ and depends on PyTorch, Hugging Face Transformers, GeoPandas, Shapely, H3, and related geospatial tooling.

Create an environment and install dependencies:

```bash
conda create --name py310 -y python=3.10
pip install -r requirement.txt
pip install jupyter
```

Notes:
- GeoPandas may require system libraries such as GDAL/GEOS/PROJ depending on your platform.
- Training and embedding generation will use CUDA if PyTorch detects a GPU.

## Data and Pretrained Weights

The current workflow expects external datasets and model weights. The repository README previously referenced the following download location:

- [Model weights and POI data](https://drive.google.com/drive/folders/1eCsvW92ZWvJDkDsUPSaNBb9tpd0ZNmsW?usp=sharing)

You will also need local shapefiles for:

- region boundaries
- OSM POIs
- OSM buildings
- OSM land use

The notebook examples assume a `data/` directory with paths such as:

```text
data/
  nyc_regions/region.shp
  gis_osm_pois_free_1/gis_osm_pois_free_1.shp
  gis_osm_buildings_a_free_1/gis_osm_buildings_a_free_1.shp
  gis_osm_landuse_a_free_1/gis_osm_landuse_a_free_1.shp
```

Important assumptions in the current code:
- Region attributes default to `BoroName` and `NTAName`.
- Region and OSM layers must have valid CRS metadata.
- Geometry columns are written and later re-read as WKT strings in CSV outputs.
- Default region aggregation uses H3 at resolution `11` from [`utils/const.py`](https://github.com/knowledge-computing/ucgis-regionlm/blob/main/utils/const.py).

## Quick Start

### 1. Extract POIs inside the target region

```bash
python 1_1_extract_osm_features_in_region.py \
  --region_shp data/nyc_regions/region.shp \
  --osm_feature_shp data/gis_osm_pois_free_1/gis_osm_pois_free_1.shp \
  --output_shp data/nyc_osm_pois/pois.shp \
  --output_csv data/nyc_osm_pois/pois.csv \
  --predicate within
```

Repeat the same step for buildings and land use if you want AOI context during pseudo-sentence generation.

### 2. Rasterize buildings or land use into AOI points

Example for buildings:

```bash
python 1_2_rasterization.py \
  --input_csv_path data/nyc_osm_buildings/buildings.csv \
  --output_csv_path data/nyc_buildings_h3.csv \
  --regioncontext_region_type h3 \
  --regioncontext_region_level 11
```

The notebook uses the same process for land use, then combines the AOI CSVs into a single `data/aoi.csv`.

### 3. Generate SpaBERT pseudo-sentences

```bash
python 1_3_generate_spabert_json.py \
  --csv_file_path data/nyc_osm_pois/pois.csv \
  --processed_aoi_csv_file_path data/aoi.csv \
  --num_neighbors 100 \
  --search_radius_meters 100
```

This creates a JSONL file under a subdirectory named after the input CSV stem, for example:

```text
data/nyc_osm_pois/pois/pois_pseudo_sentence_v2_nn100_sdm100.json
```

### 4. Train SpaBERT

```bash
python 2_1_train_predict_spabert.py \
  --mode train \
  --json_file_path data/nyc_osm_pois/pois/pois_pseudo_sentence_v2_nn100_sdm100.json \
  --model_save_dir model_weights/
```

By default, the script initializes from `bert-base-uncased` and saves `.pth` checkpoints to `model_weights/`.

If you already have pretrained weights, skip training and use them directly for prediction.

### 5. Generate POI embeddings

```bash
python 2_1_train_predict_spabert.py \
  --mode predict \
  --json_file_path data/nyc_osm_pois/pois/pois_pseudo_sentence_v2_nn100_sdm100.json \
  --model_save_dir model_weights/ \
  --csv_file_path data/nyc_osm_pois/pois/pois_pseudo_sentence_v2_nn100_sdm100_spabert_embedding.csv
```

### 6. Aggregate to region embeddings

```bash
python 2_2_region_embedding.py \
  --in_csv_file_path data/nyc_osm_pois/pois/pois_pseudo_sentence_v2_nn100_sdm100_spabert_embedding.csv \
  --out_csv_file_path data/nyc_osm_pois/pois/pois_pseudo_sentence_v2_nn100_sdm100_region_embedding.csv
```

This groups feature embeddings by spatial cell and averages embeddings within each H3 or geohash region.

### 7. Reduce dimensions and cluster regions

```bash
python 2_3_dimension_reduction_clustering.py \
  --input_region_embedding_csv data/nyc_osm_pois/pois/pois_pseudo_sentence_v2_nn100_sdm100_region_embedding.csv \
  --output_dimension_reduce_csv data/nyc_osm_pois/pois/pois_pseudo_sentence_v2_nn100_sdm100_dimreduce.csv \
  --output_cluster_csv data/nyc_osm_pois/pois/pois_pseudo_sentence_v2_nn100_sdm100_cluster.csv
```

The script currently calls clustering directly. If you also need the autoencoder-based reduction stage, uncomment the `dimension_reduction(args)` call in [`2_3_dimension_reduction_clustering.py`](https://github.com/knowledge-computing/ucgis-regionlm/blob/main/2_3_dimension_reduction_clustering.py) before running it.

## Outputs

Typical outputs produced by the pipeline:

- clipped feature shapefiles and CSVs
- AOI CSVs derived from buildings and land use
- pseudo-sentence JSONL for SpaBERT training
- POI-level SpaBERT embedding CSV
- region-level embedding CSV
- optional reduced-dimension embedding CSV
- cluster assignment CSV

## Current Limitations
- The workflow is configured around the NYC example paths shown in the notebook.
- Some region metadata columns are hard-coded (`BoroName`, `NTAName`).
- AOI CSV merging is demonstrated in the notebook rather than exposed as a dedicated CLI script.
- The final clustering script does not run dimensionality reduction unless you modify the file.
- Dependency management is incomplete in `requirements.txt`.

## Recommended Workflow

For a first run, start with the notebook [`1_0_preprocess_OSM_data.ipynb`](https://github.com/knowledge-computing/ucgis-regionlm/blob/main/1_0_preprocess_OSM_data.ipynb), then move to the individual scripts once your data layout and parameter choices are stable.

