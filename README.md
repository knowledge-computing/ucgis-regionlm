# RegionLM

RegionLM is a geospatial representation learning pipeline built around SpaBERT-style contextual embeddings for OpenStreetMap (OSM) features. It extracts features inside target regions, converts nearby spatial context into pseudo-sentences, trains or applies a spatial BERT model, aggregates feature embeddings into region embeddings, and clusters the resulting regions.

The repository is currently organized as a script-driven research workflow. The notebook [`1_0_preprocess_OSM_data.ipynb`](https://github.com/knowledge-computing/ucgis-regionlm/blob/main/1_0_preprocess_OSM_data.ipynb) shows the intended end-to-end sequence, while the numbered Python scripts provide CLI entrypoints for each stage.

## Prerequisites 
### 1. Data and Pretrained Weights

The current workflow expects external datasets and model weights:

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

### 2. GitHub Repository

The source code is available in the following repository:

https://github.com/knowledge-computing/ucgis-regionlm

Clone the repository to your local machine:

```bash
git clone https://github.com/knowledge-computing/ucgis-regionlm.git
```
  
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

## Workflow

For a step-by-step walkthrough of the complete workflow, please refer to the [tutorial notebook](https://github.com/knowledge-computing/ucgis-regionlm/blob/main/0_regionlm_tutorial.ipynb)
