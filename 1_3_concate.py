import os
import pandas as pd
from typing import List, Optional

def concat_csv_files(
    file_paths: List[str],
    output_file: str,
    columns: Optional[List[str]] = None,
    ignore_index: bool = True,
    add_source: bool = False,
    encoding: Optional[str] = None
) -> pd.DataFrame:
    """
    Concatenate multiple CSV files, optionally keeping only selected columns,
    and save the result to an output CSV file.

    Args:
        file_paths: List of input CSV file paths.
        output_file: Path to the output CSV file.
        columns: List of columns to keep. If None, keep all columns.
        ignore_index: Whether to reset the index in the concatenated result.
        add_source: Whether to add a column with the source file name.
        encoding: Optional file encoding for reading/writing CSVs.

    Returns:
        The concatenated DataFrame.
    """
    dfs = []

    for fp in file_paths:
        if not os.path.exists(fp):
            print(f"Warning: {fp} not found, skipping.")
            continue

        df = pd.read_csv(fp, encoding=encoding)

        if columns is not None:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                print(f"Warning: {fp} is missing columns {missing_cols}, skipping.")
                continue
            df = df[columns]

        if add_source:
            df["source_file"] = os.path.basename(fp)

        dfs.append(df)

    if not dfs:
        raise ValueError("No valid CSV files found.")

    result = pd.concat(dfs, ignore_index=ignore_index)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    result.to_csv(output_file, index=False, encoding=encoding)

    return result


files = ["data/nyc_buildings_h3.csv", "data/nyc_landuse_h3.csv"]

data = concat_csv_files(
    file_paths=files,
    output_file="data/aoi.csv",
    columns=["osm_id", "agg_category", "geometry", "h3_list", "poi_aoi"],
    add_source=False
)