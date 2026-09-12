import json
import logging
from pathlib import Path
import re
import time
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
import topojson as tp
from tqdm import tqdm

from spacey_falcon_9_project.config import (
    dataset_interim,
    dataset_processed,
    external_dir,
    file_name_gcat,
    file_name_ll2,
    gcat_url,
    geodata_paths,
    geodata_urls,
    interim_dir,
    ll2_url,
    processed_dir,
    random_state,
    raw_dir,
    test_set,
    training_set,
)

#=====================================================================================================================
# DATA COLLECTION
#=====================================================================================================================

def get_ll2_launches(offset: int) -> None|dict:
    """
    Function for making get requests to LL2 API

    :param offset: int; offset to be used for the API request
    :return: dict; contents of the response from the API request
    """
    ll2_api = ll2_url
    query_params = dict(
        mode="detailed", limit=100, rocket__configuration__name="Falcon 9", offset=offset
    )

    for attempt in range(3):
        try:
            response = requests.get(ll2_api, params=query_params)
            if response.status_code == 504:
                print("504 Server Error. Retrying in {} seconds...".format(2 * (attempt + 1)))
                time.sleep(2 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            print("Error: {}".format(e))
            return None

    print("Failed after 3 attempts (504 Server Error)")
    return None

def download_all_ll2_launches() -> None|list[Path]:
    """
    :return: list[Path]; file paths to the LL2 launches
    """
    file_paths = []
    for offset in tqdm(range(0, 700, 100), desc="Downloading all LL2 launches"):
        raw_dir.mkdir(parents=True, exist_ok=True)
        file_name = "ll2-api-2.3.0-launches-previous-{}.json".format(offset)
        file_path = raw_dir / file_name
        if file_path.is_file():
            file_paths.append(file_path)
            continue
        try:
            data = get_ll2_launches(offset)
            with open(file_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print("Error: {}".format(e))
            break
        file_paths.append(file_path)
    return file_paths

def download_launch_data_static(url: str, file_name: str, query_params: dict|None = None, headers: dict|None = None) -> None|Path:
    """
    :param url: str; URL to download the data from
    :param file_name: str; File name to save the data to
    :param query_params: dict or None; Default None; Parameters to be used in requesting data
    :param headers: dict or None; Default None; Headers to be used in requesting data
    :return: Path; file path to the downloaded file
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / file_name

    if file_path.is_file():
        return file_path

    response = requests.get(url, params=query_params, headers=headers)

    try:
        response.raise_for_status()

    except requests.exceptions.HTTPError as e:
        print("Error: {}".format(e))
        return None

    try:
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path
    except Exception as e:
        print("Error: {}".format(e))
        return None

# Partial Data Transformation - Merging LL2 launch, GCAT launch, and Course-provided launch data
def merge_ll2_launch_data(json_paths: None|list[Path], merged_path: None|Path = None) -> None|Path:
    """
    Function to merge LL2 launches that were extracted and downloaded with the download_all_ll2_launches() function
    :param json_paths: list[Path]; file paths to the paginated files. Particularly, the returned output of the
                       download_all_ll2_launches() function
    :param merged_path: None or Path; path to the merged files. If not specified, the merged file will be saved in
                        the parent directory > data/interim folder.
    """

    if not json_paths:
        return None

    merged = []
    for path in json_paths:
        with open(path, 'r') as f:
            launch_page = json.load(f)
        merged.extend(launch_page['results'])

    if merged_path is None:
        interim_dir.mkdir(parents=True, exist_ok=True)
        merged_path = interim_dir / file_name_ll2

    with open(merged_path, 'w') as f:
        json.dump(merged, f)

    return merged_path

def get_path(d: dict, path: str) -> Any:
    """
    A utility function for quickly accessing items in a nested dictionary
    from a specified string of keys joined by dots '.'

    :param d: dict; nested dictionary containing items to be accessed
    :param path: str; string of keys joined by dots '.'
    :return: None or Any; the item to be accessed, whatever its type
    """
    for key in path.split("."):
        if isinstance(d, list):
            d = d[0]
        d = d.get(key) if isinstance(d, dict) else None
    return d

def transform_ll2_launches(json_path: None | str | Path = None) -> pd.DataFrame:
    """
    A function to transform LL2 raw data given as a JSON file path

    :param json_path: str or Path; local path to the JSON file containing raw ll2 launches data; if None, dataset is
    assumed saved in the interim folder with the file name 'll2-api-2.3.0-launches-previous-merged.json'

    :return: pd.DataFrame; transformed LL2 launch data containing relevant columns for analysis, but requires further wrangling
    """

    if not json_path:
        json_path = interim_dir / file_name_ll2

    with open(json_path, "r") as f:
        data_json = json.load(f)

    json_keys = dict(
        launch_designator="launch_designator",
        net="net",
        booster_version="rocket.configuration.name",
        orbit="mission.orbit.name",
        mission_type="mission.type",
        launch_site="pad.name",
        landing_success="rocket.launcher_stage.landing.success",
        landing_type="rocket.launcher_stage.landing.type.abbrev",
        flights="rocket.launcher_stage.launcher.flights",
        reused="rocket.launcher_stage.reused",
        landing_pad="rocket.launcher_stage.landing.landing_location.abbrev",
        block="rocket.configuration.variant",
        flight_number="rocket.launcher_stage.launcher_flight_number",
        serial="rocket.launcher_stage.launcher.serial_number",
        longitude="pad.longitude",
        latitude="pad.latitude",
    )

    launch_dict = {}
    for col, path in json_keys.items():
        col_values = []
        for launch in data_json:
            col_values.append(get_path(launch, path))
        launch_dict.update({col: col_values})

    data_df = pd.DataFrame(launch_dict)

    # Derived column - reused_count
    data_df["reused_count"] = data_df["flight_number"] - 1
    data_df = data_df.drop("flight_number", axis=1)

    # Derived column - launch_date
    data_df["launch_date"] = pd.to_datetime(data_df["net"]).dt.date
    data_df = data_df.drop("net", axis=1)

    # Drop missing values in launch designator
    data_df = data_df.dropna(subset="launch_designator")

    # Convert landing success column to str
    data_df['landing_success'] = data_df['landing_success'].map(str)

    # Regroup: "Block 4", "v1.0", "v1.1" -> "Legacy"
    data_df['block'] = data_df['block'].map(lambda x: "Legacy" if x in ['Block 4', 'Full Thrust', 'v1.0', 'v1.1'] else x)

    return data_df.reset_index(drop=True)

def transform_gcat_data(gcat_path: None | str | Path = None) -> pd.DataFrame:
    """
    Function to transform GCAT raw data from the specified file path
    :param gcat_path: str or Path; file path to the GCAT raw data. If None, dataset is assumed to be saved in raw data
    folder with the file name 'mcdowell-gcat-launch-data.tsv'

    :return: pd.DataFrame; dataframe containing relevant data from GCAT
    """

    if not gcat_path:
        gcat_path = raw_dir / file_name_gcat

    gcat_df = pd.read_csv(gcat_path, sep="\t", skiprows=(lambda x: x in [1]))

    pattern = r"^\d{4}\s+\w{3}\s+\d{1,2}"
    format_date = re.compile(pattern)

    def format_gcat_date(x):
        match = format_date.match(x)
        if not match:
            return None
        else:
            return match.group()

    launch_date = gcat_df["Launch_Date"].map(format_gcat_date)
    gcat_df["launch_date"] = pd.to_datetime(launch_date).dt.date

    gcat_df = gcat_df[["#Launch_Tag", "launch_date", "OrbPay"]]
    gcat_df = gcat_df.rename(
        columns={"#Launch_Tag": "launch_designator", "OrbPay": "payload_mass"}
    )  # rename '#Launch_Tag' to launch designator
    gcat_df["launch_designator"] = gcat_df[
        "launch_designator"
    ].str.strip()  # remove trailing white spaces from launch_designator column

    return gcat_df.reset_index(drop=True)

def merge_launch_data(ll2_df: pd.DataFrame, gcat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to merge transformed ll2 launches data, transformed GCAT launch data, and transformed course data.
    ll2 launches and gcat data will be merged on `launch_designator`, and resulting merged DataFrame will be merged
    with course data on `launch_date`. Each merge is a one-to-one inner join. The final merged DataFrame will be
    saved to `csv_path`

    :param ll2_df: pd.DataFrame; transformed ll2 data
    :param gcat_df: pd.DataFrame; transformed gcat data
    :return: pd.DataFrame; merged DataFrame
    """

    # Merge LL2 and GCAT
    merged_df = ll2_df.merge(gcat_df, on='launch_designator', how='inner', validate='1:1')

    mask = (merged_df['launch_date_x'] == merged_df['launch_date_y'])
    if mask.all():
        merged_df['launch_date'] = merged_df['launch_date_x']
        merged_df = merged_df.drop(['launch_date_x', 'launch_date_y'], axis=1)
    else:
        raise Exception('Mismatch: launch_date in LL2 Launches does not match with launch_date in GCAT Launches')

    # Drop launches with overlapping dates
    counts = merged_df['launch_date'].value_counts()
    duplicate_dates = merged_df[merged_df['launch_date'].map(lambda x: x in counts[counts>1].index)]
    merged_df = merged_df.drop(index=duplicate_dates.index)

    return merged_df.reset_index(drop=True)

def add_class(launch_df: pd.DataFrame, save_path: None | str | Path = None) -> None | Path:
    """
    Function that adds the target 'Class' column to the launch data from the specified launch csv path,
    and saves the result to the specified save path.

    :param launch_df: pd.DataFrame; DataFrame containing launch data
    :param save_path: str or Path; Default None; file path where the result is to be saved. If None,
                      result will be saved in parent directory > data/interim folder.
    :return save_path: Path; file path where result is saved.
    """

    def outcome_map(entry: str) -> int:
        """
        Utility function that maps an outcome to either the integer 1 or 0.
        If the outcome is a landing success, the function returns 1; otherwise,
        the function returns 0

        :param entry: str; a landing outcome in the format "(Landing Success) (Landing Pad)", e.g., True ASDS.
        :return outcome: int; if the entry is a landing success, `outcome=1`; otherwise, `outcome=0`.
        """

        if entry=='True':
            landing_outcome = 1
        else:
            landing_outcome = 0

        return landing_outcome

    launch_df["class"] = launch_df["landing_success"].map(outcome_map)

    # Drop NaN or NA values
    from_cols = list(launch_df.columns)
    from_cols.remove("landing_success")
    launch_df.dropna(axis=0, inplace=True, subset=from_cols)

    if not save_path:
        interim_dir.mkdir(parents=True, exist_ok=True)
        save_path = interim_dir / dataset_interim
    else:
        save_path = Path(save_path)

    launch_df.to_csv(save_path, index=False)

    return save_path

#=====================================================================================================================
# GEODATA -- Addition of columns for distance to nearest highway, railway, and coastline
#=====================================================================================================================
def download_layers_data() -> dict[str, Path]:
    """
    Function for downloading geodata for highways, railways, and coastlines. File path where data is downloaded
    is set as parent directory > data/external folder.

    :return: dict[str, Path]; Dictionary of file paths.
    """
    external_dir.mkdir(parents=True, exist_ok=True)
    for key, file, url in zip(geodata_paths.keys(), geodata_paths.values(), geodata_urls):
        if not file.is_file():
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                with open(file, "wb") as f:
                    for chunk in tqdm(
                        r.iter_content(chunk_size=8192),
                        total=total // 8192,
                        unit="chunk",
                        desc=f"Downloading {key} geodata",
                    ):
                        f.write(chunk)

    return geodata_paths

def add_nearest_highway(data: pd.DataFrame, us_roadmap: None | str | Path = None) -> pd.DataFrame:
    """
    Function to calculate distance of corresponding launch site to nearest highway.
    Adds the distance to nearest highway column to `data`.

    :param data: pd.DataFrame; launch data

    :param us_roadmap: None, or str, or Path; Default None; file path to US roadmap geodata. If None,
                       function loads the geodata from the default path:
                       parent directory > data/external/us_road_map.json

    :return: pd.DataFrame; launch data with `nearest_highway` column
    """
    # Load US Roadmap geodata
    if not us_roadmap:
        us_roadmap = geodata_paths["US roadmap"]

    with open(us_roadmap, "rb") as f:
        roads_json = json.load(f)

    roads_topo = tp.Topology(roads_json, object_name="roads")
    roads_gdf = roads_topo.to_gdf(crs="EPSG:4326")
    mask = roads_gdf["type"] == "Major Highway"
    roads_gdf = (
        roads_gdf[mask].reset_index(drop=True).to_crs(epsg=5070)
    )  # reprojected to EPSG:5070

    # Define function -- convert reproject (lon,lat) to EPSG:5070 and calculate nearest distance
    def func(row):
        lon = row.longitude
        lat = row.latitude
        row_conv = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=5070)[0]
        return roads_gdf.geometry.distance(row_conv).min() / 1000

    # Apply function to data
    data["nearest_highway"] = data.apply(func, axis=1)

    return data


def add_nearest_railway(data: pd.DataFrame, us_railways: None | str | Path = None) -> pd.DataFrame:
    """
    Function to calculate distance of corresponding launch site to nearest railway.
    Adds the distance to nearest railway column to `data`.

    :param data: pd.DataFrame; launch data

    :param us_railways: None, or str, or Path; Default None; file path to US railways geodata. If None,
                        functions loads the geodata from the default path:
                        parent directory > data/external/us_railways.geojson

    :return: pd.DataFrame; launch data with `nearest_railway` column
    """
    # Load US Roadmap geodata
    if not us_railways:
        us_railways = geodata_paths["US railways"]

    railways_gdf = gpd.read_file(us_railways)
    mask = railways_gdf["NET"] == "M"
    railways_gdf = railways_gdf[mask][["geometry"]].to_crs(epsg=5070)  # reprojected to EPSG:5070

    # Define function -- convert reproject (lon,lat) to EPSG:5070 and calculate nearest distance
    def func(row):
        lon = row.longitude
        lat = row.latitude
        row_conv = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=5070)[0]
        return railways_gdf.geometry.distance(row_conv).min() / 1000

    # Apply function to data
    data["nearest_railway"] = data.apply(func, axis=1)

    return data


def add_nearest_coastline(
    data: pd.DataFrame,
    global_coastline: None | str | Path = None,
    florida_coastline: None | str | Path = None,
) -> pd.DataFrame:
    """
    Function to calculate distance of corresponding launch site to nearest coastline.
    Adds the distance to nearest coastline column to `data`

    For accuracy purposes, the Florida coastline geodata will be used for launch sites based in Florida;
    otherwise, the global coastline geodata will be used.

    :param data: pd.DataFrame; launch data

    :param global_coastline: None, or str, or Path; Default None; file path to global coastline geodata. If None,
                             function loads geodata from the default path:
                             parent directory > data/external/ne_coastline.zip

    :param florida_coastline: None, or str, or Path; Default None; file path to Florida coastline geodata. If None,
                              function loads geodata from the default path:
                              parent directory > data/external/florida_coastline.geojson

    :return: pd.DataFrame; launch data with `nearest_coastline` column
    """

    # Load global coastline data
    if not global_coastline:
        global_coastline = geodata_paths["Global coastline"]
    else:
        global_coastline = Path(global_coastline)

    zip_uri = f"zip://{global_coastline.as_posix()}"
    global_coastline_gdf = gpd.read_file(zip_uri).to_crs(epsg=5070)

    # Load Florida coastline data
    if not florida_coastline:
        florida_coastline = geodata_paths["Florida coastline"]

    florida_gdf = gpd.read_file(florida_coastline)
    florida_gdf["geometry"] = florida_gdf.geometry.boundary
    florida_gdf = florida_gdf.to_crs(epsg=5070)

    # Define function -- convert reproject (lon,lat) to EPSG:5070 and calculate nearest distance
    def func(row):
        lon = row.longitude
        lat = row.latitude
        row_conv = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=5070)[0]

        if lon >= -87 and lat <= 31:  # Approximate range for places in Florida
            return florida_gdf.geometry.distance(row_conv).min() / 1000

        else:
            return global_coastline_gdf.geometry.distance(row_conv).min() / 1000

    # Apply function to data
    data["nearest_coastline"] = data.apply(func, axis=1)

    return data

def add_nearest(csv_path: None | str | Path, save_path: None | str | Path = None) -> None|Path:
    """
    Utility function to run `add_nearest_highway`, `add_nearest_railway`, and `add_nearest_coastline`
    functions on the csv file containing the dataset.

    :param csv_path: str or Path; file path where csv file containing launch data is saved
    :param save_path: None, str, or Path; default None; file path where the processed csv file
                      is saved. If None, the csv file is saved in the parent directory > data/processed
                      folder
    :return: `save_path` as Path
    """
    if not csv_path:
        return None

    try:
        df = pd.read_csv(csv_path, na_filter=False)

    except FileNotFoundError as e:
        print("Please check if file exists: m{}".format(e))
        return None

    processed_dir.mkdir(parents=True, exist_ok=True)
    if not save_path:
        save_path = processed_dir / dataset_processed
    else:
        save_path = Path(save_path)

    add_nearest_highway(df)
    add_nearest_railway(df)
    add_nearest_coastline(df)

    df.to_csv(save_path, index=False)

    return save_path

#=====================================================================================================================
# CREATE TEST SET
#=====================================================================================================================

def create_test_set() -> None | list[Path]:
    """
    Function to create a test set stratified based on class.
    :return: [train_set_path, test_set_path]; list of Paths to the training and test sets.
    """
    train_set_path = processed_dir / training_set
    test_set_path = processed_dir / test_set
    csv_path = processed_dir / dataset_processed

    try:
        df = pd.read_csv(csv_path, na_filter=False)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_state, stratify=df['class'])
        train_df.to_csv(train_set_path, index=False)
        test_df.to_csv(test_set_path, index=False)
    except FileNotFoundError as e:
        print("Please check if file exists: m{}".format(e))
        return None

#=====================================================================================================================
# MAIN
#=====================================================================================================================

def main():
    logger = logging.getLogger('make_dataset')

    logger.info('Downloading launch data from Launch Library 2')
    ll2_raw = download_all_ll2_launches()

    logger.info('Downloading launch data from GCAT')
    download_launch_data_static(gcat_url, file_name_gcat)

    logger.info('Merging launch data from LL2 and GCAT')
    merge_ll2_launch_data(ll2_raw)
    ll2_df = transform_ll2_launches()
    gcat_df = transform_gcat_data()
    merged_df = merge_launch_data(ll2_df, gcat_df)

    logger.info('Adding target attribute \'class\'')
    csv_interim = add_class(merged_df)

    logger.info('Downloading geodata')
    download_layers_data()

    logger.info('Adding proximity attributes')
    add_nearest(csv_interim)

    logger.info('Creating training and test sets')
    create_test_set()

    logger.info('Done!')

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()