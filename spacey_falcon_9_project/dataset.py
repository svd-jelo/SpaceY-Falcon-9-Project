from itertools import islice
import json
from pathlib import Path
import re
import time
from typing import Any

from bs4 import BeautifulSoup
import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point
import topojson as tp
from tqdm import tqdm

#=====================================================================================================================
# DATA COLLECTION PART 1 - FROM API
#=====================================================================================================================

def get_ll2_launches(offset):
    ll2_api = "https://ll.thespacedevs.com/2.3.0/launches/previous/"
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
    file_paths = []
    for offset in range(0, 700, 100):
        parent_dir = Path.cwd().parent
        file_dir = parent_dir / "data" / "raw"
        file_dir.mkdir(parents=True, exist_ok=True)
        file_name = "ll2-api-2.3.0-launches-previous-{}.json".format(offset)
        file_path = file_dir / file_name
        if file_path.is_file():
            print("{} already exists.".format(file_name))
            continue
        try:
            data = get_ll2_launches(offset)
            if not isinstance(data, bytes):
                raise TypeError("Expected bytes, got {}".format(type(data)))
            with open(file_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print("Error: {}".format(e))
            break
        file_paths.append(file_path)
    return file_paths

def download_launch_data_static(url: str, file_name: str, query_params=None, headers=None) -> None|Path:
    parent_dir = Path.cwd().parent
    file_dir = parent_dir / "data" / "raw"
    file_dir.mkdir(parents=True, exist_ok=True)
    file_path = file_dir / file_name

    if file_path.is_file():
        print("{} already exists".format(file_name))
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
def merge_ll2_launch_data(json_paths: list[Path], merged_path: None|Path = None) -> Path:
    """
    Function to merge LL2 launches that were extracted and downloaded with the download_all_ll2_launches() function
    :param json_paths: list[Path]; file paths to the paginated files. Particularly, the returned output of the
                       download_all_ll2_launches() function
    :param merged_path: None or Path; path to the merged files. If not specified, the merged file will be saved in
                        the current working directory.
    """

    merged = []
    for path in json_paths:
        with open(path, 'r') as f:
            launch_page = json.load(f)
        merged.extend(launch_page['results'])

    if merged_path is None:
        merged_path = Path.cwd() / 'll2-api-2.3.0-launches-previous-merged.json'

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

def transform_ll2_launches(json_path: str | Path) -> pd.DataFrame:
    """
    A function to transform LL2 raw data given as a JSON file path

    :param json_path: str or Path; local path to the JSON file containing raw ll2 launches data
    :return: pd.DataFrame; transformed LL2 launch data containing relevant columns for analysis, but requires further wrangling
    """
    with open(json_path, "r") as f:
        data_json = json.load(f)

    json_keys = dict(
        launch_designator="launch_designator",
        net="net",
        booster_version="rocket.configuration.name",
        orbit="mission.orbit.name",
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

    # Derived column - outcome
    def to_str(entry: Any) -> str:
        """
        Utility function to convert any object to str.
        If the object is None, function returns the string 'None'

        :param entry: object to be converted to string.
        """
        if entry is None:
            return 'None'
        else:
            return str(entry)

    data_df["outcome"] = data_df["landing_success"].map(to_str) + " " + data_df["landing_type"].map(to_str)
    data_df = data_df.drop(["landing_success", "landing_type"], axis=1)

    # Drop missing values in launch designator and outcome
    data_df = data_df.dropna(subset="launch_designator")

    return data_df

def transform_gcat_data(gcat_path: str | Path) -> pd.DataFrame:
    """
    Function to transform GCAT raw data from the specified file path
    :param gcat_path: str or Path; file path to the GCAT raw data.
    :return: pd.DataFrame; dataframe containing relevant data from GCAT
    """

    gcat_df = pd.read_csv(gcat_path, sep="\t", skiprows=(lambda x: x in [1]))

    pattern = r"^\d{4}\s+\w{3}\s+\d{1,2}"
    format_date = re.compile(pattern)
    launch_date = gcat_df["Launch_Date"].map(lambda x: format_date.match(x).group())
    gcat_df["launch_date"] = pd.to_datetime(launch_date).dt.date

    gcat_df = gcat_df[["#Launch_Tag", "launch_date", "OrbPay"]]
    gcat_df = gcat_df.rename(
        columns={"#Launch_Tag": "launch_designator"}
    )  # rename '#Launch_Tag' to launch designator
    gcat_df["launch_designator"] = gcat_df[
        "launch_designator"
    ].str.strip()  # remove trailing white spaces from launch_designator column

    return gcat_df

def transform_course_data(course_path: str | Path) -> pd.DataFrame:
    """
    Function to transform course-provided static JSON raw data from the specified path

    :param course_path: str or Path; file path to the JSON raw data
    :return: pd.DataFrame; containing relevant data from the course-provided data
    """

    with open(course_path, "r") as f:
        json_data = json.load(f)

    json_keys = {"date_utc": "date_utc", "gridfins": "cores.gridfins", "legs": "cores.legs"}

    course_dict = {}
    for col, key in json_keys.items():
        cols = []
        for launch in json_data:
            cols.append(get_path(launch, key))
        course_dict.update({col: cols})

    course_df = pd.DataFrame(course_dict)

    course_df["launch_date"] = pd.to_datetime(course_df["date_utc"]).dt.date
    course_df = course_df.drop("date_utc", axis=1)

    return course_df

def merge_launch_data(ll2_df: pd.DataFrame, gcat_df: pd.DataFrame, course_df: pd.DataFrame, csv_path: None|str|Path = None) -> None|Path:
    """
    Function to merge transformed ll2 launches data, transformed GCAT launch data, and transformed course data.
    ll2 launches and gcat data will be merged on `launch_designator`, and resulting merged DataFrame will be merged
    with course data on `launch_date`. Each merge is a one-to-one inner join. The final merged DataFrame will be
    saved to `csv_path`

    :param ll2_df: pd.DataFrame; transformed ll2 data
    :param gcat_df: pd.DataFrame; transformed gcat data
    :param course_df: pd.DataFrame; transformed course data
    :param csv_path: None or str or Path; Default None. File path where merged DataFrame is to be saved. If None,
                     merged DataFrame will be saved in the current working directory
    :return: Path; csv_path
    """

    if not csv_path:
        csv_path = Path.cwd() / 'launch_data_1.csv'

    if isinstance(csv_path, str):
        csv_path = Path(csv_path)

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

    # Merge Course DF
    merged_df = merged_df.merge(course_df, on='launch_date', how='inner', validate='1:1')

    # Convert DF to CSV
    merged_df.to_csv(csv_path, index=False)

    return csv_path

def add_class(launch_csv: str | Path, save_path: None | str | Path = None) -> None | Path:
    """
    Function that adds the target 'Class' column to the launch data from the specified launch csv path,
    and saves the result to the specified save path.

    :param launch_csv: str or Path; file path to the launches csv file
    :param save_path: str or Path; Default None; file path where the result is to be saved. If None,
                      result will be saved in the current working directory.
    :return save_path: Path; file path where result is saved.
    """
    try:
        launch_df = pd.read_csv(launch_csv)

    except Exception as e:
        print(e)
        return None

    # Drop NaN or NA values
    launch_df.dropna(axis=0, inplace=True)

    landing_outcomes = launch_df["outcome"].unique()

    bad_outcomes = set()
    for outcome in landing_outcomes:
        if "False" in outcome or "None" in outcome:
            bad_outcomes.add(outcome)

    def outcome_map(entry: str) -> int:
        """
        Utility function that maps an outcome to either the integer 1 or 0.
        If the outcome is a landing success, the function returns 1; otherwise,
        the function returns 0

        :param entry: str; a landing outcome in the format "(Landing Success) (Landing Pad)", e.g., True ASDS.
        :return outcome: int; if the entry is a landing success, `outcome=1`; otherwise, `outcome=0`.
        """

        if entry in bad_outcomes:
            landing_outcome = 0
        else:
            landing_outcome = 1

        return landing_outcome

    launch_df["class"] = launch_df["outcome"].map(outcome_map)

    if not save_path:
        save_path = Path.cwd() / "api-launch-data-table-class.csv"

    if isinstance(save_path, str):
        save_path = Path(save_path)

    launch_df.to_csv(save_path, index=False)

    return save_path

# Download LL2 API Launches Data
#download_all_ll2_launches()

# Download static JSON provided by the course for project use
#url_ibm='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/API_call_spacex_api.json'
#file_name_ibm = 'ibm-ds-capstone-launch-data.json'
#download_launch_data_static(url_ibm, file_name_ibm)

# Download launch data from GCAT
#url_gcat='https://planet4589.org/space/gcat/tsv/launch/Falcon9.tsv'
#file_name_gcat = 'mcdowell-gcat-launch-data.tsv'
#download_launch_data_static(url_gcat, file_name_gcat)

#=====================================================================================================================
# DATA COLLECTION PART 2 - BY WEB SCRAPING
#=====================================================================================================================

# Download launch data from Wikipedia
#url_wiki = "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"
#file_name_wiki = "wikipedia-launch-data-table.html"
#headers_wiki = {
#    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#                  "AppleWebKit/537.36 (KHTML, like Gecko) "
#                  "Chrome/91.0.4472.124 Safari/537.36"
#}
#download_launch_data_static(url_wiki, file_name_wiki, headers=headers_wiki)

# Parse HTML document and convert to csv
def parse_table(table) -> pd.DataFrame:
    """
    returns a pandas DataFrame containing the parsed table
    table: bs4 table element
    """
    # Create Header Row
    header_row = table.find("tr")

    # Create List of Keys
    keys = []
    for child in header_row.children:
        if child.name:  # Filter out `\n` using the fact that the element `\n` has no name (None)
            keys.append(child.get_text())
    keys.append("Description")

    # Create Launch Dictionary
    values = []
    for i in range(len(keys)):
        values.append([])
    launch_dict = dict(zip(keys, values))

    # RegEx objects to filter launch data - to filter out sup, spans, white spaces, and falcon heavy launches
    pattern_sup_span = (
        r"\[\d+\]|\[[a-z]\]|\s+$"  # Pattern that matches sup and spans and trailing white spaces
    )
    pattern_fh = (
        r"FH\s+\d+"  # Pattern that matches Flight No.'s corresponding to Falcon Heavy launches
    )
    remove_sup_span = re.compile(pattern_sup_span)
    is_fh = re.compile(pattern_fh)

    # Parse table and populate launch dictionary
    iterator = iter([row for row in header_row.next_siblings if row.name])
    for row in iterator:
        if row.name:
            cols = [col for col in row.children if col.name]
            try:
                # Skip Falcon Heavy Launches
                if is_fh.fullmatch(cols[0].get_text()):
                    rowspan = int(cols[0]["rowspan"])
                    next_flight = next(islice(iterator, rowspan - 1, rowspan), None)
                    if not next_flight:
                        break
                    cols = [col for col in next_flight.children if col.name]

                # Populate launch dictionary
                if cols[0].name == "th":
                    rowspan = int(cols[0]["rowspan"])
                    keys = launch_dict.keys()
                    for idx, key in enumerate(keys):
                        # Obtain launch data for current iteration
                        if idx != len(keys) - 1:
                            launch_dict[key].append(remove_sup_span.sub("", cols[idx].get_text()))

                        # Obtain the description from the last iteration (if rowspan=3, then description row is 2 step from the current iteration)
                        else:
                            next_row = next(
                                islice(iterator, rowspan - 2, rowspan - 1),
                                BeautifulSoup("<p></p>", "lxml"),
                            )
                            description = [col for col in next_row.children if col.name]
                            launch_dict[key].append(
                                remove_sup_span.sub("", description[0].get_text())
                            )

            except IndexError as e:
                print("{}\nrow with column length {} was ignored".format(e, len(cols)))

    return pd.DataFrame(launch_dict)


def parse_all_tables(past_launches_tables) -> pd.DataFrame:
    """
    returns a Pandas DataFrame containing all tables parsed from the past_launches_tables
    past_launches_table: list of bs4 element tags; contents of the section of the markup containing table of past launches by period (2010-2013, 2014, etc.)
    """
    launches_df_web_scrap = parse_table(past_launches_tables[0])
    for table in past_launches_tables[1:]:
        table_df = parse_table(table)
        table_df.columns = launches_df_web_scrap.columns.values
        launches_df_web_scrap = pd.concat([launches_df_web_scrap, table_df])

    return launches_df_web_scrap.reset_index(drop=True)

def launches_html_to_csv(html_path: str|Path, csv_path:None|str = None) -> None|Path:
    """
    :param csv_path: file path where converted HTML document is to be saved
    :param html_path: file path to the HTML document.
    :return: None; the function saves the csv file to the specified path.
    """
    # Load HTML Document
    try:
        with open(html_path, "r") as f:
            html_doc = BeautifulSoup(f.read(), "lxml")
    except OSError as e:
        print('{} Conversion of HTML document to CSV file has been stopped.'.format(e))
        return None

    # Check/Create CSV Path
    if not csv_path:
        parent_dir = Path.cwd().parent                  # Parent directory
        file_dir = parent_dir / "data" / "interim"      # File directory; where the csv file is saved
        file_dir.mkdir(parents=True, exist_ok=True)     # Create file directory if it does not exist
        file_name = 'wikipedia-launch-data-table.csv'   # File name
        csv_path = file_dir / file_name                 # File path

        if csv_path.is_file():
            print("{} already exists".format(file_name))
            return csv_path

    if isinstance(csv_path, str):
        csv_path = Path(csv_path)

    # Parse HTML document and store data in a pandas DataFrame
    html_launch_section = html_doc.find("section", attrs={"aria-labelledby": "Past_launches"})
    html_launch_tables = html_launch_section.find_all("table")
    launches_df = parse_all_tables(html_launch_tables)

    # Save pandas DataFrame to CSV file
    launches_df.to_csv(csv_path, index=False)

    return csv_path

#=====================================================================================================================
# GEODATA -- Addition of columns for distance to nearest highway, railway, and coastline
#=====================================================================================================================
def download_layers_data(folder_path: None | str | Path = None) -> dict[str, Path]:
    """
    Function for downloading geodata for highways, railways, and coastlines

    :param folder_path: None, str, or Path; default None; folder path where geodata are cached
                        if None, the files are saved in the default path:
                        parent directory > data/external

    :return: dict[str, Path]; Dictionary of file paths.
    """
    if not folder_path:
        folder_path = Path.cwd().parent / "data/external"
    else:
        folder_path = Path(folder_path)

    folder_path.mkdir(parents=True, exist_ok=True)

    # File Paths
    file_paths = {
        "US roadmap": folder_path / "us_road_map.json",
        "US railways": folder_path / "us_railways.geojson",
        "Global coastline": folder_path / "ne_coastline.zip",
        "Florida coastline": folder_path / "florida_coastline.geojson",
    }

    # URLs
    urls = [
        # US Roads
        "https://gist.githubusercontent.com/bricedev/96d2113bd29f60780223/raw/957d51ac88a6de442cf73b9efa8615fce9f9577e/usroads.json",
        # US Railways
        "/".join(
            [
                "https://services.arcgis.com",
                "xOi1kZaI0eWDREZv",
                "arcgis",
                "rest",
                "services",
                "NTAD_North_American_Rail_Network_Lines",
                "FeatureServer",
                "replicafilescache",
                "NTAD_North_American_Rail_Network_Lines_-5214657740406327753.geojson",
            ]
        ),
        # Global Coastline Data - from Natural Earth
        "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip",
        # Florida Coastline Data - from ArcGIS
        "https://hub.arcgis.com/api/v3/datasets/eda0c60e98cd43af9422dc5ea54d8d56_2/downloads/data?format=geojson&spatialRefId=4326&where=1%3D1",
    ]

    for key, file, url in zip(file_paths.keys(), file_paths.values(), urls):
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

    return file_paths

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
        us_roadmap = Path.cwd().parent / "data/external/us_road_map.json"

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
        us_railways = Path.cwd().parent / "data/external/us_railways.geojson"

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
        global_coastline = Path.cwd().parent / "data/external/ne_coastline.zip"

    zip_uri = f"zip://{global_coastline.as_posix()}"
    global_coastline_gdf = gpd.read_file(zip_uri).to_crs(epsg=5070)

    # Load Florida coastline data
    if not florida_coastline:
        florida_coastline = Path.cwd().parent / "data/external/florida_coastline.geojson"

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

def add_nearest(csv_path: str | Path, save_path: None | str | Path = None) -> None|Path:
    """
    Utility function to run `add_nearest_highway`, `add_nearest_railway`, and `add_nearest_coastline`
    functions on the csv file containing the dataset.

    :param csv_path: str or Path; file path where csv file containing launch data is saved
    :param save_path: None, str, or Path; default None; file path where the processed csv file
                      is saved. If None, the csv file is saved in the current working directory.
    :return: `save_path` as Path
    """
    try:
        df = pd.read_csv(csv_path)

    except FileNotFoundError as e:
        print("Please check if file exists: {}".format(e))
        return None

    if not save_path:
        save_path = Path.cwd()
    else:
        save_path = Path(save_path)

    add_nearest_highway(df)
    add_nearest_railway(df)
    add_nearest_coastline(df)

    df.to_csv(save_path, index=False)

    return save_path