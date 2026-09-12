from pathlib import Path

# Directories
root = Path(__file__).parent.parent
raw_dir = root / "data" / "raw"
interim_dir = root / "data" / "interim"
external_dir = root / "data" / "external"
processed_dir = root / "data" / "processed"
models_dir = root / "models"

# Data Sources
ll2_url = "https://ll.thespacedevs.com/2.3.0/launches/previous/"
gcat_url = "https://planet4589.org/space/gcat/tsv/launch/Falcon9.tsv"
geodata_urls = [
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

# Filenames
file_name_gcat = "mcdowell-gcat-launch-data.tsv"
file_name_ll2 = "ll2-api-2.3.0-launches-previous-merged.json"
dataset_interim = "launch-data-table-class.csv"
dataset_processed = "dataset-processed.csv"
geodata_paths = {
    "US roadmap": external_dir / "us_road_map.json",
    "US railways": external_dir / "us_railways.geojson",
    "Global coastline": external_dir / "ne_coastline.zip",
    "Florida coastline": external_dir / "florida_coastline.geojson",
}
training_set = "training-set.csv"
test_set = "test-set.csv"

# Random State config
random_state = 42
