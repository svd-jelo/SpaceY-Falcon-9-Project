from pathlib import Path
import time

import requests


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
            return response.content

        except requests.exceptions.HTTPError as e:
            print("Error: {}".format(e))
            return None

    print("Failed after 3 attempts (504 Server Error)")
    return None

def download_all_ll2_launches():
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
            with open(file_path, "wb") as f:
                f.write(data)
        except Exception as e:
            print("Error: {}".format(e))
            break

def download_launch_data_static(url, file_name, query_params=None, headers=None):
    parent_dir = Path.cwd().parent
    file_dir = parent_dir / "data" / "raw"
    file_dir.mkdir(parents=True, exist_ok=True)
    file_path = file_dir / file_name

    if file_path.is_file():
        print("{} already exists".format(file_name))
        return None

    response = requests.get(url, params=query_params, headers=headers)

    try:
        response.raise_for_status()

    except requests.exceptions.HTTPError as e:
        print("Error: {}".format(e))
        return None

    try:
        with open(file_path, "wb") as f:
            f.write(response.content)
    except Exception as e:
        print("Error: {}".format(e))
        return None

# Download LL2 API Launches Data
download_all_ll2_launches()

# Download static JSON provided by the course for project use
url_ibm='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/API_call_spacex_api.json'
file_name_ibm = 'ibm-ds-capstone-launch-data.json'
download_launch_data_static(url_ibm, file_name_ibm)

# Download launch data from GCAT
url_gcat='https://planet4589.org/space/gcat/tsv/launch/Falcon9.tsv'
file_name_gcat = 'mcdowell-gcat-launch-data.tsv'
download_launch_data_static(url_gcat, file_name_gcat)

# Download launch data from Wikipedia
url_wiki = "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"
file_name_wiki = "wikipedia-launch-data-table.html"
headers_wiki = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/91.0.4472.124 Safari/537.36"
}
download_launch_data_static(url_wiki, file_name_wiki, headers=headers_wiki)