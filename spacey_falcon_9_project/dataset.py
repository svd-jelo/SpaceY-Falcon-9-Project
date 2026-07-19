from itertools import islice
from pathlib import Path
import re
import time

from bs4 import BeautifulSoup
import pandas as pd
import requests

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

#=====================================================================================================================
# DATA COLLECTION PART 2 - BY WEB SCRAPING
#=====================================================================================================================

# Download launch data from Wikipedia
url_wiki = "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"
file_name_wiki = "wikipedia-launch-data-table.html"
headers_wiki = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/91.0.4472.124 Safari/537.36"
}
download_launch_data_static(url_wiki, file_name_wiki, headers=headers_wiki)

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

def launches_html_to_csv(csv_path:None|str = None, html_path='/Users/jelo/spacey_falcon_9_project/data/raw/wikipedia-launch-data-table.html') -> None:
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
            return None

    # Parse HTML document and store data in a pandas DataFrame
    html_launch_section = html_doc.find("section", attrs={"aria-labelledby": "Past_launches"})
    html_launch_tables = html_launch_section.find_all("table")
    launches_df = parse_all_tables(html_launch_tables)

    # Save pandas DataFrame to CSV file
    launches_df.to_csv(csv_path, index=False)

    return None
