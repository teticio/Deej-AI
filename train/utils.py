import csv
import json
import traceback
from base64 import b64decode
from time import sleep
from typing import Any, Callable, Dict, List, Optional

import boto3
import requests
from tqdm import tqdm

# for really long playlists!
csv.field_size_limit(1000000)


def paginate(func: Callable, delay: float = 1, *args, **kwargs) -> List:
    """
    Paginates a Spotify API request.

    Args:
        func (Callable): The function to paginate.
        delay (float): The delay between requests. Default is 1.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        A list of the results of the paginated request.
    """
    results = []
    offset = 0
    while True:
        response = func(*args, limit=50, offset=offset, **kwargs)
        if "items" not in response:
            return results
        results += response["items"]
        sleep(delay)
        if response["next"] is None:
            break
        offset += 50
    return results


def request_with_proxy(method: str, url: str, headers: Dict, proxy: str) -> bytes:
    """
    Makes a request to a proxy Lambda function.

    Args:
        method (str): The HTTP method to use.
        url (str): The URL to make the request to.
        headers (Dict): The headers to use for the request.
        proxy (str): The name of the proxy Lambda function to use for the request (see README).

    Returns:
        bytes: The response body.
    """
    response = json.loads(
        boto3.client("lambda")
        .invoke(
            FunctionName=proxy,
            InvocationType="RequestResponse",
            Payload=json.dumps(
                {
                    "method": method,
                    "url": url,
                    "headers": headers,
                }
            ),
        )["Payload"]
        .read()
    )
    return b64decode(response["body"])


def get_access_token(cookie: Optional[str] = None, proxy: Optional[str] = None) -> str:
    """
    Retrieves a Spotify access token.

    Args:
        cookie (str): The sp_dc cookie to use for the request. Default is None.
        proxy (str): The name of the proxy Lambda function to use for the request (see README). Default is None.

    Returns:
        str: The access token.
    """
    url = "https://open.spotify.com/get_access_token?reason=transport&productType=web-player"
    headers = {} if cookie is None else {"cookie": f"sp_dc={cookie}"}
    while True:
        try:
            return (
                requests.get(url=url, headers=headers).json()["accessToken"]
                if proxy is None
                else json.loads(
                    request_with_proxy("GET", url=url, headers=headers, proxy=proxy)
                )["accessToken"]
            )
        except:
            traceback.print_exc()
            print("Retrying in 1 second...")
            sleep(1)
            continue


def read_playlists(playlists_file: str) -> Dict[str, List[str]]:
    """
    Reads a CSV file containing playlists.

    Args:
        playlists_file (str): The path to the CSV file.

    Returns:
        Dict[str, List[str]]: A dictionary mapping playlist IDs to track IDs.
    """
    with open(playlists_file, "r") as csvfile:
        reader = csv.reader(csvfile)
        playlists = {row[0]: row[1:] for row in tqdm(reader, desc="Reading playlists")}
    return playlists


def write_playlists(playlists: Dict[str, List[str]], playlists_file: str) -> None:
    """
    Writes a dictionary of playlists to a CSV file.

    Args:
        playlists (Dict[str, List[str]]): A dictionary mapping playlist IDs to track IDs.
        playlists_file (str): The path to the CSV file.

    Returns:
        None
    """
    with open(playlists_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for key, value in tqdm(playlists.items(), desc="Writing playlists"):
            writer.writerow([key] + value)


columns = ["artist", "title", "url", "count"]


def read_tracks(tracks_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Reads a CSV file containing tracks.

    Args:
        tracks_file (str): The path to the CSV file.

    Returns:
        Dict[str, Dict[str, str]]: A dictionary mapping track IDs to track metadata.
    """
    with open(tracks_file, "r") as csvfile:
        reader = csv.reader(csvfile)
        tracks = {
            row[0]: {columns[i]: value for i, value in enumerate(row[1:])}
            for row in tqdm(reader, desc="Reading tracks")
        }
    return tracks


def write_tracks(tracks: Dict[str, Dict[str, Any]], tracks_file: str) -> None:
    """
    Writes a dictionary of tracks to a CSV file.

    Args:
        tracks (Dict[str, Dict[str, str]]): A dictionary mapping track IDs to track metadata.
        tracks_file (str): The path to the CSV file.

    Returns:
        None
    """
    with open(tracks_file, "w", newline="") as f:
        writer = csv.writer(f)
        for key, details in tqdm(tracks.items(), desc="Writing tracks"):
            row = [key] + [details[column] for column in columns]
            writer.writerow(row)
