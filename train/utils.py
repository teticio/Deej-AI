import csv
import json
from base64 import b64decode
from time import sleep

import boto3
import requests
import traceback
from tqdm import tqdm

# for really long playlists!
csv.field_size_limit(1000000)


def paginate(func, delay=0, *args, **kwargs):
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


def request_with_proxy(method, url, headers=None, proxy=None):
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


def get_access_token(cookie=None, proxy=None):
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


def read_playlists(playlists_file):
    with open(playlists_file, "r") as csvfile:
        reader = csv.reader(csvfile)
        playlists = {row[0]: row[1:] for row in tqdm(reader, desc="Reading playlists")}
    return playlists


def write_playlists(playlists, playlists_file):
    with open(playlists_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for key, value in tqdm(playlists.items(), desc="Writing playlists"):
            writer.writerow([key] + value)


columns = ["artist", "title", "url", "count"]


def read_tracks(tracks_file):
    with open(tracks_file, "r") as csvfile:
        reader = csv.reader(csvfile)
        tracks = {
            row[0]: {columns[i]: value for i, value in enumerate(row[1:])}
            for row in tqdm(reader, desc="Reading tracks")
        }
    return tracks


def write_tracks(tracks, tracks_file):
    with open(tracks_file, "w", newline="") as f:
        writer = csv.writer(f)
        for key, details in tqdm(tracks.items(), desc="Writing tracks"):
            row = [key] + [details[column] for column in columns]
            writer.writerow(row)
