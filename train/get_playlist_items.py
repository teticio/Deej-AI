# import debugpy
# debugpy.listen(5678)

import argparse
import concurrent.futures
import gc
import json
import os
import traceback
from time import sleep

import requests
from tqdm import tqdm
from utils import (get_access_token, paginate, read_playlists, read_tracks,
                   request_with_proxy, write_playlists, write_tracks)

access_token = None


def get_playlist_items(playlist_id, limit=50, offset=0, proxy=None):
    global access_token
    for _ in range(0, 2):
        try:
            if access_token is None:
                access_token = get_access_token(proxy=proxy)
            url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks?limit={limit}&offset={offset}"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }
            items = (
                requests.get(url=url, headers=headers).json()
                if proxy is None
                else json.loads(
                    request_with_proxy("GET", url=url, headers=headers, proxy=proxy)
                )
            )
            if "error" not in items:
                return items
            print(items["error"])
        except:
            traceback.print_exc()
            pass
        sleep(1)
        access_token = get_access_token(proxy=proxy)
    print(f"Skipping {playlist_id}")
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--playlists_file",
        type=str,
        default="data/playlists.csv",
        help="Playlists CSV file",
    )
    parser.add_argument(
        "--tracks_file",
        type=str,
        default="data/tracks.csv",
        help="Tracks CSV file",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Maximum number of cores to use",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="Proxy lambda function (see https://github.com/teticio/lambda-scraper)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Batch size",
    )
    args = parser.parse_args()

    playlists = read_playlists(args.playlists_file)
    start_index = len(playlists)
    keys = list(playlists.keys())
    while start_index > 0 and len(playlists[keys[start_index - 1]]) == 0:
        start_index -= 1
    tracks = read_tracks(args.tracks_file) if os.path.exists(args.tracks_file) else {}

    for i in range(start_index, len(playlists), args.batch_size):
        print(f"{i}/{len(playlists)}")
        batch = keys[i : i + args.batch_size]

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.max_workers
        ) as executor:
            futures = {
                executor.submit(
                    paginate,
                    get_playlist_items,
                    delay=0.1,
                    playlist_id=playlist_id,
                    proxy=f"{args.proxy}-{i % args.max_workers}"
                    if args.proxy
                    else None,
                ): playlist_id
                for i, playlist_id in enumerate(tqdm(batch, desc="Setting up jobs"))
            }
            for i, future in enumerate(
                tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Getting playlist items",
                )
            ):
                playlist_id = futures[future]
                items = future.result()
                del futures[future]

                for item in items:
                    if item["track"] is None or item["track"]["id"] is None:
                        continue
                    if item["track"]["id"] in tracks:
                        tracks[item["track"]["id"]]["count"] = (
                            int(tracks[item["track"]["id"]]["count"]) + 1
                        )
                    else:
                        tracks[item["track"]["id"]] = {
                            "artist": item["track"]["artists"][0]["name"],
                            "title": item["track"]["name"],
                            "url": item["track"]["preview_url"]
                            if item["track"]["preview_url"] is not None
                            else "",
                            "count": 1,
                        }
                    playlists[playlist_id].append(item["track"]["id"])
                del items

        write_tracks(tracks, args.tracks_file)
        write_playlists(playlists, args.playlists_file)
        gc.collect()


if __name__ == "__main__":
    main()
