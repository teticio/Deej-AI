import argparse
import concurrent.futures
import os
from time import sleep

import pandas as pd
import requests
from tqdm import tqdm


def download_file(track_id: str, track_url: str, dir: str) -> None:
    """
    Downloads an MP3 preview file from a URL and saves it to disk.

    Args:
        track_id (str): The ID of the track to download.
        track_url (str): The URL of the MP3 preview file to download.
        dir (str): The directory to save the downloaded file.

    Returns:
        None
    """
    for _ in range(0, 2):
        response = requests.get(track_url, stream=True)
        if response.status_code == 200:
            with open(os.path.join(dir, f"{track_id}.mp3"), "wb") as file:
                file.write(response.content)
            return
    print(f"Skipping {track_url}")


def main() -> None:
    """
    Entry point for the download_previews script.

    Downloads MP3 preview files for a list of tracks.

    Args:
        --max_workers (int): Maximum number of cores to use. Default is the number of available CPU cores on the system,
        --previews_dir (str): Path to the directory where preview files will be saved. Default is "previews".
        --tracks_file (str): Path to the CSV file containing the list of tracks to download. Default is "data/tracks_dedup.csv".

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_workers",
        type=int,
        default=os.cpu_count() if os.cpu_count() is not None else 1,
        help="Maximum number of cores to use",
    )
    parser.add_argument(
        "--previews_dir",
        type=str,
        default="previews",
        help="Directory to save previews",
    )
    parser.add_argument(
        "--tracks_file",
        type=str,
        default="data/tracks_dedup.csv",
        help="Tracks CSV file",
    )
    args = parser.parse_args()

    track_urls = pd.read_csv(
        args.tracks_file,
        header=None,
        index_col=0,
        names=["artist", "title", "url", "count"],
    )["url"].to_dict()
    os.makedirs(args.previews_dir, exist_ok=True)
    already_downloaded = set(os.listdir(args.previews_dir))

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = {
            executor.submit(
                download_file, track_id, track_urls[track_id], args.previews_dir
            ): track_id
            for track_id in tqdm(track_urls, desc="Setting up jobs")
            if f"{track_id}.mp3" not in already_downloaded and sleep(1e-4) is None
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Downloading previews",
        ):
            track_id = futures[future]
            try:
                future.result()
            except KeyboardInterrupt:
                break
            except Exception:
                print(f"Skipping {track_id}")


if __name__ == "__main__":
    main()
