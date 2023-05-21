import argparse
import concurrent.futures
import os

import pandas as pd
import requests
from tqdm import tqdm


def download_file(track_id, track_url, dir):
    response = requests.get(track_url, stream=True)
    if response.status_code == 200:
        with open(os.path.join(dir, f"{track_id}.mp3"), "wb") as file:
            file.write(response.content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracks_file",
        type=str,
        default="data/tracks_dedup.csv",
        help="Tracks CSV file",
    )
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
    args = parser.parse_args()

    track_urls = pd.read_csv(
        args.tracks_file,
        header=None,
        index_col=0,
        names=["artist", "title", "url", "count"],
    )["url"].to_dict()
    os.makedirs(args.previews_dir, exist_ok=True)
    already_downloaded = os.listdir(args.previews_dir)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = {
            executor.submit(
                download_file, track_url, track_urls[track_url], args.previews_dir
            ): track_url
            for track_url in track_urls
            if f"{track_url}.mp3" not in already_downloaded
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Downloading previews",
        ):
            url = futures[future]
            try:
                future.result()
            except KeyboardInterrupt:
                break
            except Exception:
                print(f"Skipping {url}")


if __name__ == "__main__":
    main()
