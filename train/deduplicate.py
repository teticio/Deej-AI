import argparse
import csv

import pandas as pd
from tqdm import tqdm
from utils import read_playlists

# for really long playlists!
csv.field_size_limit(1000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dedup_playlists_file",
        type=str,
        default="data/playlists_dedup.csv",
        help="Deduplicated playlists CSV file",
    )
    parser.add_argument(
        "--dedup_tracks_file",
        type=str,
        default="data/tracks_dedup.csv",
        help="Deduplicated tracks CSV file",
    )
    parser.add_argument(
        "-drop_missing_urls",
        action="store_true",
        help="Drop tracks with missing URLs",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=10,
        help="Number of times track must appear in playlists to be included",
    )
    parser.add_argument(
        "--oov",
        type=str,
        default=None,
        help="ID for out-of-vocabulary track or None to skip",
    )
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
    args = parser.parse_args()

    playlists = read_playlists(args.playlists_file)
    tracks_df = pd.read_csv(
        args.tracks_file,
        header=None,
        names=["id", "artist", "title", "url", "count"],
    )

    tracks_df = tracks_df[tracks_df["count"] >= args.min_count]
    tracks_df["url_is_empty"] = tracks_df["url"].isna() | (tracks_df["url"] == "")
    if args.drop_missing_urls:
        tracks_df = tracks_df[~tracks_df["url_is_empty"]]
    else:
        tracks_df = tracks_df.sort_values(["url_is_empty"])

    deduped_tracks_df = (
        tracks_df.groupby(["artist", "title"])
        .agg({"id": "first", "url": "first", "count": "sum"})
        .reset_index()
    )
    # grouping doesn't preserve order
    merged_df = pd.merge(
        tracks_df,
        deduped_tracks_df,
        on=["artist", "title"],
        suffixes=("_original", "_deduped"),
        how="left",
    )
    dedup = dict(zip(merged_df["id_original"], merged_df["id_deduped"]))
    deduped_playlists = {
        playlist_id: [dedup.get(track_id, args.oov) for track_id in playlist]
        for playlist_id, playlist in playlists.items()
    }

    with open(args.dedup_playlists_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for key, value in tqdm(deduped_playlists.items(), desc="Writing playlists"):
            if args.oov is None:
                value = [v for v in value if v is not None]
            if len(value) > 0 and not all(v == args.oov for v in value):
                writer.writerow([key] + value)
    print(f"Writing tracks")
    deduped_tracks_df.set_index("id").to_csv(args.dedup_tracks_file, header=False)
