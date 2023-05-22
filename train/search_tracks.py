import argparse

import pandas as pd

pd.set_option("max_colwidth", 0)
pd.set_option("display.max_rows", 1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracks_file",
        type=str,
        default="data/tracks.csv",
        help="Tracks CSV file",
    )
    parser.add_argument(
        "--search",
        type=str,
        required=True,
        help="Search string",
    )
    args = parser.parse_args()

    tracks_df = pd.read_csv(
        args.tracks_file,
        header=None,
        index_col=0,
        names=["artist", "title", "url", "count"],
    ).fillna("")
    tracks_df["name"] = tracks_df["artist"] + " - " + tracks_df["title"]
    print(tracks_df[tracks_df["name"].str.contains(args.search, case=False)][["name"]])
