import argparse

import pandas as pd

pd.set_option("max_colwidth", 0)
pd.set_option("display.max_rows", 1000)


if __name__ == "__main__":
    """
    Entry point for the search_tracks script.

    Searches for tracks on Spotify.

    Args:
        --search (str): Search string. Default is None (interactive mode).
        --tracks_file (str): Path to the tracks CSV file. Default is "data/tracks_dedup.csv".

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Search string",
    )
    parser.add_argument(
        "--tracks_file",
        type=str,
        default="data/tracks_dedup.csv",
        help="Tracks CSV file",
    )
    args = parser.parse_args()

    tracks_df = pd.read_csv(
        args.tracks_file,
        header=None,
        index_col=0,
        names=["artist", "title", "url", "count"],
    ).fillna("")
    tracks_df["name"] = tracks_df["artist"] + " - " + tracks_df["title"]

    interactive = args.search is None
    while interactive:
        if interactive:
            args.search = input("Search: ")
        print(
            tracks_df[tracks_df["name"].str.contains(args.search, case=False)][["name"]]
        )
