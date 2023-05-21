import argparse
import csv

import gensim
import pandas as pd
from gensim.models.callbacks import CallbackAny2Vec

pd.set_option("max_colwidth", 0)
pd.set_option("display.max_rows", 1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dedup_tracks_file",
        type=str,
        default="data/tracks_dedup.csv",
        help="Deduplicated tracks CSV file",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="models/track2vec",
        help="Model file",
    )
    args = parser.parse_args()

    tracks_df = pd.read_csv(
        args.dedup_tracks_file,
        header=None,
        index_col=0,
        names=["artist", "title", "url", "count"],
    ).fillna("")
    tracks_df["name"] = tracks_df["artist"] + " - " + tracks_df["title"]
    model = gensim.models.Word2Vec.load(args.model_file)

    while True:
        search = input("Search for a track: ")
        track_ids = tracks_df[tracks_df["name"].str.contains(search, case=False)][
            ["name"]
        ]
        if len(track_ids) > 0:
            break
    print(track_ids)
    track_id = input("Enter track ID: ")
    print()

    print(
        f"\u001b]8;;{tracks_df.loc[track_id]['url']}\u001b\\{tracks_df.loc[track_id]['artist']} - {tracks_df.loc[track_id]['title']}\u001b]8;;\u001b\\"
    )
    most_similar = model.wv.most_similar(positive=[track_id], topn=8)
    for i, similar in enumerate(most_similar):
        print(
            f"{i + 1}. \u001b]8;;{tracks_df.loc[similar[0]]['url']}\u001b\\{tracks_df.loc[similar[0]]['artist']} - {tracks_df.loc[similar[0]]['title']}\u001b]8;;\u001b\\ ({similar[1]:.2f})"
        )
