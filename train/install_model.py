import argparse
import os
import pickle
import shutil

from utils import read_tracks

if __name__ == "__main__":
    """
    Entry point for the install_model script.

    Installs model to deej-ai.online app.

    Args:
        --deejai_model_dir (str): Path to the deej-ai.online model directory. Default is "../deej-ai.online-dev/model".
        --mp3tovec_model_file (str): Path to the MP3ToVec model file. Default is "models/mp3tovec.ckpt".
        --mp3tovec_file (str): Path to the MP3ToVec file. Default is "models/mp3tovec.p".
        --old_deejai_model_dir (str): Optionally merge old track metdata for backwards compatibility. Default is None.
        --track2vec_file (str): Path to the Track2Vec file. Default is "models/track2vec.p".
        --tracks_file (str): Path to the tracks CSV file. Default is "data/tracks.csv".

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deejai_model_dir",
        type=str,
        default="../deej-ai.online-dev/model",
        help="deej-ai.online model directory",
    )
    parser.add_argument(
        "--mp3tovec_model_file",
        type=str,
        default="models/speccy_model",
        help="MP3ToVec model file",
    )
    parser.add_argument(
        "--mp3tovec_file",
        type=str,
        default="models/mp3tovec.p",
        help="MP3ToVec file",
    )
    parser.add_argument(
        "--old_deejai_model_dir",
        type=str,
        default=None,
        help="Merge old track metadata (optional)",
    )
    parser.add_argument(
        "--track2vec_file",
        type=str,
        default="models/track2vec.p",
        help="Track2Vec file",
    )
    parser.add_argument(
        "--tracks_file",
        type=str,
        default="data/tracks_dedup.csv",
        help="Tracks CSV file",
    )
    args = parser.parse_args()

    track2vec = pickle.load(open(f"{args.track2vec_file}", "rb"))
    spotify2vec = pickle.load(open(f"{args.mp3tovec_file}", "rb"))
    tracks = read_tracks(args.tracks_file)

    common_tracks = set(track2vec.keys()).intersection(set(spotify2vec.keys()))
    print(f"{len(common_tracks)} tracks")
    to_delete = set(track2vec.keys()).difference(common_tracks)
    for track_id in to_delete:
        del track2vec[track_id]
    to_delete = set(spotify2vec.keys()).difference(common_tracks)
    for track_id in to_delete:
        del spotify2vec[track_id]

    spotify_tracks = {}
    spotify_urls = {}
    if args.old_deejai_model_dir is not None:
        spotify_tracks = pickle.load(
            open(os.path.join(args.old_deejai_model_dir, "spotify_tracks.p"), "rb")
        )
        spotify_urls = pickle.load(
            open(os.path.join(args.old_deejai_model_dir, "spotify_urls.p"), "rb")
        )

    for track_id in common_tracks:
        spotify_urls[track_id] = tracks[track_id]["url"]
        spotify_tracks[
            track_id
        ] = f"{tracks[track_id]['artist']} - {tracks[track_id]['title']}"

    pickle.dump(
        track2vec, open(os.path.join(args.deejai_model_dir, "tracktovec.p"), "wb")
    )
    pickle.dump(
        spotify2vec, open(os.path.join(args.deejai_model_dir, "spotifytovec.p"), "wb")
    )
    pickle.dump(
        spotify_urls, open(os.path.join(args.deejai_model_dir, "spotify_urls.p"), "wb")
    )
    pickle.dump(
        spotify_tracks,
        open(os.path.join(args.deejai_model_dir, "spotify_tracks.p"), "wb"),
    )
    shutil.copyfile(
        args.mp3tovec_model_file, os.path.join(args.deejai_model_dir, "speccy_model")
    )
