import argparse
import os
from time import sleep

import pandas as pd
import spotipy
from tqdm import tqdm
from utils import get_access_token, paginate

if __name__ == "__main__":
    """
    Entry point for the get_playlists script.

    Parses command-line arguments and retrieves playlists for a list of Spotify users.

    Args:
        --limit (int): Limit number of playlists to retrieve. Default is None (retrieve all playlists).
        --playlists_file (str): Path to the output CSV file where playlists will be saved. Default is "data/playlists.csv".
        --start_from (int): Index of the first user from which to retrieve playlists for. Default is 0.
        --users_file (str): Path to the user IDs CSV file to retrieve playlists for. Default is "data/users.csv".

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of playlists"
    )
    parser.add_argument(
        "--playlists_file",
        type=str,
        default="data/playlists.csv",
        help="Output playlists CSV file",
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=0,
        help="Start from user index",
    )
    parser.add_argument(
        "--users_file",
        type=str,
        default="data/users.csv",
        help="Input users CSV file",
    )
    args = parser.parse_args()

    users = pd.read_csv(args.users_file, header=None, index_col=None)[0]

    playlists = (
        set(pd.read_csv(args.playlists_file, header=None, index_col=None)[0])
        if os.path.exists(args.playlists_file)
        else set()
    )
    for user in tqdm(
        users[args.start_from :],
        initial=args.start_from,
        total=len(users),
        desc="Getting playlists",
    ):
        try:
            access_token = get_access_token()
            sp = spotipy.Spotify(access_token)
            playlists |= set(
                playlist["id"] for playlist in paginate(sp.user_playlists, user=user)
            )
        except KeyboardInterrupt:
            break
        except Exception:
            print(f"Skipping {user}")
            sleep(1)
        if args.limit is not None and len(playlists) >= args.limit:
            break

    print(len(playlists))
    print(f"Writing {args.playlists_file}")
    pd.DataFrame(playlists).to_csv(args.playlists_file, index=False, header=False)
