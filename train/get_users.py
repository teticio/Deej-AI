import argparse
from typing import Set

import pandas as pd
import requests
from utils import get_access_token


def get_following_and_followers(user: str, cookie: str) -> Set[str]:
    """
    Retrieves the list of users that a given user is following or followed by.

    Args:
        user (str): The username of the user to retrieve following and followers for.
        cookie (str): The cookie string to use for authentication.

    Returns:
        A set containing the usernames of the users that the given user is following or followed by.
    """

    def add_users(users: Set[str], response_json: dict) -> None:
        """
        Adds the usernames of Spotify users to a set.

        Args:
            users (set): The set to add the usernames to.
            response_json (dict): The JSON response from a Spotify API request.

        Returns:
            None
        """
        if "profiles" in response_json:
            for profile in response_json["profiles"]:
                if profile["uri"].startswith("spotify:user:"):
                    users.add(profile["uri"][len("spotify:user:") :])

    access_token = get_access_token(cookie)
    users = set()
    response = requests.get(
        f"https://spclient.wg.spotify.com/user-profile-view/v3/profile/{user}/following?market=from_token",
        headers={
            "authorization": f"Bearer {access_token}",
        },
    )
    add_users(users, response.json())
    response = requests.get(
        f"https://spclient.wg.spotify.com/user-profile-view/v3/profile/{user}/followers?market=from_token",
        headers={
            "authorization": f"Bearer {access_token}",
        },
    )
    add_users(users, response.json())
    return users


if __name__ == "__main__":
    """
    Entry point for the get_users script.

    Retrieves Spotify users.

    Args:
        --cookie (str): The sp_dc cookie to use for authentication.
        --limit (int): The maximum number of users to retrieve. Default is None (CTRL+C to stop).
        --user (str): The username of the user to start from.
        --users_file (str): The path to the CSV file to write the users to. Default is data/users.csv.

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cookie", type=str, required=True, help="sp_dc cookie")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of users")
    parser.add_argument("--user", type=str, required=True, help="Seed user")
    parser.add_argument(
        "--users_file",
        type=str,
        default="data/users.csv",
        help="Users CSV file",
    )
    args = parser.parse_args()

    users = set([args.user])
    new_users = set(users)
    while True:
        try:
            for user in new_users:
                try:
                    users |= get_following_and_followers(user, args.cookie)
                except KeyboardInterrupt:
                    raise (KeyboardInterrupt)
                except Exception:
                    print(f"Skipping {user}")
                print(len(users), end="\r")
                if args.limit is not None and len(users) >= args.limit:
                    break
            if args.limit is not None and len(users) >= args.limit:
                break
            new_users = users - new_users

        except KeyboardInterrupt:
            break

    print(f"\nWriting {args.users_file}")
    pd.DataFrame(users).to_csv(args.users_file, index=False, header=False)
