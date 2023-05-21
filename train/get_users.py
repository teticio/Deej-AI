import argparse

import pandas as pd
import requests
from utils import get_access_token


def get_following_and_followers(user, cookie):
    def add_users(users, response_json):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--cookie", type=str, required=True, help="sp_dc cookie")
    parser.add_argument("--user", type=str, required=True, help="Seed user")
    parser.add_argument(
        "--users_file",
        type=str,
        default="data/users.csv",
        help="Users CSV file",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of users")
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
