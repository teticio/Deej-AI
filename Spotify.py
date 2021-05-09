# TODO 
# limit number of tracks from same artist
# remove exact duplicates (mp3vec proximity 1)

import os
import numpy as np
import pickle
import argparse
import spotipy
import spotipy.util as util
import random
import requests
import webbrowser

# If you want to be able to load your playlists into Spotify
# you will need to get credentials from https://developer.spotify.com/dashboard/applications

scope = 'playlist-modify-public'
client_id = '194086cb37be48ebb45b9ba4ce4c5936'
client_secret = 'fb9fb4957a9841fcb5b2dbc7804e1e85'
redirect_uri = 'https://www.attentioncoach.es/'

epsilon_distance = 0.001


def download_file_from_google_drive(id, destination):
    if os.path.isfile(destination):
        return None
    print(f'Downloading {destination}')
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def add_track_to_playlist(sp, username, playlist_id, track_id, replace=False):
    if sp is not None and username is not None and playlist_id is not None:
        try:
            if replace:
                result = sp.user_playlist_replace_tracks(
                    username, playlist_id, [track_id])
            else:
                result = sp.user_playlist_add_tracks(username, playlist_id,
                                                     [track_id])
        except spotipy.client.SpotifyException:
            # token has probably gone stale
            token = util.prompt_for_user_token(username, scope, client_id,
                                               client_secret, redirect_uri)
            sp = spotipy.Spotify(token)
            if replace:
                result = sp.user_playlist_replace_tracks(
                    username, playlist_id, [track_id])
            else:
                result = sp.user_playlist_add_tracks(username, playlist_id,
                                                     [track_id])


def most_similar(mp3tovecs, weights, positive=[], negative=[], noise=0):
    if isinstance(positive, str):
        positive = [positive]  # broadcast to list
    if isinstance(negative, str):
        negative = [negative]  # broadcast to list
    similar = np.zeros((len(mp3tovecs[0]), 2, len(weights)), dtype=np.float64)
    for k, mp3tovec in enumerate(mp3tovecs):
        mp3_vec_i = np.sum([mp3tovec[i] for i in positive] +
                           [-mp3tovec[i] for i in negative],
                           axis=0)
        mp3_vec_i += np.random.normal(0, noise, len(mp3_vec_i))
        mp3_vec_i = mp3_vec_i / np.linalg.norm(mp3_vec_i)
        for j, track_j in enumerate(mp3tovec):
            if track_j in positive or track_j in negative:
                continue
            mp3_vec_j = mp3tovec[track_j]
            similar[j, 0, k] = j
            similar[j, 1, k] = np.dot(mp3_vec_i, mp3_vec_j)
    return sorted(similar, key=lambda x: -np.dot(x[1], weights))


def most_similar_by_vec(mp3tovecs,
                        weights,
                        positives=None,
                        negatives=None,
                        noise=0):
    similar = np.zeros((len(mp3tovecs[0]), 2, len(weights)), dtype=np.float64)
    positive = negative = []
    for k, mp3tovec in enumerate(mp3tovecs):
        if positives is not None:
            positive = positives[k]
        if negatives is not None:
            negative = negatives[k]
        if isinstance(positive, str):
            positive = [positive]  # broadcast to list
        if isinstance(negative, str):
            negative = [negative]  # broadcast to list
        mp3_vec_i = np.sum([i for i in positive] + [-i for i in negative],
                           axis=0)
        mp3_vec_i += np.random.normal(0, noise, len(mp3_vec_i))
        mp3_vec_i = mp3_vec_i / np.linalg.norm(mp3_vec_i)
        for j, track_j in enumerate(mp3tovec):
            mp3_vec_j = mp3tovec[track_j]
            similar[j, 0, k] = j
            similar[j, 1, k] = np.dot(mp3_vec_i, mp3_vec_j)
    return sorted(similar, key=lambda x: -np.dot(x[1], weights))


# create a musical journey between given track "waypoints"
def join_the_dots(sp, username, playlist_id, mp3tovecs, weights, ids, \
                  tracks, track_ids, n=5, noise=0, replace=True):
    playlist = []
    playlist_tracks = [tracks[_] for _ in ids]
    end = start = ids[0]
    start_vec = [mp3tovec[start] for k, mp3tovec in enumerate(mp3tovecs)]
    for end in ids[1:]:
        end_vec = [mp3tovec[end] for k, mp3tovec in enumerate(mp3tovecs)]
        playlist.append(start)
        add_track_to_playlist(sp, username, playlist_id, playlist[-1], replace
                              and len(playlist) == 1)
        print(f'{len(playlist)}.* {tracks[playlist[-1]]}')
        for i in range(n):
            candidates = most_similar_by_vec(mp3tovecs,
                                             weights,
                                             [[(n - i + 1) / n * start_vec[k] +
                                               (i + 1) / n * end_vec[k]]
                                              for k in range(len(mp3tovecs))],
                                             noise=noise)
            for candidate in candidates:
                track_id = track_ids[int(candidate[0][0])]
                if track_id not in playlist + ids and tracks[
                        track_id] not in playlist_tracks:
                    break
            playlist.append(track_id)
            playlist_tracks.append(tracks[track_id])
            add_track_to_playlist(sp, username, playlist_id, playlist[-1])
            print(f'{len(playlist)}. {tracks[playlist[-1]]}')
        start = end
        start_vec = end_vec
    playlist.append(end)
    add_track_to_playlist(sp, username, playlist_id, playlist[-1])
    print(f'{len(playlist)}.* {tracks[playlist[-1]]}')
    return playlist

def make_playlist(sp, username, playlist_id, mp3tovecs, weights, seed_tracks, \
                  tracks, track_ids, size=10, lookback=3, noise=0, replace=True):
    playlist = seed_tracks
    playlist_tracks = [tracks[_] for _ in playlist]
    for i in range(0, len(seed_tracks)):
        add_track_to_playlist(sp, username, playlist_id, playlist[i], replace
                              and len(playlist) == 1)
        print(f'{i+1}.* {tracks[seed_tracks[i]]}')
    for i in range(len(seed_tracks), size):
        candidates = most_similar(mp3tovecs,
                                  weights,
                                  positive=playlist[-lookback:],
                                  noise=noise)
        for candidate in candidates:
            track_id = track_ids[int(candidate[0][0])]
            if track_id not in playlist and tracks[
                    track_id] not in playlist_tracks:
                break
        playlist.append(track_id)
        playlist_tracks.append(tracks[track_id])
        add_track_to_playlist(sp, username, playlist_id, playlist[-1])
        print(f'{i+1}. {tracks[playlist[-1]]}')
    return playlist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', type=str, help='Spotify username')
    parser.add_argument('--playlist',
                        type=str,
                        help='Playlist name (must already exist')
    parser.add_argument('--n',
                        type=int,
                        help='Size of playlist to generate (default 5)')
    parser.add_argument('--creativity',
                        type=float,
                        help='Discover something new? (0-1, default 0.5)')
    parser.add_argument(
        '--lookback',
        type=int,
        help='Number of previous tracks to consider (default 3)')
    parser.add_argument('--noise',
                        type=float,
                        help='Degree of randomness (0-1, default 0)')
    parser.add_argument('--mp3',
                        type=str,
                        help='Start with sommething that sounds like this')
    parser.add_argument('--mp3tovec',
                        type=str,
                        help='MP3ToVecs file (full path)')
    parser.add_argument('--extend',
                        action='store_true',
                        help='Extend existing playlist')
    args = parser.parse_args()
    username = args.user
    playlist_name = args.playlist
    size = args.n
    creativity = args.creativity
    lookback = args.lookback
    noise = args.noise
    mp3_filename = args.mp3
    user_mp3tovecs_filename = args.mp3tovec
    if size is None:
        size = 5
    if creativity is None:
        creativity = 0.5
    if lookback is None:
        lookback = 3
    if noise is None:
        noise = 0
    if args.extend:
        replace = False
    else:
        replace = True
    sp = playlist_id = None
    if username is not None and playlist_name is not None:
        token = util.prompt_for_user_token(username, scope, client_id,
                                           client_secret, redirect_uri)
        if token is not None:
            sp = spotipy.Spotify(token)
            if sp is not None:
                try:
                    playlists = sp.user_playlists(username)
                    if playlists is not None:
                        playlist_ids = [
                            playlist['id'] for playlist in playlists['items']
                            if playlist['name'] == playlist_name
                        ]
                        if len(playlist_ids) > 0:
                            playlist_id = playlist_ids[0]
                except:
                    pass
        if playlist_id is None:
            print(
                f'Unable to access playlist {playlist_name} for user {username}'
            )
    download_file_from_google_drive('1Mg924qqF3iDgVW5w34m6Zaki5fNBdfSy',
                                    'spotifytovec.p')
    download_file_from_google_drive('1geEALPQTRBNUvkpI08B-oN4vsIiDTb5I',
                                    'tracktovec.p')
    download_file_from_google_drive('1Qre4Lkym1n5UTpAveNl5ffxlaAmH1ntS',
                                    'spotify_tracks.p')
    mp3tovecs = pickle.load(open('spotifytovec.p', 'rb'))
    mp3tovecs = dict(
        zip(mp3tovecs.keys(),
            [mp3tovecs[_] / np.linalg.norm(mp3tovecs[_]) for _ in mp3tovecs]))
    tracktovecs = pickle.load(open('tracktovec.p', 'rb'))
    tracktovecs = dict(
        zip(tracktovecs.keys(), [
            tracktovecs[_] / np.linalg.norm(tracktovecs[_])
            for _ in tracktovecs
        ]))
    tracks = pickle.load(open('spotify_tracks.p', 'rb'))
    track_ids = [_ for _ in mp3tovecs]
    if mp3_filename is None or user_mp3tovecs_filename is None:
        user_input = input('Search keywords: ')
        input_tracks = []
        while True:
            if user_input == '':
                break
            ids = sorted([
                track for track in mp3tovecs
                if all(word in tracks[track].lower()
                       for word in user_input.lower().split())
            ],
                         key=lambda x: tracks[x])
            for i, id in enumerate(ids):
                print(f'{i+1}. {tracks[id]}')
            while True:
                user_input = input(
                    'Input track number, ENTER to finish, or search keywords: '
                )
                if user_input == '':
                    break
                if user_input.isdigit() and len(ids) > 0:
                    if int(user_input) - 1 >= len(ids):
                        continue
                    id = ids[int(user_input) - 1]
                    input_tracks.append(id)
                    print(f'Added {tracks[id]} to playlist')
                else:
                    break
        print()
        if len(input_tracks) == 0:
            ids = [track for track in mp3tovecs]
            input_tracks.append(ids[random.randint(0, len(ids))])
        if len(input_tracks) > 1:
            playlist = join_the_dots(sp,
                                     username,
                                     playlist_id, [mp3tovecs, tracktovecs],
                                     [creativity, 1 - creativity],
                                     input_tracks,
                                     tracks,
                                     track_ids,
                                     n=size,
                                     noise=noise,
                                     replace=replace)
        else:
            playlist = make_playlist(sp,
                                     username,
                                     playlist_id, [mp3tovecs, tracktovecs],
                                     [creativity, 1 - creativity],
                                     input_tracks,
                                     tracks,
                                     track_ids,
                                     size=size,
                                     lookback=lookback,
                                     noise=noise,
                                     replace=replace)
    else:
        user_mp3tovecs = pickle.load(open(user_mp3tovecs_filename, 'rb'))
        ids = most_similar_by_vec(mp3tovecs, [user_mp3tovecs[mp3_filename]],
                                  topn=10)
        for i, id in enumerate(ids):
            print(f'{i+1}. {tracks[id[0]]} [{id[1]:.2f}]')
        user_input = input('Input track number: ')
        if user_input.isdigit(
        ) and int(user_input) > 0 and int(user_input) < len(ids):
            print()
            playlist = make_playlist(sp,
                                     username,
                                     playlist_id, [mp3tovecs, tracktovecs],
                                     [creativity, 1 - creativity],
                                     [ids[int(user_input) - 1][0]],
                                     size=size,
                                     lookback=lookback,
                                     noise=noise,
                                     replace=replace)
    if sp is None or username is None or playlist_id is None:
        with open("playlist.html", "w") as text_file:
            for id in playlist:
                text_file.write(
                    f'<iframe src="https://open.spotify.com/embed/track/{id}" width="100%" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>'
                )
        webbrowser.open('file://' + os.path.realpath('playlist.html'))
