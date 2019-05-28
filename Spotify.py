# TODO
# download embeddings and spotify_tracks.csv (without URL) if not there
# limit number of tracks from same artist
# remove exact duplicates (mp3vec proximity 1)

import os
import numpy as np
import pickle
import argparse
import spotipy
import spotipy.util as util
import webbrowser
from tqdm import tqdm

# You can get credentials from https://developer.spotify.com/dashboard/applications
scope = 'playlist-modify-public'
client_id='fill this in with your details'
client_secret='fill this in with your details'
redirect_uri='fill this in with your details'

epsilon_distance = 0.001

def spotify_playlist(sp, username, playlist_id, track_details):
    with open("playlist.html", "w") as text_file:
        track_ids = [track_detail for track_detail in track_details]
        if sp is None or username is None or playlist_id is None:
            for track_detail in track_details:
                text_file.write(f'<iframe src="https://open.spotify.com/embed/track/{track_detail}" width="100%" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>')
        else:
            try:
                result = sp.user_playlist_replace_tracks(username, playlist_id, track_ids)
            except spotipy.client.SpotifyException:
                # token has probably gone stale
                token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)
                sp = spotipy.Spotify(token)
                result = sp.user_playlist_replace_tracks(username, playlist_id, track_ids)
            text_file.write(f'<iframe src="https://open.spotify.com/embed/user/{username}/playlist/{playlist_id}" width="100%" height="100%" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>')

def most_similar(mp3tovecs, weights, positive=[], negative=[], topn=5, noise=0):
    if isinstance(positive, str):
        positive = [positive] # broadcast to list
    if isinstance(negative, str):
        negative = [negative] # broadcast to list
    similar = [('', 0)] * len(mp3tovecs[0])
    for i, mp3tovec in enumerate(mp3tovecs):
        mp3_vec_i = np.sum([mp3tovec[i] for i in positive] + [-mp3tovec[i] for i in negative], axis=0)
        mp3_vec_i += np.random.normal(0, noise * np.linalg.norm(mp3_vec_i), len(mp3_vec_i))
        for j, track_j in enumerate(mp3tovec):
            if track_j in positive or track_j in negative:
                continue
            mp3_vec_j = mp3tovec[track_j]
            cos_proximity = np.dot(mp3_vec_i, mp3_vec_j) / (np.linalg.norm(mp3_vec_i) * np.linalg.norm(mp3_vec_j))
            similar[j] = (track_j, similar[j][1] + weights[i] * cos_proximity)
    return sorted(similar, key=lambda x:-x[1])[:topn]

def make_playlist(mp3tovecs, weights, seed_tracks, size=10, lookback=3, noise=0):
    max_tries = 10
    playlist = seed_tracks
    for i in tqdm(range(len(seed_tracks), size)):
        similar = most_similar(mp3tovecs, weights, positive=playlist[-lookback:], topn=5, noise=noise)
        candidates = [candidate[0] for candidate in similar if candidate[0] != playlist[-1]]
        for candidate in candidates:
            if not candidate in playlist:
                break
        playlist.append(candidate)
    return playlist

def most_similar_by_vec(mp3tovec, positive=[], negative=[], topn=5, noise=0):
    if isinstance(positive, str):
        positive = [positive] # broadcast to list
    if isinstance(negative, str):
        negative = [negative] # broadcast to list
    mp3_vec_i = np.sum([i for i in positive] + [-i for i in negative], axis=0)
    mp3_vec_i += np.random.normal(0, noise * np.linalg.norm(mp3_vec_i), len(mp3_vec_i))
    similar = []
    for track_j in mp3tovec:
        mp3_vec_j = mp3tovec[track_j]
        cos_proximity = np.dot(mp3_vec_i, mp3_vec_j) / (np.linalg.norm(mp3_vec_i) * np.linalg.norm(mp3_vec_j))
        similar.append((track_j, cos_proximity))
    return sorted(similar, key=lambda x:-x[1])[:topn]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', type=str, help='Spotify username')
    parser.add_argument('--playlist', type=str, help='Playlist name (must already exist')
    parser.add_argument('--n', type=int, help='Size of playlist to generate (default 20)')
    parser.add_argument('--creativity', type=float, help='Discover something new? (0-1, default 0.5)')
    parser.add_argument('--lookback', type=int, help='Number of previous tracks to consider (default 3)')
    parser.add_argument('--noise', type=float, help='Degree of randomness (0-1, default 0)')
    parser.add_argument('--mp3', type=str, help='Start with sommething that sounds like this')
    parser.add_argument('--mp3tovec', type=str, help='MP3ToVecs file (full path)')
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
        size = 20
    if creativity is None:
        creativity = 0.5
    if lookback is None:
        lookback = 3
    if noise is None:
        noise = 0
    sp = playlist_id = None
    if username is not None and playlist_name is not None:
        token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)
        if token is not None:
            sp = spotipy.Spotify(token)
            if sp is not None:
                playlists = sp.user_playlists(username)
                if playlists is not None:
                    playlist_ids = [playlist['id'] for playlist in playlists['items'] if playlist['name'] == playlist_name]
                    if len(playlist_ids) > 0:
                        playlist_id = playlist_ids[0]
        if playlist_id is None:
            print(f'Unable to access playlist {playlist_name} for user {username}')
    # download embeddings and tracks.csv if not exists here
    mp3tovecs = pickle.load(open('spotifytovec.p', 'rb'))
    tracktovecs = pickle.load(open('tracktovec.p', 'rb'))
    tracks = pickle.load(open('spotify_tracks.p', 'rb'))
    if mp3_filename is None or user_mp3tovecs_filename is None:
        user_input = input('Search keywords: ')
        while True:
            ids = sorted([track for track in mp3tovecs if all(word in tracks[track].lower() for word in user_input.lower().split())], key = lambda x: tracks[x])
            for i, id in enumerate(ids):
                print(f'{i+1}. {tracks[id]}')
            while True:
                user_input = input('Input track number, 0 to finish, or search keywords: ')
                if user_input == '0':
                    break
                if user_input.isdigit() and len(ids) > 0:
                    if int(user_input)-1 >= len(ids):
                        continue
                    id = ids[int(user_input)-1]
                    playlist = make_playlist([mp3tovecs, tracktovecs], [creativity, 1-creativity], [id], size=size, lookback=lookback, noise=noise)
                    spotify_playlist(sp, username, playlist_id, playlist)
                    webbrowser.open('file://' + os.path.realpath('playlist.html'))
                else:
                    break
            if user_input == '0':
                break
    else:
        user_mp3tovecs = pickle.load(open(user_mp3tovecs_filename, 'rb'))
        ids = most_similar_by_vec(mp3tovecs, [user_mp3tovecs[mp3_filename]], topn=10)
        for i, id in enumerate(ids):
            print(f'{i+1}. {tracks[id[0]]} [{id[1]:.2f}]')
        user_input = input('Input track number: ')
        if user_input.isdigit() and int(user_input) > 0 and int(user_input) < len(ids):
            playlist = make_playlist([mp3tovecs, tracktovecs], [creativity, 1-creativity], [ids[int(user_input)-1][0]], size=size, lookback=lookback, noise=noise)
            spotify_playlist(sp, username, playlist_id, playlist)
            webbrowser.open('file://' + os.path.realpath('playlist.html'))