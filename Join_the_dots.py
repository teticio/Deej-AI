import pickle
import numpy as np
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.id3 import ID3, ID3NoHeaderError
import subprocess as sp
import random
import argparse
import re

max_duration = 10 * 60 # avoid adding mixes to mix

def get_track_duration(filename):
    duration = 0
    if filename[-3:].lower() == 'mp3':
        duration = MP3(filename).info.length
    elif filename[-3:].lower() == 'm4a':
        duration = MP4(filename).info.length
    return duration

def most_similar(positive=[], negative=[], topn=5, noise=0):
    if isinstance(positive, str):
        positive = [positive] # broadcast to list
    if isinstance(negative, str):
        negative = [negative] # broadcast to list
    mp3_vec_i = np.sum([mp3tovec[i] for i in positive] + [-mp3tovec[i] for i in negative], axis=0)
    mp3_vec_i += np.random.normal(0, noise * np.linalg.norm(mp3_vec_i), len(mp3_vec_i))
    similar = []
    for track_j in mp3tovec:
        if track_j in positive or track_j in negative:
            continue
        mp3_vec_j = mp3tovec[track_j]
        cos_proximity = np.dot(mp3_vec_i, mp3_vec_j) / (np.linalg.norm(mp3_vec_i) * np.linalg.norm(mp3_vec_j))
        similar.append((track_j, cos_proximity))
    return sorted(similar, key=lambda x:-x[1])[:topn]

def most_similar_by_vec(positive=[], negative=[], topn=5, noise=0):
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

def make_playlist(seed_tracks, size=10, lookback=3, noise=0):
    max_tries = 10
    playlist = seed_tracks
    while len(playlist) < size:
        similar = most_similar(positive=playlist[-lookback:], topn=max_tries, noise=noise)
        candidates = [candidate[0] for candidate in similar if candidate[0] != playlist[-1]]
        for candidate in candidates:
            if not candidate in playlist and get_track_duration(candidate) < max_duration:
                break
        playlist.append(candidate)
    return playlist

def join_the_dots(tracks, n=5, noise=0): # create a musical journey between given track "waypoints"
    max_tries = 10
    playlist = []
    end = start = tracks[0]
    start_vec = mp3tovec[start]
    for end in tracks[1:]:
        end_vec = mp3tovec[end]
        playlist.append(start)
        for i in range(n-1):
            similar = most_similar_by_vec(positive=[(n-i+1)/n * start_vec + (i+1)/n * end_vec], topn=max_tries, noise=noise)
            candidates = [candidate[0] for candidate in similar if candidate[0] != playlist[-1]]
            for candidate in candidates:
                if not candidate in playlist and candidate != end and get_track_duration(candidate) < max_duration:
                    break
            playlist.append(candidate)
        start = end
        start_vec = end_vec
    playlist.append(end)
    return playlist

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mp3tovec', type=str, help='MP3ToVecs file (full path)')
    parser.add_argument('--inputs', type=str, help='Text file with list of songs')
    parser.add_argument('output', type=str, help='Output MP3 filename')
    parser.add_argument('n', type=int, help='Number of songs to add between input songs')
    parser.add_argument('--noise', type=float, help='Degree of randomness (0-1)')
    args = parser.parse_args()
    mp3tovec_filename = args.mp3tovec
    tracks_filename = args.inputs
    mix_filename = args.output
    n = args.n
    mp3tovec = pickle.load(open(mp3tovec_filename, 'rb'))
    noise = 0
    if args.noise is not None:
        noise = args.noise
    input_tracks = []
    if tracks_filename is not None:
        with open(tracks_filename, 'rt') as file:
            for track in file:
                input_tracks.append(track.replace('\n',''))
    else:
        user_input = input('Search keywords: ')
        while True:
            tracks = sorted([mp3 for mp3 in mp3tovec if all(word in mp3.lower() for word in user_input.lower().split())])
            for i, track in enumerate(tracks):
                print(f'{i+1}. {track}')
            while True:
                user_input = input('Input track number to add, 0 to finish, or search keywords: ')
                if user_input == '0':
                    break
                if user_input.isdigit() and len(tracks) > 0:
                    if int(user_input)-1 >= len(tracks):
                        continue
                    input_tracks.append(tracks[int(user_input)-1])
                    print(f'Added {tracks[int(user_input)-1]} to playlist')
                else:
                    break
            if user_input == '0':
                break
        print()
    total_duration = 0
    if len(input_tracks) == 0:
        tracks = [mp3 for mp3 in mp3tovec]
        input_tracks.append(tracks[random.randint(0, len(tracks))])
    if len(input_tracks) > 1:
        playlist = join_the_dots(input_tracks, n=n, noise=noise)
    else:
        playlist = make_playlist(input_tracks, size=n, lookback=3, noise=noise)
    tracks = []
    for i, track in enumerate(playlist):
        tracks.append('-i')
        tracks.append(track)
        total_duration += get_track_duration(track)
        if n == 0 and i == 0 or n != 0 and i % n == 0:
            print(f'{i+1}.* {track}')
        else:
            print(f'{i+1}. {track}')
    print(f'Total duration = {total_duration//60//60:.0f}:{total_duration//60%60:02.0f}:{total_duration%60:02.0f}s')
    print('')
    print(f'Creating mix {mix_filename}')
    pipe = sp.Popen(['ffmpeg',
                    '-y', # replace if exists
                    '-i', 'static/meta_data.txt'] + # use this meta data
                    tracks + # append playlist tracks
                    ['-filter_complex', f'loudnorm=I=-14,concat=n={len(playlist)}:v=0:a=1[out]', # normalize and concatenate
                    '-map', '[out]', # final output
                    mix_filename], # output file
                   stdin=sp.PIPE,stdout=sp.PIPE, stderr=sp.PIPE)
    pipe.communicate()