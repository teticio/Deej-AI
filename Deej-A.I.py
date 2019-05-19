from __future__ import print_function, division
import dash
from dash.dependencies import Input, Output, State
import dash_daq as daq
import dash_core_components as dcc
import dash_html_components as html
from flask import send_from_directory
import os
from io import BytesIO
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.id3 import ID3, ID3NoHeaderError
from PIL import Image
import base64
import numpy as np
import pandas as pd
import keras
from keras.models import load_model
from keras import backend as K
import librosa
import pickle
import random
import shutil
import time
import argparse

default_lookback = 3 # number of previous tracks to take into account
default_noise = 0    # amount of randomness to throw in the mix

app = dash.Dash()
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

@app.server.route('/static/<path:path>')
def static_file(path):
    static_folder = os.path.join(os.getcwd(), 'static')
    return send_from_directory(static_folder, path)

theme = {
    'dark': True,
    'detail': '#007439',
    'primary': '#00EA64', 
    'secondary': '#6E6E6E'
}

upload = html.Div(
    [
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select File'),
                ' and wait a bit...'
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center'
            },
            # Do not allow multiple files to be uploaded
            multiple=False
        ),
        html.Br()
    ],
    style={
        'width': '98vw',
        'textAlign': 'center',
        'margin': 'auto'
    }
)

shared = html.Div(
    id='shared-info',
    style={
        'display': 'none'
    }
)

lookback_knob = html.Div(
    [
        daq.Knob(
            id='lookback-knob',
            label='Keep on',
            size=90,
            max=10,
            value=default_lookback,
            scale={
                'start': 0,
                'labelInterval': 10,
                'interval': 1
            },
            theme=theme
        ),
        html.Div(
            id='lookback-knob-output',
            style={
                'display': 'none'
            }
        )
    ],
    style={
        'width': '10%',
        'display': 'inline-block'
    }
)

noise_knob = html.Div(
    [
        daq.Knob(
            id='noise-knob',
            label='Drunk',
            size=90,
            min=0,
            max=1,
            value=default_noise,
            scale={
                'start': 0,
                'labelInterval': 10,
                'interval': 0.1
            },
            theme=theme
        ),
        html.Div(
            id='noise-knob-output',
            style={
                'display': 'none'
            }
        )
    ],
    style={
        'width': '10%',
        'display': 'inline-block'
    }
)

app.layout = html.Div(
    [
        html.Link(
            rel='stylesheet',
            href='/static/custom.css'
        ),
        html.Div(
            [
                html.Div(
                    id='output-image-upload',
                    children=[
                        upload,
                        shared
                    ]
                )
            ]
        ),
        html.Div(
            [
                lookback_knob,
                html.Div(
                    style={
                        'width': '79%',
                        'display': 'inline-block'
                    }
                ),
                noise_knob
            ]
        )        
    ]
)

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

def make_playlist(seed_tracks, size=10, lookback=5, noise=0.01):
    max_tries = 10
    playlist = seed_tracks
    while len(playlist) < size:
        similar = most_similar(positive=playlist[-lookback:], topn=max_tries+2, noise=noise)
        candidates = [candidate[0] for candidate in similar if candidate[0] != playlist[-1]]
        for i in range(max_tries):
            if not candidates[i] in playlist:
                break
        playlist.append(candidates[i])
    return playlist

def get_mp3tovec(content_string, filename):
    sr                = 22050
    n_fft             = 2048
    hop_length        = 512
    n_mels            = 96 # model.layers[0].input_shape[1]
    slice_size        = 216 # model.layers[0].input_shape[2]
    slice_time        = slice_size * hop_length / sr
    start = time.time()
    print(f'Analyzing {filename}')
    decoded = base64.b64decode(content_string)
    with open('dummy.' + filename[-3:], 'wb') as file: # this is really annoying!
        shutil.copyfileobj(BytesIO(decoded), file, length=131072)
    y, sr = librosa.load('dummy.' + filename[-3:], mono=True)
    os.remove('dummy.' + filename[-3:])
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=sr/2)
    x = np.ndarray(shape=(S.shape[1] // slice_size, n_mels, slice_size, 1), dtype=float)
    for slice in range(S.shape[1] // slice_size):
        log_S = librosa.power_to_db(S[:, slice * slice_size : (slice+1) * slice_size], ref=np.max)
        if np.max(log_S) - np.min(log_S) != 0:
            log_S = (log_S - np.min(log_S)) / (np.max(log_S) - np.min(log_S))
        x[slice, :, :, 0] = log_S
    # need to put semaphore around this
    K.clear_session()
    model = load_model(model_file)
    new_vecs = model.predict(x)
    K.clear_session()
    print(f'Spectrogram analysis took {time.time() - start:0.0f}s')
    start = time.time()
    mp3s = {}
    dropout = batch_size / len(mp3tovec) # only process a selection of MP3s
    for vec in mp3tovec:
        if random.uniform(0, 1) > dropout:
            continue
        pickle_filename = (vec[:-3]).replace('\\', '_').replace('/', '_').replace(':','_') + 'p'
        unpickled = pickle.load(open(dump_directory + '/' + pickle_filename, 'rb'))
        mp3s[unpickled[0]] = unpickled[1]
    new_idfs = []
    for vec_i in new_vecs:
        idf = 1 # because each new vector is in the new mp3 by definition
        for mp3 in mp3s:
            for vec_j in mp3s[mp3]:
                if 1 - np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j)) < epsilon_distance:
                    idf += 1
                    break
        new_idfs.append(-np.log(idf / (len(mp3s) + 1))) # N + 1
    vec = 0
    for i, vec_i in enumerate(new_vecs):
        tf = 0
        for vec_j in new_vecs:
            if 1 - np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j)) < epsilon_distance:
                tf += 1
        vec += vec_i * tf * new_idfs[i]
    similar = most_similar_by_vec([vec], topn=1, noise=0)
    print(f'TF-IDF analysis took {time.time() - start:0.0f}s')
    return similar[0][0]
    
def get_track_info(filename):
    artwork = None
    artist = track = album = None
    duration = 0
    if filename[-3:].lower() == 'mp3':
        try:
            audio = ID3(filename)
            if audio.get('APIC:') is not None:
                pict = audio.get('APIC:').data
                im = Image.open(BytesIO(pict)).convert('RGB')
                buff = BytesIO()
                im.save(buff, format='jpeg')
                artwork = base64.b64encode(buff.getvalue()).decode('utf-8')
            if audio.get('TPE1') is not None:
                artist = audio['TPE1'].text[0]
            if audio.get('TIT2') is not None:
                track = audio['TIT2'].text[0]
            if audio.get('TALB') is not None:
                album = audio['TALB'].text[0]
        except ID3NoHeaderError:
            pass
        duration = MP3(filename).info.length
    elif filename[-3:].lower() == 'm4a':
        audio = MP4(filename)
        if audio.get("covr") is not None:
            pict = audio.get("covr")[0]
            im = Image.open(BytesIO(pict)).convert('RGB')
            buff = BytesIO()
            im.save(buff, format='jpeg')
            artwork = base64.b64encode(buff.getvalue()).decode('utf-8')
        if audio.get('\xa9ART') is not None:
            artist = audio.get('\xa9ART')[0]
        if audio.get('\xa9nam') is not None:
            track = audio.get('\xa9nam')[0]
        if audio.get('\xa9alb') is not None:
            album = audio.get('\xa9alb')[0]
        duration = audio.info.length
    if artwork == None:
        artwork = base64.b64encode(open('./static/record.jpg', 'rb').read()).decode()
    if (artist, track, album) == (None, None, None):
        artist = filename
    return artwork, artist, track, album, duration

def play_track(tracks, durations):
    artwork, artist, track, album, duration = get_track_info(tracks[-1])
    print(f'{len(tracks)}. {artist} - {track} ({album})')
    df = pd.DataFrame({'tracks': tracks, 'durations': durations + [duration]})
    jsonifed_data = df.to_json()
    return html.Div(
        [
            html.H1(
                f'{len(tracks)}. {artist} - {track} ({album})'
            ),
            html.Div(
                dcc.Upload(
                    id='upload-image',
                    style={
                            'display': 'none'
                    }
                )
            ),
            html.Audio(
                id='audio',
                src='data:audio/mp3;base64,{}'.format(base64.b64encode(open(tracks[-1], 'rb').read()).decode()),
                controls=False,
                autoPlay=True,
                style={
                    'display': 'none'
                }
            ),
            html.Div(
                [
                    html.Div(
                        html.Img(
                            src='data:image/jpeg;base64,{}'.format(artwork),
                            style={
                                'width': '85vh',
                                'margin': 'auto',
                                'display': 'inline-block'
                            }
                        ),
                        style={
                            'textAlign': 'center',
                        }
                    )
                ]
            ),
            html.Div(
                id='shared-info',
                style={
                    'display': 'none'
                },
                children=jsonifed_data
            )
        ]
    )

@app.callback(
    Output('lookback-knob-output', 'children'),
    [Input('lookback-knob', 'value')])
def update_output(value):
    print(f'lookback changed to {value}')
    return int(value)

@app.callback(
    Output('noise-knob-output', 'children'),
    [Input('noise-knob', 'value')])
def update_output(value):
    print(f'noise changed to {value}')
    return value

@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents'),
               Input('shared-info', 'children')],
              [State('upload-image', 'filename'),
               State('lookback-knob-output', 'children'),
               State('noise-knob-output', 'children')])
def update_output(contents, jsonified_data, filename, lookback, noise):
    print(f'lookback={lookback}, noise={noise}')
    if lookback is None:
        lookback = default_lookback
    if noise is None:
        noise = default_noise
    if jsonified_data is not None:
        # next time round
        df = pd.read_json(jsonified_data)
        durations = df['durations'].tolist()
        if demo is not None:
            time.sleep(demo)
        else:
            time.sleep(durations[-1])
        tracks = df['tracks'].tolist()
        tracks = make_playlist(tracks, size=len(tracks)+1, noise=noise, lookback=lookback)
        return play_track(tracks, durations)
    if contents is not None and filename is not None:
        # first time round
        content_type, content_string = contents.split(',')
        track = get_mp3tovec(content_string, filename)
        return play_track([track], [])
    # make sure we get called back
    time.sleep(1)
    return [upload, shared]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pickles', type=str, help='Directory of pickled TrackToVecs')
    parser.add_argument('mp3tovec', type=str, help='Filename (without extension) of pickled MP3ToVecs')
    parser.add_argument('--demo', type=int, help='Only play this number of seconds of each track')
    parser.add_argument('--model', type=str, help='Load spectrogram to Track2Vec model (default: "speccy_model")')
    parser.add_argument('--batchsize', type=int, help='Number of MP3s to process in each batch (default: 100)')
    parser.add_argument('--epsilon', type=float, help='Epsilon distance (default: 0.001)')
    args = parser.parse_args()
    dump_directory = args.pickles
    mp3tovec_file = args.mp3tovec
    demo = args.demo
    model_file = args.model
    batch_size = args.batchsize
    epsilon_distance = args.epsilon
    if model_file == None:
        model_file = 'speccy_model'
    if batch_size == None:
        batch_size = 100
    if epsilon_distance == None:
        epsilon_distance = 0.001 # should be small, but not too small
    mp3tovec = pickle.load(open(dump_directory + '/mp3tovecs/' + mp3tovec_file + '.p', 'rb'))
    print(f'{len(mp3tovec)} MP3s')
    app.run_server(debug=False)
