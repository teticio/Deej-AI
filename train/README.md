# Train Deej-A.I.

Since I last trained Deej-A.I. nearly 5 years ago, a lot of things have changed. For one thing, there is a lot of new music not included in the model, but also my Python programming has improved, I now have my own server with GPUs (as opposed to being limited to Google Colab) and I have learned ways to improve upon what I did originally. Even so, people still continue to use the website, so I think it is time to train a new model.

You can run `make help` from the project root directory. If you run

```bash
make install COOKIE=... USER=... MAX_WORKERS=1 PROXY=
```

then by default it will make all the steps below necessary to create the datasets, train the models and install them to the `../deej-ai.online-dev/model`. You will need to specify the `sp_dc` cookie and Spotify user to start from (see below). You will also need to ensure that the `test_track_ids` in the `config/track2vec.yaml` and `config/mp3tovec.yaml` files are present by searching in the `data/tracks_dedup.csv` file. The workflow would typically be

```bash
# Create datasets
make tracks COOKIE=... USER=... MAX_WORKERS=1 PROXY=
make search
make install
```

If you want to use more than one worker, you will need to set up proxy Lambda functions to avoid throttling issues with the Spotify API as described below. More detailed descriptions of the scripts follow.

## Setup

```bash
pip install -r train/requirements.txt
mkdir data
mkdir models
mkdir previews
mkdir spectrograms
```

## Get data

The first step is to produce a dataset of Spotify user playlists. Previously I did this by searching for all plalists containg 'a', 'b' and so on, but a better (but less legitamate) way is to first create a list of Spotify user IDs and then get all their public playlists.

Use the inspector in your browser to get the `sp_dc` cookie from the web player when you are logged in to Spotify. Then run the following script to get at least 1,000,000 Spotify user IDs. It does this by crawling the followers of the seed user, who they are following and their followers and so on. Unfortunately, there is no way to do this via the regular Spotify API.

```bash
python train/get_users.py --cookie=... --user=<seed_user> --limit=1000000
```

To get a list of just over 1,000,000 public playlists of these users, run this script.

```bash
python train/get_playlists.py --limit=1000000
```

Then the following script will download the track lists for these playlists. This can take weeks to run on a single CPU core. In order to run in parallel, it is necessary to use a proxy pool, otherwise Spotify will throttle the requests (see https://github.com/teticio/lambda-scraper).

```bash
# Assuming you have 32 cores and have deployed Lambda functions proxy-0 ... proxy-31 for the proxy pool 
python train/get_tracks.py --max_workers=32 --proxy=proxy
```

To de-duplicate tracks by the same artist with the same title and drop tracks with no preview URL or less than 10 references, run the following script. The track IDs in playlists for dropped tracks will be replaced with an ID specified by `--oov` or skipped if this is not set.

```bash
python train/deduplicate.py --min_count=10
```

Use this script to find particular tracks (useful for model evaluation - see `config/track2vec.yaml`).

```bash
python train/search_tracks.py --search="James Brown"
```

## Train models

Now you can train a Track2Vec model with
```bash
python train/track2vec.py
python train/test2vec.py
```

To train the MP3ToVec model, you will first need to download the 30 second previews of the tracks. Each preview is around 350k in size.

```bash
python train/download_previews.py
```

Then, you will need to calculate the spectograms of the first 5 seconds of the previews.
```bash
python train/calc_spectrograms.py
```

Now you can train the MP3ToVec model with
```bash
python train/train_mp3tovec.py
```

Use this script to calculate the Mp3ToVec embeddings for the previews.
```bash
python train/calc_mp3tovecs.py
```

Finally, to reduce these to a single vector per track using the TF-IDF algorithm, run
```bash
python train/calc_tfidf.py
```

## Install models

The MP3ToVec model can be converted from PyTorch to TensorFlow with
```bash
python train/pt_to_tf.py
```

The `spotifytovec.p`, `tracktovec.p`, `spotify_tracks.p`, `spotify_urls.p` and `speccy_model` files are installed to the directory where [`deej-ai.online`](https://github.com/teticio/deej-ai.online-app) is running with the following command. This also ensures that there are the same tracks in all the files.
```bash
python train/install_model.py
```

# TODO
* Makefile
