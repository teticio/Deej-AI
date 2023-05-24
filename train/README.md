# Train Deej-A.I.

Since I last trained Deej-A.I. nearly 5 years ago, a lot of things have changed. For one thing, there is a lot of new music not included in the model, but also my Python programming has improved, I now have my own server with GPUs (as opposed to being limited to Google Colab) and I have learned ways to improve upon what I did originally. Even so, people still continue to use the website, so I think it is time to train a new model.

## Setup

The first step is to produce a dataset of Spotify user playlists. Previously I did this by searching for all plalists containg 'a', 'b' and so on, but a better (but less legitamate) way is to first create a list of Spotify user IDs and then get all their public playlists.

Use the inspector in your browser to get the `sp_dc` cookie from the web player when you are logged in to Spotify. Then run the following script to get at least 1,000,000 Spotify user IDs. It does this by crawling the followers of the seed user, who they are following and their followers and so on. Unfortunately, there is no way to do this via the regular Spotify API.

```bash
pip install -r train/requirements.txt
mkdir data
mkdir models
mkdir previews
mkdir spectrograms
```

## Get data

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
python train/get_playlist_items.py --max_workers=32 --proxy=proxy
```

To de-duplicate tracks by the same artist with the same title and drop tracks with no preview URL or less than 10 references, run the following script. The track IDs in playlists for dropped tracks will be replaced with an ID specified by `--oov` or skipped if this is not set.

```bash
python train/deduplicate.py --min_count=10
```

Now you can download the 30 second previews of these tracks. Each preview is around 350k in size.

```bash
python train/download_previews.py --min_count=10
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

To train the MP3ToVec model, you will first need to calculate the spectograms of the first 5 seconds of the previews.
```bash
python train/calc_spectrograms.py
```

Then you can train the model with
```bash
python train/train_mp3tovec.py
```

Finally, use this script to calculate the Mp3ToVec embeddings for the previews.
```bash
python train/calc_mp3tovecs.py
```

To reduce these to a single vector per track using the TF-IDF algorithm, run
```bash
python train/calc_tfidf.py
```

## Install models

The MP3ToVec model can be converted from PyTorch to TensorFlow with
```bash
python train/pt_to_tf.py
```

The `spotifytovec.p`, `tracktovec.p`, `spotify_tracks.p`, `spotify_urls.p` and `speccy_model` files are installed to the directory where `deej-ai.online` is running with the following command. This also ensures that there are the same tracks in all the files.
```bash
python train/install_model.py
```
