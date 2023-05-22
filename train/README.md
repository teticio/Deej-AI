# Train Deej-A.I.

Since I last trained Deej-A.I. nearly 5 years ago, a lot of things have changed. For one thing, there is a lot of new music not included in the model, but also my Python programming has improved, I now have my own server with GPUs (as opposed to being limited to Google Colab) and I have learned ways to improve upon what I did originally. Even so, people still continue to use the website, so I think it is time to train a new model.

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
python train/get_playlist_items.py --max_workers=32 --proxy=proxy
```

To de-duplicate tracks by the same artist with the same title and drop tracks with no preview URL or less than 10 references, run the following script. The track IDs in playlists for dropped tracks will be replaced with an ID specified by `--oov` or skipped if this is not set.

```bash
python train/deduplicate.py --min_count=10 -drop_missing_urls
```

Now you can download the 30 second previews of these tracks. Each preview is around 350k in size.

```bash
python train/download_previews.py --min_count=10
```

Use this script to find particular tracks (useful for model evaluation - see `config/track2vec.yaml`).

```bash
python train/search_tracks.py --search="James Brown"
```

Now you can train a track2vec model with
```bash
python train/track2vec.py
python train/test2vec.py
```

To train the mp3tovec model, you will first need to get the spectograms of the first 5 seconds of the previews.
```bash
python train/get_spectrograms.py
```

Then you can train the model with
```bash train/train_mp3tovec.py```
```

# TODO

* Train mp3tovec
* Run model on Spotify previews in parallel
* config for AudioEncoder
* Tensorboard
* Remove missing tracks
* Script to map from PyTorch to TensorFlow