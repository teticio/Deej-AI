LIMIT_USERS ?= 1000000
LIMIT_PLAYLISTS ?= 1000000
MAX_WORKERS ?= 32
PROXY ?= proxy
DEEJAI_MODEL_DIR ?= ../deej-ai.online-dev/models

.PHONY: help
help: ## show help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  \033[36m\033[0m\n"} /^[$$()% a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: all
all: setup users playlists tracks deduplicate search track2vec download spectrogram mp3tovec mp3tovecs tfidf install ## run all steps

.PHONY: setup
setup: ## setup environment for training
	@pip install -r train/requirements.txt
	@mkdir -p models
	@mkdir -p data
	@mkdir -p previews
	@mkdir -p spectrograms

.PHONY: users
users: ## scrape user IDs from Spotify
	python train/get_users.py --cookie=$(COOKIE) --user=$(USER) --limit=$(LIMIT_USERS)

.PHONY: playlists
playlists: ## get user playlist IDs from Spotify
	python train/get_playlists.py --limit=$(LIMIT_PLAYLISTS)

.PHONY: tracks
tracks: ## get playlist track IDs from Spotify
	python train/get_tracks.py --max_workers=$(MAX_WORKERS) --proxy=$(PROXY)

.PHONY: deduplicate
deduplicate: ## deduplicate tracks
	python train/deduplicate.py

.PHONY: search
search: ## search for track IDs for testing in config/track2vec.yaml and config/mp3tovec.yaml
	bash -c 'trap "exit 0" SIGINT; python train/search_tracks.py'

.PHONY: track2vec
track2vec: ## train Track2Vec model
	python train/train_track2vec.py --max_workers=$(MAX_WORKERS)

.PHONY: download
download: ## download previews
	python train/download_previews.py --max_workers=$(MAX_WORKERS)

.PHONY: spectrogram
spectrogram: ## generate spectrograms
	python train/calc_spectrograms.py --max_workers=$(MAX_WORKERS)

.PHONY: mp3tovec
mp3tovec: ## train MP3ToVec model
	python train/train_mp3tovec.py

.PHONY: mp3tovecs
mp3tovecs: ## calculate MP3ToVec embeddings for the previews
	python train/train_mp3tovecs.py --max_workers=$(MAX_WORKERS)

.PHONY: tfidf
tfidf: ## calculate MP3ToVec embedding per preview using TF-IDF
	python train/train_tfidf.py --max_workers=$(MAX_WORKERS)

.PHONY: install
install: ## install model in deej-ai.online application
	python train/install_model.py --deejai_model_dir=$(DEEJAI_MODEL_DIR)
