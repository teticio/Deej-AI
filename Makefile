LIMIT_USERS ?= 1000000
LIMIT_PLAYLISTS ?= 1000000
MAX_WORKERS ?= 32
PROXY ?= proxy
DEEJAI_MODEL_DIR ?= ../deej-ai.online-dev/models

.PHONY: help
help: ## Show help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  \033[36m\033[0m\n"} /^[$$()% a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: all
all: setup users playlists tracks deduplicate search track2vec download spectrograms mp3tovec mp3tovecs tfidf tf install ## Run all steps

.PHONY: setup
setup: ## Setup environment for training
	@pip install -r train/requirements.txt
	@mkdir -p models
	@mkdir -p data
	@mkdir -p previews
	@mkdir -p spectrograms

.PHONY: users
users: ## Scrape user IDs from Spotify
	python train/get_users.py --cookie=$(COOKIE) --user=$(USER) --limit=$(LIMIT_USERS)

.PHONY: playlists
playlists: ## Get user playlist IDs from Spotify
	python train/get_playlists.py --limit=$(LIMIT_PLAYLISTS)

.PHONY: tracks
tracks: ## Get playlist track IDs from Spotify
	python train/get_tracks.py --max_workers=$(MAX_WORKERS) --proxy=$(PROXY)

.PHONY: deduplicate
deduplicate: ## Deduplicate tracks
	python train/deduplicate.py

.PHONY: search
search: ## Search for track IDs for testing in config/track2vec.yaml and config/mp3tovec.yaml
	bash -c 'trap "exit 0" SIGINT; python train/search_tracks.py'

.PHONY: track2vec
track2vec: ## Train Track2Vec model
	python train/train_track2vec.py --max_workers=$(MAX_WORKERS)

.PHONY: download
download: ## Download previews
	python train/download_previews.py --max_workers=$(MAX_WORKERS)

.PHONY: spectrograms
spectrogram: ## Generate spectrograms
	python train/calc_spectrograms.py --max_workers=$(MAX_WORKERS)

.PHONY: mp3tovec
mp3tovec: ## Train MP3ToVec model
	python train/train_mp3tovec.py

.PHONY: mp3tovecs
mp3tovecs: ## Calculate MP3ToVec embeddings for the previews
	python train/calc_mp3tovecs.py --max_workers=$(MAX_WORKERS)

.PHONY: tfidf
tfidf: ## Calculate MP3ToVec embedding per preview using TF-IDF
	python train/calc_tfidf.py --max_workers=$(MAX_WORKERS)

.PHONY: tf
tf: ## Convert PyTorch model to TensorFlow
	python train/pt_to_tf.py

.PHONY: install
install: ## Install model in deej-ai.online application
	python train/install_model.py --deejai_model_dir=$(DEEJAI_MODEL_DIR)
