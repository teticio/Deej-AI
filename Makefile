DATA_DIR ?= data
DEEJAI_MODEL_DIR ?= ../deej-ai.online-dev/model
LIMIT_PLAYLISTS ?= 1000000
LIMIT_USERS ?= 1000000
MAX_WORKERS ?= 32
MIN_COUNT ?= 10
MODELS_DIR ?= models
PREVIEWS_DIR ?= previews
PROXY ?= proxy
SPECTROGRAMS_DIR ?= spectrograms

.PHONY: help
help: ## Show help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  \033[36m\033[0m\n"} /^[$$()% a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: setup
setup: ## Setup environment for training
	@pip install -r train/requirements.txt

.PHONY: users
users: $(DATA_DIR)/users.csv ## Scrape user IDs from Spotify
$(DATA_DIR)/users.csv:
	@mkdir -p $(DATA_DIR)
	python train/get_users.py --cookie=$(COOKIE) --user=$(USER) --limit=$(LIMIT_USERS) --users_file=$(DATA_DIR)/users.csv

.PHONY: playlists
playlists: ## Get user playlist IDs from Spotify
$(DATA_DIR)/playlists.csv: $(DATA_DIR)/users.csv
	python train/get_playlists.py --limit=$(LIMIT_PLAYLISTS) --users_file=$(DATA_DIR)/users.csv --playlists_file=$(DATA_DIR)/playlists.csv

.PHONY: tracks
tracks: $(DATA_DIR)/tracks.csv ## Get playlist track IDs from Spotify
$(DATA_DIR)/tracks.csv $(DATA_DIR)/playlist_details.csv: $(DATA_DIR)/playlists.csv
	python train/get_tracks.py --max_workers=$(MAX_WORKERS) --proxy=$(PROXY) --playlists_file=$(DATA_DIR)/playlists.csv --playlist_details_file=$(DATA_DIR)/playlist_details.csv --tracks_file=$(DATA_DIR)/tracks.csv

.PHONY: deduplicate
deduplicate: $(DATA_DIR)/tracks_dedup.csv $(DATA_DIR)/playlists_dedup.csv ## Deduplicate tracks
$(DATA_DIR)/tracks_dedup.csv $(DATA_DIR)/playlists_dedup.csv: $(DATA_DIR)/playlist_details.csv $(DATA_DIR)/tracks.csv
	python train/deduplicate.py --dedup_tracks_file=$(DATA_DIR)/tracks_dedup.csv --dedup_playlists_file=$(DATA_DIR)/playlists_dedup.csv --min_count=$(MIN_COUNT) --tracks_file=$(DATA_DIR)/tracks.csv --playlists_file=$(DATA_DIR)/playlist_details.csv

.PHONY: search
search: $(DATA_DIR)/tracks_dedup.csv ## Search for track IDs to use for testing in config/track2vec.yaml and config/mp3tovec.yaml
	bash -c 'trap "exit 0" SIGINT; python train/search_tracks.py' --tracks_file=$(DATA_DIR)/tracks_dedup.csv

.PHONY: track2vec
track2vec: $(MODELS_DIR)/track2vec ## Train Track2Vec model
$(MODELS_DIR)/track2vec: $(DATA_DIR)/tracks_dedup.csv $(DATA_DIR)/playlists_dedup.csv
	@mkdir -p $(MODELS_DIR)
	python train/train_track2vec.py --max_workers=$(MAX_WORKERS) --tracks_file=$(DATA_DIR)/tracks_dedup.csv --playlists_file=$(DATA_DIR)/playlists_dedup.csv --track2vec_model_file=$(MODELS_DIR)/track2vec

.PHONY: download
download: $(PREVIEWS_DIR)/ ## Download 30 second previews
$(PREVIEWS_DIR)/: $(DATA_DIR)/tracks_dedup.csv 
	@mkdir -p $(PREVIEWS_DIR)
	@touch $(PREVIEWS_DIR)
	python train/download_previews.py --max_workers=$(MAX_WORKERS) --tracks_file=$(DATA_DIR)/tracks_dedup.csv --previews_dir=$(PREVIEWS_DIR)

.PHONY: spectrograms
spectrograms: $(SPECTROGRAMS_DIR)/ ## Generate spectrograms for first audio slice of each preview
$(SPECTROGRAMS_DIR)/: $(PREVIEWS_DIR)/
	@mkdir -p $(SPECTROGRAMS_DIR)
	@touch $(SPECTROGRAMS_DIR)
	python train/calc_spectrograms.py --max_workers=$(MAX_WORKERS) --previews_dir=$(PREVIEWS_DIR) --spectrograms_dir=$(SPECTROGRAMS_DIR)

.PHONY: mp3tovec
mp3tovec: $(MODELS_DIR)/mp3tovec.ckpt ## Train MP3ToVec model
$(MODELS_DIR)/mp3tovec.ckpt: $(SPECTROGRAMS_DIR)/ $(MODELS_DIR)/track2vec $(DATA_DIR)/tracks_dedup.csv
	python train/train_mp3tovec.py --spectrograms_dir=$(SPECTROGRAMS_DIR) --track2vec_model_file=$(MODELS_DIR)/track2vec --tracks_file=$(DATA_DIR)/tracks_dedup.csv --mp3tovec_model_dir=$(MODELS_DIR)

.PHONY: mp3tovecs
mp3tovecs: $(MODELS_DIR)/mp3tovecs.p ## Calculate MP3ToVec embeddings for the previews
$(MODELS_DIR)/mp3tovecs.p: $(PREVIEWS_DIR)/ $(MODELS_DIR)/mp3tovec.ckpt
	python train/calc_mp3tovecs.py --max_workers=$(MAX_WORKERS) --mp3tovec_model_file=$(MODELS_DIR)/mp3tovec.ckpt --mp3tovecs_file=$(MODELS_DIR)/mp3tovecs.p --mp3s_dir=$(PREVIEWS_DIR)

.PHONY: tfidf
tfidf: $(MODELS_DIR)/mp3tovec.p ## Calculate MP3ToVec embedding per preview using TF-IDF
$(MODELS_DIR)/mp3tovec.p: $(MODELS_DIR)/mp3tovecs.p
	python train/calc_tfidf.py --max_workers=$(MAX_WORKERS) --mp3tovecs_file=$(MODELS_DIR)/mp3tovecs.p --mp3tovec_file=$(MODELS_DIR)/mp3tovec.p

.PHONY: tf
tf: $(MODELS_DIR)/speccy_model ## Convert PyTorch model to TensorFlow
$(MODELS_DIR)/speccy_model: $(MODELS_DIR)/mp3tovec.ckpt
	python train/pt_to_tf.py --pt_model_file=$(MODELS_DIR)/mp3tovec.ckpt --tf_model_file=$(MODELS_DIR)/speccy_model

.PHONY: install ## Install model in deej-ai.online app
install: $(MODELS_DIR)/mp3tovec.ckpt $(MODELS_DIR)/mp3tovec.p $(MODELS_DIR)/speccy_model $(DATA_DIR)/tracks_dedup.csv $(MODELS_DIR)/track2vec.p ## Install model in deej-ai.online application
	python train/install_model.py --deejai_model_dir=$(DEEJAI_MODEL_DIR) --mp3tovec_model_file=$(MODELS_DIR)/speccy_model --mp3tovec_file=$(MODELS_DIR)/mp3tovec.p --tracks_file=$(DATA_DIR)/tracks_dedup.csv --track2vec_file=$(MODELS_DIR)/track2vec.p
