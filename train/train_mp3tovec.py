import argparse
import os
import pickle

import gensim
import numpy as np
import lightning.pytorch as pl
import torch
import yaml
from audiodiffusion.audio_encoder import AudioEncoder
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from utils import read_tracks


def image_to_tensor(image_path):
    image = Image.open(image_path)
    image = np.frombuffer(image.tobytes(), dtype="uint8").reshape(
        (1, image.height, image.width)
    )
    image = torch.from_numpy((image / 255) * 2 - 1).type(torch.float32)
    return image


class Mp3Dataset(Dataset):
    def __init__(self, dir, track2vec):
        self.dir = dir
        self.files = os.listdir(dir)
        self.track2vec = track2vec

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mp3_file = self.files[idx]
        item = image_to_tensor(
            os.path.join(self.dir, f"{mp3_file[: -len('.mp3')]}.png")
        )
        return item, self.track2vec[mp3_file[: -len(".mp3")]]


class Mp3ToVecModel(pl.LightningModule):
    def __init__(self):
        super(Mp3ToVecModel, self).__init__()
        self.model = AudioEncoder()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = (1 - self.cosine_similarity(y_hat, y)).mean()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)


def train_val_split(dataset, train_frac=0.8):
    dataset_size = len(dataset)
    train_size = int(train_frac * dataset_size)
    val_size = dataset_size - train_size
    return random_split(dataset, [train_size, val_size])


def create_dataloaders(directory, track2vec, batch_size=32):
    dataset = Mp3Dataset(directory, track2vec)
    train_dataset, val_dataset = train_val_split(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class TestCallback(Callback):
    def __init__(self, tracks, track2vec_model, test_track_ids, test_batch):
        self.track2vec_model = track2vec_model
        self.tracks = tracks
        self.test_track_ids = test_track_ids
        self.test_batch = test_batch

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        with torch.no_grad():
            vecs = pl_module(self.test_batch.to(pl_module.device))
        pl_module.train()

        print()
        for i, track_id in enumerate(self.test_track_ids):
            print(
                f"\u001b]8;;{self.tracks[track_id]['url']}\u001b\\{self.tracks[track_id]['artist']} - {self.tracks[track_id]['title']}\u001b]8;;\u001b\\"
            )
            most_similar = self.track2vec_model.wv.similar_by_vector(
                np.array(vecs[i].cpu()), topn=8
            )
            for i, similar in enumerate(most_similar):
                print(
                    f"{i + 1}. \u001b]8;;{self.tracks[similar[0]]['url']}\u001b\\{self.tracks[similar[0]]['artist']} - {self.tracks[similar[0]]['title']}\u001b]8;;\u001b\\ ({similar[1]:.2f})"
                )
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dedup_tracks_file",
        type=str,
        default="data/tracks_dedup.csv",
        help="Deduplicated tracks CSV file",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/mp3tovec.yaml",
        help="Model configuation file",
    )
    parser.add_argument(
        "--spectrograms_dir",
        type=str,
        default="spectrograms",
        help="Spectrograms directory",
    )
    parser.add_argument(
        "--track2vec_model_file",
        type=str,
        default="models/track2vec",
        help="Track2Vec model file",
    )
    parser.add_argument(
        "--mp3tovec_model_file",
        type=str,
        default="models/track2vec",
        help="MP3toVec model save file",
    )
    parser.add_argument(
        "--tracks_file",
        type=str,
        default="data/tracks_dedup.csv",
        help="Tracks CSV file",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=min(5, os.cpu_count()),
        help="Maximum number of cores to use in DataLoader",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as stream:
        config = yaml.safe_load(stream)

    track2vec = pickle.load(open(f"{args.track2vec_model_file}.p", "rb"))

    train_loader, val_loader = create_dataloaders(
        args.spectrograms_dir, track2vec, batch_size=config["data"]["batch_size"]
    )

    track2vec_model = gensim.models.Word2Vec.load(args.track2vec_model_file)
    tracks = read_tracks(args.tracks_file)
    test_track_ids = config["data"]["test_track_ids"]
    test_batch = torch.stack(
        [
            image_to_tensor(os.path.join(args.spectrograms_dir, f"{test_track_id}.png"))
            for test_track_id in test_track_ids
        ]
    )
    test_callback = TestCallback(tracks, track2vec_model, test_track_ids, test_batch)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        dirpath=args.mp3tovec_model_file,
        filename="mp3tovec-{epoch:02d}-{val_loss:.2f}",
    )

    model = Mp3ToVecModel()
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, test_callback], **config["trainer"]
    )
    trainer.fit(model, train_loader, val_loader)
