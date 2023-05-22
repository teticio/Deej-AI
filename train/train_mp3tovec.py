import argparse
import os
import pickle

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from audiodiffusion.audio_encoder import AudioEncoder
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


class Mp3Dataset(Dataset):
    def __init__(self, dir, mel, track2vec):
        self.dir = dir
        self.files = os.listdir(dir)
        self.mel = mel
        self.ref = mel.n_fft // 2
        self.track2vec = track2vec

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mp3_file = self.files[idx]

        image = torch.rand((1, 216, 256))
        image = Image.open(
            os.path.join(self.dir, f"{mp3_file[: -len('.mp3')]}.png"), mode="L"
        )
        image = np.frombuffer(image.tobytes(), dtype="uint8").reshape(
            (1, image.height, image.width)
        )
        image = torch.from_numpy((image / 255) * 2 - 1).type(torch.float32)
        return image, self.track2vec[mp3_file[: -len(".mp3")]]


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


def create_dataloaders(directory, transform, track2vec, batch_size=32):
    dataset = Mp3Dataset(directory, transform, track2vec)
    train_dataset, val_dataset = train_val_split(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


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

    model = Mp3ToVecModel()
    trainer = pl.Trainer(**config["trainer"])
    trainer.fit(model, train_loader, val_loader)
