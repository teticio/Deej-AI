import argparse
import os
import pickle
from typing import Dict, List, Tuple

import gensim
import lightning.pytorch as pl
import numpy as np
import torch
import yaml
from audiodiffusion.audio_encoder import AudioEncoder
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from utils import read_tracks


def image_to_tensor(image_path: str) -> torch.Tensor:
    """
    Converts an image to a PyTorch tensor.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: A PyTorch tensor of shape (1, 96, 216) representing the image.
    """
    image = Image.open(image_path)
    image = np.frombuffer(image.tobytes(), dtype="uint8").reshape(
        (1, image.height, image.width)
    )
    image = torch.from_numpy((image / 255) * 2 - 1).type(torch.float32)
    return image


class Mp3Dataset(Dataset):
    """
    A PyTorch Dataset for loading MP3 spectrograms and taget Track2Vec vectors.

    Args:
        dir (str): Path to the directory containing the MP3 files.
        track2vec (dict): Dictionary mapping MP3 filenames to TrackToVec vectors.
    """

    def __init__(self, dir: str, track2vec: dict) -> None:
        self.dir = dir
        self.files: List = os.listdir(dir)
        self.track2vec = track2vec

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the MP3 spectrogram and the target Track2Vec vector corresponding to an index.

        Args:
            idx (int): Index of the MP3 file.

        Returns:
            torch.Tensor: A PyTorch tensor of shape (1, 96, 216) representing the MP3 spectrogram images.
            torch.Tensor: A PyTorch tensor of shape (1, 100) representing the target Track2Vec vectors.
        """
        mp3_file = self.files[idx]
        item = image_to_tensor(
            os.path.join(self.dir, f"{mp3_file[: -len('.mp3')]}.png")
        )
        return item, self.track2vec[mp3_file[: -len(".mp3")]]


class Mp3ToVecModel(pl.LightningModule):
    """
    A PyTorch Lightning model for training an MP3ToVec model.
    """

    def __init__(self) -> None:
        super(Mp3ToVecModel, self).__init__()
        self.model = AudioEncoder()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = (1 - self.cosine_similarity(y_hat, y)).mean()
        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self.step(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=0.001)


def train_val_split(
    dataset: Mp3Dataset, train_frac: float = 0.8
) -> List[Subset[Mp3Dataset]]:
    """
    Splits a dataset into a training and validation set.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to split.
        train_frac (float): The fraction of the dataset to use for training.

    Returns:
        torch.utils.data.Dataset: The training dataset.
        torch.utils.data.Dataset: The validation dataset.
    """
    dataset_size = len(dataset)
    train_size = int(train_frac * dataset_size)
    val_size = dataset_size - train_size
    return random_split(dataset, [train_size, val_size])


def create_dataloaders(
    directory: str, track2vec: dict, batch_size: int = 32, num_workers: int = 1
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates dataloaders for training and validation.

    Args:
        directory (str): Path to the directory containing the MP3 files.
        track2vec (dict): Dictionary mapping MP3 filenames to TrackToVec vectors.

    Returns:
        torch.utils.data.DataLoader: The training dataloader.
        torch.utils.data.DataLoader: The validation dataloader.
    """
    dataset = Mp3Dataset(directory, track2vec)
    train_dataset, val_dataset = train_val_split(dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


class TestCallback(Callback):
    """
    Callback for printing the most similar tracks to each test track after each epoch.

    Args:
        tracks (dict): Dictionary mapping track IDs to track metadata.
        track2vec_model (torch.nn.Module): The trained Track2Vec model.
        test_track_ids (list): List of track IDs to use for testing.
        test_batch (torch.Tensor): A batch of spectrograms corresponding to the test tracks.
    """

    def __init__(
        self,
        tracks: Dict,
        track2vec_model: gensim.models.Word2Vec,
        test_track_ids: list,
        test_batch: torch.Tensor,
    ) -> None:
        self.track2vec_model = track2vec_model
        self.tracks = tracks
        self.test_track_ids = test_track_ids
        self.test_batch = test_batch

    @rank_zero_only
    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Prints the most similar tracks to each test track after each epoch.

        Args:
            trainer (pytorch_lightning.Trainer): The PyTorch Lightning trainer.
            pl_module (pytorch_lightning.LightningModule): The PyTorch Lightning module.
        """
        pl_module.eval()
        with torch.no_grad():
            vecs = pl_module(self.test_batch.to(pl_module.device))
        pl_module.train()

        print()
        for i, track_id in enumerate(self.test_track_ids):
            print(
                f"\u001b]8;;{self.tracks[track_id]['url']}\u001b\\{self.tracks[track_id]['artist']} - {self.tracks[track_id]['title']}\u001b]8;;\u001b\\"
            )  # type: ignore
            most_similar = self.track2vec_model.wv.similar_by_vector(
                np.array(vecs[i].cpu()), topn=8
            )
            for i, similar in enumerate(most_similar):
                print(
                    f"{i + 1}. \u001b]8;;{self.tracks[similar[0]]['url']}\u001b\\{self.tracks[similar[0]]['artist']} - {self.tracks[similar[0]]['title']}\u001b]8;;\u001b\\ ({similar[1]:.2f})"
                )
            print()


if __name__ == "__main__":
    """
    Entry point for the train_mp3tovec script.

    Trains the MP3ToVec model.

    Args:
        --config_file (str): Model configuation file. Defaults to config/mp3tovec.yaml.
        --mp3tovec_model_dir (str): MP3ToVec model save directory. Defaults to models.
        --spectrograms_dir (str): Spectrograms directory. Defaults to spectrograms.
        --track2vec_model_file (str): Track2Vec model file. Defaults to models/track2vec.
        --tracks_file (str): Track metadata file. Defaults to data/tracks_dedup.json.

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/mp3tovec.yaml",
        help="Model configuation file",
    )
    parser.add_argument(
        "--mp3tovec_model_dir",
        type=str,
        default="models",
        help="MP3ToVec model save directory",
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
        "--tracks_file",
        type=str,
        default="data/tracks_dedup.csv",
        help="Tracks CSV file",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as stream:
        config = yaml.safe_load(stream)

    track2vec = pickle.load(open(f"{args.track2vec_model_file}.p", "rb"))

    train_loader, val_loader = create_dataloaders(
        args.spectrograms_dir,
        track2vec,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
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
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=args.mp3tovec_model_dir,
        filename="mp3tovec",
    )

    model = Mp3ToVecModel()
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, test_callback], **config["trainer"]
    )
    trainer.fit(model, train_loader, val_loader)
