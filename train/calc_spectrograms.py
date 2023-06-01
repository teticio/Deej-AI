import argparse
import concurrent.futures
import os
from time import sleep

import yaml
from audiodiffusion.mel import Mel
from tqdm import tqdm


def calc_spectrogram(mp3_file: str, previews_dir: str, spectrograms_dir: str, mel: Mel):
    """
    Calculates the spectrogram of the first slice of an MP3 file and saves it to disk.

    Args:
        mp3_file (str): Filename of MP3.
        previews_dir (str): Directory of MP3.
        spectrograms_dir (str): Path to the directory where spectrogram images will be saved.
        mel (Mel): Mel object.
    """
    mel.load_audio(os.path.join(previews_dir, mp3_file))
    image = mel.audio_slice_to_image(slice=0, ref=float(mel.n_fft // 2))
    image.save(os.path.join(spectrograms_dir, mp3_file[: -len(".mp3")] + ".png"))


def main():
    """
    Main function for the calc_spectrograms script.

    Calculates spectrograms for a directory of MP3 files.

    Args:
        --config_file (str): Path to the model configuration file. Default is "config/mp3tovec.yaml".
        --max_workers (int): Maximum number of cores to use. Default is the number of available CPU cores on the system.
        --previews_dir (str): Path to the directory where preview MP3s will be read from. Default is "previews".
        --spectrograms_dir (str): Path to the directory where spectrogram images will be saved. Default is "spectrograms".
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/mp3tovec.yaml",
        help="Model configuation file",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=os.cpu_count() if os.cpu_count() is not None else 1,
        help="Maximum number of cores to use",
    )
    parser.add_argument(
        "--previews_dir",
        type=str,
        default="previews",
        help="Previews directory",
    )
    parser.add_argument(
        "--spectrograms_dir",
        type=str,
        default="spectrograms",
        help="Directory to save spectrograms",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as stream:
        config = yaml.safe_load(stream)

    mel = Mel(**config["mel"])

    mp3_files = os.listdir(args.previews_dir)
    os.makedirs(args.spectrograms_dir, exist_ok=True)
    already_done = set(os.listdir(args.spectrograms_dir))

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = {
            executor.submit(
                calc_spectrogram,
                mp3_file,
                args.previews_dir,
                args.spectrograms_dir,
                mel,
            ): mp3_file
            for mp3_file in tqdm(mp3_files, desc="Setting up jobs")
            if mp3_file.endswith(".mp3")
            and f"{mp3_file[: -len('.mp3')]}.png" not in already_done
            and sleep(1e-4) is None
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Calculating spectrograms",
        ):
            mp3_file = futures[future]
            try:
                future.result()
            except KeyboardInterrupt:
                break
            except Exception:
                print(f"Skipping {mp3_file}")


if __name__ == "__main__":
    main()
