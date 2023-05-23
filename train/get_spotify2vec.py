import argparse
import concurrent.futures
import multiprocessing as mp
import os
import pickle
from time import sleep

import torch
from audiodiffusion.audio_encoder import AudioEncoder
from tqdm import tqdm


def encode_file(model, mp3_file, dir):
    return model.encode([os.path.join(dir, mp3_file)])[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_workers",
        type=int,
        default=os.cpu_count() if os.cpu_count() is not None else 1,
        help="Maximum number of cores to use",
    )
    parser.add_argument(
        "--spotifytovec_file",
        type=str,
        default="models/spotify2vec.p",
        help="SpotifytoVec file",
    )
    parser.add_argument(
        "--mp3tovec_model_file",
        type=str,
        default="models/mp3tovec.ckpt",
        help="MP3ToVec model file",
    )
    parser.add_argument(
        "--previews_dir",
        type=str,
        default="previews",
        help="Directory to save previews",
    )
    args = parser.parse_args()

    model = AudioEncoder()
    model.load_state_dict(
        {
            k.replace("model.", ""): v
            for k, v in torch.load(args.mp3tovec_model_file)["state_dict"].items()
        }
    )

    mp3tovec = (
        pickle.loads(open(args.spotifytovec_file, "rb"))
        if os.path.exists(args.spotifytovec_file)
        else {}
    )

    torch.multiprocessing.set_start_method("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = {
            executor.submit(encode_file, model, mp3_file, args.previews_dir): mp3_file
            for mp3_file in tqdm(os.listdir(args.previews_dir), desc="Setting up jobs")
            if f"{mp3_file[:-len('.mp3')]}" not in mp3tovec and sleep(1e-4) is None
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Encoding previews",
        ):
            mp3_file = futures[future]
            try:
                mp3tovec[mp3_file[: -len(".mp3")]] = future.result()
            except KeyboardInterrupt:
                break
            except Exception:
                print(f"Skipping {mp3_file}")


if __name__ == "__main__":
    main()
