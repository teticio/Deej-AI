import argparse
import concurrent.futures
import os
import pickle
from time import sleep

import numpy as np
from tqdm import tqdm


def calc_idf(mp3s, mp3_indices, close):
    vec_in_mp3 = np.zeros((close.shape[0], len(mp3s)))
    for i, mp3 in enumerate(mp3s):
        vec_in_mp3[:, i] = close[:, mp3_indices[mp3]].any(axis=1)
    idfs = -np.log((vec_in_mp3.sum(axis=1)) / len(mp3s))
    return idfs


def calc_tf_and_mp3tovec(mp3s, mp3_indices, close, idfs, mp3_vecs):
    mp3tovec = {}
    for mp3 in mp3s:
        tf = np.sum(close[mp3_indices[mp3], :][:, mp3_indices[mp3]], axis=1)
        mp3tovec[mp3] = np.sum(
            mp3_vecs[mp3_indices[mp3]]
            * tf[:, np.newaxis]
            * idfs[mp3_indices[mp3]][:, np.newaxis],
            axis=0,
        )
    return mp3tovec


def calc_mp3tovec(mp3s, mp3tovecs, epsilon, dims):
    mp3tovec = {}
    mp3_indices = {}
    mp3_vecs = np.empty((sum(len(mp3tovecs[mp3]) for mp3 in mp3s), dims))
    start = 0
    for mp3 in mp3s:
        end = start + len(mp3tovecs[mp3])
        mp3_vecs[start:end] = np.array(mp3tovecs[mp3])
        norms = np.linalg.norm(mp3_vecs[start:end], axis=1)
        mp3_vecs[start:end] /= norms[:, np.newaxis]
        mp3_indices[mp3] = list(range(start, end))
        start = end
    assert start == len(mp3_vecs)

    close = (
        1
        - np.einsum(
            "ij,kj->ik",
            mp3_vecs.astype(np.float16),
            mp3_vecs.astype(np.float16),
            dtype=np.float16,
        )
        < epsilon
    ).astype(bool)

    idfs = calc_idf(mp3s, mp3_indices, close)
    mp3tovec = calc_tf_and_mp3tovec(mp3s, mp3_indices, close, idfs, mp3_vecs)
    return mp3tovec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for TF-IDF calculation",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.001,
        help="Minimum cosine proximity for two vectors to be considered equal",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=os.cpu_count() if os.cpu_count() is not None else 1,
        help="Maximum number of cores to use",
    )
    parser.add_argument(
        "--mp3tovecs_file",
        type=str,
        default="models/mp3tovecs.p",
        help="Mp3ToVecs input file",
    )
    parser.add_argument(
        "--mp3tovec_file",
        type=str,
        default="models/mp3tovec.p",
        help="Mp3ToVec output file",
    )
    parser.add_argument(
        "--pool",
        type=bool,
        default=False,
        help="Just pool the vectors taking the average",
    )
    args = parser.parse_args()

    mp3tovecs = pickle.load(open(args.mp3tovecs_file, "rb"))
    dims = mp3tovecs[list(mp3tovecs.keys())[0]][0].shape[0]

    if args.pool:
        mp3tovec = {k: np.mean(v, axis=0) for k, v in mp3tovecs.items()}
    else:
        mp3tovec = {}
        keys = list(mp3tovecs.keys())

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.max_workers
        ) as executor:
            futures = {
                executor.submit(
                    calc_mp3tovec,
                    keys[i : i + args.batch_size],
                    mp3tovecs,
                    args.epsilon,
                    dims,
                ): i
                for i in tqdm(
                    range(0, len(mp3tovecs), args.batch_size), desc="Setting up jobs"
                )
                if sleep(1e-4) is None
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Calculating TF-IDF",
            ):
                futures[future]
                for k, v in future.result().items():
                    mp3tovec[k] = v

    pickle.dump(mp3tovec, open(args.mp3tovec_file, "wb"))


if __name__ == "__main__":
    main()
