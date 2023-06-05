import argparse
import concurrent.futures
import os
import pickle
from time import sleep
from typing import Dict, List

import numpy as np
from tqdm import tqdm


def calc_idf(
    mp3s: List[str], mp3_indices: Dict[str, List], close: np.ndarray
) -> np.ndarray:
    """
    Calculates the inverse document frequency (IDF) for a list of MP3 files.

    Args:
        mp3s (list[str]): A list of MP3 IDs.
        mp3_indices (dict[str, List]): A map of MP3 IDs to a list of indices in the list of vectors.
        close (np.ndarray): close[i, j] = True if cosine proximity of the ith and jth vectors is less than epsilon.

    Returns:
        np.ndarray: IDF[i] = 1 / log of number of MP3s in which ith vector appears at least once.
    """
    vec_in_mp3 = np.zeros((close.shape[0], len(mp3s)))
    for i, mp3 in enumerate(mp3s):
        vec_in_mp3[:, i] = close[:, mp3_indices[mp3]].any(axis=1)
    idfs = -np.log((vec_in_mp3.sum(axis=1)) / len(mp3s))
    return idfs


def calc_tf_and_mp3tovec(
    mp3s: List[str],
    mp3_indices: Dict[str, List],
    close: np.ndarray,
    idfs: np.ndarray,
    mp3_vecs: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Calculates the term frequency (TF) and TF-IDF vectors for a list of MP3 files.

    Args:
        mp3s (List[str]): A list of MP3 IDs.
        mp3_indices (dict[str, List]): A map of MP3 IDs to a list of indices in the list of vectors.
        close (np.ndarray): close[i, j] = True if cosine proximity of the ith and jth vectors is less than epsilon.
        idfs (np.ndarray): 1 / log of number of MP3s in which ith vector appears at least once.
        mp3_vecs (np.ndarray): A list of all MP3 vectors.

    Returns:
        Dict[str, np.ndarray]: A map of MP3 IDs to a single MP3ToVec vector using TF-IDF.
    """
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


def calc_mp3tovec(
    mp3s: List[str], mp3tovecs: Dict[str, np.ndarray], epsilon: float, dims: int
) -> Dict[str, np.ndarray]:
    """
    Calculates the feature vectors for a list of MP3 files.

    Args:
        mp3s (List[str]): A list of MP3 IDs.
        mp3tovecs (Dict[str, np.ndarray]): A map of MP3 IDs to an array of MP3ToVec vectors.
        epsilon (float): A small value used to determine whether two vectors are the close in cosine proximity.
        dims (int): The number of dimensions of the MP3ToVec vectors.

    Returns:
        Dict[str, np.ndarray]: A map of MP3 IDs to a single MP3ToVec vector using TF-IDF.
    """
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


def main() -> None:
    """
    Main function for the calc_tfidf script.

    Calculates single MP3ToVec vector from a list of MP3ToVec vectors per MP3 using TF-IDF.

    Args:
        --batch_size (int): Batch size for TF-IDF calculation. Default is 100.
        --epsilon (float): Minimum cosine proximity for two vectors to be considered equal. Default is 0.001.
        --max_workers (int): Maximum number of cores to use. Default is the number of cores on the machine.

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
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
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save MP3ToVec every N batches",
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
            for i, future in enumerate(
                tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Calculating TF-IDF",
                )
            ):
                futures[future]
                for k, v in future.result().items():
                    mp3tovec[k] = v
                if (i + 1) % args.save_every == 0:
                    pickle.dump(mp3tovec, open(args.mp3tovec_file, "wb"))

    pickle.dump(mp3tovec, open(args.mp3tovec_file, "wb"))


if __name__ == "__main__":
    main()
