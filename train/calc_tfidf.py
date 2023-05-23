import argparse
import os
import pickle

import numpy as np


def main():
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
        "--pool",
        type=bool,
        default=False,
        help="Just pool the vectors taking the average",
    )
    args = parser.parse_args()

    mp3tovecs = pickle.loads(open(args.mp3tovecs_file, "rb"))

    if args.pool:
        mp3tovecs = {k: np.mean(v, axis=0) for k, v in mp3tovecs.items()}
    else:
        assert False, "Not implemented yet"

    pickle.dump(mp3tovecs, open(args.mp3tovec_file, "wb"))
