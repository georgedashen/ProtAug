#!/usr/bin/env python3
"""Sequence-quality metrics from ProtAug.

variation   normalized Hamming distance between an original CSV and an augmented CSV
repetition  fraction of length covered by runs of three identical amino acids
shannon     mean Shannon diversity of amino-acid composition
"""

import argparse
import sys
from collections import Counter
from math import log

import numpy as np
import pandas as pd

SEQ_COLUMNS = (
    "sequences",
    "protein_sequence",
    "protein_seq",
    "sequence",
)


def sequence_column(df):
    for name in SEQ_COLUMNS:
        if name in df.columns:
            return name
    return df.columns[1]


def load_sequences(path, limit=None):
    df = pd.read_csv(path)
    seqs = df[sequence_column(df)].astype(str).tolist()
    if limit is not None:
        seqs = seqs[:limit]
    return seqs


def variation(original, augmented):
    distances = []
    lengths = []
    for left, right in zip(original, augmented):
        n = min(len(left), len(right))
        if n == 0:
            continue
        distances.append(sum(a != b for a, b in zip(left[:n], right[:n])))
        lengths.append(n)
    if not lengths:
        return float("nan")
    return float(np.sum(distances) / np.sum(lengths))


def repetition_ratio(seq):
    if len(seq) == 0:
        return 0.0
    # Number of AAA-style triplets, divided by sequence length (Appendix C).
    triplets = sum(
        seq[i] == seq[i - 1] == seq[i - 2]
        for i in range(2, len(seq))
    )
    return triplets / len(seq)


def shannon(seq):
    if len(seq) == 0:
        return 0.0
    counts = Counter(seq)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        prop = count / total
        entropy -= prop * log(prop)
    return entropy


def main():
    parser = argparse.ArgumentParser(description="ProtAug sequence-quality metrics")
    sub = parser.add_subparsers(dest="command", required=True)

    vary = sub.add_parser("variation", help="normalized Hamming distance")
    vary.add_argument("--d1", required=True, help="original csv")
    vary.add_argument("--d2", required=True, help="augmented csv")
    vary.add_argument("--test", action="store_true", help="score the first 10 sequences")

    rep = sub.add_parser("repetition", help="mean amino-acid triplet repetition ratio")
    rep.add_argument("csv")
    rep.add_argument("--test", action="store_true")

    shan = sub.add_parser("shannon", help="mean Shannon diversity")
    shan.add_argument("csv")
    shan.add_argument("--test", action="store_true")

    args = parser.parse_args()
    limit = 10 if args.test else None

    if args.command == "variation":
        left = load_sequences(args.d1, limit)
        right = load_sequences(args.d2, limit)
        if len(left) != len(right):
            n = min(len(left), len(right))
            print(f"warning: {len(left)} vs {len(right)} sequences; comparing {n}", file=sys.stderr)
            left, right = left[:n], right[:n]
        print(f"{variation(left, right):.6f}")
        return

    seqs = load_sequences(args.csv, limit)
    if args.command == "repetition":
        print(f"{np.mean([repetition_ratio(seq) for seq in seqs]):.6f}")
    else:
        print(f"{np.mean([shannon(seq) for seq in seqs]):.6f}")


if __name__ == "__main__":
    main()
