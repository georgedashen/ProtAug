import pandas as pd
import numpy as np
import argparse

if __name__ == "__main__":
    example_usage='python genFasta.py --input data/mf-training.csv --output data/mf-training.fasta'
    parser = argparse.ArgumentParser(description='Convert sequences to FASTA format', epilog=example_usage)
    parser.add_argument('--input', type=str, help='input sequence csv file')
    parser.add_argument('--output', type=str, default='out.fasta')
    args = parser.parse_args()

    outfile = f'data/{args.output}'

    df = pd.read_csv(args.input)

    with open(outfile, 'w') as f:
        for i in range(len(df)):
            if df['proteins'][i] == 0:
                f.write(f'>seq_{i}\n')
            else:
                f.write(f">{df['proteins'][i]}\n")

            f.write(f"{df['sequences'][i]}\n")

