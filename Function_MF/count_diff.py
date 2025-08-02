import argparse
import sys
import pandas as pd
import numpy as np

example_usage='python count_diff.py --d1 data/mf-training.csv --d2 data/mf-augmented_training_Random_k0.023.csv'
parser = argparse.ArgumentParser(description='iteratively predict masked amino acids from encoder pLM', epilog=example_usage)
parser.add_argument('--d1', type=str, help='first sequence file to be compared')
parser.add_argument('--d2', type=str, default=0.15, help='second sequence file to be compared')
parser.add_argument('--truc', type=str, default='', help='truncated length of sequences, default is empty')
parser.add_argument('--test', action='store_true', default=False, help='only load a small subset of data')
args = parser.parse_args()

seq1 = args.d1
seq2 = args.d2

q = pd.read_csv(seq1)
p = pd.read_csv(seq2)
q = q['sequences'].tolist()
p = p['sequences'].tolist()

if args.truc:
    args.truc = int(args.truc)
    q = q[:int(sys.argv[3])]
    p = p[:int(sys.argv[3])]

if args.test:
    q = q[:10]
    p = p[:10]

def count_diff(q,p):
    d = []
    for i in range(len(q)):
        d.append(sum(1 for a,b in zip(q[i],p[i]) if a!=b))
    return d

diff_stat = count_diff(q,p)

len_stat = []
for i in range(len(q)):
    len_stat.append(len(q[i]))

ratio_stat = np.array(diff_stat) / np.array(len_stat)

df = pd.DataFrame({'diff':np.array(diff_stat),'len':np.array(len_stat),'ratio':ratio_stat})

print(df.describe())

print(f'\nMean variation level: {df['ratio'].mean()}')
