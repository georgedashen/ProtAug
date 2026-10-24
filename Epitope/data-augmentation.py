import argparse
import random
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

example_usage = 'python data-augmentation.py --method PAM1 --k 2 --seed 42'
parser = argparse.ArgumentParser(description='data augmentation by simple replacement', epilog=example_usage)
parser.add_argument('--method', type=str, default='PAM1', help='which folder to load embeddings')
parser.add_argument('--k', type=float, default=2, help='control percentage of length to be changed')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--output', type=str, default='')
parser.add_argument('--test', action='store_true', default=False, help='use the first 10 sequences')
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

# Generate PAM matrix
list_amino = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
probabilities = np.array([[9867,2,9, 10,3,8, 17, 21,2,6,4,2,6,2, 22, 35, 32,0,2, 18],
[1,9913,1,0,1, 10,0,0, 10,3,1, 19,4,1,4,6,1,8,0,1],
[4,1, 9822, 36,0,4,6,6, 21,3,1, 13,0,1,2, 20,9,1,4,1],
[6,0, 42, 9859,0,6, 53,6,4,1,0,3,0,0,1,5,3,0,0,1],
[1,1,0,0, 9973,0,0,0,1,1,0,0,0,0,1,5,1,0,3,2],
[3,9,4,5,0, 9876, 27,1, 23,1,3,6,4,0,6,2,2,0,0,1],
[10,0,7, 56,0, 35, 9865,4,2,3,1,4,1,0,3,4,2,0,1,2],
[21,1, 12, 11,1,3,7, 9935,1,0,1,2,1,1,3, 21,3,0,0,5],
[1,8, 18,3,1, 20,1,0, 9912,0,1,1,0,2,3,1,1,1,4,1],
[2,2,3,1,2,1,2,0,0, 9872,9,2, 12,7,0,1,7,0,1, 33],
[3,1,3,0,0,6,1,1,4, 22, 9947,2, 45, 13,3,1,3,4,2, 15],
[2,37, 25,6,0, 12,7,2,2,4,1, 9926, 20,0,3,8, 11,0,1,1],
[1,1,0,0,0,2,0,0,0,5,8,4, 9874,1,0,1,2,0,0,4],
[1,1,1,0,0,0,0,1,2,8,6,0,4, 9946,0,2,1,3, 28,0],
[13,5,2,1,1,8,3,2,5,1,2,2,1,1, 9926, 12,4,0,0,2],
[28,11, 34,7, 11,4,6, 16,2,2,1,7,4,3, 17, 9840, 38,5,2,2],
[22,2, 13,4,1,3,2,2,1, 11,2,8,6,1,5, 32, 9871,0,2,9],
[0,2,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0, 9976,1,0],
[1,0,3,0,3,0,1,0,4,1,1,0,0, 21,0,1,1,2, 9945,1],
[13,2,1,1,3,2,2,3,3, 57, 11,1, 17,1,3,2, 10,0,2, 9901]]) / 10000

if args.method != 'PAM1':
    probabilities = np.load(f'{args.method}.npy') / 100

probabilities = probabilities/probabilities.sum(axis=0,keepdims=1)

t0 = time.time()

# Load data
df_train = pd.read_csv('data/train_split.csv')
if args.test:
    df_train = df_train[:10]

# Generate new data
K_VALUE = args.k

def generate_augmentation(df):
  sample_ids = range(len(df))
  df_final = {'parent_protein_id': [], 'protein_seq': []}
  
  for i in tqdm(sample_ids):
    prot = list(df.iloc[i, 1])

    swaps = int(len(prot) * K_VALUE)
    idx = np.random.choice(np.where(np.in1d(prot, list_amino))[0], size=swaps)

    for j in idx:
      prot[j] = np.random.choice(list_amino, p=probabilities[:, list_amino.index(prot[j])].tolist())

    df_final['parent_protein_id'].append(str(df.iloc[i, 0]))
    df_final['protein_seq'].append(''.join(prot))
  
  return pd.DataFrame(df_final)

generated_df = pd.DataFrame(columns=['parent_protein_id','protein_seq'])

generated_df = pd.concat([generated_df, generate_augmentation(df_train)], ignore_index=True)

print(f'Augmentation time: {time.time() - t0}, method: {args.method}', file=open('augmentation_time.txt', 'a'))
print(f'Augmentation time: {time.time() - t0}')

if len(args.output) > 0:
    generated_df.to_csv(f'data/aug_{args.output}.csv', index=False)
else:
    generated_df.to_csv(f'data/aug_{args.method}_k{args.k}.csv', index=False)
