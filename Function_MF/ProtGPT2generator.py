# Step 1: Import libraries
import argparse
import os
import numpy as np
import pandas as pd
import torch
import random
import time
from tqdm import tqdm
from transformers import pipeline

import warnings
warnings.filterwarnings("ignore")

example_usage='python ProtGPT2generator.py --k 0.10'
example_usage='python ProtGPT2generator.py --rho 0.10'
example_usage='python ProtGPT2generator.py --k 0.10 --num_chunk 5 --chunk_id 0'
parser = argparse.ArgumentParser(description='iteratively predict masked amino acids from decoder pLM', epilog=example_usage)
parser.add_argument('--model', type=str, default='ProtGPT2')
parser.add_argument('--k', type=float, default=0.15, help='variation fraction control')
parser.add_argument('--K_MAX', type=float, default=0.8)
parser.add_argument('--rho', type=str, default='', help='variation level')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch', type=int, default=128) #could be larger, recommend 128,8,16,32,64
parser.add_argument('--test', action='store_true', default=False, help='only load a small subset of data')
parser.add_argument('--test_id', type=int, default=-1, help='which seq to test')
parser.add_argument('--num_chunk', type=int, default=1, help='whether to partition the dataset')
parser.add_argument('--chunk_id', type=int, default=0, help='out of 5 chunks, which one to process')
parser.add_argument('--output', type=str, default='')
args = parser.parse_args()

if not args.rho:
    args.method = f'ProtGPT2_k{args.k}'
else:
    args.rho = float(args.rho)
    args.method = f'ProtGPT2_rho{args.rho}'

np.random.seed(args.seed)
random.seed(args.seed)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["PYTHONHASHSEED"] = str(args.seed)

t0 = time.time()
df_train = pd.read_csv('data/mf-training.csv') # mf-training_2.csv replace ProtGPT2-unrecognized letters (e.g. X, B, O) in the original mf-training.csv with L (Leucine) to avoid errors
if args.test:
    df_train = df_train[:10]
    if args.test_id>0:
        df_train = df_train[args.test_id:args.test_id+1]
        args.num_chunk = 1

if args.num_chunk > 1:
    if len(df_train) % args.num_chunk == 0:
        size = int(len(df_train) / args.num_chunk)
    else:
        size = int(len(df_train) / args.num_chunk) + 1
    df_chunks = [df_train.iloc[i:i+size,:] for i in range(0, len(df_train), size)]
    df_train = df_chunks[args.chunk_id]

Nseq = len(df_train)

# Step 2: Load the ProtGPT2 model and tokenizer
if args.model == 'ProtGPT2':
    protgpt2 = pipeline('text-generation', model="nferruz/ProtGPT2", device=0, batch_size=1)

SLICE_LEN = 1000

aa_list = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']

for i_seq in tqdm(range(len(df_train))):
    seq = df_train.iloc[i_seq, 1]
    tseq = seq

    # Step 5: Mask k% of the words in the sentence
    if args.rho:
        num_round = 10000
        k = args.K_MAX if args.rho > args.K_MAX else args.rho
    elif args.k>args.K_MAX:
        num_round = int(args.k // args.K_MAX) + 1
        prop_to_mask_last = args.k - args.K_MAX*(num_round-1)
        k = args.K_MAX
    else:
        num_round = 1
        prop_to_mask_last = 0
        k = args.k
    
    chunks = [tseq[i:i+SLICE_LEN] for i in range(0, len(tseq), SLICE_LEN)]
    #if len(tokens) < SLICE_LEN:
    #    num_to_mask = int(len(tokens) * args.k)
    #num_slice = num_to_mask // len(tokens) + 1
    #mask_indices = np.random.choice(len(tokens), num_to_mask, replace=False)

    for ichunk,chunk in enumerate(chunks):
        original_chunk = chunk
        original_length = len(chunk)
        for iround in range(num_round):
            if iround == num_round-1 and prop_to_mask_last:
                num_to_mask = int(len(chunk) * prop_to_mask_last)
                if prop_to_mask_last < 0.1:
                    mask_chunk_num = 3
                elif prop_to_mask_last < 0.4:
                    mask_chunk_num = 5
                else:
                    mask_chunk_num = 10
            else:
                num_to_mask = int(len(chunk) * k)
                if k < 0.1:
                    mask_chunk_num = 3
                elif k < 0.4:
                    mask_chunk_num = 5
                else:
                    mask_chunk_num = 10

            chunk_size = len(chunk) // mask_chunk_num + 1
            mask_chunk_size = num_to_mask // mask_chunk_num
            mask_chunk_size_last = num_to_mask - mask_chunk_num * mask_chunk_size
            mask_indices_t = np.array([],dtype=np.int32)
            mask_indices = np.array([],dtype=np.int32)
            for offset in range(0, len(chunk), chunk_size):
                mask_indices_t = np.append(mask_indices_t, np.random.choice(chunk_size, 1)+offset)
            for idx in mask_indices_t:
                for j in range(idx,idx+mask_chunk_size):
                    if j not in mask_indices and j < len(chunk):
                        mask_indices = np.append(mask_indices, j)
                if idx == mask_indices_t[0]:
                    for j in range(idx+mask_chunk_size, idx+mask_chunk_size+mask_chunk_size_last):
                        if j not in mask_indices and j < len(chunk):
                            mask_indices = np.append(mask_indices, j)
            mask_indices = mask_indices[mask_indices != 0]
            chunk = list(chunk)
            for idx in mask_indices:
                chunk[idx] = '!'
            chunk = ''.join(chunk)
            slices = []
            start_index = 0
            delim_count = 0
            delimiter = '!'
            for i in range(len(chunk)):
                if chunk[i:i+len(delimiter)] == delimiter:
                    delim_count += 1
                    end_index = i+len(delimiter)
                    if i == len(chunk) - 1:
                        slices.append((chunk[start_index:end_index-delim_count], delim_count))
                else:
                    if delim_count > 0:
                        slices.append((chunk[start_index:end_index-delim_count], delim_count))
                        start_index = end_index
                        delim_count = 0
                    if i == len(chunk) - 1:
                        slices.append((chunk[start_index:], delim_count))

    # Step 8: Convert input IDs to TensorFlow tensors
            acc_seq = ''
            acc_seq_t = ''
            for i, (s, l) in enumerate(slices):
                bad_count = -1
                if l > 0:
                    while(len(acc_seq_t) != len(acc_seq+s)+l): #at most 3 return sequences
                        if bad_count == 2:
                            max_idx = np.argmax([len(re['generated_text']) for re in result])
                            acc_seq_t = result[max_idx]['generated_text'][:len(acc_seq+s)+l]
                            acc_seq_t = acc_seq_t + ''.join(np.random.choice(aa_list, len(acc_seq+s)+l-len(acc_seq_t)).tolist()) #random filled
                            break

                        bad_count += 1
                        result = protgpt2(acc_seq+s, max_length=len(acc_seq+s)+l, max_new_tokens=None, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=1+bad_count, pad_token_id=0, eos_token_id=0)
                        acc_seq_t = result[-1]['generated_text'][:len(acc_seq+s)+l]

                    acc_seq = acc_seq_t
                else:
                    acc_seq = acc_seq + s

            chunk = acc_seq
            chunk = chunk.replace('\n', 'L')

            if args.rho:
                delta = sum(1 for x,y in zip(chunk,original_chunk) if x!=y)
                if args.test:
                    print(f'variation for current chunk {iround}iter: {delta/original_length}, {original_length}aa') #test
                if delta > args.rho * original_length:
                    break

        chunks[ichunk] = chunk

    newseq = ''.join(token for chunk in chunks for token in chunk)
    df_train.iloc[i_seq, 1] = newseq

print(f'Augmentation time: {time.time() - t0}, method: {args.method}, Nchunk: {args.num_chunk}, chunk_id: {args.chunk_id}, Nseq:{Nseq}, seed: {args.seed}', file=open('augmentation_time.txt', 'a'))
print(f'Augmentation time: {time.time() - t0}')

if args.num_chunk > 1:
    flag = f'_chunk{args.chunk_id}'
else:
    flag = ''

if args.output:
    df_train.to_csv(f'data/{args.output}', index=False)
else:
    df_train.to_csv(f'data/mf-augmented_training_{args.method}{flag}.csv', index=False)
