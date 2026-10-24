# Step 1: Import libraries
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
from protaug_paths import resolve_model

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import time
from tqdm import tqdm
from transformers import AutoTokenizer, TFAutoModelForMaskedLM

exec(open('utils.py').read())

example_usage='python MLMgenerator.py --model Esm2 --k 0.55'
example_usage='python MLMgenerator.py --model Esm2 --rho 0.55'
example_usage='python MLMgenerator.py --model Esm2 --k 0.55 --num_chunk 5 --chunk_id 0'
parser = argparse.ArgumentParser(description='iteratively predict masked amino acids from encoder pLM', epilog=example_usage)
parser.add_argument('--model', type=str, default='prot_bert_bfd', choices=['DistilProtBert','prot_bert_bfd','Esm2'])
parser.add_argument('--k', type=float, default=0.15, help='fraction control factor, enable when rho is disable')
parser.add_argument('--K_MAX', type=float, default=0.15)
parser.add_argument('--rho', type=str, default='', help='variation level')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch', type=int, default=128) #could be larger, recommend 128,8,16,32,64
parser.add_argument('--test', action='store_true', default=False, help='only load a small subset of data')
parser.add_argument('--test_id', type=int, default=-1, help='which seq to test')
parser.add_argument('--num_chunk', type=int, default=1, help='number larger than 1 to partition the dataset')
parser.add_argument('--chunk_id', type=int, default=0, help='which chunk to process, allow multiple processes')
parser.add_argument('--output', type=str, default='', help='predefined output file')
args = parser.parse_args()

if not args.rho:
    args.method = f'{args.model}_MLM_k{args.k}'
else:
    args.rho = float(args.rho)
    args.method = f'{args.model}_MLM_rho{args.rho}'

np.random.seed(args.seed)
tf.random.set_seed(args.seed)
random.seed(args.seed)
tf.experimental.numpy.random.seed(args.seed)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["PYTHONHASHSEED"] = str(args.seed)

t0 = time.time()
df_train = pd.read_csv('data/mf-training.csv')
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

# Step 2: Load the BERT model and tokenizer
model_name = resolve_model(args.model)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForMaskedLM.from_pretrained(model_name)
if tokenizer.sep_token == None: #Esm2
    tokenizer.sep_token = '<eos>'
if args.model == 'Esm2':
    SLICE_LEN = 1000
elif args.model == 'prot_bert_bfd' or args.model == 'DistilProtBert':
    SLICE_LEN = 500

for i_seq in tqdm(range(len(df_train))):
    seq = df_train.iloc[i_seq, 1]
    tseq = seq
    #if len(seq) > 1024:
    #    start_idx = np.random.choice(range(len(seq)-1023))
    #    end_idx = start_idx + 1024
    #    tseq = tseq[start_idx:end_idx]
    tseq = ' '.join(tseq)
    tokens = tokenizer.tokenize(tseq)

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
    
    chunks = [tokens[i:i+SLICE_LEN] for i in range(0, len(tokens), SLICE_LEN)]
    #if len(tokens) < SLICE_LEN:
    #    num_to_mask = int(len(tokens) * args.k)
    #num_slice = num_to_mask // len(tokens) + 1
    #mask_indices = np.random.choice(len(tokens), num_to_mask, replace=False)

    for ichunk,chunk in enumerate(chunks):
        original_chunk = np.array(chunk)
        for iround in range(num_round):
            if iround == num_round-1 and prop_to_mask_last:
                num_to_mask = int(len(chunk) * prop_to_mask_last)
            else:
                num_to_mask = int(len(chunk) * k)
            mask_indices = np.random.choice(len(chunk), num_to_mask, replace=False)
            original_length = len(original_chunk)
            original_aa = original_chunk[mask_indices]
            for idx in mask_indices:
                chunk[idx] = tokenizer.mask_token

    # Step 6: Convert tokens to input IDs
            input_ids = tokenizer.convert_tokens_to_ids(chunk)

    # Step 7: Add special tokens [CLS] and [SEP]
            input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    # Step 8: Convert input IDs to TensorFlow tensors
            input_tensor = tf.constant(input_ids)[None, :]  # Batch size: 1

    # Step 9: Predict the masked words
            outputs = model(input_tensor)
            predictions = outputs.logits.numpy()[0, mask_indices+1, :]

    # Step 10: Convert predictions to tokens
        #predicted_tokens = [tokenizer.convert_ids_to_tokens(int(np.argmax(predictions[i,:]))) for i in range(predictions.shape[0])]
            predicted_tokens = tokenizer.convert_ids_to_tokens(np.argmax(predictions, axis=1))

    #print(predicted_tokens)
            replace_choice = np.random.rand(len(mask_indices))
            for i,idx in enumerate(mask_indices):
                if replace_choice[i] < 0.1: #original aa
                    chunk[idx] = original_aa[i]
                elif replace_choice[i] > 0.9: #random replacement
                    chunk[idx] = tokenizer.convert_ids_to_tokens(np.random.choice(20, 1) + 4)[0]
                else: #most probable prediction
                    chunk[idx] = predicted_tokens[i]

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
