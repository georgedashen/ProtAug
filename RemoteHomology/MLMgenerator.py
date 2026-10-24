# Step 1: Import libraries
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
from protaug_paths import resolve_model

import argparse
import os
import re
import numpy as np
import pandas as pd
import random
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

example_usage='python MLMgenerator.py --model Esm2 --k 0.55'
example_usage='python MLMgenerator.py --model Esm2 --rho 0.55'
example_usage='python MLMgenerator.py --model Esm2 --k 0.55 --num_chunk 5 --chunk_id 0'
parser = argparse.ArgumentParser(description='iteratively predict masked amino acids from encoder pLM', epilog=example_usage)
parser.add_argument('--model', type=str, default='Esm2')
parser.add_argument('--k', type=float, default=0.15, help='fraction control factor, enable when rho is disable')
parser.add_argument('--K_MAX', type=float, default=0.15)
parser.add_argument('--rho', type=str, default='', help='variation level')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch', type=int, default=128) #could be larger, recommend 128,8,16,32,64
parser.add_argument('--test', action='store_true', default=False, help='only load a small subset of data')
parser.add_argument('--test_id', type=int, default=-1, help='which seq to test')
parser.add_argument('--num_chunk', type=int, default=1, help='number larger than 1 to partition the dataset')
parser.add_argument('--chunk_id', type=int, default=0, help='which chunk to process, allow multiple processes')
parser.add_argument('--gpu', default='0', type=str, help='which gpu card in your device to use')
parser.add_argument('--output', type=str, default='', help='predefined output file')
parser.add_argument('--input', type=str, default='data/train_family.csv', help='training csv: data/train_family.csv, data/train_superfamily.csv, or data/train_fold.csv')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpu.split(','))
os.environ["TF_USE_LEGACY_KERAS"]= "1"

exec(open('utils.py').read())

if not args.rho:
    args.method = f'{args.model}_MLM_k{args.k}'
else:
    args.rho = float(args.rho)
    args.method = f'{args.model}_MLM_rho{args.rho}'

np.random.seed(args.seed)
random.seed(args.seed)
os.environ["PYTHONHASHSEED"] = str(args.seed)
# Torch deterministic settings
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

t0 = time.time()
df_train = pd.read_csv(args.input)
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
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()
if tokenizer.sep_token == None: #Esm2
    tokenizer.sep_token = '<eos>'
if args.model.endswith('Esm2'):
    SLICE_LEN = 1000
elif args.model == 'prot_bert_bfd' or args.model == 'DistilProtBert':
    SLICE_LEN = 500

# Configure GPU device and optional mixed precision for faster inference (PyTorch)
if torch.cuda.is_available():
    device = torch.device('cuda')
    try:
        torch.cuda.set_device(0)
    except Exception:
        pass
    use_amp = True
    print(f'Using CUDA device, count: {torch.cuda.device_count()}')
else:
    device = torch.device('cpu')
    use_amp = False
    print('No CUDA detected, running on CPU')

# Move model to the selected device
try:
    model.to(device)
    if device.type == 'cuda':
        # optionally convert weights to fp16 to save memory/boost perf
        try:
            model.half()
        except Exception:
            pass
except Exception:
    print('Warning: unable to move model to device')

# Helper: batch inference to reduce per-call overhead
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.sep_token_id

def _process_batch(batch_input_ids, batch_mask_positions, batch_chunks, batch_original_aas):
    if len(batch_input_ids) == 0:
        return
    # pad sequences
    maxlen = max(len(x) for x in batch_input_ids)
    padded = [x + [pad_token_id] * (maxlen - len(x)) for x in batch_input_ids]
    input_tensor = torch.tensor(padded, dtype=torch.long, device=device)
    # build attention mask (1 for real tokens, 0 for padding)
    attention_mask = torch.tensor([[1]*len(x) + [0]*(maxlen - len(x)) for x in batch_input_ids], dtype=torch.long, device=device)
    # run model on selected device
    model.eval()
    with torch.no_grad():
        if use_amp and device.type == 'cuda':
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model(input_ids=input_tensor, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_tensor, attention_mask=attention_mask)
    # get predicted token ids per position (B, L)
    pred_ids = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

    # apply predictions back to chunks
    for i, mask_pos in enumerate(batch_mask_positions):
        if len(mask_pos) == 0:
            continue
        # shift by 1 due to CLS token (we add CLS token earlier when building inputs)
        shifted = [p + 1 for p in mask_pos]
        predicted_ids = pred_ids[i, shifted].tolist()
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids)
        replace_choice = np.random.rand(len(mask_pos))
        for j, idx in enumerate(mask_pos):
            if replace_choice[j] < 0.1:
                batch_chunks[i][idx] = batch_original_aas[i][j]
            elif replace_choice[j] > 0.9:
                rnd_id = int(np.random.randint(0, 20) + 4)
                batch_chunks[i][idx] = tokenizer.convert_ids_to_tokens([rnd_id])[0]
            else:
                batch_chunks[i][idx] = re.sub(r"[UZOB]","L", predicted_tokens[j])

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
        num_round = 300
        prop_to_mask_last = 0
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

    # Prepare batch accumulators to collect masked inputs for batched inference
    batch_input_ids = []
    batch_mask_positions = []
    batch_chunks_refs = []
    batch_original_aas = []

    for ichunk,chunk in enumerate(chunks):
        original_chunk = np.array(chunk)
        for iround in range(num_round):
            if iround == num_round-1 and prop_to_mask_last:
                num_to_mask = int(len(chunk) * prop_to_mask_last)
            else:
                num_to_mask = int(len(chunk) * k)
            if num_to_mask <= 0:
                continue
            mask_indices = np.random.choice(len(chunk), num_to_mask, replace=False)
            original_length = len(original_chunk)
            original_aa = original_chunk[mask_indices]
            for idx in mask_indices:
                chunk[idx] = tokenizer.mask_token

            # Convert tokens to ids and add special tokens
            input_ids = tokenizer.convert_tokens_to_ids(chunk)
            input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

            batch_input_ids.append(input_ids)
            batch_mask_positions.append(mask_indices.tolist())
            batch_chunks_refs.append(chunk)
            batch_original_aas.append(original_aa.tolist())

            # If batch is full, process it
            if len(batch_input_ids) >= args.batch:
                _process_batch(batch_input_ids, batch_mask_positions, batch_chunks_refs, batch_original_aas)
                # clear batches
                batch_input_ids = []
                batch_mask_positions = []
                batch_chunks_refs = []
                batch_original_aas = []

            # after processing (maybe), check stopping condition for this chunk
            if args.rho:
                delta = sum(1 for x,y in zip(chunk,original_chunk) if x!=y)
                if args.test:
                    print(f'variation for current chunk {iround}iter: {delta/original_length}, {original_length}aa') #test
                if delta > args.rho * original_length:
                    break

        # process any remaining items in batch before moving to next chunk
        if len(batch_input_ids) > 0:
            _process_batch(batch_input_ids, batch_mask_positions, batch_chunks_refs, batch_original_aas)
            batch_input_ids = []
            batch_mask_positions = []
            batch_chunks_refs = []
            batch_original_aas = []

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
    df_train.to_csv(f'data/aug_{args.method}{flag}.csv', index=False)
