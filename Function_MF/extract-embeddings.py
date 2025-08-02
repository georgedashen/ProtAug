import argparse
import os

example_usage = 'CUDA_VISIBLE_DEVICES=0 python extract-embeddings.py --model_name Esm2 --model_path fix_weights/Esm2-finetune-noaug/test --noaug --output Esm2-fix-noaug'
example_usage = 'CUDA_VISIBLE_DEVICES=0 python extract-embeddings.py --model_name Esm2 --model_path fix_weights/Esm2-finetune-duplicate/test --duplicate --output Esm2-fix-duplicate'
example_usage = 'CUDA_VISIBLE_DEVICES=0 python extract-embeddings.py --model_name Esm2 --model_path fix_weights/Esm2-finetune-aug_Esm2_MLM_rho0.55/test --augfile _Esm2_MLM_rho0.55 --output Esm2-fix-Esm2_MLM_rho0.55'
parser = argparse.ArgumentParser(description='extract embeddings from pLM', epilog=example_usage)
parser.add_argument('--subset', type=str, default='full', choices=['full','train','aug','val','test'], help='only encode one dataset')
parser.add_argument('--noaug', action='store_true', default=False, help='Whether no augmentation used')
parser.add_argument('--duplicate', action='store_true', default=False, help='Whether duplicate training data')
parser.add_argument('--augfile', type=str, default='', help='provide augmentation file, in the format of _42, _12345')
parser.add_argument('--maxlen', type=int, default=500, help='max sequence len before truncating')
parser.add_argument('--overlap', type=int, default=250, help='overlap sequence')
parser.add_argument('--model_name', type=str, default='DistilProtBert', help='base model')
parser.add_argument('--model_path', type=str, default='', help='finetuned weights')
parser.add_argument('--batch', type=int, default=128, help='batch to encode')
parser.add_argument('--output', type=str, default='DistilProtBert', help='folder to save the embeddings')
parser.add_argument('--test', action='store_true', default=False, help='enter test mode to load a small subset of data')
parser.add_argument('--lora', action='store_true', default=False, help='is lora model?')
parser.add_argument('--gpu', default='0', type=str, help='which gpu card in your device to use')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpu.split(','))

os.environ["TF_USE_LEGACY_KERAS"]= "1"

import ktrain
from ktrain import text
import tensorflow as tf
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
from transformers import *
import time
from npy_append_array import NpyAppendArray
ext = compile(open('utils.py').read(), 'utils.py', 'exec')
exec(ext)

os.makedirs(f'data/embedding/{args.output}', exist_ok=True)
MAX_LEN = args.maxlen
args.model_name = f'../Models/{args.model_name}'

mp = os.listdir(os.path.dirname(args.model_path))
if len(mp)==1:
    args.model_path = os.path.dirname(args.model_path) + '/' + mp[0]
else:
    raise ValueError(f'Multiple files found in {os.path.dirname(args.model_path)}. Please specify the exact file.')

# Load data
df_train = pd.read_csv('data/mf-training.csv')
df_val = pd.read_csv('data/mf-validation.csv')
df_test = pd.read_csv('data/mf-test.csv')

if args.test:
    df_train = df_train[:10]
    df_val = df_val[:100]
    df_test = df_test[:100]

ontologies_names = df_val.columns[2:].values

# Generate slices from sliding window technique
X_train, y_train, positions_train = generate_data(df_train, subseq=args.maxlen, overlap=args.overlap)
if not args.noaug:
    data_augmentation = pd.read_csv(f'data/mf-augmented_training{args.augfile}.csv')
    if args.test:
        data_augmentation = data_augmentation[:10]
    X_aug, y_aug, positions_aug = generate_data(data_augmentation, subseq=args.maxlen, overlap=args.overlap)

if args.duplicate:
    X_aug, y_aug, positions_aug = X_train, y_train, positions_train

X_val, y_val, positions_val = generate_data(df_val, subseq=args.maxlen, overlap=args.overlap)

X_test, y_test, positions_test = generate_data(df_test, subseq=args.maxlen, overlap=args.overlap)


################ Training #####################
if args.subset == 'full' or args.subset == 'train':
    model, tokenizer, ds = get_model(args.model_name, args.model_path, args.output, 'train', args.lora)
    file_path = f'data/embedding/{args.output}/mf-X_train.npy'
    t0 = time.time()
    if args.batch == 1:
        get_embeddings_noBatch(X_train, model, tokenizer, file_path)
    else:
        get_embeddings(ds, model, tokenizer, file_path)
    print(f'Embedding time for train: {time.time()-t0}', file = open('Embedding_time.txt', 'a'))
    print(f'Embedding time for train: {time.time()-t0}')

################ Augmentation #####################
if (args.subset == 'full' or args.subset == 'aug') and not args.noaug:
    file_path = f'data/embedding/{args.output}/mf-X_aug{args.augfile}.npy'
    model, tokenizer, ds = get_model(args.model_name, args.model_path, args.output, 'aug', args.lora)
    t0 = time.time()
    if not args.noaug:
        if args.batch == 1:
            get_embeddings_noBatch(X_aug, model, tokenizer, file_path)
        else:
            get_embeddings(ds, model, tokenizer, file_path)
        print(f'Embedding time for aug: {time.time()-t0}', file = open('Embedding_time.txt', 'a'))
        print(f'Embedding time for aug: {time.time()-t0}')

################ Validation #####################
if args.subset == 'full' or args.subset == 'val':
    file_path = f'data/embedding/{args.output}/mf-X_val.npy'
    model, tokenizer, ds = get_model(args.model_name, args.model_path, args.output, 'val', args.lora)
    t0 = time.time()
    if args.batch == 1:
        get_embeddings_noBatch(X_val, model, tokenizer, file_path)
    else:
        get_embeddings(ds, model, tokenizer, file_path)
    print(f'Embedding time for val: {time.time()-t0}', file = open('Embedding_time.txt', 'a'))
    print(f'Embedding time for val: {time.time()-t0}')

################ Test #####################
if args.subset == 'full' or args.subset == 'test':
    file_path = f'data/embedding/{args.output}/mf-X_test.npy'
    model, tokenizer, ds = get_model(args.model_name, args.model_path, args.output, 'test', args.lora)
    t0 = time.time()
    if args.batch == 1:
        get_embeddings_noBatch(X_test, model, tokenizer, file_path)
    else:
        get_embeddings(ds, model, tokenizer, file_path)
    print(f'Embedding time for test: {time.time()-t0}', file = open('Embedding_time.txt', 'a'))
    print(f'Embedding time for test: {time.time()-t0}')
