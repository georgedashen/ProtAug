#############################
## Revised by Zhuoyang CHEN
## Date: 2023/11/03
## Content: more printing and testing
############################

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
from protaug_paths import resolve_model

import argparse
import os, sys
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
import time
from npy_append_array import NpyAppendArray

example_usage = 'python extract-embeddings.py --model_name ../Models/prot_bert_bfd --model_path weights/mf-fine-tuned/weights-06.hdf5 --output finetune'
parser = argparse.ArgumentParser(description='extract embeddings from pLM', epilog=example_usage)
parser.add_argument('--subset', type=str, default='full', choices=['full','train','aug','val','test'], help='only encode one dataset')
parser.add_argument('--noaug', action='store_true', default=False, help='Whether no augmentation used')
parser.add_argument('--augfile', type=str, default='', help='provide augmentation file, in the format of _42, _12345')
parser.add_argument('--maxlen', type=int, default=500, help='max sequence len before truncating')
parser.add_argument('--overlap', type=int, default=250, help='overlap sequence')
parser.add_argument('--model_name', type=str, default='DistilProtBert', help='base model')
parser.add_argument('--model_path', type=str, default='', help='finetuned weights')
parser.add_argument('--batch', type=int, default=128, help='batch to encode')
parser.add_argument('--output', type=str, default='DistilProtBert', help='folder to save the embeddings')
parser.add_argument('--test', action='store_true', default=False, help='enter test mode to load a small subset of data')
parser.add_argument('--gpu', default='0', type=str, help='which gpu card in your device to use')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpu.split(','))

os.environ["TF_USE_LEGACY_KERAS"]= "1"

import ktrain
from ktrain import text
from transformers import *
import tensorflow as tf

ext = compile(open('utils.py').read(), 'utils.py', 'exec')
exec(ext)

if args.model_path:
    if not os.path.exists(args.model_path):
        sys.exit(0)
    elif os.path.basename(args.model_path) != 'hdf5':
        files = os.listdir(args.model_path)
        if len(files)==10:
            File = list(filter(lambda x:'10' in x, files))[0]
        else:
            File = np.sort(files)[-1]
        args.model_path = os.path.join(args.model_path, File)
        print(args.model_path)

if args.model_path:
    if not os.path.exists(os.path.dirname(args.model_path)):
        sys.exit(0)
    mp = os.listdir(os.path.dirname(args.model_path))
    if len(mp)==1:
        args.model_path = os.path.dirname(args.model_path) + '/' + mp[0]
    elif 'weights-10.hdf5' in mp:
        args.model_path = os.path.dirname(args.model_path) + '/' + 'weights-10.hdf5'
    else:
        args.model_path = os.path.dirname(args.model_path) + '/' + mp[-1]

os.makedirs(f'data/embedding/{args.output}', exist_ok=True)
MAX_LEN = args.maxlen
args.model_name = resolve_model(args.model_name)

# Load data
df_train = pd.read_csv('data/train_split.csv')
df_val = pd.read_csv('data/val_split.csv')
df_test = pd.read_csv('data/test_split.csv')

if args.test:
    df_train = df_train[:1000]
    df_val = df_val[:100]
    df_test = df_test[:100]

# Generate slices from sliding window technique
X_train, y_train, positions_train = generate_data(df_train, subseq=args.maxlen, overlap=args.overlap)
if not args.noaug:
    data_augmentation = pd.read_csv(f'data/aug{args.augfile}.csv')
    if args.test:
        data_augmentation = data_augmentation[:1000]
    X_aug, y_aug, positions_aug = generate_data(data_augmentation, subseq=args.maxlen, overlap=args.overlap)

X_val, y_val, positions_val = generate_data(df_val, subseq=args.maxlen, overlap=args.overlap)

X_test, y_test, positions_test = generate_data(df_test, subseq=args.maxlen, overlap=args.overlap)


################ Training #####################
if args.subset == 'full' or args.subset == 'train':
    model, tokenizer, ds = get_model(args.model_name, args.model_path, args.output, 'train')
    file_path = f'data/embedding/{args.output}/X_train.npy'
    t0 = time.time()
    if args.batch == 1:
        get_embeddings_noBatch(X_train, model, tokenizer, file_path)
    else:
        get_embeddings(ds, model, tokenizer, file_path)
    tf.keras.backend.clear_session()
    print(f'Embedding time for train: {time.time()-t0}', file = open('Embedding_time.txt', 'a'))
    print(f'Embedding time for train: {time.time()-t0}')

################ Augmentation #####################
if (args.subset == 'full' or args.subset == 'aug') and not args.noaug:
    file_path = f'data/embedding/{args.output}/X_aug{args.augfile}.npy'
    model, tokenizer, ds = get_model(args.model_name, args.model_path, args.output, 'aug')
    t0 = time.time()
    if not args.noaug:
        if args.batch == 1:
            get_embeddings_noBatch(X_aug, model, tokenizer, file_path)
        else:
            get_embeddings(ds, model, tokenizer, file_path)
        tf.keras.backend.clear_session()
        print(f'Embedding time for aug: {time.time()-t0}', file = open('Embedding_time.txt', 'a'))
        print(f'Embedding time for aug: {time.time()-t0}')

################ Validation #####################
if args.subset == 'full' or args.subset == 'val':
    file_path = f'data/embedding/{args.output}/X_val.npy'
    model, tokenizer, ds = get_model(args.model_name, args.model_path, args.output, 'val')
    t0 = time.time()
    if args.batch == 1:
        get_embeddings_noBatch(X_val, model, tokenizer, file_path)
    else:
        get_embeddings(ds, model, tokenizer, file_path)
    tf.keras.backend.clear_session()
    print(f'Embedding time for val: {time.time()-t0}', file = open('Embedding_time.txt', 'a'))
    print(f'Embedding time for val: {time.time()-t0}')

################ Test #####################
if args.subset == 'full' or args.subset == 'test':
    file_path = f'data/embedding/{args.output}/X_test.npy'
    model, tokenizer, ds = get_model(args.model_name, args.model_path, args.output, 'test')
    t0 = time.time()
    if args.batch == 1:
        get_embeddings_noBatch(X_test, model, tokenizer, file_path)
    else:
        get_embeddings(ds, model, tokenizer, file_path)
    tf.keras.backend.clear_session()
    print(f'Embedding time for test: {time.time()-t0}', file = open('Embedding_time.txt', 'a'))
    print(f'Embedding time for test: {time.time()-t0}')
