########################################
## Revised by Zhuoyang CHEN
## Date: 2023/11/07
## Add arguments and more testing
#######################################

import argparse
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import time
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import os, sys
import csv

example_usage = 'python meta-classifier.py --embedding Esm2-original --augfile data/embedding/X_aug_Esm2_MLM_k0.05.npy'
parser = argparse.ArgumentParser(description='extract embeddings from pLM', epilog=example_usage)
parser.add_argument('--embedding', type=str, default='original_PAM1', help='which folder to load embeddings')
parser.add_argument('--noaug', action='store_true', default=False, help='whether use data augmentation')
parser.add_argument('--only_aug', action='store_true', default=False, help='only use augmented data')
parser.add_argument('--duplicate', action='store_true', default=False, help='duplicate training data')
parser.add_argument('--augfile', type=str, default='', help='provide specific X augmentation file')
parser.add_argument('--augcsv', type=str, default='', help='provide specific csv file for augmentation info')
parser.add_argument('--testset', type=str, default='test', help='which set of data to evaluate and output: train, val and test?')
parser.add_argument('--maxlen', type=int, default=500, help='max sequence len before truncating')
parser.add_argument('--overlap', type=int, default=250, help='overlap sequence')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu', default='0', type=str, help='which gpu card in your device to use')
parser.add_argument('--gpu_mem', type=int, default=0, help='per-process GPU memory cap in MiB (0 = grow on demand)')
parser.add_argument('--test', action='store_true', default=False, help='only load a small set')
parser.add_argument('--save', type=int, default=1)
args = parser.parse_args()

args.save = bool(args.save)

if not os.path.exists(f'data/embedding/{args.embedding}'):
    sys.exit(0)

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    g.strip() for g in str(args.gpu).split(",") if g.strip()
)

os.environ["TF_USE_LEGACY_KERAS"]= "1"
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

import ktrain
from ktrain import text
from transformers import *
import tensorflow as tf

def _configure_tensorflow_gpu(memory_limit_mb=None):
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return
    if memory_limit_mb is not None:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=int(memory_limit_mb))],
        )
    else:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

_configure_tensorflow_gpu(args.gpu_mem if args.gpu_mem > 0 else None)

exec(open('utils.py').read())
#exec(open('evaluate.py').read())
MAX_LEN = args.maxlen

#seed = 42
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
random.seed(args.seed)
tf.experimental.numpy.random.seed(args.seed)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["PYTHONHASHSEED"] = str(args.seed)

if args.duplicate:
    args.noaug = True

# Load data
df_train = pd.read_csv('data/train_split.csv')
df_val = pd.read_csv('data/val_split.csv')
df_test = pd.read_csv('data/test_split.csv')

if args.test:
    df_train = df_train[:10]
    df_val = df_val[:10]

# Generate slices from sliding window technique
__, y_train, positions_train = generate_data(df_train, subseq=500, overlap=250)
__, y_val, positions_val = generate_data(df_val, subseq=500, overlap=250)
__, y_test, positions_test = generate_data(df_test, subseq=500, overlap=250)

X_train = np.load(f'data/embedding/{args.embedding}/X_train.npy') #make sure if it is augmented
X_val = np.load(f'data/embedding/{args.embedding}/X_val.npy')
X_test = np.load(f'data/embedding/{args.embedding}/X_test.npy')
if args.test:
    X_train = X_train[:len(y_train),:]
    X_val = X_val[:len(y_val)]

if not args.noaug:
    if len(args.augfile) >0:
        X_aug = np.load(args.augfile) #use aug file other strategy
    else:
        args.augfile = list(filter(lambda x:'aug' in x, os.listdir(f'data/embedding/{args.embedding}')))[0]
        X_aug = np.load(f'data/embedding/{args.embedding}/{args.augfile}')
    if args.test:
        X_aug = X_aug[:len(y_train),:]
    if not args.augcsv:
        args.augcsv = f"data/{os.path.basename(args.augfile).replace('X_', '').replace('.npy', '.csv')}"
    df_aug = pd.read_csv(args.augcsv)
    __, y_aug, positions_aug = generate_data(df_aug, subseq=500, overlap=250)
    if args.only_aug:
        X_train = X_aug
    else:
        X_train = np.concatenate([X_train, X_aug], axis=0)
        positions_train = positions_train + positions_aug
        y_train = np.concatenate([y_train, y_aug], axis=0)

if args.duplicate:
    X_train = np.concatenate([X_train, X_train], axis=0)
    positions_train = positions_train + positions_train
    y_train = np.concatenate([y_train, y_train], axis=0)

#Embeddings aggregation
X_train, y_train = protein_embedding(X_train, y_train, positions_train)
X_val, y_val = protein_embedding(X_val, y_val, positions_val)
X_test, y_test = protein_embedding(X_test, y_test, positions_test)


# Meta-classifier fit
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(500, input_shape=(X_train.shape[1], ), activation='relu'))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mse')

es = tf.keras.callbacks.EarlyStopping(patience=15, verbose=1, restore_best_weights=True)
lr = tf.keras.callbacks.ReduceLROnPlateau(patience=5)

t0 = time.time()
model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_val, y_val), callbacks=[es, lr])
training_time = time.time() - t0
print(f'Training time: {training_time}')

if args.testset == 'test':
    pred = model.predict(X_test)
    mse = mean_squared_error(pred, y_test)
    mae = np.mean(np.abs(pred-y_test))
    cor,p = spearmanr(pred,y_test)
elif args.testset == 'train':
    pred = model.predict(X_train)
    mse = mean_squared_error(pred, y_train)
    mae = np.mean(np.abs(pred-y_train))
    cor,p = spearmanr(pred,y_train)
    args.save = False
else:
    pred = model.predict(X_val)
    mse = mean_squared_error(pred, y_val)
    mae = np.mean(np.abs(pred-y_val))
    cor,p = spearmanr(pred,y_val)
    args.save = False
    
print(f'mse on {args.testset} set: {mse}')
print(f'spearman correlation on {args.testset} set: {cor}, p-value: {p}')

if args.save:
    np.save(f'predictions/{args.embedding}.npy', pred)
    flag = 'noaug' if args.noaug else 'aug'
    flag = 'duplicate' if args.duplicate else flag
    flag = 'only_aug' if args.only_aug else flag
    fn = f'predictions/{args.embedding}.csv'
    if not os.path.exists(fn):
        with open(fn, 'w') as f:
            csv.writer(f).writerow(['mse', 'mae', 'cor', 'p', 'training_time', 'seed', 'aug', 'augfile'])
    with open(fn, 'a') as f:
        csv.writer(f).writerow([mse, mae, cor, p, training_time, args.seed, flag, args.augfile])
