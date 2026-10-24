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
import os, sys
import csv

example_usage = 'python meta-classifier.py --embedding Esm2-original --augfile data/embedding/X_aug_Esm2_MLM_k0.05.npy'
parser = argparse.ArgumentParser(description='extract embeddings from pLM', epilog=example_usage)
parser.add_argument('--label', type=str, default='family', help='property to retrieve', choices=['fold', 'superfamily', 'family'])
parser.add_argument('--embedding', type=str, default='prot_bert_bfd-original', help='which folder to load embeddings')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
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
parser.add_argument('--test', action='store_true', default=False, help='only load a small set')
parser.add_argument('--save', type=int, default=1)
args = parser.parse_args()

args.save = bool(args.save)

if not os.path.exists(f'data/embedding/{args.embedding}'):
    print(f'Embedding folder data/embedding/{args.embedding} not found!')
    sys.exit(0)

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpu.split(','))

os.environ["TF_USE_LEGACY_KERAS"]= "1"

import ktrain
from ktrain import text
from transformers import *
import tensorflow as tf

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
df_train = pd.read_csv(f'data/train_{args.label}.csv')
df_val = pd.read_csv(f'data/valid_{args.label}.csv')
df_test = pd.read_csv(f'data/test_{args.label}.csv')

if args.test:
    df_train = df_train[:10]
    df_val = df_val[:10]

# Generate slices from sliding window technique
__, y_train, positions_train = generate_data(df_train, subseq=500, overlap=250)
__, y_val, positions_val = generate_data(df_val, subseq=500, overlap=250)
__, y_test, positions_test = generate_data(df_test, subseq=500, overlap=250)

X_train = np.load(f'data/embedding/{args.embedding}/X_train.npy') #make sure if it is augmented
X_val = np.load(f'data/embedding/{args.embedding}/X_val.npy')
X_test = np.load(f'data/embedding/{args.embedding}/X_test_{args.label}.npy')
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
        args.augcsv = f"data/{args.label}/{os.path.basename(args.augfile).replace('X_', '').replace('.npy', '.csv')}"
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

# Convert integer class labels (0..1194) to one-hot vectors
# User labels are categorical from 0 to 1194 (1195 classes)
if args.label == 'fold':
    NUM_CLASSES = 1195
elif args.label == 'superfamily':
    NUM_CLASSES = 1961
elif args.label == 'family':
    NUM_CLASSES = 3888

y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)


# Meta-classifier fit
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1200, input_shape=(X_train.shape[1], ), activation='relu'))
# Final layer: softmax over classes for categorical labels
model.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))

# Use categorical_crossentropy for multi-class one-hot labels
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

es = tf.keras.callbacks.EarlyStopping(patience=15, verbose=1, restore_best_weights=True)
lr = tf.keras.callbacks.ReduceLROnPlateau(patience=5)

t0 = time.time()
model.fit(X_train, y_train, epochs=1000, batch_size=args.batch_size, validation_data=(X_val, y_val), callbacks=[es, lr])
training_time = time.time() - t0
print(f'Training time: {training_time}')

if args.testset == 'test':
    pred = model.predict(X_test)
    # calulate and report accuracy
    acc = np.sum(np.argmax(pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]

elif args.testset == 'train':
    pred = model.predict(X_train)
    acc = np.sum(np.argmax(pred, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
    args.save = False
else:
    pred = model.predict(X_val)
    acc = np.sum(np.argmax(pred, axis=1) == np.argmax(y_val, axis=1)) / y_val.shape[0]
    args.save = False
    
print(f'accuracy on {args.testset} set: {acc}')
if not os.path.exists(f'predictions/{args.label}'):
    os.mkdir(f'predictions/{args.label}')

if args.save:
    np.save(f'predictions/{args.label}/{args.embedding}.npy', pred)
    flag = 'noaug' if args.noaug else 'aug'
    flag = 'duplicate' if args.duplicate else flag
    flag = 'only_aug' if args.only_aug else flag
    fn = f'predictions/{args.label}/{args.embedding}.csv'
    if not os.path.exists(fn):
        with open(fn, 'w') as f:
            csv.writer(f).writerow(['acc', 'training_time', 'seed', 'aug', 'augfile'])
    with open(fn, 'a') as f:
        csv.writer(f).writerow([acc, training_time, args.seed, flag, args.augfile])
