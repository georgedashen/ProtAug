import argparse
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import time
import os
import pickle
import csv

example_usage = 'python meta-classifier.py --embedding Esm2-fix-noaug --noaug --test'
example_usage = 'python meta-classifier.py --embedding Esm2-fix-duplicate --duplicate --test'
example_usage = 'python meta-classifier.py --embedding Esm2-fix-Esm2_MLM_rho0.55 --only_aug --test'
example_usage = 'python meta-classifier.py --embedding Esm2-fix-Esm2_MLM_rho0.55 --test'
parser = argparse.ArgumentParser(description='extract embeddings from pLM', epilog=example_usage)
parser.add_argument('--aspect', type=str, default='mf', choices=['bp','mf','cc'])
parser.add_argument('--embedding', type=str, default='original_PAM1', help='which folder to load embeddings')
parser.add_argument('--noaug', action='store_true', default=False, help='whether use data augmentation')
parser.add_argument('--only_aug', action='store_true', default=False, help='only use augmented data')
parser.add_argument('--duplicate', action='store_true', default=False, help='duplicate training data')
parser.add_argument('--augfile', type=str, default='', help='provide specific X augmentation file')
parser.add_argument('--testset', type=str, default='test', help='which set of data to evaluate and output: train, val and test?')
parser.add_argument('--maxlen', type=int, default=500, help='max sequence len before truncating')
parser.add_argument('--overlap', type=int, default=250, help='overlap sequence')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--ic', type=str, default='mf_ic.pkl')
parser.add_argument('--test', action='store_true', default=False, help='only load a small set')
args = parser.parse_args()

args.save = bool(args.save)
exec(open('utils.py').read())
exec(open('evaluate.py').read())
MAX_LEN = args.maxlen

os.environ["TF_USE_LEGACY_KERAS"]= "1"
import ktrain
from ktrain import text
import tensorflow as tf
from transformers import *

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
df_train = pd.read_csv('data/mf-training.csv')
df_val = pd.read_csv('data/mf-validation.csv')
df_test = pd.read_csv('data/mf-test.csv')
ontologies_names = df_val.columns[2:].values

if args.test:
    df_train = df_train[:10]
    df_val = df_val[:100]

# Load ontology
ontology = generate_ontology('go.obo', specific_space=True, name_specific_space='molecular_function')

# Generate slices from sliding window technique
__, y_train, positions_train = generate_data(df_train, subseq=500, overlap=250)
__, y_val, positions_val = generate_data(df_val, subseq=500, overlap=250)
__, y_test, positions_test = generate_data(df_test, subseq=500, overlap=250)
if not args.noaug:
    y_aug, positions_aug = y_train, positions_train

X_train = np.load(f'data/embedding/{args.embedding}/X_train.npy') #make sure if it is augmented
X_val = np.load(f'data/embedding/{args.embedding}/X_val.npy')
X_test = np.load(f'data/embedding/{args.embedding}/X_test.npy')
if args.test:
    X_train = X_train[:len(y_train),:]
    X_val = X_val[:len(y_val)]


if not args.noaug:
    if len(args.augfile) >0:
        # if duplicate, use the train embedding
        X_aug = np.load(args.augfile) #use aug file other strategy
    else:
        args.augfile = list(filter(lambda x:'aug' in x, os.listdir(f'data/embedding/{args.embedding}')))[0]
        X_aug = np.load(f'data/embedding/{args.embedding}/{args.augfile}')
    if args.test:
        X_aug = X_aug[:len(y_aug),:]
    if args.only_aug:
        X_train = X_aug
        positions_train = positions_aug
        y_train = y_aug
    else:
        X_train = np.concatenate([X_train, X_aug], axis=0)
        positions_train = positions_train + positions_aug
        y_train = np.concatenate([y_train, y_aug], axis=0)

if args.duplicate:
    X_train = np.concatenate([X_train, X_train], axis=0)
    positions_train = positions_train + positions_train
    y_train = np.concatenate([y_train, y_train], axis=0)

# Embeddings aggregation
X_train, y_train = protein_embedding(X_train, y_train, positions_train)
X_val, y_val = protein_embedding(X_val, y_val, positions_val)
X_test, y_test = protein_embedding(X_test, y_test, positions_test)


# Meta-classifier fit
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1000, input_shape=(X_train.shape[1], ), activation='relu'))
model.add(tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')

es = tf.keras.callbacks.EarlyStopping(patience=15, verbose=1, restore_best_weights=True)
lr = tf.keras.callbacks.ReduceLROnPlateau(patience=5)

t0 = time.time()
model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_val, y_val), callbacks=[es, lr])
training_time = time.time() - t0
print(f'Training time: {training_time}')

if args.ic:
    ic = pickle.load(open(args.ic,'rb'))
else:
    ic = None

if args.testset == 'test':
    pred = model.predict(X_test)
    perf = evaluate(pred, y_test, ic)
elif args.testset == 'train':
    pred = model.predict(X_train)
    perf = evaluate(pred, y_train, ic)
    args.save = False
else:
    pred = model.predict(X_val)
    perf = evaluate(pred, y_val, ic)
    args.save = False
    
if args.save:
    os.system(f'mkdir -p predictions')
    np.save(f'predictions/mf-temprot-{args.embedding}.npy', pred)
    flag = 'noaug' if args.noaug else 'aug'
    flag = 'duplicate' if args.duplicate else flag
    flag = 'only_aug' if args.only_aug else flag
    fn = f'predictions/mf-temprot-{args.embedding}.csv'
    if not os.path.exists(fn):
        with open(fn, 'w') as f:
            csv.writer(f).writerow(['fmax', 'threshold', 'smin', 'aupr', 'Iaupr', 'training_time', 'seed', 'aug', 'augfile'])
    with open(fn, 'a') as f:
        csv.writer(f).writerow([perf['fmax'], perf['threshold'], perf['smin'], perf['aupr'], perf['Iaupr'], training_time, args.seed, flag, args.augfile])
