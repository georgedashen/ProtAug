# ktrain code edited
# 1: ktrain/text/preprocessor.py (bypass length checking)
# 2: ktrain/lroptimize/triangular.py (np.Inf -> np.inf)

import argparse
import os
import pandas as pd
import numpy as np
import random
import time
from tqdm import tqdm
import re

example_usage = 'CUDA_VISIBLE_DEVICES=0 python fine-tuning-fix.py --model Esm2 --method noaug --batch 4 --lr 1e-4'
example_usage = 'CUDA_VISIBLE_DEVICES=0 python fine-tuning-fix.py --model Esm2 --method duplicate --batch 4 --lr 1e-4'
example_usage = 'CUDA_VISIBLE_DEVICES=0 python fine-tuning-fix.py --model Esm2 --method aug --augfile  _Esm2_MLM_rho0.55 --batch 4 --lr 1e-4'
parser = argparse.ArgumentParser(description='finetuning pLM with frozen-backbone settings', epilog=example_usage)
parser.add_argument('--aspect', type=str, default='mf', choices=['bp','mf','cc'])
parser.add_argument('--model', type=str, default='prot_bert_bfd', choices=['DistilProtBert','prot_bert_bfd','Esm2'], help='base model')
parser.add_argument('--method', type=str, default='noaug', choices=['noaug','duplicate','aug'], help='noaug: no augmentation, duplicate: duplicate training data, aug: use augmentation')
parser.add_argument('--augfile', type=str, default='', help='provide augmentation file, in the format of _42, _12345')
parser.add_argument('--maxlen', type=int, default=500, help='max sequence len before truncating')
parser.add_argument('--overlap', type=int, default=250, help='overlap sequence')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch', type=int, default=4) #could be larger, recommend 128,8,16,32,64
parser.add_argument('--test', action='store_true', default=False, help='only load a small subset of data')
parser.add_argument('--lr', type=float, default=1e-5) #recommend 3e-5, 3e-4, 1e-4, 5e-5, 3e-5
parser.add_argument('--epoch', type=int, default=10) #recommend 4, but early_strop deployed
parser.add_argument('--patience', default=None, help='reduce lr when val_loss not reduce for this num of epochs')
parser.add_argument('--gpu', default='0', type=str, help='which gpu card in your device to use')
parser.add_argument('--resume', action='store_true', default=False, help='whether resume from the last checkpoint')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpu.split(','))
os.environ["TF_USE_LEGACY_KERAS"]= '1' #use legacy keras to avoid error in lora layer

import ktrain
from ktrain import text
import tensorflow as tf

assert args.batch % len(args.gpu.split(',')) == 0, "Can not assign equal samples to each GPU!"
args.batch = int(args.batch / len(args.gpu.split(',')))

exec(open('utils.py').read())
MAX_LEN = args.maxlen
model_name = f'../Models/{args.model}'

noaug = False
if args.method == 'noaug' or args.method == 'duplicate':
    noaug = True

if args.patience:
    args.patience = int(args.patience)

np.random.seed(args.seed)
tf.random.set_seed(args.seed)
random.seed(args.seed)
tf.experimental.numpy.random.seed(args.seed)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["PYTHONHASHSEED"] = str(args.seed)

# Load data
df_train = pd.read_csv('data/mf-training.csv')
df_val = pd.read_csv('data/mf-validation.csv')

if args.test:
    df_train = df_train[:10]
    df_val = df_val[:100]

if not noaug:
    data_augmentation = pd.read_csv(f'data/mf-augmented_training{args.augfile}.csv')
    if args.test:
        data_augmentation = data_augmentation[:100]

ontologies_names = df_val.columns[2:].values

# Generate slices from sliding window technique
t0 = time.time()
X_train, y_train, positions_train = generate_data(df_train, subseq=500, overlap=250)
X_val, y_val, positions_val = generate_data(df_val, subseq=500, overlap=250)
if not noaug:
    X_aug, y_aug, positions_aug = generate_data(data_augmentation, subseq=500, overlap=250)
print(f'Sequence Slicing time: {time.time()-t0}, method: {args.method}', file = open('seqSlice_time.txt', 'a'))
print(f'Sequence Slicing time: {time.time()-t0}')

if args.method == 'duplicate':
    X_train = X_train + X_train
    y_train = np.concatenate([y_train, y_train], axis=0)
    positions_train = positions_train + positions_train

if not noaug:
    X_train = X_train + X_aug
    y_train = np.concatenate([y_train, y_aug], axis=0)
    positions_train = positions_train + positions_aug

# Fine-tuning the model
t0 = time.time()
ms = tf.distribute.MirroredStrategy()
t = text.Transformer(model_name, maxlen=MAX_LEN, class_names=ontologies_names)
trn = t.preprocess_train(X_train, y_train)
val = t.preprocess_test(X_val, y_val)
with ms.scope():
    model = t.get_classifier()
    if args.resume:
        weight_file = os.listdir(f'fix_weights/{args.model}-finetune-{args.method}{args.augfile}')[-1]
        model.load_weights(f'fix_weights/{args.model}-finetune-{args.method}{args.augfile}/{weight_file}')

# fix parameters
if args.model == 'Esm2':
    for i,layer in enumerate(model.esm.encoder.layer):
        if i < len(model.esm.encoder.layer)-2:
            layer.trainable = False
if args.model in ['DistilProtBert', 'prot_bert_bfd']:
    for i,layer in enumerate(model.bert.encoder.layer):
        if i < len(model.bert.encoder.layer)-2:
            layer.trainable = False

learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=args.batch) 
learner.autofit(args.lr, args.epoch, early_stopping=1, reduce_on_plateau=args.patience, checkpoint_folder=f'fix_weights/{args.model}-finetune-{args.method}{args.augfile}')
print(f'Fine-tuning time: {time.time()-t0}, model: {args.model}, method: {args.method}, batch_size: {args.batch}, lr: {args.lr}, augfile: {args.augfile}, seed: {args.seed}', file = open('finetuning_time.txt', 'a'))
print(f'Fine-tuning time: {time.time()-t0}')


# remove weights
checkpoint_dir = f'fix_weights/{args.model}-finetune-{args.method}{args.augfile}'
pattern = re.compile(r'weights-\d+-([\d.]+)\.hdf5')
best_file = None
best_loss = float('inf')

for fname in os.listdir(checkpoint_dir):
    match = pattern.match(fname)
    if match:
        loss = float(match.group(1))
        if loss < best_loss:
            best_loss = loss
            best_file = fname

# Remove all but the best checkpoint
for fname in os.listdir(checkpoint_dir):
    if fname != best_file and pattern.match(fname):
        os.remove(os.path.join(checkpoint_dir, fname))
