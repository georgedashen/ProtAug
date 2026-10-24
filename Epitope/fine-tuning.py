##############
## revised by Zhuoyang CHEN
## date: 2023/11/03
## more printing and testing lines
##############

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
from protaug_paths import resolve_model

import argparse
import os
import pandas as pd
import numpy as np
import random
import time
from tqdm import tqdm
import csv
from transformers import AutoConfig  # Required for flash attention configuration

example_usage = 'python fine-tuning.py --model Esm2 --method noaug --batch 4 --lr 5e-6 --gpu 1 --test'
parser = argparse.ArgumentParser(description='extract embeddings from pLM', epilog=example_usage)
parser.add_argument('--model', type=str, default='prot_bert_bfd')
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--method', type=str, default='noaug', help='which folder to save models')
parser.add_argument('--augfile', type=str, default='', help='provide augmentation file, in the format of _42, _12345')
parser.add_argument('--maxlen', type=int, default=500, help='max sequence len before truncating')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch', type=int, default=4) #could be larger, recommend 128,8,16,32,64
parser.add_argument('--test', action='store_true', default=False, help='only load a small subset of data')
parser.add_argument('--lr', type=float, default=1e-4) #recommend 3e-5, 3e-4, 1e-4, 5e-5, 3e-5
parser.add_argument('--epoch', type=int, default=10) #recommend 4, but early_strop deployed
parser.add_argument('--patience', default=1, help='reduce lr when val_loss not reduce for this num of epochs')
parser.add_argument('--gpu', default='0', type=str, help='which gpu card in your device to use')
parser.add_argument('--gpu_mem', type=int, default=0, help='per-process GPU memory cap in MiB (0 = grow on demand)')
parser.add_argument('--output', default='finetune_logits', type=str)
parser.add_argument('--small', type=float, default=1.0, help='fraction of training sequences to use')
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--dropsars', action='store_true', default=False, help='not use sars as training data')
parser.add_argument('--resume', action='store_true', default=False, help='whether resume from the last checkpoint')
parser.add_argument('--predict', action='store_true', default=False, help='Enable predict mode')
parser.add_argument('--unfreeze_plm', action='store_true', default=False, help='unfreeze pLM parameters during training')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpu.split(','))

os.environ["TF_USE_LEGACY_KERAS"]= "1"
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

import ktrain
from ktrain import text
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

assert args.batch % len(args.gpu.split(',')) == 0, "Can not assign equal samples to each GPU!"
args.batch = int(args.batch / len(args.gpu.split(',')))

exec(open('utils.py').read())
exec(open('evaluate.py').read())
MAX_LEN = args.maxlen
model_name = resolve_model(args.model)

only_aug = args.method == 'only_aug'
noaug = False
if args.method == 'noaug' or args.method == 'duplicate':
    args.augfile = ""
    noaug = True
elif only_aug and not args.augfile:
    parser.error('only_aug requires --augfile (e.g. _homology_rho0.02)')

weight_tag = f'{args.model}-finetune-{args.method}{args.augfile}-seed{args.seed}'
weight_dir = f'weights/{weight_tag}'

if args.patience:
    args.patience = int(args.patience)

args.save = int(args.save)

np.random.seed(args.seed)
tf.random.set_seed(args.seed)
random.seed(args.seed)
tf.experimental.numpy.random.seed(args.seed)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["PYTHONHASHSEED"] = str(args.seed)

# Load data
df_train = pd.read_csv('data/train_split.csv')
df_val = pd.read_csv('data/val_split.csv')
df_test = pd.read_csv('data/test_split.csv')
info1 = pd.read_csv('data/input_bcell.csv')
if not args.dropsars:
    info2 = pd.read_csv('data/input_sars.csv')
    info = pd.concat([info1,info2], axis=0)
    info.reset_index(inplace=True,drop=True)
else:
    info = info1

if args.test:
    df_train = df_train[:2]
    df_val = df_val[:2]

if not noaug:
    data_augmentation = pd.read_csv(f'data/aug{args.augfile}.csv')
    if args.test:
        data_augmentation = data_augmentation[:100]

# Generate slices from sliding window technique
t0 = time.time()
X_train, y_train, positions_train = create_data(df_train, info, args.maxlen)
X_val, y_val, positions_val = create_data(df_val, info, args.maxlen)
if not noaug:
    X_aug, y_aug, positions_aug = create_data(data_augmentation, info, args.maxlen)
print(f'Sequence Slicing time: {time.time()-t0}, method: {args.method}', file = open('seqSlice_time.txt', 'a'))
print(f'Sequence Slicing time: {time.time()-t0}')

n_train = int(len(y_train) * args.small)
X_train = X_train[:n_train]
y_train = y_train[:n_train]
positions_train = positions_train[:n_train]

if args.method == 'duplicate':
    X_train = X_train + X_train
    y_train = np.concatenate([y_train, y_train], axis=0)
    positions_train = positions_train + positions_train

if only_aug:
    n_aug = int(len(X_aug) * args.small)
    X_train = X_aug[:n_aug]
    y_train = y_aug[:n_aug]
    positions_train = positions_aug[:n_aug]
elif not noaug:
    n_aug = min(len(X_aug), len(y_train))
    X_train = X_train + X_aug[:n_aug]
    y_train = np.concatenate([y_train, y_aug[:n_aug]], axis=0)
    positions_train = positions_train + positions_aug[:n_aug]

# Fine-tuning the model
t0 = time.time()
ms = tf.distribute.MirroredStrategy()
t = text.Transformer(model_name, maxlen=MAX_LEN, class_names=['no','yes'])
trn = t.preprocess_train(X_train, y_train)
val = t.preprocess_test(X_val, y_val)

for i,(start,end) in enumerate(positions_train):
    trn.x[i,2,:] = 0
    trn.x[i,2,start+1:end+1] = 1 #first token is for CLS
for i,(start,end) in enumerate(positions_val):
    val.x[i,2,:] = 0
    val.x[i,2,start+1:end+1] = 1 #first token is for CLS

from transformers import TFAutoModel, AutoTokenizer
from tensorflow.keras.models import Model

class EpitopeModel(Model):
    def __init__(self, model_name, unfreeze_plm=False):
        super(EpitopeModel, self).__init__()

        self.pLM = TFAutoModel.from_pretrained(model_name)
        # Optionally freeze pretrained language model parameters so they are not updated during fine-tuning
        if not unfreeze_plm:
            # This disables training for all variables of the pretrained model
            self.pLM.trainable = False
        self.linear1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.linear2 = tf.keras.layers.Dense(32, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        #self.linear3 = tf.keras.layers.Dense(2, activation='sigmoid') #logits
        self.linear3 = tf.keras.layers.Dense(2)
        # some says do not use sigmoid and softmax before cross-entropy loss
        #self.act = tf.keras.layers.Activation('softmax') #prob

    def call(self, inputs, training=None):
        token = inputs[:,0,:]
        mask = inputs[:,1,:]
        positions = inputs[:,2,:]
        
        outputs = self.pLM(token, mask)
        outputs = outputs[0]
        output = tf.einsum('ijk,ij->ik', outputs, tf.cast(positions,tf.float32))
        # mean pooling
        factor = tf.reduce_sum(tf.cast(positions,tf.float32),1,keepdims=True)
        output = output / factor
        output = self.linear1(output)
        output = self.dropout1(output)
        output = self.linear2(output)
        output = self.dropout2(output)
        #return self.act(self.linear3(output)) #prob
        return self.linear3(output) #logits

    def train_step(self, inputs):
        inputs, labels = inputs
        with tf.GradientTape() as tape:
            pred = self(inputs)
            loss = self.compiled_loss(labels,pred,regularization_losses=self.losses)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        self.compiled_metrics.update_state(labels, pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        inputs, labels = inputs
        pred = self(inputs)
        self.compiled_loss(labels,pred,regularization_losses=self.losses)
        self.compiled_metrics.update_state(labels, pred)
        return {m.name: m.result() for m in self.metrics}
            

model = EpitopeModel(model_name, unfreeze_plm=args.unfreeze_plm)
if not args.predict:
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001, weight_decay=0.01)
    # Use focal loss with from_logits=True because the model returns raw logits.
    loss_fn = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, gamma=2, from_logits=True)
    #model.compile(optimizer=opt, loss='binary_crossentropy')
    model.compile(optimizer=opt, loss=loss_fn, metrics=['AUC'])
    #results = model(trn.x[:10])
    #es = tf.keras.callbacks.EarlyStopping(patience=1, verbose=1, restore_best_weights=True)
    #lr = tf.keras.callbacks.ReduceLROnPlateau(patience=args.patience)
    with ms.scope():
        #model = t.get_classifier()
        if args.resume:
            weight_file = os.listdir(weight_dir)[-1]
            model(trn.x[:4,:,:])
            model.load_weights(f'{weight_dir}/{weight_file}')
        else:
            if os.path.exists(weight_dir):
                os.system(f'rm {weight_dir}/*')
    #model.fit(trn.x, trn.y, epochs=args.epoch, batch_size=args.batch, validation_data=(val.x, val.y), callbacks=[es, lr])
    learner = ktrain.get_learner(model, train_data=(trn.x,trn.y), val_data=(val.x,val.y), batch_size=args.batch) 
    history = learner.autofit(args.lr, args.epoch, early_stopping=2, reduce_on_plateau=args.patience, checkpoint_folder=weight_dir)
    if not args.model_path:
        args.model_path = f'{weight_dir}/test'
    training_time = time.time()-t0
    print(f'Fine-tuning time: {time.time()-t0}, model: {args.model}, method: {args.method}, batch_size: {args.batch}, lr: {args.lr}', file = open('finetuning_time.txt', 'a'))
    print(f'Fine-tuning time: {time.time()-t0}')
    sub_history = {key: history.history.get(key) for key in ['loss', 'auc', 'val_loss', 'val_auc']}
    df_loss = pd.DataFrame(sub_history)
    os.system('mkdir -p loss_stat')
    df_loss.to_csv(f'loss_stat/{weight_tag}.csv')
else:
    if not args.model_path:
        args.model_path = f'{weight_dir}/test'
    mp = os.listdir(os.path.dirname(args.model_path))
    if len(mp)==1:
        args.model_path = os.path.dirname(args.model_path) + '/' + mp[0]
    elif 'weights-10.hdf5' in mp:
        args.model_path = os.path.dirname(args.model_path) + '/' + 'weights-10.hdf5'
    else:
        args.model_path = os.path.dirname(args.model_path) + '/' + mp[-1]
    model(trn.x[:4,:,:])
    model.load_weights(args.model_path)
    training_time = 0

if args.save:
    X_test, y_test, positions_test = create_data(df_test, info, args.maxlen)
    test = t.preprocess_train(X_test, y_test)
    for i,(start,end) in enumerate(positions_test):
        test.x[i,2,:] = 0
        test.x[i,2,start+1:end+1] = 1
    # model.predict returns logits (we configured the model to return logits). Convert to probabilities
    logits = model.predict(test.x)
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    perf = evaluate(probs[:,1], y_test)
    print(perf)
    np.save(f'predictions/{args.output}_seed{args.seed}.npy', probs)
    fn = f'predictions/{args.output}.csv'
    if not os.path.exists(fn):
        with open(fn, 'w') as f:
            csv.writer(f).writerow(['acc', 'roc_auc', 'f1', 'fmax', 'threshold', 'training_time', 'seed', 'model', 'aug', 'augfile', 'small'])
    with open(fn, 'a') as f:
        csv.writer(f).writerow([perf['acc'], perf['roc_auc'], perf['f1'], perf['fmax'], perf['tmax'], training_time, args.seed, args.model, args.method, args.augfile or '', args.small])

# remove weights
mp = os.listdir(os.path.dirname(args.model_path))
if len(mp)==1:
    pass
elif 'weights-10.hdf5' in mp:
    for i in mp:
        if i != 'weights-10.hdf5':
            os.system(f'rm {os.path.dirname(args.model_path)}/{i}')
else:
    for i in mp:
        if i != mp[-1]:
            os.system(f'rm {os.path.dirname(args.model_path)}/{i}')
