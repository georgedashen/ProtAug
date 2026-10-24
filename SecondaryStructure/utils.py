import ktrain
from ktrain import text
from tqdm import tqdm
import os
import numpy as np
import tensorflow as tf

#### CREATE DATA FROM AUGMENTED SEQS
# this function creates data by centering the peptide in a windowed segment
def create_data(df_seq, df_info, maxlen=500):
    #df_seq is the train/val/test seq dataframe that contains only seq_id and amino acid seq
    #df_info is the original dataframe contains start/end pos and target label
    protein_list = df_seq['parent_protein_id'].tolist()
    df_seq.set_index('parent_protein_id', inplace=True)
    parent_protein_id = []
    peptide_seq = []
    target = []
    idx = []
    for i in range(len(df_info)):
        if df_info['parent_protein_id'][i] in protein_list:
            parent_protein_id.append(df_info['parent_protein_id'][i])
            peptide_len = len(df_info['peptide_seq'][i])
            protein_len = len(df_info['protein_seq'][i])
            start = df_info['start_position'][i] - (maxlen-peptide_len)//2 - 1
            end = start + maxlen
            peptide_start = (maxlen-peptide_len)//2
            peptide_end = peptide_start + peptide_len
            if start < 0:
                start = 0
                end = maxlen #can run when max_len > protein_len
                peptide_start = df_info['start_position'][i] - 1
                peptide_end = df_info['end_position'][i]
            elif end > protein_len:
                end = protein_len
                if protein_len > maxlen:
                    peptide_end = maxlen - (protein_len-df_info['end_position'][i])
                    peptide_start = peptide_end - peptide_len
                    start = end - maxlen

            peptide_seq.append(' '.join(df_seq.loc[df_info['parent_protein_id'][i],'protein_seq'][start:end]))
            idx.append((peptide_start,peptide_end))
            target.append(df_info['target'][i])

    return peptide_seq, np.array(target, dtype=int), idx


# this function combined the windowed segment for the peptide and the peptide itself
def create_data2(df_seq, df_info, model='prot_bert_bfd', maxlen=480):
    #df_seq is the train/val/test seq dataframe that contains only seq_id and amino acid seq
    #df_info is the original dataframe contains start/end pos and target label
    protein_list = df_seq['parent_protein_id'].tolist()
    df_seq.set_index('parent_protein_id', inplace=True)
    parent_protein_id = []
    peptide_seq = []
    target = []
    idx = []
    for i in range(len(df_info)):
        if df_info['parent_protein_id'][i] in protein_list:
            parent_protein_id.append(df_info['parent_protein_id'][i])
            peptide_len = len(df_info['peptide_seq'][i])
            protein_len = len(df_info['protein_seq'][i])
            start = df_info['start_position'][i] - (maxlen-peptide_len)//2 - 1
            end = start + maxlen
            peptide_start = (maxlen-peptide_len)//2
            peptide_end = peptide_start + peptide_len
            if start < 0:
                start = 0
                end = maxlen #can run when max_len > protein_len
                peptide_start = df_info['start_position'][i] - 1
                peptide_end = df_info['end_position'][i]
            elif end > protein_len:
                end = protein_len
                if protein_len > maxlen:
                    peptide_end = maxlen - (protein_len-df_info['end_position'][i])
                    peptide_start = peptide_end - peptide_len
                    start = end - maxlen

            new_seq = ' '.join(df_seq.loc[df_info['parent_protein_id'][i],'protein_seq'][start:end])
            if model == 'Esm2':
                peptide_seq.append(f"{' '.join(df_info['peptide_seq'][i])} <eos> {new_seq}")
            else:
                peptide_seq.append(f"{' '.join(df_info['peptide_seq'][i])} [SEP] {new_seq}")
            idx.append((peptide_start,peptide_end))
            target.append(df_info['target'][i])

    return peptide_seq, np.array(target, dtype=int), idx


#### EMBEDDINGS EXTRACTION #### 
def save_numpy(path, file):
  with open(path, 'wb') as f:
    np.save(f, file)

def get_model(model_name, model_path, output, dataset):
    #mask_names = ['target']
    t = text.Transformer(model_name, maxlen=MAX_LEN, class_names=['0','1'])
    ## for ktrain to adapt Esm, manually edit ktrain/text/preprocess to bypass token_type_ids length consistency check such as 'token_type_ids = attention_mask
    if dataset == 'train':
        ds = t.preprocess_train(X_train, y_train)
    elif dataset == 'val':
        ds = t.preprocess_train(X_val, y_val)
    elif dataset == 'test':
        ds = t.preprocess_train(X_test, y_test)
    elif dataset == 'aug' and not args.noaug:
        ds = t.preprocess_train(X_aug, y_aug)
    
    model = t.get_classifier()
    if len(model_path) > 0:
        model.load_weights(model_path)
    learner = ktrain.get_learner(model, batch_size=args.batch, train_data=ds)
    predictor = ktrain.get_predictor(learner.model, preproc=t)
    full_model_name = f'weights/{args.output}-full-model'
    learner.model.save_pretrained(full_model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return TFAutoModel.from_pretrained(full_model_name), tokenizer, ds


def get_embeddings_pep(X, position, model, tokenizer, file_path, batch_size):
    batch_num = len(X)
    with NpyAppendArray(file_path) as npaa:
        for i in tqdm(range(batch_num)):
            with tf.device('/device:GPU:0'):
                mean_embeddings = []
                outputs = model(X[i][0][:,0,:], X[i][0][:,1,:]) #input_ids, attention_mask
                for j, (start, end) in enumerate(position[i*batch_size:(i+1)*batch_size]):
                    mean_embeddings.append(np.mean(outputs[0][j,start+1:end+1,:].numpy(),axis=0))#not from the CLS
                npaa.append(np.stack(mean_embeddings))

def get_embeddings(X, position, model, tokenizer, file_path, batch_size):
    batch_num = len(X)
    with NpyAppendArray(file_path) as npaa:
        for i in tqdm(range(batch_num)):
            with tf.device('/device:GPU:0'):
                outputs = model(X[i][0][:,0,:], X[i][0][:,1,:]) #input_ids, attention_mask
                outputs = outputs[0][:,0,:].numpy()
                npaa.append(outputs)


def get_embeddings_noBatch(X, model, tokenizer, file_path):
    with NpyAppendArray(file_path, delete_if_exists=True) as npaa:
        for i in tqdm(range(len(X))):
            with tf.device('/device:GPU:0'):
                emb = np.array([]).tolist()
                input_ids = tf.constant(tokenizer.encode(X[i]))[None, :]
                outputs = model(input_ids)
                last_hidden_states = outputs[0]
                npaa.append(last_hidden_states.numpy()[0, 0, :])

from transformers import TFAutoModel, AutoTokenizer
from tensorflow.keras.models import Model

class EpitopeModel(Model):
    def __init__(self, model_name):
        super(EpitopeModel, self).__init__()
        self.pLM = TFAutoModel.from_pretrained(model_name)
        self.linear1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.linear2 = tf.keras.layers.Dense(32, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.linear3 = tf.keras.layers.Dense(2, activation='sigmoid')

    def call(self, inputs, training=None):
        token = inputs[:,0,:]
        mask = inputs[:,1,:]
        positions = inputs[:,2,:]
        outputs = self.pLM(token, mask)
        outputs = outputs[0]
        output = tf.einsum('ijk,ij->ik', outputs, tf.cast(positions,tf.float32))
        factor = tf.reduce_sum(tf.cast(positions,tf.float32),1,keepdims=True)
        output = output / factor
        output = self.linear1(output)
        output = self.dropout1(output)
        output = self.linear2(output)
        output = self.dropout2(output)
        return self.linear3(output)

    def train_step(self, inputs):
        inputs, labels = inputs
        with tf.GradientTape() as tape:
            logits = self(inputs)
            loss = self.compiled_loss(labels,logits,regularization_losses=self.losses)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        inputs, labels = inputs
        logits = self(inputs)
        self.compiled_loss(labels,logits,regularization_losses=self.losses)
        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}
