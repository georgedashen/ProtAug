import os
import numpy as np
from tqdm import tqdm

os.environ["TF_USE_LEGACY_KERAS"]= '1' #use legacy keras to avoid error in lora layer
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

import ktrain
from ktrain import text
import tensorflow as tf

def configure_tensorflow_gpu(memory_limit_mb=None):
    """Avoid reserving all GPU memory so multiple jobs can share one card."""
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

#### LOAD ONTOLOGY ####
def get_ancestors(ontology, term):
  list_of_terms = []
  list_of_terms.append(term)
  data = []
  
  while len(list_of_terms) > 0:
    new_term = list_of_terms.pop(0)

    if new_term not in ontology:
      break
    data.append(new_term)
    for parent_term in ontology[new_term]['parents']:
      if parent_term in ontology:
        list_of_terms.append(parent_term)
  
  return data

def generate_ontology(file, specific_space=False, name_specific_space=''):
  ontology = {}
  gene = {}
  flag = False
  with open(file) as f:
    for line in f.readlines():
      line = line.replace('\n','')
      if line == '[Term]':
        if 'id' in gene:
          ontology[gene['id']] = gene
        gene = {}
        gene['parents'], gene['alt_ids'] = [], []
        flag = True
        
      elif line == '[Typedef]':
        flag = False
      
      else:
        if not flag:
          continue
        items = line.split(': ')
        if items[0] == 'id':
          gene['id'] = items[1]
        elif items[0] == 'alt_id':
          gene['alt_ids'].append(items[1])
        elif items[0] == 'namespace':
          if specific_space:
            if name_specific_space == items[1]:
              gene['namespace'] = items[1]
            else:
              gene = {}
              flag = False
          else:
            gene['namespace'] = items[1]
        elif items[0] == 'is_a':
          gene['parents'].append(items[1].split(' ! ')[0])
        elif items[0] == 'name':
          gene['name'] = items[1]
        elif items[0] == 'is_obsolete':
          gene = {}
          flag = False
    
    key_list = list(ontology.keys())
    for key in key_list:
      ontology[key]['ancestors'] = get_ancestors(ontology, key)
      for alt_ids in ontology[key]['alt_ids']:
        ontology[alt_ids] = ontology[key]
    
    for key, value in ontology.items():
      if 'children' not in value:
        value['children'] = []
      for p_id in value['parents']:
        if p_id in ontology:
          if 'children' not in ontology[p_id]:
            ontology[p_id]['children'] = []
          ontology[p_id]['children'].append(key)
    
  return ontology

def get_and_print_children(ontology, term):
  children = {}
  if term in ontology:
    for i in ontology[term]['children']:
      children[i] = ontology[i]
      print(i, ontology[i]['name'])
  return children



#### GENERATE DATA BASED ON SLIDING WINDOW TECHNIQUE ####
def generate_data(df, subseq=100, overlap=0):
  X = []
  y = []
  positions = []
  sequences = df.iloc[:, 1].values

  for i in tqdm(range(len(sequences))):

    len_seq = int(np.ceil(len(sequences[i]) / subseq))

    for idx in range(len_seq):
      if idx != len_seq - 1:
        X.append(' '.join(list(sequences[i][idx * subseq : (idx + 1) * subseq])))
      else:
        X.append(' '.join(list(sequences[i][idx * subseq : ])))
      positions.append(i)
      y.append(df.iloc[i, 2:])

    if overlap > 0:
      init = overlap
      while True:
        if init + subseq >= len(sequences[i]):
          break
        X.append(' '.join(list(sequences[i][init : init + subseq])))
        positions.append(i)
        y.append(df.iloc[i, 2:])
        init += subseq
  
  return X, np.array(y, dtype=int), positions



#### EMBEDDINGS EXTRACTION #### 
def save_numpy(path, file):
  with open(path, 'wb') as f:
    np.save(f, file)

import math
class LoraLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        original_layer,
        rank=16,
        alpha=32,
        trainable=False,
        **kwargs,
    ):
        original_layer_config = original_layer.get_config()
        name = original_layer_config["name"]

        kwargs.pop("name", None)

        super().__init__(name=name, trainable=trainable, **kwargs)

        self.rank = rank
        self.alpha = alpha

        self._scale = alpha / rank

        self.original_layer = original_layer
        self.original_layer.trainable = False

        self.A = tf.keras.layers.Dense(
            units=rank,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=math.sqrt(5), mode="fan_in", distribution="uniform"
            ),
            trainable=trainable,
            name=f"lora_A",
        )
        self.B = tf.keras.layers.EinsumDense(
            equation= 'abc,cde->abde',
            output_shape=original_layer_config["units"],
            kernel_initializer="zeros",
            trainable=trainable,
            name=f"lora_B",
        )

    def call(self, inputs):
        original_output = self.original_layer(inputs)
        if self.trainable:
            lora_output = self.B(self.A(inputs)) * self._scale
            return original_output + lora_output

        return original_output


def get_model(model_name, model_path, output, dataset, lora=False):
    t = text.Transformer(model_name, maxlen=MAX_LEN, class_names=ontologies_names)
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
        if lora:
            if os.path.basename(model_name) == "Esm2":
                layers = model.esm.encoder.layer
            elif os.path.basename(model_name) in ['DistilProtBert','prot_bert_bfd']:
                layers = model.bert.encoder.layer
            RANK = 16
            ALPHA = 32
            for layer in layers:
                if os.path.basename(model_name) == "Esm2":
                    qd = layer.attention.self.query
                    vd = layer.attention.self.value
                elif os.path.basename(model_name) in ['DistilProtBert','prot_bert_bfd']:
                    qd = layer.attention.self_attention.query
                    vd = layer.attention.self_attention.value
                # Change query dense layer.
                qd = LoraLayer(
                    qd,
                    rank=RANK,
                    alpha=ALPHA,
                    trainable=True,
                )
                # Change value dense layer.
                vd = LoraLayer(
                    vd,
                    rank=RANK,
                    alpha=ALPHA,
                    trainable=True,
                )
        model.load_weights(model_path)

    learner = ktrain.get_learner(model, batch_size=args.batch, train_data=ds)
    predictor = ktrain.get_predictor(learner.model, preproc=t)
    full_model_name = f'weights/{args.output}-full-model'
    learner.model.save_pretrained(full_model_name)
    #tokenizer = AutoTokenizer.from_pretrained(model_name)

    return TFAutoModel.from_pretrained(full_model_name), learner, ds


def get_embeddings(X, model, tokenizer, file_path, gpu=0):
    os.system(f'mkdir -p {os.path.dirname(file_path)}') #create filepath if not exists
    batch_num = len(X)
    if os.path.exists(file_path):
        print(f'File {file_path} already exists. Remove existing file.')
        os.remove(file_path)

    with NpyAppendArray(file_path) as npaa:
        for i in tqdm(range(batch_num)):
            with tf.device(f'/device:GPU:{gpu}'):
                outputs = model(X[i][0][:,0,:], X[i][0][:,1,:]) #input_ids, attention_mask
                outputs = outputs[0][:,0,:].numpy() # last_hidden_states, CLS token
                npaa.append(outputs)

def get_embeddings_noBatch(X, model, tokenizer, file_path, gpu=0):
    with NpyAppendArray(file_path, delete_if_exists=True) as npaa:
        for i in tqdm(range(len(X))):
            with tf.device(f'/device:GPU:{gpu}'):
                emb = np.array([]).tolist()
                input_ids = tf.constant(tokenizer.encode(X[i]))[None, :]
                outputs = model(input_ids)
                last_hidden_states = outputs[0]
                npaa.append(last_hidden_states.numpy()[0, 0, :])


#### COMBINE EMBEDDINGS ####
def protein_embedding(X, y, pos):
  n_X = []
  last_pos = pos[0]
  cur_emb = []
  n_y = [y[0]]
    
  for i in range(len(pos)):
    cur_pos = pos[i]
    if last_pos == cur_pos:
      cur_emb.append(X[i])
    else:
      n_X.append(np.mean(np.array(cur_emb), axis=0))
      last_pos = cur_pos
      cur_emb = [X[i]]
      n_y.append(y[i])

  n_X.append(np.mean(np.array(cur_emb), axis=0))
    
  return np.array(n_X), np.array(n_y)

