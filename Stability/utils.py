import ktrain
from ktrain import text
import os
import tensorflow as tf


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
      if len(df.columns) > 4:
        y.append(df.iloc[i, 4])
      else:
        y.append(df.iloc[i, 2])

    if overlap > 0:
      init = overlap
      while True:
        if init + subseq >= len(sequences[i]):
          break
        X.append(' '.join(list(sequences[i][init : init + subseq])))
        positions.append(i)
        if len(df.columns) > 4:
          y.append(df.iloc[i, 4])
        else:
          y.append(df.iloc[i, 2])
        init += subseq
  
  return X, np.array(y, dtype=int), positions



#### EMBEDDINGS EXTRACTION #### 
def save_numpy(path, file):
  with open(path, 'wb') as f:
    np.save(f, file)

def get_model(model_name, model_path, output, dataset):
    t = text.Transformer(model_name, maxlen=MAX_LEN)
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


def get_embeddings(X, model, tokenizer, file_path):
    os.system(f'mkdir -p {os.path.dirname(file_path)}') #create filepath if not exists
    batch_num = len(X)
    with NpyAppendArray(file_path) as npaa:
        for i in tqdm(range(batch_num)):
            with tf.device('/device:GPU:0'):
                outputs = model(X[i][0][:,0,:], X[i][0][:,1,:]) #input_ids, attention_mask
                outputs = outputs[0][:,0,:].numpy() # last_hidden_states, CLS token
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


import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

def knn_distance_coverage(X_train, X_test, k=1, metric='euclidean'):
    """返回测试集到训练集第k近邻的距离向量（越小表示被覆盖越好）"""
    nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1)
    nn.fit(X_train)
    dist, _ = nn.kneighbors(X_test)
    return dist[:, -1]          # shape = (n_test,)

def rbf_mmd(X_train, X_test, gamma=1.0):
    """RBF核MMD^2，值≈0表示同分布"""
    from sklearn.metrics import pairwise_kernels
    K_ss = pairwise_kernels(X_train, X_train, metric='rbf', gamma=gamma)
    K_tt = pairwise_kernels(X_test,  X_test,  metric='rbf', gamma=gamma)
    K_st = pairwise_kernels(X_train, X_test,  metric='rbf', gamma=gamma)
    n, m = K_ss.shape[0], K_tt.shape[0]
    mmd2 = (K_ss.sum() / (n*n) +
            K_tt.sum() / (m*m) -
            2 * K_st.sum() / (n*m))
    return float(mmd2)

def domain_auc(X_train, X_test, clf=None):
    """训练0/1域分类器并返回AUC（0.5表示分不清）"""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    if clf is None:
        clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    X = np.vstack([X_train, X_test])
    y = np.array([0]*len(X_train) + [1]*len(X_test))
    preds = np.zeros(len(y))
    for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y):
        clf.fit(X[tr], y[tr])
        preds[va] = clf.predict_proba(X[va])[:, 1]
    return float(roc_auc_score(y, preds))

