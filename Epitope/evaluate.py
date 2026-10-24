import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from collections import Counter
import math

def evaluate_multi(predictions, ground_truth):
  precisions = []
  recalls = []
  f1s = []
  thresholds = []
  f1_max_value = -1
  f1_max_threshold = -1
  len_gt = ground_truth.shape[1]

  for i in tqdm(range(1, 101)):
    threshold = i/100
    p, r = 0, 0
    number_of_proteins = 0

    for idx_protein in range(len(predictions)):
      len_pred = sum(predictions[idx_protein,:])
      tp = sum(predictions[idx_protein,:]==ground_truth[idx_protein,:])
      if len_pred > 0:
        number_of_proteins += 1
        p += tp / len_pred
      r += tp / len_gt

    if number_of_proteins > 0:
      threshold_p = p / number_of_proteins
    else:
      threshold_p = 0

    threshold_r = r / len(predictions)

    precisions.append(threshold_p)
    recalls.append(threshold_r)
    
    f1 = 0
    if threshold_p > 0 or threshold_r > 0:
      f1 = (2 * threshold_p * threshold_r) / (threshold_p + threshold_r)
    
    f1s.append(f1)
    thresholds.append(threshold)

    if f1 > f1_max_value:
      f1_max_value = f1
      f1_max_threshold = threshold

  print('\nFmax:', f1_max_value)
  print('F1 threshold:', f1_max_threshold)

  precisions = np.array(precisions)
  recalls = np.array(recalls)
  sorted_index = np.argsort(recalls)
  recalls = recalls[sorted_index]
  precisions = precisions[sorted_index]
  aupr = np.trapz(precisions, recalls)
  print('AuPRC:', aupr)

  new_p = []
  new_r = []
  for i in range(1, 101):
    if len(np.where(recalls >= i/100)[0]) != 0:
      idx = np.where(recalls >= i/100)[0][0]
      new_r.append(i/100)
      new_p.append(max(precisions[idx:]))

  Iaupr = np.trapz(new_p, new_r)
  print('IAuPRC:', Iaupr)
    
  perf = {'fmax':f1_max_value, 'threshold':f1_max_threshold, 'smin':0, 'aupr':aupr, 'Iaupr':Iaupr}
  return perf 

from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc, roc_auc_score
def evaluate(y_score, y_true, threshold=0.5):
    y_pred = (y_score>threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2*recall*precision/(recall+precision)
    index = np.nanargmax(f1_scores)
    fmax = f1_scores[index]
    best_threshold = thresholds[index]
    '''
    thresholds = []
    recalls = []
    precisions = []
    for i in tqdm(range(1, 101)):
        t = i/100
        thresholds.append(i/100)
        tp = sum((y_score>t).astype(int)==y_true)
        recalls.append( tp/sum(y_true) )
        precisions.append( tp/sum((y_score>t).astype(int)) )
    
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    f1_scores = 2*recalls*precisions/(recalls+precisions)
    index = np.argmax(f1_scores)
    fmax = f1_scores[index]
    best_threshold = thresholds[index]
    f1 = f1_scores[int(threshold//0.01)-1]
    '''
    perf = {'acc':acc, 'roc_auc':roc_auc, 'f1':f1, 'fmax':fmax, 'tmax':best_threshold}

    return perf
    
