import sys
import pandas as pd
import numpy as np

seq1 = sys.argv[1]

q = pd.read_csv(seq1)
q = q['sequences'].tolist()

def calculate_ratio(seq):
    aas = ' '.join(q[i]).split() 
    rep_count = sum((aa == aas[i-1] and aa == aas[i-2]) for i,aa in enumerate(aas[2:]))
    return rep_count / len(aas)

ratio_stat = []
for i in range(len(q)):
    ratio_stat.append(calculate_ratio(q[i]))

print(np.mean(ratio_stat))
