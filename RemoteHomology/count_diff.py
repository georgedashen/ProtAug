import sys
import pandas as pd
import numpy as np

seq1 = sys.argv[1]
seq2 = sys.argv[2]

q = pd.read_csv(seq1)
p = pd.read_csv(seq2)
q = q['sequences'].tolist()
p = p['sequences'].tolist()

if len(sys.argv) == 4:
    q = q[:int(sys.argv[3])]
    p = p[:int(sys.argv[3])]

def count_diff(q,p):
    d = []
    for i in range(len(q)):
        d.append(sum(1 for a,b in zip(q[i],p[i]) if a!=b))
    return d

diff_stat = count_diff(q,p)

len_stat = []
for i in range(len(q)):
    len_stat.append(len(q[i]))

ratio_stat = np.array(diff_stat) / np.array(len_stat)

df = pd.DataFrame({'diff':np.array(diff_stat),'len':np.array(len_stat),'ratio':ratio_stat})

print(df.describe())
