import sys
import pandas as pd
import numpy as np
from collections import Counter
from math import log

def shannon_diversity(seq):
    aa_counts = Counter(seq)
    total_aa = sum(aa_counts.values())
    shannon = 0.0
    for count in aa_counts.values():
        prop = count/total_aa
        shannon -= prop * log(prop)
    return shannon

seq1 = sys.argv[1]

q = pd.read_csv(seq1)
q = q.iloc[:,1].tolist()

shannon_stat = []
for i in range(len(q)):
    shannon_stat.append(shannon_diversity(q[i]))

print(np.mean(shannon_stat))
