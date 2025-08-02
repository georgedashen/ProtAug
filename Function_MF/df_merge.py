import sys
import pandas as pd

pattern = sys.argv[1]
num_chunk = int(sys.argv[2])

li = []

for i in range(num_chunk):
    df = pd.read_csv(f'data/mf-augmented_training_{pattern}_chunk{i}.csv', index_col=None)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
frame.to_csv(f'data/mf-augmented_training_{pattern}.csv', index=False)

