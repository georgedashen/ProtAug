# run filterUniProt.sh first, resulting diamond blastp file: training_uniprot_tvt.txt
# ensure that you have random augmentation training data
# if homologous seqs are not found, it will use random augmentation data instead

import os
from Bio import SeqIO
from tqdm import tqdm
import pandas as pd
import argparse

# python homology.py --rho 0.3 --output data/aug_homology_rho0.3.csv --random data/aug_Identity_k0.4.csv
# 0.023, 0.12, 0.4, 0.77, 1.7

example_usage = 'python homology.py --rho 0.3 --output data/mf-augmented_training_homology_rho0.3.csv --random data/mf-augmented_training_Identity_k0.4.csv'
parser = argparse.ArgumentParser(description='homology search across UniProt_SwissProt', epilog=example_usage)
parser.add_argument('--input', type=str, default='data/training_uniprot_tvt.txt', help='diamond blastp result file')
parser.add_argument('--uniprot', type=str, default='data/uniprot_filt_tvt.fasta', help='uniprot swissprot fasta file')
parser.add_argument('--data', type=str, default='data/train_split.csv', help='random augmentation training data file')
parser.add_argument('--random', type=str, default='', help='random augmentation training data file')
parser.add_argument('--no_random', action='store_true', help='do not use random augmentation data')
parser.add_argument('--rho', type=float, default=0.7, help='diversity level, homology with max (1-rho) threshold')
parser.add_argument('--higher', action='store_true', help='use higher identity threshold for homologous search')
parser.add_argument('--output', type=str, default='')
parser.add_argument('--test', action='store_true', help='test mode with limited data')
args = parser.parse_args()

df_match = pd.read_csv(args.input, sep='\t', header=None)
df_match.columns = ['query_id', 'subject_id', 'perc_identity', 'alignment_length', 'mismatches', 'gap_opens', 'q_start', 'q_end', 's_start', 's_end', 'evalue', 'bit_score']

# read uniprot fasta into dict
uniprot_dict = {}
for record in SeqIO.parse(args.uniprot, "fasta"):
    uniprot_dict[record.id] = str(record.seq)

df = pd.read_csv(args.data)

# read random augmentation data
if not args.random:
    args.random = f'data/aug_Identity_rho{str(args.rho)}.csv'

if not args.no_random:
    random = pd.read_csv(args.random)   
    # check if the protein same between df and random
    assert all(df['seq_id'] == random['seq_id']), "Proteins not match between original and random augmentation data."

if args.test:
    df = df.head(100)
    if not args.no_random:
        random = random.head(100)

# filter by evalue < 0.001 and perc_identity < args.ident
if args.higher:
    df_match = df_match[(df_match['evalue'] < 0.001) & (df_match['perc_identity'] > (1-args.rho) * 100)]
else:
    df_match = df_match[(df_match['evalue'] < 0.001) & (df_match['perc_identity'] < (1-args.rho) * 100)]

# for each protein in df, find id in df_match, get subject_id with highest perc_identity
# get sequence from uniprot_dict
count = 0
df_final = {'seq_id': [], 'protein_sequence': []}
ident = []
for term in df.columns[2:].values.tolist():
    df_final[term] = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    protein_id = row['seq_id']
    matches = df_match[df_match['query_id'] == protein_id]
    if len(matches) == 0:
        if args.no_random:
            seq = row['protein_sequence']
        else:
            # no homologous found, use random augmentation
            seq = random.loc[random['seq_id'] == protein_id, 'protein_sequence'].values[0]
    else:
        best_match = matches.loc[matches['perc_identity'].idxmax()]
        subject_id = best_match['subject_id']
        seq = uniprot_dict.get(subject_id, None)
        if seq is None:
            if args.no_random:
                seq = row['protein_sequence']
            else:
                # subject_id not found in uniprot_dict, use random augmentation
                seq = random.loc[random['seq_id'] == protein_id, 'protein_sequence'].values[0]
        else:
            count += 1
            ident.append(best_match['perc_identity'])
    df_final['seq_id'].append(protein_id)
    df_final['protein_sequence'].append(seq)
    for key, value in zip(df.columns[2:].values.tolist(), df.iloc[idx, 2:].values.tolist()):
        df_final[key].append(value)

# save augmented data
flag = ''
if args.higher:
    flag = '_higher'
if not args.output:
    args.output = f'data/aug_homology_rho{str(args.rho)}{flag}.csv'
generated_df = pd.DataFrame(df_final)
generated_df.to_csv(args.output, index=False)
print(f'Number of homologous sequences found: {count} out of {len(df)}')
print(f'Average percent identity of homologous sequences: {sum(ident)/len(ident) if len(ident) > 0 else 0}')

if args.test:
    os.system(f'python count_diff.py {args.data} {args.output} 100')
else:
    os.system(f'python count_diff.py {args.data} {args.output}')