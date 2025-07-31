# ProtAug
ProtAug: Utilizing Self-Supervised Protein Language Models for Effective Protein Sequence Augmentation

## Requirements
* python=3.12
* pandas
* numpy
* tqdm
* tensorflow=2.19.0
* transformers
* ktrain

## Models
` DistilProtBert: yarongef/DistilProtBert
` prot_bert_bfd: Rostlab/prot_bert_bfd
` Esm2: facebook/esm2_t33_650M_UR50D

## Run

Single batch: (remove `--test` to run all data)
```bash
python data-augmentation.py
python MLMgenerator.py
python ProtGPT2generator.py
python extract-embeddings.py
python meta-classification.py
python evaluate.py
```

Multiple batch: (remove `--test` to run all data)
```
python MLMgenerator.py --model Esm2 --rho 0.55 --num_chunk 5 --chunk_id [from 0-4]
python df_merge.py Esm2_MLM_rho0.55 5
```

## Quality Assess

```bash
python count_diff.py
python cal_rep_ratio.py
python cal_shannon.py
```
