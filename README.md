# ProtAug
ProtAug: Utilizing Self-Supervised Protein Language Models for Effective Protein Sequence Augmentation

## Requirements
* python=3.12
* pandas
* numpy
* tqdm
* tensorflow=2.19.0
* pytorch=2.7.1
* transformers
* ktrain

## Models
* **DistilProtBert**: yarongef/DistilProtBert
* **prot_bert_bfd**: Rostlab/prot_bert_bfd
* **Esm2**: facebook/esm2_t33_650M_UR50D

## Run

### Baseline
```
python data-augmentation.py
```

### ProtAug_Esm
Single batch: (remove `--test` to run all data)
```bash
python MLMgenerator.py
```

Multiple batch: (remove `--test` to run all data)
```
python MLMgenerator.py --model Esm2 --rho 0.55 --num_chunk 5 --chunk_id [from 0-4] --test
python df_merge.py Esm2_MLM_rho0.55 5
```

### ProtAug_GPT
Single batch: (remove `--test` to run all data)
```bash
python ProtGPT2generator.py
```

Multiple batch: (remove `--test` to run all data)
```
python ProtGPT2generator.py --model ProtGPT2 --rho 0.55 --num_chunk 5 --chunk_id [from 0-4]
python df_merge.py Esm2_MLM_rho0.55 5
```

## Quality Assess

```bash
python count_diff.py
python cal_rep_ratio.py
python cal_shannon.py
```

### Training and Evaluation
```bash
python extract-embeddings.py
python meta-classification.py
python evaluate.py
```
