# ProtAug
ProtAug: Utilizing Self-Supervised Protein Language Models for Effective Protein Sequence Augmentation

## Requirements
Under CUDA 12.8 with NVIDIA RTX3090 GPU with 24G GPU

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
We would provide pipeline for fasta input in the future. For now this procedure is dataset-specific. Take augmentation in Function_MF as an example.
```bash
cd Function_MF
```

### Baseline
To generate the Random and PAM-guided substitutiona at 10% variation, use the following: (remove `--test` to run all data)
```
python data-augmentation.py --method Random --k 0.023 --test
python data-augmentation.py --method PAM1 --k 10 --test
```

You can use the `count_diff.py` to evaluate variation level: (remove `--test` to run all data)
```bash
python count_diff.py --d1 data/mf-training.csv --d2 data/mf-augmented_training_Random_k0.023.csv --test
```

### ProtAug_Esm
Single batch: (remove `--test` to run all data)
```bash
python MLMgenerator.py --rho 0.55 --test
```

Split batches and merge: (remove `--test` to run all data)
```
python MLMgenerator.py --model Esm2 --rho 0.55 --num_chunk 5 --chunk_id [from 0-4] --test
python df_merge.py Esm2_MLM_rho0.55 5
```

### ProtAug_GPT
Single batch: (remove `--test` to run all data)
```bash
python ProtGPT2generator.py --model ProtGPT2 --rho 0.55 --test
```

Split batches and merge: (remove `--test` to run all data)
```
python ProtGPT2generator.py --model ProtGPT2 --rho 0.55 --num_chunk 5 --chunk_id [from 0-4] --test
python df_merge.py ProtGPT2_rho0.55 5
```

## Quality Assess

```bash
python cal_rep_ratio.py
python cal_shannon.py
```

### Training and Evaluation
```bash
python extract-embeddings.py
python meta-classification.py
python evaluate.py
```
