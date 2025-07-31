# ProtAug
ProtAug: Utilizing Self-Supervised Protein Language Models for Effective Protein Sequence Augmentation

## Requirements
` python=3.12
` pandas
` numpy
` tqdm
` tensorflow=2.19.0
` transformers
` ktrain

## Models
` DistilProtBert: yarongef/DistilProtBert
` prot_bert_bfd: Rostlab/prot_bert_bfd
` Esm2: facebook/esm2_t33_650M_UR50D

## Run

```bash
python data-augmentation.py
python MLMgenerator.py
python ProtGPT2generator.py
python extract-embeddings.py
python meta-classification.py
python evaluate.py
```

