Encoder checkpoints are loaded from this folder when it contains a `config.json`. Otherwise the scripts fall back to the Hugging Face ids below. ProtAug_GPT downloads `nferruz/ProtGPT2` on first use.

```bash
pip install -U huggingface_hub
huggingface-cli download facebook/esm2_t33_650M_UR50D --local-dir Models/Esm2
huggingface-cli download Rostlab/prot_bert_bfd --local-dir Models/prot_bert_bfd
huggingface-cli download yarongef/DistilProtBert --local-dir Models/DistilProtBert
```

* DistilProtBert: `yarongef/DistilProtBert`
* prot_bert_bfd: `Rostlab/prot_bert_bfd`
* Esm2: `facebook/esm2_t33_650M_UR50D` (33 layers, 650M, UR50D)
* ProtGPT2: `nferruz/ProtGPT2`
