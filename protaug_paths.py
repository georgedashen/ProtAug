"""Resolve pLM checkpoints shipped with this repository or on Hugging Face."""

import os

# Local folder name -> public Hugging Face id used in the paper.
HF_IDS = {
    "Esm2": "facebook/esm2_t33_650M_UR50D",
    "prot_bert_bfd": "Rostlab/prot_bert_bfd",
    "DistilProtBert": "yarongef/DistilProtBert",
    "ProtGPT2": "nferruz/ProtGPT2",
}


def resolve_model(name, models_dir=None):
    """Return Models/<name> when that folder has a config, otherwise a Hugging Face id.

    `name` may already be a filesystem path or a Hugging Face repo id.
    """
    if not name:
        raise ValueError("model name is empty")
    if os.path.isdir(name) and os.path.isfile(os.path.join(name, "config.json")):
        return name
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models")
    local = os.path.join(models_dir, name)
    if os.path.isdir(local) and os.path.isfile(os.path.join(local, "config.json")):
        return local
    return HF_IDS.get(name, name)
