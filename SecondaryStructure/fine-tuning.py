#!/usr/bin/env python3
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
from protaug_paths import resolve_model

"""
Protein secondary structure prediction script
Using pre-trained model and shallow neural network
"""

import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import AutoModel, AutoTokenizer
import argparse
import random
import sys
from tqdm import tqdm
import time
import csv
import ast
import warnings

# supress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

class ProteinSSDataset(Dataset):
    """Protein secondary structure dataset"""
    
    def __init__(self, csv_file, tokenizer, max_length=1650, test_mode=False):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Test mode: use only first 100 samples
        if test_mode:
            self.data = self.data.head(100)
            print(f"Test mode: using {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        protein_id = row.iloc[0]
        sequence = row.iloc[1]
        ss_labels = row.iloc[2]
        
        # Handle secondary structure labels (assume string format list)
        if isinstance(ss_labels, str):
            try:
                ss_labels = ast.literal_eval(ss_labels)
            except:
                # If space-separated string
                ss_labels = [int(x) for x in ss_labels.strip('[]').split()]
        
        # Tokenize the sequence
        inputs = self.tokenizer(
            ' '.join(list(re.sub(r"[UZOB]","X",sequence))),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Align labels with tokenized sequence (considering special tokens)
        #tokenized_sequence = self.tokenizer.tokenize(' '.join(sequence))
        aligned_labels = self._align_labels(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0)), ss_labels)
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(aligned_labels, dtype=torch.long),
            'protein_id': protein_id,
            'sequence_length': len(sequence)
        }
    
    def _align_labels(self, tokenized_seq, original_labels):
        """Align labels with tokenized sequence"""
        aligned_labels = []
        pos = 0
        
        for token in tokenized_seq:
            if token in ['<cls>', '<pad>', '<eos>', '<unk>','[PAD]','[CLS]','[SEP]','[UNK]']:
                # Special token, use -1 to ignore
                aligned_labels.append(-1)
            elif token in ['<mask>','[MASK]']:
                aligned_labels.append(-1)
            elif pos < len(original_labels):
                aligned_labels.append(original_labels[pos])
                pos += 1
            else:
                aligned_labels.append(-1)
        
        # Pad to max length
        while len(aligned_labels) < self.max_length:
            aligned_labels.append(-1)
        
        return aligned_labels[:self.max_length]

def load_plm(plm_path):
    """Load PLM; use FAEsmModel for flash_Esm2 checkpoints with modeling_faesm.py."""
    plm_path = os.path.abspath(plm_path)
    faesm_py = os.path.join(plm_path, 'modeling_faesm.py')
    if os.path.isfile(faesm_py):
        import sys
        if plm_path not in sys.path:
            sys.path.insert(0, plm_path)
        from modeling_faesm import FAEsmModel, flash_attn_installed
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        plm = FAEsmModel.from_pretrained(plm_path, torch_dtype=dtype)
        attn_cls = type(plm.encoder.layer[0].attention).__name__
        print(f'Loaded FAEsmModel from {plm_path}')
        print(f'  flash_attn_installed={flash_attn_installed}, attention={attn_cls}, dtype={dtype}')
        return plm
    plm = AutoModel.from_pretrained(plm_path, trust_remote_code=True)
    print(f'Loaded {type(plm).__name__} from {plm_path}')
    return plm


class ProteinSSPredictor(nn.Module):
    """Protein secondary structure prediction model"""

    def __init__(self, plm_name='facebook/esm2_t6_8M_UR50D', num_classes=3, unfreeze_plm=False):
        super(ProteinSSPredictor, self).__init__()

        self.plm = load_plm(plm_name)
        self.hidden_size = self.plm.config.hidden_size
        
        # Freeze PLM parameters (default)
        if not unfreeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
            print("PLM parameters frozen")
        else:
            print("PLM parameters unfrozen, fine-tuning enabled")
        
        # Shallow classifier
        plm_dtype = next(self.plm.parameters()).dtype
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        ).to(dtype=plm_dtype)
        
    def forward(self, input_ids, attention_mask):
        # Get PLM output
        outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Classification prediction
        logits = self.classifier(sequence_output)
        
        return logits

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        
        # Compute loss (ignore -1 labels)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=-1)
        mask = labels != -1
        
        all_preds.extend(preds[mask].cpu().numpy())
        all_labels.extend(labels[mask].cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    protein_accuracies = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            protein_ids = batch['protein_id']
            seq_lengths = batch['sequence_length']
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Compute loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
            
            # Compute accuracy for each protein
            preds = torch.argmax(logits, dim=-1)
            
            for i in range(len(protein_ids)):
                protein_mask = (labels[i] != -1)
                protein_preds = preds[i][protein_mask].cpu().numpy()
                protein_true = labels[i][protein_mask].cpu().numpy()
                
                if len(protein_true) > 0:
                    protein_acc = accuracy_score(protein_true, protein_preds)
                    protein_accuracies.append(protein_acc)
                
                all_preds.extend(protein_preds)
                all_labels.extend(protein_true)
    
    avg_protein_accuracy = np.mean(protein_accuracies) if protein_accuracies else 0
    overall_accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, overall_accuracy, avg_protein_accuracy

def main():
    example_usage = 'python fine-tuning.py --augfile data/aug_Esm2_MLM_rho0.1.csv --epochs 5 --batch_size 32 --learning_rate 1e-4 --model flash_Esm2'
    parser = argparse.ArgumentParser(description='Protein secondary structure prediction', epilog=example_usage)
    parser.add_argument('--model', type=str, default='Esm2', help='Pre-trained model name or Models/ folder')
    parser.add_argument('--model_path', type=str, default='', help='Path to load pretrained model weights')
    parser.add_argument('--method', type=str, default='noaug', help='Method name for logging', choices=['aug','noaug', 'duplicate'])
    parser.add_argument('--noaug', action='store_true', default=False, help='No data augmentation')
    parser.add_argument('--only_aug', action='store_true', default=False, help='Only use augmented data')
    parser.add_argument('--duplicate', action='store_true', default=False, help='Duplicate training data')
    parser.add_argument('--augfile', type=str, default='', help='Path to input augmented file')
    parser.add_argument('--test', action='store_true', default=False, help='Test mode (use small sample)')
    parser.add_argument('--unfreeze_plm', action='store_true', default=False, help='Unfreeze PLM parameters')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=1650, help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu card in your device to use')
    parser.add_argument('--gpu_mem', type=int, default=0, help='per-process GPU memory cap in MiB (0 = no cap)')
    parser.add_argument('--small', type=float, default=1.0, help='fraction of training sequences to use')
    parser.add_argument('--output', type=str, default='', help='Output results CSV basename under predictions/')
    parser.add_argument('--evaluate', action='store_true', default=False, help='Only evaluate the model')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpu.split(','))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.gpu_mem > 0 and torch.cuda.is_available():
        total_mib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        frac = min(1.0, args.gpu_mem / total_mib)
        torch.cuda.set_per_process_memory_fraction(frac, 0)
    
    if args.method == 'noaug':
        args.noaug = True
        args.augfile = ''
    if args.method == 'duplicate':
        args.duplicate = True
        args.noaug = True
        args.augfile = ''
    if args.method == 'aug':
        args.noaug = False
        args.duplicate = False

    aug_suffix = ''
    if args.method == 'aug' and args.augfile:
        base = os.path.splitext(os.path.basename(args.augfile))[0]
        aug_suffix = f'_{base}' if base else ''

    weight_tag = f'{args.model}-finetune-{args.method}{aug_suffix}-seed{args.seed}'
    weight_dir = f'weights/{weight_tag}'
    best_model = f'{weight_dir}/best.pth'

    plm_path = resolve_model(args.model)
    if args.output == '':
        args.output = f'{args.model}_original'

    tokenizer = AutoTokenizer.from_pretrained(plm_path, trust_remote_code=True)

    # if args.model.endswith('Esm2'):
    #     args.max_length = min(args.max_length, 1024)
    # else:
    #     args.max_length = min(args.max_length, 512)
    
    # Create dataset
    train_set = ProteinSSDataset(
        'data/train.csv', 
        tokenizer, 
        max_length=args.max_length,
        test_mode=args.test
    )
    
    val_set = ProteinSSDataset(
        'data/valid.csv', 
        tokenizer, 
        max_length=args.max_length,
        test_mode=args.test
    )

    test_set = ProteinSSDataset(
        'data/test.csv', 
        tokenizer, 
        max_length=args.max_length,
        test_mode=args.test
    )
    
    n_train = int(len(train_set.data) * args.small)
    train_set.data = train_set.data.iloc[:n_train].reset_index(drop=True)

    if args.duplicate:
        train_set.data = pd.concat([train_set.data, train_set.data], ignore_index=True)
        print(f"Duplicated training data, new size: {len(train_set.data)}")

    if not args.noaug and args.augfile:
        aug_data = pd.read_csv(args.augfile)
        if args.test:
            aug_data = aug_data.head(100)
            print(f"Test mode: using {len(aug_data)} augmented samples")
        n_aug = min(len(aug_data), len(train_set.data))
        aug_data = aug_data.iloc[:n_aug]
        if args.only_aug:
            train_set.data = aug_data.reset_index(drop=True)
        else:
            train_set.data = pd.concat([train_set.data, aug_data], ignore_index=True)
        print(f"Training size after augmentation: {len(train_set.data)}")
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    
    model = ProteinSSPredictor(
        plm_name=plm_path,
        unfreeze_plm=args.unfreeze_plm
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)


    if not args.evaluate:
        os.makedirs(weight_dir, exist_ok=True)
        
        # optimizer
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        
        print(f"Training samples: {len(train_set)}")
        print(f"Validation samples: {len(val_set)}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Training loop
        best_val_accuracy = 0
        
        t0 = time.time()
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            print(f"Training loss: {train_loss:.4f}, Training accuracy: {train_acc:.4f}")
            
            # Validate
            val_loss, val_acc, val_protein_acc = evaluate(model, val_loader, criterion, device)
            print(f"Validation loss: {val_loss:.4f}")
            print(f"Validation accuracy: {val_acc:.4f}")
            print(f"Protein average accuracy: {val_protein_acc:.4f}")
            
            # Save best model
            if val_protein_acc > best_val_accuracy:
                best_val_accuracy = val_protein_acc
                torch.save(model.state_dict(), best_model)
                print(f"Best model saved to {best_model}")
            else:
                print("No improvement")
                break
        training_time = time.time()-t0
        
        print(f"\nTraining complete!")
        print(f"Total training time: {training_time/60:.2f} minutes")
        print(f"Best protein average accuracy in Val: {best_val_accuracy:.4f}")

    # eval test_set
    # load best model
    if args.evaluate:
        load_path = args.model_path or best_model
        if not os.path.exists(load_path):
            print(f"Model weights not found: {load_path}")
            sys.exit(1)
        model.load_state_dict(torch.load(load_path, map_location=device))
        training_time = 0.0
    else:
        if not os.path.exists(best_model):
            print(f"Best model not found: {best_model}")
            sys.exit(1)
        model.load_state_dict(torch.load(best_model, map_location=device))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    test_loss, test_acc, test_protein_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test protein average accuracy: {test_protein_acc:.4f}")

    # make predictions directory if not exists
    os.makedirs('predictions', exist_ok=True)

    fn = f'predictions/{args.output}.csv'
    if not os.path.exists(fn):
        with open(fn, 'w') as f:
            csv.writer(f).writerow(['acc', 'finetune_time', 'seed', 'model', 'aug', 'augfile', 'small'])
    with open(fn, 'a') as f:
        csv.writer(f).writerow([
            test_protein_acc, training_time, args.seed, args.model,
            args.method, args.augfile or '', args.small,
        ])

if __name__ == "__main__":
    main()