import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from pycocotools.coco import COCO

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """Required by DataLoader for Mask R-CNN's variable-size targets"""
    return tuple(zip(*batch))


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss':                 loss,
    }, path)


def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def plot_training_curve(log_path, output_path):
    """
    Expects a JSONL log file where each line is:
    {"epoch": 1, "train_loss": 0.42, "val_loss": 0.38}
    """
    epochs, train_losses, val_losses = [], [], []
    
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line)
            epochs.append(entry['epoch'])
            train_losses.append(entry['train_loss'])
            val_losses.append(entry['val_loss'])
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train loss')
    plt.plot(epochs, val_losses,   label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")