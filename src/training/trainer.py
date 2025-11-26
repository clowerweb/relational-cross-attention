"""
Training and Evaluation Functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict


def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                device: torch.device) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_cells = 0

    for batch in dataloader:
        example_inputs = batch['example_inputs'].to(device)
        example_outputs = batch['example_outputs'].to(device)
        test_input = batch['test_input'].to(device)
        test_output = batch['test_output'].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(example_inputs, example_outputs, test_input)  # (B, H, W, C)

        # Compute loss
        B, H, W, C = logits.shape
        loss = F.cross_entropy(logits.view(-1, C), test_output.view(-1))

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * B
        predictions = logits.argmax(dim=-1)  # (B, H, W)
        total_correct += (predictions == test_output).sum().item()
        total_cells += B * H * W

    return {
        'loss': total_loss / len(dataloader.dataset),
        'accuracy': total_correct / total_cells
    }


def evaluate(model: nn.Module,
             dataloader: DataLoader,
             device: torch.device,
             return_per_transform: bool = False) -> Dict:
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_cells = 0

    # Track per-transform accuracy
    transform_correct = {}
    transform_total = {}

    with torch.no_grad():
        for batch in dataloader:
            example_inputs = batch['example_inputs'].to(device)
            example_outputs = batch['example_outputs'].to(device)
            test_input = batch['test_input'].to(device)
            test_output = batch['test_output'].to(device)
            transforms = batch['transforms']

            # Forward pass
            logits = model(example_inputs, example_outputs, test_input)

            # Compute loss
            B, H, W, C = logits.shape
            loss = F.cross_entropy(logits.view(-1, C), test_output.view(-1))

            # Track metrics
            total_loss += loss.item() * B
            predictions = logits.argmax(dim=-1)
            correct_mask = (predictions == test_output)
            total_correct += correct_mask.sum().item()
            total_cells += B * H * W

            # Per-transform tracking
            if return_per_transform:
                for i, t in enumerate(transforms):
                    if t not in transform_correct:
                        transform_correct[t] = 0
                        transform_total[t] = 0
                    transform_correct[t] += correct_mask[i].sum().item()
                    transform_total[t] += H * W

    results = {
        'loss': total_loss / len(dataloader.dataset),
        'accuracy': total_correct / total_cells
    }

    if return_per_transform:
        results['per_transform'] = {
            t: transform_correct[t] / transform_total[t]
            for t in transform_correct
        }

    return results
