"""
Main Experiment Runners
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Dict
import random
import time

from src.data.datasets import SyntheticARCDataset, ARCDataset, collate_arc_episodes, collate_arc_tasks
from src.models.arc_reasoner import ARCReasonerRelational, ARCReasonerStandard
from src.training.trainer import train_epoch, evaluate


def run_experiment(seed: int,
                   train_transforms: List[str],
                   test_transforms: List[str],
                   model_type: str = 'relational',
                   epochs: int = 30,
                   batch_size: int = 32,
                   grid_size: int = 8,
                   embed_dim: int = 128,
                   num_layers: int = 4,
                   num_examples: int = 3,
                   lr: float = 1e-3,
                   device: torch.device = None) -> Dict:
    """
    Run a single experiment.

    Args:
        model_type: 'relational' or 'standard'

    Returns dict with train/test metrics and timing.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)

    # Create datasets
    train_dataset = SyntheticARCDataset(
        transforms=train_transforms,
        grid_size=grid_size,
        num_examples=num_examples,
        episodes_per_epoch=1000
    )

    test_seen_dataset = SyntheticARCDataset(
        transforms=train_transforms,  # Same transforms as training
        grid_size=grid_size,
        num_examples=num_examples,
        episodes_per_epoch=200
    )

    test_unseen_dataset = SyntheticARCDataset(
        transforms=test_transforms,  # Held-out transforms
        grid_size=grid_size,
        num_examples=num_examples,
        episodes_per_epoch=200
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_arc_episodes)
    test_seen_loader = DataLoader(test_seen_dataset, batch_size=batch_size,
                                   shuffle=False, collate_fn=collate_arc_episodes)
    test_unseen_loader = DataLoader(test_unseen_dataset, batch_size=batch_size,
                                     shuffle=False, collate_fn=collate_arc_episodes)

    # Create model
    if model_type == 'relational':
        model = ARCReasonerRelational(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=4
        ).to(device)
    elif model_type == 'standard':
        model = ARCReasonerStandard(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=4
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Optimizer with cosine annealing
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    start_time = time.time()

    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Acc = {100*train_metrics['accuracy']:.1f}%")

    total_time = time.time() - start_time

    # Final evaluation
    final_seen = evaluate(model, test_seen_loader, device, return_per_transform=True)
    final_unseen = evaluate(model, test_unseen_loader, device, return_per_transform=True)

    return {
        'seed': seed,
        'model_type': model_type,
        'train_transforms': train_transforms,
        'test_transforms': test_transforms,
        'final_seen_accuracy': final_seen['accuracy'],
        'final_unseen_accuracy': final_unseen['accuracy'],
        'final_seen_per_transform': final_seen.get('per_transform', {}),
        'final_unseen_per_transform': final_unseen.get('per_transform', {}),
        'total_time': total_time
    }


def run_arc_experiment(data_dir: str, epochs: int = 30, device: torch.device = None,
                       low_memory: bool = True):
    """
    Train on ARC training set, evaluate on ARC evaluation set.

    Args:
        low_memory: If True, use memory-efficient settings (smaller batch, gradient accumulation)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("REAL ARC EXPERIMENT")
    print("=" * 70)
    print()

    # Memory-efficient settings
    if low_memory:
        batch_size = 3
        accumulation_steps = 4  # Effective batch size = 3 * 4 = 12
        print("⚡ Low-memory mode enabled:")
        print(f"   Batch size: {batch_size}")
        print(f"   Gradient accumulation: {accumulation_steps} steps")
        print(f"   Effective batch size: {batch_size * accumulation_steps}")
        print()
    else:
        batch_size = 8
        accumulation_steps = 1

    # Load datasets
    train_dataset = ARCDataset(data_dir, split='training', max_grid_size=30)
    eval_dataset = ARCDataset(data_dir, split='evaluation', max_grid_size=30)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_arc_tasks)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_arc_tasks)

    # Create model (larger for real ARC)
    model = ARCReasonerRelational(
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        num_colors=10,
        max_grid_size=30
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    print(f"{'Epoch':<8} {'Train Acc':<12} {'Eval Cell':<12} {'Eval Task':<12}")
    print("-" * 70)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_cells = 0

        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            example_inputs = batch['example_inputs'].to(device)
            example_outputs = batch['example_outputs'].to(device)
            test_input = batch['test_input'].to(device)
            test_output = batch['test_output'].to(device)
            mask = batch['mask'].to(device)
            B = example_inputs.shape[0]

            logits = model(example_inputs, example_outputs, test_input)

            # Flatten logits and targets, but filter by mask
            # logits: (B, H, W, C) -> logits[mask]: (N_valid_pixels, C)
            active_logits = logits[mask]
            active_targets = test_output[mask]

            loss = F.cross_entropy(active_logits, active_targets)

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()

            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * B * accumulation_steps
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions[mask] == test_output[mask]).sum().item()
            total_cells += mask.sum().item()

        scheduler.step()
        train_acc = total_correct / total_cells

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            eval_correct = 0
            eval_cells = 0
            task_correct = 0
            task_total = 0

            with torch.no_grad():
                for batch in eval_loader:
                    example_inputs = batch['example_inputs'].to(device)
                    example_outputs = batch['example_outputs'].to(device)
                    test_input = batch['test_input'].to(device)
                    test_output = batch['test_output'].to(device)

                    mask = batch['mask'].to(device) # Get mask
                    logits = model(example_inputs, example_outputs, test_input)
                    predictions = logits.argmax(dim=-1)

                    # Cell accuracy
                    eval_correct += (predictions[mask] == test_output[mask]).sum().item()
                    eval_cells += mask.sum().item()

                    # Task accuracy (100% of cells correct)
                    for i in range(predictions.shape[0]):
                        # Extract valid region for this specific example
                        m = mask[i]
                        if torch.equal(predictions[i][m], test_output[i][m]):
                            task_correct += 1
                        task_total += 1

            eval_acc = eval_correct / eval_cells
            task_acc = task_correct / task_total

            print(f"{epoch+1:<8} {100*train_acc:<12.1f} {100*eval_acc:<12.1f} {100*task_acc:<12.1f}")

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Cell Accuracy: {100*eval_acc:.2f}%")
    print(f"  Task Accuracy: {100*task_acc:.2f}% ({task_correct}/{task_total} tasks)")
    print("=" * 70)
