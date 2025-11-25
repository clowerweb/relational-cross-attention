## Full README soon!

Code to hold you over:

```python
"""
Relational Cross-Attention for ARC-like Transform Learning

This experiment demonstrates that multi-layer cross-attention between input and output
grids enables better generalization on unseen geometric and color transformations.

Architecture features (Relational Model):
- Cross-attention pathway: Learns spatial mappings (rotate, flip, translate)
- Dual prediction heads with learned gating:
  * Spatial head: Uses attended embeddings (for geometric transforms)
  * Color head: Uses attended embeddings + one-hot colors (for color transforms)
  * Gate network: Learns which head to trust for each position
- Prevents task interference between spatial and color learning

Baseline (Standard Model):
- Standard transformer with self-attention over all examples + test
- No explicit relational structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict
import random
import math
from datetime import datetime
import json
from glob import glob

# =============================================================================
# GPU Detection and Setup
# =============================================================================

def detect_gpu():
    """Detect and report GPU backend (ROCm/CUDA/CPU)"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_props = torch.cuda.get_device_properties(0)

        # Check if it's ROCm (AMD GPU)
        is_rocm = 'AMD' in device_name or 'Radeon' in device_name or hasattr(torch.version, 'hip')

        print("=" * 60)
        if is_rocm:
            print("[ROCm] GPU DETECTED!")
            if hasattr(torch.version, 'hip'):
                print(f"   HIP Version: {torch.version.hip}")
        else:
            print("[CUDA] GPU DETECTED")
            print(f"   CUDA Version: {torch.version.cuda}")

        print(f"   Device: {device_name}")
        print(f"   Compute Capability: {device_props.major}.{device_props.minor}")
        print(f"   Total Memory: {device_props.total_memory / 1024**3:.2f} GB")
        print("=" * 60)
        return torch.device("cuda"), is_rocm
    else:
        print("=" * 60)
        print("[WARNING] NO GPU DETECTED - Using CPU")
        print("=" * 60)
        return torch.device("cpu"), False


# =============================================================================
# Synthetic ARC-Like Task Generator
# =============================================================================

class SyntheticARCDataset(Dataset):
    """
    Generates synthetic ARC-like episodes.
    Each episode: N example pairs + 1 test input → test output

    The model must infer the transformation from examples and apply it.
    """

    TRANSFORMS = {
        # Geometric transforms
        'rotate_90': lambda g: torch.rot90(g, 1, [0, 1]),
        'rotate_180': lambda g: torch.rot90(g, 2, [0, 1]),
        'rotate_270': lambda g: torch.rot90(g, 3, [0, 1]),
        'flip_horizontal': lambda g: torch.flip(g, [1]),
        'flip_vertical': lambda g: torch.flip(g, [0]),
        'transpose': lambda g: g.T,

        # Translation transforms
        'translate_right': lambda g: torch.roll(g, 1, dims=1),
        'translate_left': lambda g: torch.roll(g, -1, dims=1),
        'translate_down': lambda g: torch.roll(g, 1, dims=0),
        'translate_up': lambda g: torch.roll(g, -1, dims=0),

        # Color transforms
        'increment_colors': lambda g: (g + 1) % 10,
        'decrement_colors': lambda g: (g - 1) % 10,
        'invert_colors': lambda g: 9 - g,
        'double_colors': lambda g: (g * 2) % 10,

        # Identity (control)
        'identity': lambda g: g.clone(),
    }

    def __init__(self,
                 transforms: List[str],
                 grid_size: int = 8,
                 num_colors: int = 10,
                 num_examples: int = 3,
                 episodes_per_epoch: int = 1000):
        """
        Args:
            transforms: List of transform names to use
            grid_size: Size of square grids
            num_colors: Number of possible colors (0 to num_colors-1)
            num_examples: Number of example pairs per episode
            episodes_per_epoch: How many episodes constitute one epoch
        """
        self.transforms = transforms
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.num_examples = num_examples
        self.episodes_per_epoch = episodes_per_epoch

        # Validate transforms
        for t in transforms:
            if t not in self.TRANSFORMS:
                raise ValueError(f"Unknown transform: {t}")

    def __len__(self):
        return self.episodes_per_epoch

    def __getitem__(self, idx):
        # Pick a random transform for this episode
        transform_name = random.choice(self.transforms)
        transform_fn = self.TRANSFORMS[transform_name]

        # Generate example pairs
        example_inputs = []
        example_outputs = []
        for _ in range(self.num_examples):
            inp = torch.randint(0, self.num_colors, (self.grid_size, self.grid_size))
            out = transform_fn(inp)
            example_inputs.append(inp)
            example_outputs.append(out)

        # Generate test case
        test_input = torch.randint(0, self.num_colors, (self.grid_size, self.grid_size))
        test_output = transform_fn(test_input)

        return {
            'example_inputs': torch.stack(example_inputs),    # (N, H, W)
            'example_outputs': torch.stack(example_outputs),  # (N, H, W)
            'test_input': test_input,                         # (H, W)
            'test_output': test_output,                       # (H, W)
            'transform': transform_name
        }


def collate_arc_episodes(batch: List[Dict]) -> Dict:
    """Custom collate function for ARC episodes."""
    return {
        'example_inputs': torch.stack([b['example_inputs'] for b in batch]),    # (B, N, H, W)
        'example_outputs': torch.stack([b['example_outputs'] for b in batch]),  # (B, N, H, W)
        'test_input': torch.stack([b['test_input'] for b in batch]),            # (B, H, W)
        'test_output': torch.stack([b['test_output'] for b in batch]),          # (B, H, W)
        'transforms': [b['transform'] for b in batch]
    }


class ARCDataset(Dataset):
    """
    Real ARC dataset loader.

    Each task has multiple train examples and one or more test examples.
    We treat each task as one episode (like our synthetic data).
    """

    def __init__(self, data_dir: str, split: str = 'training', max_grid_size: int = 30):
        """
        Args:
            data_dir: Path to ARC-AGI/data folder
            split: 'training' or 'evaluation'
            max_grid_size: Pad/crop grids to this size
        """
        self.max_grid_size = max_grid_size
        self.tasks = []

        pattern = f"{data_dir}/{split}/*.json"
        for filepath in sorted(glob(pattern)):
            with open(filepath) as f:
                task = json.load(f)
                task['filename'] = filepath.split('\\')[-1].split('/')[-1]  # Handle both Windows and Unix
                self.tasks.append(task)

        if len(self.tasks) == 0:
            raise ValueError(f"No tasks found at {pattern}")

        print(f"Loaded {len(self.tasks)} ARC tasks from {split}")

    def __len__(self):
        return len(self.tasks)

    def pad_grid(self, grid: List[List[int]]) -> torch.Tensor:
        """Pad grid to max_grid_size x max_grid_size."""
        h, w = len(grid), len(grid[0]) if grid else 0
        padded = torch.zeros(self.max_grid_size, self.max_grid_size, dtype=torch.long)

        # Copy grid into top-left corner
        for i in range(min(h, self.max_grid_size)):
            for j in range(min(w, self.max_grid_size)):
                padded[i, j] = grid[i][j]

        return padded

    def __getitem__(self, idx):
        task = self.tasks[idx]

        # Get train examples (use as our "examples")
        train_examples = task['train']

        # Get test example (use first one)
        test_example = task['test'][0]

        # Convert to tensors
        example_inputs = []
        example_outputs = []

        for ex in train_examples:
            example_inputs.append(self.pad_grid(ex['input']))
            example_outputs.append(self.pad_grid(ex['output']))

        test_input = self.pad_grid(test_example['input'])
        test_output = self.pad_grid(test_example['output'])

        # Create mask for actual grid content (not padding)
        h, w = len(test_example['input']), len(test_example['input'][0])
        mask = torch.zeros(self.max_grid_size, self.max_grid_size, dtype=torch.bool)
        mask[:h, :w] = True  # True for real cells, False for padding

        return {
            'example_inputs': torch.stack(example_inputs),    # (N, H, W)
            'example_outputs': torch.stack(example_outputs),  # (N, H, W)
            'test_input': test_input,                          # (H, W)
            'test_output': test_output,                        # (H, W)
            'mask': mask,                                      # (H, W) - True for real cells
            'task_id': task['filename'],
            'num_examples': len(train_examples),
            'original_size': (h, w)
        }


def collate_arc_tasks(batch: List[Dict]) -> Dict:
    """
    Custom collate that handles variable numbers of examples per task.
    Pads to max examples in batch.
    """
    max_examples = max(b['num_examples'] for b in batch)

    example_inputs = []
    example_outputs = []

    for b in batch:
        n = b['num_examples']
        inp = b['example_inputs']  # (N, H, W)
        out = b['example_outputs']

        # Pad to max_examples by repeating last example
        if n < max_examples:
            pad_inp = inp[-1:].repeat(max_examples - n, 1, 1)
            pad_out = out[-1:].repeat(max_examples - n, 1, 1)
            inp = torch.cat([inp, pad_inp], dim=0)
            out = torch.cat([out, pad_out], dim=0)

        example_inputs.append(inp)
        example_outputs.append(out)

    return {
        'example_inputs': torch.stack(example_inputs),      # (B, max_N, H, W)
        'example_outputs': torch.stack(example_outputs),    # (B, max_N, H, W)
        'test_input': torch.stack([b['test_input'] for b in batch]),
        'test_output': torch.stack([b['test_output'] for b in batch]),
        'mask': torch.stack([b['mask'] for b in batch]),
        'task_ids': [b['task_id'] for b in batch]
    }


# =============================================================================
# ARC Relational Reasoning Model
# =============================================================================

class ARCReasonerRelational(nn.Module):
    """
    Relational cross-attention model for ARC-like transformation learning.
    Handles both spatial transforms (rotate, flip, translate) and color transforms.

    Architecture:
    1. Embed input and output grids separately with positional encodings
    2. For each example: Use multi-layer cross-attention to learn input→output mapping
    3. Average learned mappings across examples
    4. Apply learned mapping to test input via cross-attention
    5. Dual head predictions:
       - Spatial head: Predicts from attended embedding only
       - Color head: Predicts from attended embedding + one-hot input colors
    6. Gate network learns to combine the two heads

    Design rationale:
    - Spatial head: Good for geometric transforms (uses positional relationships)
    - Color head: Good for color transforms (has access to raw color values)
    - Gating: Prevents task interference, lets model discover which head to use
    """

    def __init__(self, embed_dim=128, num_layers=4, num_heads=4, num_colors=10, max_grid_size=16):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_colors = num_colors

        # Separate embeddings for input and output grids
        self.color_embed_in = nn.Embedding(num_colors, embed_dim)
        self.color_embed_out = nn.Embedding(num_colors, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_grid_size**2, embed_dim) * 0.02)

        # Multiple cross-attention layers for learning the transformation mapping
        self.cross_attn_layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                'norm1': nn.LayerNorm(embed_dim),
                'self_attn': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                'norm2': nn.LayerNorm(embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim)
                ),
                'norm3': nn.LayerNorm(embed_dim)
            })
            for _ in range(num_layers)
        ])

        # Apply learned mapping to test input
        self.test_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Dual prediction heads to prevent task interference
        # Spatial head: Uses only attended embedding (good for rotate, flip, translate)
        self.spatial_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_colors)
        )

        # Color head: Uses attended embedding + one-hot color (good for increment, invert)
        self.color_head = nn.Sequential(
            nn.Linear(embed_dim + num_colors, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_colors)
        )

        # Gating network: Learns which head to trust (2-way mixture)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim + num_colors, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 2),
            nn.Softmax(dim=-1)
        )

    def embed_input(self, grid):
        """Embed an input grid with color + position embeddings."""
        B, H, W = grid.shape
        emb = self.color_embed_in(grid).view(B, H*W, -1)
        return emb + self.pos_embed[:, :H*W, :]

    def embed_output(self, grid):
        """Embed an output grid with color + position embeddings."""
        B, H, W = grid.shape
        emb = self.color_embed_out(grid).view(B, H*W, -1)
        return emb + self.pos_embed[:, :H*W, :]

    def forward(self, example_inputs, example_outputs, test_input):
        """
        Forward pass for relational reasoning.

        Args:
            example_inputs: (B, N, H, W) - N example input grids
            example_outputs: (B, N, H, W) - N example output grids
            test_input: (B, H, W) - test input grid

        Returns:
            logits: (B, H, W, num_colors) - predicted test output
        """
        B, N, H, W = example_inputs.shape
        C = H * W

        # Step 1: Learn transformation mapping from each example
        mappings = []
        for i in range(N):
            inp_emb = self.embed_input(example_inputs[:, i])   # (B, C, D)
            out_emb = self.embed_output(example_outputs[:, i]) # (B, C, D)

            # Start with output embedding
            x = out_emb

            # Multiple rounds of cross-attention to input
            for layer in self.cross_attn_layers:
                # Cross-attend to input
                normed = layer['norm1'](x)
                cross_out, _ = layer['cross_attn'](normed, inp_emb, inp_emb)
                x = x + cross_out

                # Self-attention within positions
                normed = layer['norm2'](x)
                self_out, _ = layer['self_attn'](normed, normed, normed)
                x = x + self_out

                # Feed-forward network
                normed = layer['norm3'](x)
                x = x + layer['ffn'](normed)

            mappings.append(x)

        # Step 2: Average mappings across examples
        learned_mapping = torch.stack(mappings, dim=1).mean(dim=1)  # (B, C, D)

        # Step 3: Apply learned mapping to test input
        test_emb = self.embed_input(test_input)
        output_emb, _ = self.test_cross_attn(learned_mapping, test_emb, test_emb)

        # Step 4: Get one-hot color representation
        test_colors_onehot = F.one_hot(test_input.view(B, -1), num_classes=self.num_colors).float()

        # Step 5: Dual head predictions
        # Spatial head: Uses only attended embedding (for spatial transforms)
        spatial_logits = self.spatial_head(output_emb)  # (B, C, num_colors)

        # Color head: Uses attended embedding + one-hot color (for color transforms)
        combined = torch.cat([output_emb, test_colors_onehot], dim=-1)
        color_logits = self.color_head(combined)  # (B, C, num_colors)

        # Step 6: Gating - learn which head to trust
        gate_weights = self.gate(combined)  # (B, C, 2)

        # Combine predictions using learned gates
        final_logits = (gate_weights[:, :, 0:1] * spatial_logits +
                       gate_weights[:, :, 1:2] * color_logits)

        return final_logits.view(B, H, W, -1)


class ARCReasonerStandard(nn.Module):
    """
    Standard transformer baseline for comparison.
    Uses self-attention over all examples + test, without explicit relational structure.
    """

    def __init__(self, embed_dim=128, num_layers=4, num_heads=4, num_colors=10, max_grid_size=16):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_colors = num_colors

        # Embeddings
        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_grid_size**2, embed_dim) * 0.02)
        self.role_embed = nn.Embedding(3, embed_dim)  # 0=input, 1=output, 2=test

        # Standard transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                'norm1': nn.LayerNorm(embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim)
                ),
                'norm2': nn.LayerNorm(embed_dim)
            })
            for _ in range(num_layers)
        ])

        # Output head
        self.output_head = nn.Linear(embed_dim, num_colors)

    def embed_grid(self, grid, role):
        """Embed a grid with color + position + role."""
        B, H, W = grid.shape
        emb = self.color_embed(grid).view(B, H*W, -1)
        emb = emb + self.pos_embed[:, :H*W, :]
        emb = emb + self.role_embed(torch.full((B, H*W), role, device=grid.device, dtype=torch.long))
        return emb

    def forward(self, example_inputs, example_outputs, test_input):
        """
        Forward pass: Concatenate all grids and run through transformer.
        """
        B, N, H, W = example_inputs.shape
        C = H * W

        # Encode all grids into one sequence
        tokens = []
        for i in range(N):
            tokens.append(self.embed_grid(example_inputs[:, i], role=0))  # Input
            tokens.append(self.embed_grid(example_outputs[:, i], role=1))  # Output

        # Add test input
        test_tokens = self.embed_grid(test_input, role=2)
        tokens.append(test_tokens)
        test_start_idx = sum(t.size(1) for t in tokens[:-1])

        # Concatenate all tokens
        x = torch.cat(tokens, dim=1)  # (B, total_seq_len, embed_dim)

        # Apply transformer layers
        for layer in self.layers:
            # Self-attention
            normed = layer['norm1'](x)
            attn_out, _ = layer['self_attn'](normed, normed, normed)
            x = x + attn_out

            # FFN
            normed = layer['norm2'](x)
            x = x + layer['ffn'](normed)

        # Extract test region and predict
        test_out = x[:, test_start_idx:test_start_idx + C, :]
        logits = self.output_head(test_out)
        return logits.view(B, H, W, -1)


# =============================================================================
# Training and Evaluation
# =============================================================================

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


# =============================================================================
# Main Experiment Runner
# =============================================================================

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
    import time

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
    import time

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


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the relational cross-attention experiment."""

    # Detect GPU
    device, is_rocm = detect_gpu()

    if is_rocm:
        # Enable optimizations for AMD GPUs
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print("[OK] ROCm optimizations enabled\n")

    print("=" * 70)
    print("Relational Cross-Attention Experiment")
    print("=" * 70)
    print()
    print("Choose experiment:")
    print("  1. Synthetic ARC-lite (default)")
    print("  2. Real ARC dataset")
    print()

    choice = input("Enter choice [1]: ").strip() or "1"
    print()

    if choice == "2":
        data_dir = input("Enter path to ARC-AGI/data folder: ").strip()
        epochs = input("Enter number of epochs [50]: ").strip() or "50"
        run_arc_experiment(data_dir, epochs=int(epochs), device=device)
        return

    # Experiment configuration - testing both spatial AND color transforms
    train_transforms = ['rotate_90', 'flip_horizontal', 'translate_right', 'increment_colors']
    test_transforms = ['rotate_180', 'flip_vertical', 'translate_down', 'invert_colors']
    seeds = [42, 867, 7200, 4995, 789]
    epochs = 30

    print(f"TRAIN transforms: {train_transforms}")
    print(f"TEST transforms:  {test_transforms}")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {epochs}")
    print(f"Model: embed_dim=128, num_layers=4, num_heads=4")
    print()

    # Store results for both models
    all_results = {
        'standard': [],
        'relational': []
    }

    # Run experiment for each seed and model type
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED: {seed}")
        print(f"{'='*70}")

        for model_type in ['standard', 'relational']:
            print(f"\n[{model_type.upper()}] Training...")

            result = run_experiment(
                seed=seed,
                train_transforms=train_transforms,
                test_transforms=test_transforms,
                model_type=model_type,
                epochs=epochs,
                device=device
            )

            all_results[model_type].append(result)

            print(f"  Seen: {100*result['final_seen_accuracy']:.2f}% | "
                  f"Unseen: {100*result['final_unseen_accuracy']:.2f}% | "
                  f"Gap: {100*(result['final_seen_accuracy'] - result['final_unseen_accuracy']):.2f}% | "
                  f"Time: {result['total_time']:.1f}s")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print()

    for model_type in ['standard', 'relational']:
        print(f"\n{model_type.upper()} MODEL:")
        print(f"{'Seed':<8} {'Seen Acc':<12} {'Unseen Acc':<12} {'Gap':<10} {'Time':<10}")
        print("-" * 70)

        seen_accs = []
        unseen_accs = []
        gaps = []
        times = []

        for result in all_results[model_type]:
            seen = result['final_seen_accuracy']
            unseen = result['final_unseen_accuracy']
            gap = seen - unseen
            time_taken = result['total_time']

            seen_accs.append(seen)
            unseen_accs.append(unseen)
            gaps.append(gap)
            times.append(time_taken)

            print(f"{result['seed']:<8} {100*seen:<12.2f} {100*unseen:<12.2f} "
                  f"{100*gap:<10.2f} {time_taken:<10.1f}s")

        # Averages
        avg_seen = sum(seen_accs) / len(seen_accs)
        avg_unseen = sum(unseen_accs) / len(unseen_accs)
        avg_gap = sum(gaps) / len(gaps)
        avg_time = sum(times) / len(times)

        print("-" * 70)
        print(f"{'AVERAGE':<8} {100*avg_seen:<12.2f} {100*avg_unseen:<12.2f} "
              f"{100*avg_gap:<10.2f} {avg_time:<10.1f}s")

        # Standard deviations
        std_seen = (sum((x - avg_seen)**2 for x in seen_accs) / len(seen_accs))**0.5
        std_unseen = (sum((x - avg_unseen)**2 for x in unseen_accs) / len(unseen_accs))**0.5

        print(f"Std Dev:  ±{100*std_seen:<11.2f} ±{100*std_unseen:<11.2f}")

    # Per-transform breakdown comparison
    print("\n" + "=" * 70)
    print("PER-TRANSFORM COMPARISON (Unseen, Last Seed)")
    print("=" * 70)
    print()

    std_last = all_results['standard'][-1]
    rel_last = all_results['relational'][-1]

    print(f"{'Transform':<20} {'Standard':<12} {'Relational':<12} {'Delta':<10}")
    print("-" * 70)

    for transform in sorted(std_last['final_unseen_per_transform'].keys()):
        std_acc = std_last['final_unseen_per_transform'][transform]
        rel_acc = rel_last['final_unseen_per_transform'][transform]
        delta = rel_acc - std_acc

        print(f"{transform:<20} {100*std_acc:<12.2f} {100*rel_acc:<12.2f} "
              f"{'+' if delta > 0 else ''}{100*delta:<9.2f}")

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
```

## Sample Synthetic Run (RX 7900 XT with ROCm on Windows 11 Pro)

```
======================================================================
Relational Cross-Attention Experiment
======================================================================

TRAIN transforms: ['rotate_90', 'flip_horizontal', 'translate_right', 'increment_colors']
TEST transforms:  ['rotate_180', 'flip_vertical', 'translate_down', 'invert_colors']
Seeds: [42, 867, 7200, 4995, 789]
Epochs: 30
Model: embed_dim=128, num_layers=4, num_heads=4


======================================================================
SEED: 42
======================================================================

[STANDARD] Training...
  Epoch 10/30: Train Acc = 33.2%
  Epoch 20/30: Train Acc = 32.1%
  Epoch 30/30: Train Acc = 33.3%
  Seen: 33.04% | Unseen: 12.30% | Gap: 20.73% | Time: 30.5s

[RELATIONAL] Training...
  Epoch 10/30: Train Acc = 33.1%
  Epoch 20/30: Train Acc = 36.5%
  Epoch 30/30: Train Acc = 38.1%
  Seen: 37.87% | Unseen: 17.26% | Gap: 20.61% | Time: 24.8s

======================================================================
SEED: 867
======================================================================

[STANDARD] Training...
  Epoch 10/30: Train Acc = 32.4%
  Epoch 20/30: Train Acc = 31.2%
  Epoch 30/30: Train Acc = 33.8%
  Seen: 33.32% | Unseen: 12.61% | Gap: 20.71% | Time: 29.5s

[RELATIONAL] Training...
  Epoch 10/30: Train Acc = 32.6%
  Epoch 20/30: Train Acc = 31.2%
  Epoch 30/30: Train Acc = 33.4%
  Seen: 33.21% | Unseen: 11.99% | Gap: 21.22% | Time: 24.8s

======================================================================
SEED: 7200
======================================================================

[STANDARD] Training...
  Epoch 10/30: Train Acc = 33.1%
  Epoch 20/30: Train Acc = 31.4%
  Epoch 30/30: Train Acc = 31.5%
  Seen: 34.73% | Unseen: 12.38% | Gap: 22.34% | Time: 29.5s

[RELATIONAL] Training...
  Epoch 10/30: Train Acc = 32.6%
  Epoch 20/30: Train Acc = 36.3%
  Epoch 30/30: Train Acc = 36.4%
  Seen: 39.57% | Unseen: 17.09% | Gap: 22.48% | Time: 24.8s

======================================================================
SEED: 4995
======================================================================

[STANDARD] Training...
  Epoch 10/30: Train Acc = 32.0%
  Epoch 20/30: Train Acc = 33.0%
  Epoch 30/30: Train Acc = 32.6%
  Seen: 35.42% | Unseen: 12.14% | Gap: 23.28% | Time: 29.6s

[RELATIONAL] Training...
  Epoch 10/30: Train Acc = 33.4%
  Epoch 20/30: Train Acc = 37.8%
  Epoch 30/30: Train Acc = 37.5%
  Seen: 40.26% | Unseen: 17.23% | Gap: 23.02% | Time: 24.8s

======================================================================
SEED: 789
======================================================================

[STANDARD] Training...
  Epoch 10/30: Train Acc = 35.1%
  Epoch 20/30: Train Acc = 31.8%
  Epoch 30/30: Train Acc = 31.0%
  Seen: 33.97% | Unseen: 12.37% | Gap: 21.60% | Time: 29.6s

[RELATIONAL] Training...
  Epoch 10/30: Train Acc = 34.9%
  Epoch 20/30: Train Acc = 32.1%
  Epoch 30/30: Train Acc = 35.6%
  Seen: 38.41% | Unseen: 17.05% | Gap: 21.37% | Time: 24.8s

======================================================================
SUMMARY COMPARISON
======================================================================


STANDARD MODEL:
Seed     Seen Acc     Unseen Acc   Gap        Time
----------------------------------------------------------------------
42       33.04        12.30        20.73      30.5      s
867      33.32        12.61        20.71      29.5      s
7200     34.73        12.38        22.34      29.5      s
4995     35.42        12.14        23.28      29.6      s
789      33.97        12.37        21.60      29.6      s
----------------------------------------------------------------------
AVERAGE  34.10        12.36        21.73      29.7      s
Std Dev:  ±0.88        ±0.15

RELATIONAL MODEL:
Seed     Seen Acc     Unseen Acc   Gap        Time
----------------------------------------------------------------------
42       37.87        17.26        20.61      24.8      s
867      33.21        11.99        21.22      24.8      s
7200     39.57        17.09        22.48      24.8      s
4995     40.26        17.23        23.02      24.8      s
789      38.41        17.05        21.37      24.8      s
----------------------------------------------------------------------
AVERAGE  37.86        16.12        21.74      24.8      s
Std Dev:  ±2.47        ±2.07

======================================================================
PER-TRANSFORM COMPARISON (Unseen, Last Seed)
======================================================================

Transform            Standard     Relational   Delta
----------------------------------------------------------------------
flip_vertical        10.14        16.12        +5.98
invert_colors        20.07        20.35        +0.28
rotate_180           10.33        15.91        +5.58
translate_down       9.95         16.20        +6.25

======================================================================
EXPERIMENT COMPLETE
======================================================================
```

## Sample ARC-AGI Run (15 epochs, 7M params, batch size 3, effective batch 12)

```
Epoch    Train Acc    Eval Cell    Eval Task   
----------------------------------------------------------------------
1        63.1         61.3         2.5         
5        69.6         64.9         2.8         
10       71.7         62.4         2.2         
15       72.3         61.2         2.2         

======================================================================
FINAL RESULTS
======================================================================
  Cell Accuracy: 61.21%
  Task Accuracy: 2.25% (9/400 tasks)
======================================================================
```

The model starts sliding backwards after the 5th epoch likely due to overfitting (model too small). I don't currently have the hardware to run this test with a larger model size, but I'd love to see results from someone who can!
