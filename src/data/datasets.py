"""
Synthetic ARC-Like Task Generator and Real ARC Dataset Loader
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict
import random
import json
from glob import glob


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
