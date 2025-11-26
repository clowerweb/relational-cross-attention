"""
ARC Relational Reasoning Model Architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
