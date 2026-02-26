# -*- coding: utf-8 -*-
"""
Model Heads:
- RegressionHead: MLP that outputs predictions (MSE loss)
- RegressionHeadWithUncertainty: outputs mean + variance (NLL loss, for future use)
- GeneEncoder: encodes gene expression vectors for contrastive retrieval (BLEEP-style)
- ProjectionHead: projects features to embedding space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class RegressionHead(nn.Module):
    """
    Simple regression head: MLP → prediction

    Architecture:
        features → MLP [hidden_dims] → Linear → pred (n_target_genes)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = [512, 256],
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) features
        Returns:
            pred: (batch, output_dim) predictions
        """
        return self.mlp(x)


class RegressionHeadWithUncertainty(nn.Module):
    """
    Regression head that outputs both mean and log-variance
    for uncertainty estimation (CV-based selection)

    Output:
        mean: predicted cell type counts
        log_var: log variance for each prediction
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,  # n_cell_types
        hidden_dims: list = [512, 256],
        dropout: float = 0.1
    ):
        super().__init__()

        # Shared feature extraction
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        # Separate heads for mean and variance
        self.mean_head = nn.Linear(prev_dim, output_dim)
        self.log_var_head = nn.Linear(prev_dim, output_dim)

        # Initialize log_var head to output small variance initially
        nn.init.zeros_(self.log_var_head.weight)
        nn.init.constant_(self.log_var_head.bias, -2.0)  # exp(-2) ≈ 0.14

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim) features

        Returns:
            dict with 'mean', 'log_var', 'std'
        """
        shared_feat = self.shared(x)

        mean = self.mean_head(shared_feat)
        log_var = self.log_var_head(shared_feat)

        # Clamp log_var for numerical stability
        log_var = torch.clamp(log_var, min=-10, max=10)

        return {
            'mean': mean,
            'log_var': log_var,
            'std': torch.exp(0.5 * log_var)
        }

    def compute_cv(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute coefficient of variation (CV) = std / |mean|
        Lower CV = more confident

        Returns:
            cv: (batch,) CV for each sample
        """
        mean = outputs['mean']
        std = outputs['std']
        cv = (std / (mean.abs() + 1e-8)).mean(dim=-1)
        return cv


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning (from BLEEP)

    Architecture:
        x -> Linear -> GELU -> Linear -> Dropout -> Residual -> LayerNorm
    """
    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.projection = nn.Linear(input_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected  # residual
        x = self.layer_norm(x)
        return x


class BranchClassifier(nn.Module):
    """
    Soft mixing network for spot-level branch blending (residual formulation).

    Instead of hard binary selection (reg or ret), outputs a continuous
    mixing weight alpha = 0.5 + delta, where delta is learned.

    Key design:
        pred = alpha * pred_ret + (1 - alpha) * pred_reg
        alpha = 0.5 + 0.5 * tanh(network(features))

    Guarantees:
        - Zero-init last layer => starts as simple average (alpha=0.5)
        - Regularization on delta => stays near 0.5 unless confident
        - Cannot perform worse than simple average by design

    Architecture:
        input_dim → hidden_dim → spot_repr_dim → 1 (delta)
    """
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        spot_repr_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()

        # Spot encoder: input_dim → hidden_dim → spot_repr_dim
        self.spot_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, spot_repr_dim),
            nn.LayerNorm(spot_repr_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Spot-level head: spot_repr_dim → 1 (raw logit for delta)
        self.spot_head = nn.Linear(spot_repr_dim, 1)

        # Zero-init last layer => delta=0 => alpha=0.5 at initialization
        nn.init.zeros_(self.spot_head.weight)
        nn.init.zeros_(self.spot_head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Per-spot forward. Returns raw logits (before tanh).

        Args:
            features: (B, input_dim) backbone features
        Returns:
            logits: (B, 1) raw logits
        """
        spot_hidden = self.spot_encoder(features)  # (B, spot_repr_dim)
        return self.spot_head(spot_hidden)  # (B, 1)

    def get_delta(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get delta values (deviation from 0.5).

        Args:
            features: (N, input_dim) backbone features
        Returns:
            delta: (N,) values in (-0.5, 0.5)
        """
        raw = self.forward(features).squeeze(-1)  # (N,)
        return 0.5 * torch.tanh(raw)

    def get_alpha(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get mixing weights alpha = 0.5 + delta.

        Args:
            features: (N, input_dim) backbone features
        Returns:
            alpha: (N,) mixing weights in (0, 1)
                   alpha near 1 => use retrieval
                   alpha near 0 => use regression
                   alpha = 0.5  => simple average
        """
        return 0.5 + self.get_delta(features)


class GeneEncoder(nn.Module):
    """
    Encodes gene expression vectors to embedding space for contrastive retrieval.

    Architecture:
        gene_expression -> MLP layers -> ProjectionHead -> L2-normalized embedding
    """
    def __init__(
        self,
        input_dim: int,  # n_target_genes
        projection_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []

        # First layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ])

        # Middle layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])

        self.encoder = nn.Sequential(*layers)

        # Final projection
        self.projection = ProjectionHead(
            input_dim=hidden_dim,
            projection_dim=projection_dim,
            dropout=dropout
        )

    def forward(self, gene_expression: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gene_expression: (batch, n_target_genes) normalized gene expression

        Returns:
            embedding: (batch, projection_dim) L2-normalized embedding
        """
        x = self.encoder(gene_expression)
        x = self.projection(x)
        x = F.normalize(x, dim=-1)
        return x
