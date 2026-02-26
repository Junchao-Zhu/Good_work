# -*- coding: utf-8 -*-
"""
Dual-Backbone Model for Gene Expression Prediction

Two separate backbones:
    - DenseNet-121 (end-to-end) for BLEEP contrastive retrieval
    - ViT-B (CONCH pretrained, end-to-end) for regression

Inference:
    Image -> DenseNet-121 -> BLEEP retrieval (with cell type filter) -> pred_ret
    Image -> ViT-B -> RegressionHead -> pred_reg
    BranchClassifier(ViT-B features) -> alpha
    Final = alpha * pred_ret + (1 - alpha) * pred_reg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .backbones import get_backbone
from .heads import RegressionHead, GeneEncoder, ProjectionHead, BranchClassifier


def bleep_cross_entropy(preds, targets, reduction='none'):
    """Cross entropy with soft targets (BLEEP style)"""
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


class DualBackboneModel(nn.Module):
    """
    Dual-backbone model: DenseNet-121 for retrieval + ViT-B for regression.

    Stage 1 - BLEEP Retrieval:
        - DenseNet-121 (end-to-end) -> global pool (1024-d) -> ProjectionHead -> 256-d
        - GeneEncoder -> 256-d
        - Contrastive learning (BLEEP, temperature=0.07)

    Stage 2 - Regression:
        - ViT-B (CONCH, end-to-end) -> 768-d -> RegressionHead -> n_target_genes
        - MSE loss

    Stage 3 - Classifier:
        - BranchClassifier(ViT-B features) -> alpha
        - Both backbones frozen
    """

    def __init__(
        self,
        n_target_genes: int = 46,
        projection_dim: int = 256,
        temperature: float = 0.07,
        top_k: int = 50,
        pretrained: bool = True,
    ):
        super().__init__()

        self.n_target_genes = n_target_genes
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.top_k = top_k

        # 1. BLEEP branch: DenseNet-121 (end-to-end)
        self.bleep_encoder = get_backbone(
            name='densenet121',
            pretrained=pretrained,
            trainable=True
        )
        # Projection head: 1024 -> projection_dim
        self.bleep_projection = ProjectionHead(
            input_dim=self.bleep_encoder.feature_dim,
            projection_dim=projection_dim,
            dropout=0.1
        )

        # 2. Gene Encoder (for BLEEP contrastive)
        self.gene_encoder = GeneEncoder(
            input_dim=n_target_genes,
            projection_dim=projection_dim,
            hidden_dim=512,
            num_layers=2,
            dropout=0.1
        )

        # 3. Regression branch: ViT-B CONCH (end-to-end)
        self.reg_encoder = get_backbone(
            name='vit_b',
            pretrained=pretrained,
            trainable=True
        )
        self.regression_head = RegressionHead(
            input_dim=self.reg_encoder.feature_dim,  # 768
            output_dim=n_target_genes,
            hidden_dims=[512, 256],
            dropout=0.1
        )

        # 4. Branch Classifier (Stage 3, uses ViT-B features)
        self.branch_classifier = BranchClassifier(
            input_dim=self.reg_encoder.feature_dim,  # 768
            hidden_dim=128,
            spot_repr_dim=64,
            dropout=0.2
        )

    def freeze_bleep(self):
        """Freeze BLEEP branch (DenseNet + projection + gene encoder) after Stage 1"""
        for param in self.bleep_encoder.parameters():
            param.requires_grad = False
        for param in self.bleep_projection.parameters():
            param.requires_grad = False
        for param in self.gene_encoder.parameters():
            param.requires_grad = False
        self.bleep_encoder.eval()
        self.bleep_projection.eval()
        self.gene_encoder.eval()

    def freeze_regression(self):
        """Freeze regression branch (ViT-B + reg head) after Stage 2"""
        for param in self.reg_encoder.parameters():
            param.requires_grad = False
        for param in self.regression_head.parameters():
            param.requires_grad = False
        self.reg_encoder.eval()
        self.regression_head.eval()

    def freeze_all_except_classifier(self):
        """Freeze everything except branch classifier for Stage 3"""
        self.freeze_bleep()
        self.freeze_regression()

    def get_bleep_image_embed(self, images: torch.Tensor) -> torch.Tensor:
        """Get L2-normalized image embedding from DenseNet-121 BLEEP branch"""
        backbone_out = self.bleep_encoder(images)
        features = backbone_out['features']  # (B, 1024)
        embed = self.bleep_projection(features)  # (B, projection_dim)
        embed = F.normalize(embed, dim=-1)
        return embed

    def compute_contrastive_loss(
        self,
        image_embed: torch.Tensor,
        gene_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        BLEEP-style contrastive loss with soft targets.
        Temperature = 0.07 (standard CLIP/BLEEP).
        """
        logits = (image_embed @ gene_embed.T) / self.temperature

        image_similarity = image_embed @ image_embed.T
        gene_similarity = gene_embed @ gene_embed.T
        targets = F.softmax(
            ((image_similarity + gene_similarity) / 2) / self.temperature, dim=-1
        )

        image_loss = bleep_cross_entropy(logits, targets, reduction='none')
        gene_loss = bleep_cross_entropy(logits.T, targets.T, reduction='none')
        loss = (image_loss + gene_loss) / 2.0

        return loss.mean()

    def retrieve(
        self,
        image_embed: torch.Tensor,
        gene_embeds_db: torch.Tensor,
        target_genes_db: torch.Tensor,
        query_cell_comp: Optional[torch.Tensor] = None,
        db_cell_comp: Optional[torch.Tensor] = None,
        count_threshold: float = 0.5,
        comp_sim_threshold: float = 0.5,
        cell_sim_weight: float = 0.3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve similar samples with optional cell composition-based filtering.

        Two-stage filtering (when cell comp provided):
        1. Total cell count: filter if |q_sum - d_sum| / max > count_threshold
        2. Composition similarity: filter if cosine_sim < comp_sim_threshold

        Final score = (1 - cell_sim_weight) * gene_sim + cell_sim_weight * comp_sim
        """
        B = image_embed.shape[0]
        N = gene_embeds_db.shape[0]

        gene_sim = image_embed @ gene_embeds_db.T  # (B, N)

        if query_cell_comp is not None and db_cell_comp is not None:
            query_count = query_cell_comp.sum(dim=-1, keepdim=True)  # (B, 1)
            db_count = db_cell_comp.sum(dim=-1, keepdim=True).T  # (1, N)

            count_diff = torch.abs(query_count - db_count)
            count_max = torch.maximum(query_count, db_count) + 1e-6
            count_mask = (count_diff / count_max) <= count_threshold

            query_prop = F.normalize(query_cell_comp, p=1, dim=-1)
            db_prop = F.normalize(db_cell_comp, p=1, dim=-1)
            query_norm = F.normalize(query_prop, p=2, dim=-1)
            db_norm = F.normalize(db_prop, p=2, dim=-1)
            comp_sim = query_norm @ db_norm.T  # (B, N)

            comp_mask = comp_sim >= comp_sim_threshold
            final_mask = count_mask & comp_mask

            gene_sim_masked = gene_sim.masked_fill(~final_mask, float('-inf'))
            combined_sim = (1 - cell_sim_weight) * gene_sim + cell_sim_weight * comp_sim
            combined_sim = combined_sim.masked_fill(~final_mask, float('-inf'))
        else:
            combined_sim = gene_sim

        top_sims, top_indices = combined_sim.topk(self.top_k, dim=-1)
        top_genes = target_genes_db[top_indices]

        valid_mask = top_sims > float('-inf')
        top_sims = torch.where(valid_mask, top_sims, torch.zeros_like(top_sims))

        weights = F.softmax(top_sims / self.temperature, dim=-1)
        pred_ret = (weights.unsqueeze(-1) * top_genes).sum(dim=1)

        return pred_ret, top_genes, top_sims

    @torch.no_grad()
    def inference(
        self,
        images: torch.Tensor,
        gene_embeds_db: torch.Tensor,
        target_genes_db: torch.Tensor,
        query_cell_comp: Optional[torch.Tensor] = None,
        db_cell_comp: Optional[torch.Tensor] = None,
        count_threshold: float = 0.5,
        comp_sim_threshold: float = 0.5,
        cell_sim_weight: float = 0.3,
    ) -> Dict[str, torch.Tensor]:
        """
        Full inference: both branches + retrieval.

        Returns:
            dict with pred_reg, pred_ret, features (ViT-B), image_embed, top_sims
        """
        self.eval()

        # Regression branch (ViT-B)
        reg_out = self.reg_encoder(images)
        features = reg_out['features']  # (B, 768) for classifier
        pred_reg = self.regression_head(features)

        # BLEEP retrieval branch (DenseNet-121)
        image_embed = self.get_bleep_image_embed(images)
        pred_ret, top_genes, top_sims = self.retrieve(
            image_embed, gene_embeds_db, target_genes_db,
            query_cell_comp=query_cell_comp,
            db_cell_comp=db_cell_comp,
            count_threshold=count_threshold,
            comp_sim_threshold=comp_sim_threshold,
            cell_sim_weight=cell_sim_weight
        )

        return {
            'pred_reg': pred_reg,
            'pred_ret': pred_ret,
            'features': features,
            'image_embed': image_embed,
            'top_sims': top_sims,
        }

    @torch.no_grad()
    def inference_ret_only(
        self,
        images: torch.Tensor,
        gene_embeds_db: torch.Tensor,
        target_genes_db: torch.Tensor,
        query_cell_comp: Optional[torch.Tensor] = None,
        db_cell_comp: Optional[torch.Tensor] = None,
        count_threshold: float = 0.5,
        comp_sim_threshold: float = 0.5,
        cell_sim_weight: float = 0.3,
    ) -> Dict[str, torch.Tensor]:
        """Retrieval-only inference (no ViT-B needed, for Stage 1 eval)"""
        self.bleep_encoder.eval()
        self.bleep_projection.eval()

        image_embed = self.get_bleep_image_embed(images)
        pred_ret, top_genes, top_sims = self.retrieve(
            image_embed, gene_embeds_db, target_genes_db,
            query_cell_comp=query_cell_comp,
            db_cell_comp=db_cell_comp,
            count_threshold=count_threshold,
            comp_sim_threshold=comp_sim_threshold,
            cell_sim_weight=cell_sim_weight
        )

        return {
            'pred_ret': pred_ret,
            'image_embed': image_embed,
            'top_sims': top_sims,
        }

    def get_optimizer_params_bleep(self):
        """Parameters for Stage 1: DenseNet + projection + gene encoder"""
        return list(self.bleep_encoder.parameters()) + \
               list(self.bleep_projection.parameters()) + \
               list(self.gene_encoder.parameters())

    def get_optimizer_params_reg(self):
        """Parameters for Stage 2: ViT-B + regression head"""
        return list(self.reg_encoder.parameters()) + \
               list(self.regression_head.parameters())

    def get_optimizer_params_cls(self):
        """Parameters for Stage 3: branch classifier only"""
        return list(self.branch_classifier.parameters())

    def print_parameter_summary(self):
        """Print parameter summary"""
        def count_params(module):
            return sum(p.numel() for p in module.parameters())

        bleep_enc = count_params(self.bleep_encoder)
        bleep_proj = count_params(self.bleep_projection)
        gene_enc = count_params(self.gene_encoder)
        reg_enc = count_params(self.reg_encoder)
        reg_head = count_params(self.regression_head)
        cls = count_params(self.branch_classifier)

        print("\n" + "="*70)
        print("DUAL-BACKBONE MODEL PARAMETER SUMMARY")
        print("="*70)
        print(f"\n  BLEEP Branch (Stage 1):")
        print(f"    DenseNet-121:     {bleep_enc:>12,}")
        print(f"    ProjectionHead:   {bleep_proj:>12,}")
        print(f"    GeneEncoder:      {gene_enc:>12,}")
        print(f"    Subtotal:         {bleep_enc + bleep_proj + gene_enc:>12,}")
        print(f"\n  Regression Branch (Stage 2):")
        print(f"    ViT-B (CONCH):    {reg_enc:>12,}")
        print(f"    RegressionHead:   {reg_head:>12,}")
        print(f"    Subtotal:         {reg_enc + reg_head:>12,}")
        print(f"\n  BranchClassifier (Stage 3):")
        print(f"    Classifier:       {cls:>12,}")
        total = bleep_enc + bleep_proj + gene_enc + reg_enc + reg_head + cls
        print(f"\n  TOTAL:              {total:>12,}")
        print("="*70)


@torch.no_grad()
def build_gene_embed_database(
    model: DualBackboneModel,
    target_genes_all: torch.Tensor,
    batch_size: int = 256,
    device: torch.device = None
) -> torch.Tensor:
    """
    Precompute gene embeddings for all training samples.

    Args:
        model: DualBackboneModel
        target_genes_all: (N, n_target_genes) all training target genes
        batch_size: batch size for processing
        device: device to use

    Returns:
        gene_embeds: (N, projection_dim) gene embeddings
    """
    model.gene_encoder.eval()
    device = device or next(model.parameters()).device

    gene_embeds = []
    for i in range(0, len(target_genes_all), batch_size):
        batch = target_genes_all[i:i+batch_size].to(device)
        embed = model.gene_encoder(batch)
        gene_embeds.append(embed.cpu())

    return torch.cat(gene_embeds, dim=0)
