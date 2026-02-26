# -*- coding: utf-8 -*-
"""
Vision Encoder Backbones with Multi-Scale Feature Output

Each backbone outputs:
    - features: (B, feat_dim) global-pooled features for regression
    - multi_scale_feats: list of intermediate feature maps for adapters

Backbones are TRAINABLE, updated by regression loss.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, List, Tuple, Dict


class ResNet50Backbone(nn.Module):
    """
    ResNet-50 backbone with multi-scale feature output

    Outputs:
        features: (B, 2048) global-pooled
        multi_scale_feats: [
            (B, 256, 56, 56),   # after layer1
            (B, 512, 28, 28),   # after layer2
            (B, 1024, 14, 14),  # after layer3
            (B, 2048, 7, 7),    # after layer4
        ]
    """
    def __init__(self, pretrained: bool = True, trainable: bool = True):
        super().__init__()

        resnet = timm.create_model('resnet50', pretrained=pretrained)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.act1 = resnet.act1
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.global_pool = resnet.global_pool

        self.feature_dim = 2048
        self.stage_channels = [256, 512, 1024, 2048]
        self.feat_type = 'cnn'  # for adapter to know feature format

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        # Stages
        feat1 = self.layer1(x)       # (B, 256, 56, 56)
        feat2 = self.layer2(feat1)    # (B, 512, 28, 28)
        feat3 = self.layer3(feat2)    # (B, 1024, 14, 14)
        feat4 = self.layer4(feat3)    # (B, 2048, 7, 7)

        # Global pool
        features = self.global_pool(feat4)  # (B, 2048)

        return {
            'features': features,
            'multi_scale_feats': [feat1, feat2, feat3, feat4],
        }


class DenseNet121Backbone(nn.Module):
    """
    DenseNet-121 backbone with multi-scale feature output

    Outputs:
        features: (B, 1024) global-pooled
        multi_scale_feats: [
            (B, 256, 56, 56),   # after denseblock1
            (B, 512, 28, 28),   # after denseblock2
            (B, 1024, 14, 14),  # after denseblock3
            (B, 1024, 7, 7),    # after denseblock4
        ]
    """
    def __init__(self, pretrained: bool = True, trainable: bool = True):
        super().__init__()

        densenet = timm.create_model('densenet121', pretrained=pretrained)

        self.features = densenet.features
        self.global_pool = densenet.global_pool

        self.feature_dim = 1024
        self.stage_channels = [256, 512, 1024, 1024]
        self.feat_type = 'cnn'

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Stem
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.pool0(x)

        # DenseBlock1
        feat1 = self.features.denseblock1(x)       # (B, 256, 56, 56)
        x = self.features.transition1(feat1)

        # DenseBlock2
        feat2 = self.features.denseblock2(x)        # (B, 512, 28, 28)
        x = self.features.transition2(feat2)

        # DenseBlock3
        feat3 = self.features.denseblock3(x)        # (B, 1024, 14, 14)
        x = self.features.transition3(feat3)

        # DenseBlock4
        feat4 = self.features.denseblock4(x)        # (B, 1024, 7, 7)
        x = self.features.norm5(feat4)

        # Global pool
        features = self.global_pool(x)  # (B, 1024)

        return {
            'features': features,
            'multi_scale_feats': [feat1, feat2, feat3, feat4],
        }


class CONCHBackbone(nn.Module):
    """
    CONCH ViT-B/16 backbone with multi-scale feature output

    For ViT, we extract features from blocks [2, 5, 8, 11] (0-indexed)
    to get 4 intermediate representations.

    Outputs:
        features: (B, 768) CLS token from final layer
        multi_scale_feats: [
            (B, 197, 768),  # after block 2
            (B, 197, 768),  # after block 5
            (B, 197, 768),  # after block 8
            (B, 197, 768),  # after block 11
        ]
    """
    def __init__(
        self,
        pretrained: bool = True,
        trainable: bool = True,
        model_path: Optional[str] = None
    ):
        super().__init__()
        self.feature_dim = 768
        self.feat_type = 'transformer'
        self.extract_layers = [2, 5, 8, 11]  # 4 intermediate layers

        if pretrained:
            try:
                from conch.open_clip_custom import create_model_from_pretrained
                model, preprocess = create_model_from_pretrained(
                    'conch_ViT-B-16',
                    checkpoint_path=model_path if model_path else "hf_hub:MahmoodLab/conch"
                )
                self.model = model.visual
                self.feature_dim = 512
                self.preprocess = preprocess
            except ImportError:
                print("CONCH not installed. Using timm ViT-B/16 as fallback.")
                vit = timm.create_model(
                    'vit_base_patch16_224', pretrained=True, num_classes=0
                )
                self._setup_vit(vit)
                self.preprocess = None
        else:
            vit = timm.create_model(
                'vit_base_patch16_224', pretrained=False, num_classes=0
            )
            self._setup_vit(vit)
            self.preprocess = None

        # stage_channels for ViT: all same dim
        self.stage_channels = [self.feature_dim] * len(self.extract_layers)

        for p in self.parameters():
            p.requires_grad = trainable

    def _setup_vit(self, vit):
        """Extract ViT components for manual forward"""
        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.model = None  # mark that we use manual forward

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.model is not None:
            # CONCH model: use hooks or simple forward
            # For now, return final features only
            out = self.model(x)
            if isinstance(out, tuple):
                out = out[0]
            return {
                'features': out,
                'multi_scale_feats': [],  # CONCH: TODO add hook-based extraction
            }

        # Manual ViT forward with intermediate extraction
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        multi_scale_feats = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.extract_layers:
                multi_scale_feats.append(x)  # (B, 197, 768)

        x = self.norm(x)
        features = x[:, 0]  # CLS token

        return {
            'features': features,
            'multi_scale_feats': multi_scale_feats,
        }


class ViTBBackbone(nn.Module):
    """
    ViT-B/16 backbone with CONCH pathology pretrained weights.

    Extracts intermediate features from blocks [2, 5, 8, 11] (0-indexed).
    Patch tokens are reshaped to spatial feature maps (B, 768, 14, 14)
    so that CNNStageAdapter (Conv1x1 + GlobalAvgPool) can be used,
    leveraging full spatial information from all 196 patches.

    Outputs:
        features: (B, 768) CLS token from final layer
        multi_scale_feats: [
            (B, 768, 14, 14),  # after block 2
            (B, 768, 14, 14),  # after block 5
            (B, 768, 14, 14),  # after block 8
            (B, 768, 14, 14),  # after block 11
        ]
    """
    CONCH_PATH = "./weights/conch/pytorch_model.bin"

    def __init__(self, pretrained: bool = True, trainable: bool = True):
        super().__init__()

        self.feature_dim = 768
        self.feat_type = 'cnn'  # output spatial maps → use CNNStageAdapter
        self.extract_layers = [2, 5, 8, 11]
        self.stage_channels = [768] * len(self.extract_layers)

        if pretrained:
            from conch.open_clip_custom import create_model_from_pretrained
            model = create_model_from_pretrained(
                'conch_ViT-B-16',
                checkpoint_path=self.CONCH_PATH,
                force_image_size=224,
                return_transform=False
            )
            trunk = model.visual.trunk
        else:
            trunk = timm.create_model(
                'vit_base_patch16_224', pretrained=False, num_classes=0
            )

        self.patch_embed = trunk.patch_embed
        self.cls_token = trunk.cls_token
        self.pos_embed = trunk.pos_embed
        self.pos_drop = trunk.pos_drop
        self.blocks = trunk.blocks
        self.norm = trunk.norm

        # 224 / 16 = 14
        self.grid_size = 14

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        x = self.patch_embed(x)
        # CONCH patch_embed outputs (B, H, W, D); timm outputs (B, N, D)
        if x.dim() == 4:
            x = x.reshape(B, -1, x.shape[-1])  # (B, 196, 768)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        multi_scale_feats = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.extract_layers:
                # Remove CLS token, reshape to spatial map
                patch_tokens = x[:, 1:]  # (B, 196, 768)
                spatial = patch_tokens.transpose(1, 2).reshape(
                    B, -1, self.grid_size, self.grid_size
                )  # (B, 768, 14, 14)
                multi_scale_feats.append(spatial)

        x = self.norm(x)
        features = x[:, 0]  # CLS token

        return {
            'features': features,
            'multi_scale_feats': multi_scale_feats,
        }


def get_backbone(name: str, pretrained: bool = True, trainable: bool = True, **kwargs):
    """
    Factory function to get backbone by name

    All backbones return dict with 'features' and 'multi_scale_feats'.
    """
    backbones = {
        'resnet50': ResNet50Backbone,
        'densenet121': DenseNet121Backbone,
        'conch': CONCHBackbone,
        'vit_b': ViTBBackbone,
    }

    if name not in backbones:
        raise ValueError(f"Unknown backbone: {name}. Choose from {list(backbones.keys())}")

    return backbones[name](pretrained=pretrained, trainable=trainable, **kwargs)
