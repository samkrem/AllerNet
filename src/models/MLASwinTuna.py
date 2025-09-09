import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from AllerNet.src.config import IMG_SIZE

class AttentionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.attn(x)
        return x * w

class MLAHead(nn.Module):
    def __init__(self, feature_channels, decoder_dim, num_classes, dropout_p=0.3):
        super().__init__()
        # Project each feature map to decoder_dim channels
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, decoder_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(inplace=True)
            ) for in_ch in feature_channels # Dynamically set input channels
        ])

        # Pyramid pooling for global context
        pool_scales = (1, 2, 3, 6)
        self.ppm = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(decoder_dim, decoder_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(inplace=True)
            ) for scale in pool_scales
        ])

        # Fuse all features + PPM
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * len(feature_channels) + decoder_dim * len(pool_scales),
                      decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )

        self.attn_fuse = AttentionFusion(decoder_dim)
        self.dropout = nn.Dropout2d(dropout_p)
        self.pred = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, features):
        projected = []
        target_size = features[0].shape[2:] # Use spatial size of the first feature map as target

        for i, feat in enumerate(features):
            # Permute from (N, H, W, C) to (N, C, H, W)
            feat = feat.permute(0, 3, 1, 2).contiguous()
            proj = self.projections[i](feat)
            # Interpolate features to the lowest resolution size (target_size)
            proj = F.interpolate(proj, size=target_size, mode='bilinear', align_corners=False)
            projected.append(proj)

        # Apply PPM on the highest-level feature (last feature in the list after permuting)
        ppm_feats = []
        # The highest level feature corresponds to the deepest stage, which has the smallest spatial dimensions.
        # Apply PPM to the projected and resized highest-level feature
        high_level_feat_for_ppm = projected[-1] # Use the already projected and resized highest-level feature
        for ppm in self.ppm:
            pooled = ppm(high_level_feat_for_ppm)
            # PPM output is already at the target size due to AdaptiveAvgPool2d
            # We still need to ensure the interpolation mode is consistent if the scale != 1
            if pooled.shape[2:] != target_size:
                 pooled = F.interpolate(pooled, size=target_size, mode='bilinear', align_corners=False)
            ppm_feats.append(pooled)


        # Concatenate projected features and PPM features
        fused = torch.cat(projected + ppm_feats, dim=1)
        fused = self.fuse(fused)
        fused = self.attn_fuse(fused)
        fused = self.dropout(fused)
        out = self.pred(fused)
        return out

class SwinTransformerWithMLA(nn.Module):
    def __init__(self, num_classes=104, decoder_dim=256, backbone_name='swin_base_patch4_window7_224', img_size=IMG_SIZE):
        super().__init__()
        # Backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=(0,1,2,3), # Ensure we get features from all stages
            img_size=img_size
        )
        # Decoder
        # Get feature channel sizes from the backbone
        feature_channels = [f['num_chs'] for f in self.backbone.feature_info]
        self.decoder = MLAHead(
            feature_channels=feature_channels, # Pass the dynamic channels
            decoder_dim=decoder_dim,
            num_classes=num_classes
        )

    def forward(self, x):
        feats = self.backbone(x)  # list of 4 feature maps, potentially NWHC or NCHW
        out = self.decoder(feats)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out