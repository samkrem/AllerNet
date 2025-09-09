import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp
import timm
from ..config import IMG_SIZE
class AttentionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential( 
            nn.AdaptiveAvgPool2d(1),  #each channel has an average val (N,C,1,1)
            nn.Conv2d(channels, channels // 4, 1, bias=False), #(N,C/4,1,1), 1x1 convolution reduces number of channels (which is like a fully connected  layer), helps compress ingredient scores for images (ie learn common most important patterns)
            nn.Conv2d(channels // 4, channels, 1, bias=False), #expand to original size
            nn.Sigmoid()
        ) #(N,C,H,W) -> (N,C,1,1), 

    def forward(self, x):
        w = self.attn(x)
        return x * w #fusion

class MLAHead(nn.Module):
    def __init__(self, feature_channels, decoder_dim, num_classes, dropout_p=0.3):
        super().__init__()

        # fuse/project each feature map to same channel size
        self.linear_c = nn.ModuleList([ 
            nn.Sequential(
                nn.Conv2d(in_ch, decoder_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(inplace=True)
            ) for in_ch in feature_channels
        ])

        # pyramid pooling module for global context
        pool_scales = (1, 2, 3, 6) #1: global - > 6: local
        self.ppm = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(decoder_dim, decoder_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(inplace=True)
            ) for scale in pool_scales
        ])

        self.linear_fuse = nn.Sequential( #process conctenates tensor to managable decoder dim 
            nn.Conv2d(decoder_dim * len(feature_channels) + decoder_dim * len(pool_scales),
                      decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )

        self.attn_fuse = AttentionFusion(decoder_dim) #emphasize key channels

        self.dropout = nn.Dropout2d(p=dropout_p)

        self.pred = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, features):
        projected = []
        target_size = features[0].shape[2:]  # target spatial size (H, W)

        for i, feature in enumerate(features):
            if feature.shape[-1] != feature.shape[1]:
                feature = feature.permute(0, 3, 1, 2).contiguous()
            proj = self.linear_c[i](feature)
            proj = F.interpolate(proj, size=target_size, mode='bilinear', align_corners=False)
            projected.append(proj)

        # apply ppm on fused high-level feature
        ppm_feats = [] 
        high_level_feat = projected[-1]
        for ppm in self.ppm:
            pooled = ppm(high_level_feat)
            pooled = F.interpolate(pooled, size=target_size, mode='bilinear', align_corners=False)
            ppm_feats.append(pooled)

        # concat mla projections + ppm context
        fused = torch.cat(projected + ppm_feats, dim=1)
        fused = self.linear_fuse(fused)

        fused = self.attn_fuse(fused)
        fused = self.dropout(fused)

        out = self.pred(fused)
        return out


class SwinTransformerWithMLA(nn.Module):
    def __init__(self, num_classes = 104, decoder_dim=256, img_size= IMG_SIZE): # Added img_size parameter
        super().__init__()
        self.backbone = timm.create_model( # swin encoder
            'swin_base_patch4_window7_224',
            pretrained=True,
            features_only=True, 
            img_size=img_size 
        )

        self.decoder = MLAHead( #process multiscale features from backbone, process them, produce final segmentation map
            feature_channels=[f['num_chs'] for f in self.backbone.feature_info],
            decoder_dim=decoder_dim,
            num_classes=num_classes
        )

    def forward(self, x):
        feats = self.backbone(x)  
        out = self.decoder(feats)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)  # upscale to input size
        return out

