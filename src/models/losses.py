import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

food_label_distribution = config.FOOD_LABEL_DISTRIBUTION
food_dist_df = pd.read_csv(food_label_distribution)
df = pd.read_csv(food_label_distribution)
pixel_counts = df["Count"].values

frequencies = pixel_counts / pixel_counts.sum()
alpha = 0.1  # smoothing factor
smoothed_freqs = alpha + (1 - alpha) * frequencies
median = np.median(smoothed_freqs[smoothed_freqs > 0])
weights = median / smoothed_freqs
weights = np.clip(weights, 0, 10)
class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

cross_entropy = nn.CrossEntropyLoss(weight=class_weights_tensor)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        logits = F.interpolate(logits, size=targets.shape[1:], mode="bilinear", align_corners=False) #(N,C,128,128) ->  (N,C,512,512)

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2) #match target dim w logit dim, (N,H,W) -> (N,C,H,W)

        focal_weight = (1 - probs) ** self.gamma
        focal_log_probs = focal_weight * log_probs

        loss = - (targets_one_hot * focal_log_probs)

        if self.weight is not None:
            weight_tensor = self.weight.view(1, -1, 1, 1)  # (1, C, 1, 1)
            loss = loss * weight_tensor

        return loss.sum(dim=1).mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = F.interpolate(logits, size=targets.shape[1:], mode="bilinear", align_corners=False)

        num_classes = logits.shape[1]
        logits = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        intersection = (logits * targets_one_hot).sum(dim=(2, 3))
        union = logits.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
def combined_loss(logits, targets, ce_weight=0.5, dice_weight=0.5):
    ce = cross_entropy(logits, targets)
    dc = DiceLoss()(logits, targets)
    return ce_weight * ce + dice_weight * dc
# def combined_loss(logits, targets, focal_weight=0.7, dice_weight=0.3):
#     focal = FocalLoss(weight=class_weights_tensor, gamma=2.0)(logits, targets)
#     dc = DiceLoss()(logits, targets)
#     return focal_weight * focal + dice_weight * dc
