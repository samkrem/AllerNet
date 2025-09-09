import numpy as np
import torch
from torchvision import datasets, models, transforms
import cv2 
from skimage.measure import label, regionprops

# from scipy.ndimage import median_filter, label as ndimage_label
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from PIL import Image
import pandas as pd
from ..postprocessing.refinement import apply_crf, smooth_segmentation_mask
from .metrics import get_transforms
from ..config import IMG_SIZE



def load_image_and_mask(index, image_dir, mask_dir):
    files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', 'png', 'jpeg')))
    print(files)
    fn = files[index]
    img_path = os.path.join(image_dir, fn)
    base = os.path.splitext(fn)[0]

    for ext in ['.png', '.jpg']:
        mp = os.path.join(mask_dir, base + ext)
        if os.path.exists(mp):
            gt_mask = Image.open(mp)
            break
    else:
        gt_mask = None
    print(img_path)

    return img_path, gt_mask

def annotate_mask(ax, mask, title, class_to_food):
    ax.imshow(mask, cmap="jet", vmin=0, vmax=103)
    ax.axis('off')
    ax.set_title(title)

    for cls in np.unique(mask): #connected regions
        if cls == 0:
            continue  #not labeling background
        binary_mask = (mask == cls).astype(int)
        labeled = label(binary_mask) 
        props = regionprops(labeled)
        for prop in props:
            y, x = prop.centroid
            label_text = f"{cls}: {class_to_food.get(cls, f'Class {cls}')}"
            ax.text(x, y, label_text, color='white', fontsize=14, ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
            
def resize_mask_to_pred(msks, preds): #this version for pytorch, ie during training
    msks = msks.unsqueeze(1).float() #interpolate excpects channel dimension
    msks_resized = F.interpolate(msks, size=preds.shape[1:], mode='nearest')  
    return msks_resized.squeeze(1).long()
def resize_mask(mask_np, target_size): #this version for post/pre visualization
    mask_pil = Image.fromarray(mask_np.astype(np.uint8))
    return np.array(mask_pil.resize(target_size, resample=Image.NEAREST))
def show_segmentation_result(img_tensor, pred_mask_tensor, true_mask_tensor):
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    pred = pred_mask_tensor.detach().cpu().numpy()
    true = true_mask_tensor.detach().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img)
    axs[0].set_title("Input Image")
    axs[1].imshow(true, cmap='tab20')
    axs[1].set_title("Ground Truth")
    axs[2].imshow(pred, cmap='tab20')
    axs[2].set_title("Prediction")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
def visualize_results(img, class_to_food, pred_mask_np, gt_mask_np=None):
    n_plots = 2 if gt_mask_np is not None else 1
    fig, axes = plt.subplots(1, n_plots + 1, figsize=(7 * (n_plots + 1), 7))

    if n_plots == 1:
        axes = [axes]

    axes[0].imshow(cv2.resize(np.array(img), (640, 640)))
    axes[0].axis('off')
    axes[0].set_title("Input Image")

    if gt_mask_np is not None:
        annotate_mask(axes[1], gt_mask_np, "Ground Truth Mask", class_to_food)
        print("Unique ground truth class values:", np.unique(gt_mask_np))

    annotate_mask(axes[-1], pred_mask_np, "Predicted Mask", class_to_food)

    plt.tight_layout()
    plt.show()
def predict_and_show(
    model, class_to_food,
    *,
    test_image_dir=None,
    test_mask_dir=None,
    index: int = None,
    custom_image_path: str = None,
    threshold: float = 0.1,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    model.eval()

    if index is not None:
        img_path, gt_mask = load_image_and_mask(index, test_image_dir, test_mask_dir)
    elif custom_image_path is not None:
        img_path = custom_image_path
        gt_mask = None
    else:
        raise ValueError("Must specify either index or custom_image_path")

    img = Image.open(img_path).convert("RGB")
    transform = get_transforms()
    inp = transform(img).unsqueeze(0).to(device)  # (1, C, H, W)

    with torch.no_grad():
        logits = model(inp) 
        soft_mask = torch.nn.functional.softmax(logits, dim=1).squeeze(0).cpu()  #(num_classes, H, W)

    img_np = np.array(img)
    pred_resized_np = apply_crf(img_np, soft_mask, num_classes=soft_mask.shape[0])  # (H, W)

    pred_resized_np = smooth_segmentation_mask(pred_resized_np, filter_size=5, min_region_size=150)

    gt_resized_np = None
    if gt_mask is not None:
        gt_resized_np = resize_mask(np.array(gt_mask), IMG_SIZE)

    visualize_results(img, class_to_food, pred_resized_np, gt_resized_np)