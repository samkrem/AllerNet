from scipy.ndimage import median_filter
from scipy.ndimage import label as ndimage_label
import os
from PIL import Image
import torch
import cv2
from skimage.segmentation import slic
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import numpy as np
from torchvision import  transforms
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from ..config import IMG_SIZE

def apply_superpixel_majority_voting(pred_mask, image, n_segments=250, compactness=10):
    image = cv2.resize(image, (pred_mask.shape[1], pred_mask.shape[0]))

    if len(image.shape) == 3 and image.shape[2] == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image  

    segments = slic(image_gray, n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=None) #no channel axis bc grayscale

    out = np.copy(pred_mask)
    for seg_val in np.unique(segments):
        mask = segments == seg_val
        labels, counts = np.unique(pred_mask[mask], return_counts=True)
        if len(counts) > 0:
            majority_class = labels[np.argmax(counts)]
            out[mask] = majority_class

    return cv2.resize(out.astype(np.uint8), IMG_SIZE, interpolation=cv2.INTER_NEAREST)
def smooth_segmentation_mask(mask, filter_size=5, min_region_size=100):
    smoothed = median_filter(mask, size=filter_size)

    cleaned_mask = np.zeros_like(smoothed)
    for class_id in np.unique(smoothed): #discard small noisy regions
        binary_mask = (smoothed == class_id).astype(np.uint8)
        labeled_array, num_features = ndimage_label(binary_mask)
        for region_label in range(1, num_features + 1):
            region = (labeled_array == region_label)
            if np.sum(region) >= min_region_size:
                cleaned_mask[region] = class_id
    return cleaned_mask

def apply_tta(model, image_tensor, device):
    flips = [lambda x: x, lambda x: torch.flip(x, dims=[-1]), lambda x: torch.flip(x, dims=[-2])] #flip horizontally, vertically, original
    preds = []
    with torch.no_grad(): 
        model.eval() 
        for flip in flips:
            flipped = flip(image_tensor)
       
            pred = model(flipped.to(device)).detach().cpu()
            if flip == flips[1]: # flip back
                 pred = torch.flip(pred, dims=[-1])
            elif flip == flips[2]: #flip back
                 pred = torch.flip(pred, dims=[-2])
            preds.append(pred)

    # Average the soft predictions from TTA
    mean_pred = torch.mean(torch.stack(preds), dim=0)  #average the three
    return mean_pred 


def apply_crf(image, soft_mask, num_classes = 104):

    c, h, w = soft_mask.shape

    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if hasattr(soft_mask, "numpy"):
        soft_mask = soft_mask.numpy()

    softmax = np.clip(soft_mask, 1e-8, 1.0) #avoid log0

    d = dcrf.DenseCRF2D(w, h, num_classes)
    unary = unary_from_softmax(softmax)  # shape (C, H*W)
    d.setUnaryEnergy(unary)

    # Add pairwise potentials
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)

    Q = d.inference(5)
    result = np.argmax(Q, axis=0).reshape((h, w)).astype(np.uint8)

    result = cv2.resize(result, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

    return result
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

def get_transforms():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
def predict_mask(model, image_tensor, gt_mask_tensor, device, class_to_food):
    with torch.no_grad():
        model.eval()
        logits = apply_tta(model, image_tensor, device)

        print("Logits shape:", logits.shape)

        prob_map = torch.nn.functional.softmax(logits, dim=1)
        pred_mask = torch.argmax(prob_map, dim=1).squeeze().cpu().numpy()

        if gt_mask_tensor is not None:
            gt_mask = gt_mask_tensor.squeeze().cpu().numpy()
            gt_mask = np.round(gt_mask * 255).astype(np.uint8)
            # make sure GT mask matches predicted mask shape
            if gt_mask.shape != pred_mask.shape:
                print(f"Resizing GT mask from {gt_mask.shape} to {pred_mask.shape}")
                gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            pred_classes = set(np.unique(pred_mask))
            gt_classes = set(np.unique(gt_mask))
            def label_set_to_string(label_set):
                return [f"{cls}: {class_to_food.get(cls, f'Class {cls}')}" for cls in sorted(label_set)]
            print("Unique predicted class values:        ", label_set_to_string(pred_classes))
            print("Unique ground truth class values:     ", label_set_to_string(gt_classes))
            print("Classes in both prediction and GT:    ", label_set_to_string(pred_classes & gt_classes))
            print("Classes only in prediction:           ", label_set_to_string(pred_classes - gt_classes))
            print("Classes only in ground truth:         ", label_set_to_string(gt_classes - pred_classes))

            for cls in sorted(pred_classes | gt_classes): #computer miou for each class
                pred_inds = (pred_mask == cls)
                gt_inds = (gt_mask == cls)
                intersection = np.logical_and(pred_inds, gt_inds).sum()
                union = np.logical_or(pred_inds, gt_inds).sum()
                iou = intersection / union if union > 0 else 0
                print(f"Class {cls} {class_to_food[cls]}  IoU: {iou:.4f}")
        else:
            print("No ground truth mask provided.")

        return pred_mask

def resize_mask(mask_np, target_size):
    mask_pil = Image.fromarray(mask_np.astype(np.uint8))
    return np.array(mask_pil.resize(target_size, resample=Image.NEAREST))

    # def predict_and_show(
#     model,
#     *,
#     test_image_dir=None,
#     test_mask_dir=None,
#     index: int = None,
#     custom_image_path: str = None,
#     threshold: float = 0.1,
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
# ):
#     model.eval()
#     if index is not None:
#         img_path, gt_mask = load_image_and_mask(index, test_image_dir, test_mask_dir)
#     elif custom_image_path is not None:
#         img_path = custom_image_path
#         gt_mask = None
#     else:
#         raise ValueError("Must specify either index or custom_image_path")

#     # Load and preprocess image
#     img = Image.open(img_path).convert("RGB")
#     transform = get_transforms()
#     inp = transform(img).unsqueeze(0).to(device)

#     # Perform Test-Time Augmentation (TTA)
#     if False:
#         pred_mask_np = apply_tta(model, inp, device)
#     else:
#         with torch.no_grad():
#             pred_mask_np = model(inp).detach().cpu()

#     # Run prediction (no resizing involved)
#     # Pass the ground truth mask tensor (gt_mask) to predict_mask
#     gt_mask_tensor = transforms.ToTensor()(gt_mask).unsqueeze(0).to(device) if gt_mask is not None else None  # Convert to tensor if gt_mask exists
#     pred_mask_np = predict_mask(model, inp, gt_mask_tensor, device)

#     # Do not resize the image, retain original dimensions for prediction mask
#     pred_resized_np = pred_mask_np  # No resizing

#     if True:
#         img_np = np.array(img)
#         pred_resized_np = apply_crf(img_np, pred_mask_np, 104)

#     # Apply Superpixel Majority Voting (optional)
#     # if True:
#     #     img_np = np.array(img)
#     #     pred_resized_np = apply_superpixel_majority_voting(pred_resized_np, img_np)
#     pred_resized_np = smooth_segmentation_mask(pred_resized_np, filter_size=5, min_region_size=150)

#     # If ground truth exists, resize it to the original image size
#     gt_resized_np = None
#     if gt_mask is not None:
#         gt_resized_np = resize_mask(np.array(gt_mask), (512,512))
#     # Visualize input, prediction, and GT (if available)
#     visualize_results(img, pred_resized_np, gt_resized_np)
