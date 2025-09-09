from sklearn.metrics import confusion_matrix
from collections import defaultdict
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from ..config import IMG_SIZE

from ..postprocessing.refinement import resize_mask, apply_tta,apply_crf, apply_superpixel_majority_voting, get_transforms
from .visualization import visualize_results
def fast_hist(preds, targets, num_classes):
    k = (targets >= 0) & (targets < num_classes)
    return torch.bincount(
        num_classes * targets[k].view(-1) + preds[k].view(-1),
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)

def compute_mIoU_from_confmat(confmat):
    intersection = torch.diag(confmat)
    union = confmat.sum(1) + confmat.sum(0) - intersection

    valid = union > 0  # Ignore classes with zero union (i.e., not present at all)
    iou = intersection[valid] / union[valid].clamp(min=1e-6)

    return iou, iou.mean()
def compute_pixel_accuracy(confmat):
    correct = torch.diag(confmat).sum()
    total = confmat.sum()
    return (correct / total).item()


def get_transforms():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
def process_and_compare_masks(test_input_path, test_mask_path, class_to_food, model, device):
    """Process the predicted mask and compare with ground truth mask."""

    # print(test_input_path)
    img = Image.open(test_input_path).convert("RGB").resize(IMG_SIZE, resample=Image.BILINEAR)

    # plt.imshow(np.array(img))
    # plt.axis("off")
    # plt.show()
    transform = get_transforms()
    inp = transform(img).unsqueeze(0).to(device)

    gt_mask_pil = Image.open(test_mask_path).resize(IMG_SIZE, resample=Image.NEAREST)
    gt_mask = np.array(gt_mask_pil)
    # print(gt_mask.shape)
    # plt.imshow(gt_mask)
    # plt.axis("off")
    # plt.show()
    with torch.no_grad():
        model.to(device)
        model.eval()
        soft_pred = model(inp).detach().cpu()
    soft_pred = soft_pred.squeeze(0).numpy()
    pred_resized_np = apply_crf(np.array(img), soft_pred, 104)
    pred_resized_np = apply_superpixel_majority_voting(pred_resized_np, np.array(img))

    if pred_resized_np.shape != gt_mask.shape:
        pred_resized_np = np.array(Image.fromarray(pred_resized_np.astype(np.uint8)).resize(gt_mask.shape[::-1], resample=Image.NEAREST))

    pred_flattened = pred_resized_np.flatten()
    gt_flattened = gt_mask.flatten()

    labels = sorted(set(gt_flattened.tolist() + pred_flattened.tolist()))
    cm = confusion_matrix(gt_flattened, pred_flattened, labels=labels)

    gt_counts = {}
    pred_counts = {}

    for idx, label_id in enumerate(labels):
        if cm.sum(axis=1)[idx] > 0:
            food_name = class_to_food.get(label_id, f"Unknown_{label_id}")
            gt_counts[food_name] = cm.sum(axis=1)[idx]
        if cm.sum(axis=0)[idx] > 0:
            food_name = class_to_food.get(label_id, f"Unknown_{label_id}")
            pred_counts[food_name] = cm.sum(axis=0)[idx]
    # print("GT Mask shape:", gt_mask.shape)
    # print("Prediction shape:", pred_resized_np.shape)
    # print("Image shape:", np.array(img).shape)

    # visualize_results(img, pred_resized_np, gt_mask, csv_path=labels_dir + "/food_allergens.csv")
    return gt_counts, pred_counts

def process_all_images(test_im_dir, test_mask_dir, class_to_food, model, device):
    """Process all images and summarize GT and predicted food pixel counts."""
    all_gt_counts = defaultdict(int)
    all_pred_counts = defaultdict(int)

    test_images = sorted([f for f in os.listdir(test_im_dir) if f.endswith(('.png', '.jpg'))])
    test_masks = sorted([f for f in os.listdir(test_mask_dir) if f.endswith(('.png', '.jpg'))])

    for test_img, test_mask in tqdm(zip(test_images, test_masks), total=len(test_images), desc="Processing images"):
        test_input_path = os.path.join(test_im_dir, test_img)
        test_mask_path = os.path.join(test_mask_dir, test_mask)

        gt_counts, pred_counts = process_and_compare_masks(test_input_path, test_mask_path, class_to_food, model, device)

        for food, count in gt_counts.items():
            all_gt_counts[food] += count
        for food, count in pred_counts.items():
            all_pred_counts[food] += count

    total_gt_pixels = sum(all_gt_counts.values())
    total_pred_pixels = sum(all_pred_counts.values())

    def sort_counts_with_percentages(counts_dict, total_pixels):
        sorted_items = sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)
        return [(food, count, 100 * count / total_pixels if total_pixels > 0 else 0) for food, count in sorted_items]

    sorted_gt_counts = sort_counts_with_percentages(all_gt_counts, total_gt_pixels)
    sorted_pred_counts = sort_counts_with_percentages(all_pred_counts, total_pred_pixels)

    # Display stats
    print("\nTop 5 Ground Truth Foods by Pixel Count and Percentage:")
    for food, count, percent in sorted_gt_counts[:5]:
        print(f"{food:<25}: {count:>8} pixels ({percent:.2f}%)")

    print("\nBottom 5 Ground Truth Foods by Pixel Count and Percentage:")
    for food, count, percent in sorted_gt_counts[-5:]:
        print(f"{food:<25}: {count:>8} pixels ({percent:.2f}%)")

    print("\nTop 5 Predicted Foods by Pixel Count and Percentage:")
    for food, count, percent in sorted_pred_counts[:5]:
        print(f"{food:<25}: {count:>8} pixels ({percent:.2f}%)")

    print("\nBottom 5 Predicted Foods by Pixel Count and Percentage:")
    for food, count, percent in sorted_pred_counts[-5:]:
        print(f"{food:<25}: {count:>8} pixels ({percent:.2f}%)")

    # Create DataFrames
    gt_df = pd.DataFrame(sorted_gt_counts, columns=["Food", "GT Pixel Count", "GT Percentage"])
    pred_df = pd.DataFrame(sorted_pred_counts, columns=["Food", "Pred Pixel Count", "Pred Percentage"])

    # Merge GT and Predicted info into one DataFrame (optional)
    merged_df = pd.merge(gt_df, pred_df, on="Food", how="outer").fillna(0)

    return merged_df, gt_df, pred_df
def calculate_class_set_metrics(model, test_image_dir, test_mask_dir, num_images, threshold=0.5, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    class_set_accuracies = []
    class_set_precisions = []
    class_set_recalls = []

    files = sorted(f for f in os.listdir(test_image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg')))

    for i in tqdm(range(num_images), desc="Evaluating"):
        try:
            fn = files[i]
            base = os.path.splitext(fn)[0]

            mask_path_png = os.path.join(test_mask_dir, base + ".png")
            mask_path_jpg = os.path.join(test_mask_dir, base + ".jpg")
            mp = mask_path_png if os.path.exists(mask_path_png) else mask_path_jpg
            if not os.path.exists(mp):
                print(f"Mask not found for {fn}, skipping.")
                continue

            # Load and preprocess ground truth mask (multiclass)
            gt_mask = np.array(Image.open(mp))
            gt_mask = gt_mask.astype(np.uint8)

            # Image preprocessing
            img_tf = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            img_path = os.path.join(test_image_dir, fn)
            img = Image.open(img_path).convert("RGB")
            w, h = img.size  # Move this here before resizing
            inp = img_tf(img).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                tta_out = apply_tta(model, inp, device)  # Shape: (1, C, H, W)
                prob_map = tta_out.squeeze().cpu().numpy()  # Shape: (C, H, W)

            # --- Skip CRF and superpixel for debugging ---
            # pred_mask = apply_crf(...) + superpixel majority voting
            pred_mask = np.argmax(prob_map, axis=0)  # Multiclass prediction

            # Resize prediction mask to original size
            pred_pil = Image.fromarray(pred_mask.astype(np.uint8))
            pred_up = pred_pil.resize((w, h), resample=Image.NEAREST)
            pred_mask = np.array(pred_up)

            # --- Calculate set-based metrics ---
            gt_classes = set(np.unique(gt_mask))
            pred_classes = set(np.unique(pred_mask))
            true_classes = gt_classes.intersection(pred_classes)

            class_set_accuracy = len(true_classes) / len(gt_classes) if len(gt_classes) > 0 else 0
            class_set_precision = len(true_classes) / len(pred_classes) if len(pred_classes) > 0 else 0
            class_set_recall = len(true_classes) / len(gt_classes) if len(gt_classes) > 0 else 0

            class_set_accuracies.append(class_set_accuracy)
            class_set_precisions.append(class_set_precision)
            class_set_recalls.append(class_set_recall)

        except Exception as e:
            print(f"Error processing image {i} ({fn}): {e}")
            continue

    avg_class_set_accuracy = np.mean(class_set_accuracies) if class_set_accuracies else 0
    avg_class_set_precision = np.mean(class_set_precisions) if class_set_precisions else 0
    avg_class_set_recall = np.mean(class_set_recalls) if class_set_recalls else 0

    print("\nOverall Metrics:")
    print(f"Average Class Set Accuracy: {avg_class_set_accuracy:.4f}")
    print(f"Average Class Set Precision: {avg_class_set_precision:.4f}")
    print(f"Average Class Set Recall: {avg_class_set_recall:.4f}")

    return avg_class_set_accuracy, avg_class_set_precision, avg_class_set_recall
def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int):
    epsilon = 1e-7
    total_accuracy = np.mean(pred_mask == gt_mask)

    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_iou = 0
    num_valid_classes = 0

    per_class_metrics = {}  # Optional: useful for debugging

    for cls in range(num_classes): #average across class
        pred_cls = (pred_mask == cls)
        gt_cls = (gt_mask == cls)

        # Skip if class not present in ground truth
        if np.sum(gt_cls) == 0:
            continue

        tp = np.sum(pred_cls & gt_cls)
        fp = np.sum(pred_cls & ~gt_cls)
        fn = np.sum(~pred_cls & gt_cls)

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        iou = tp / (tp + fp + fn + epsilon)

        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_iou += iou
        num_valid_classes += 1

    # Avoid division by zero
    if num_valid_classes == 0:
        return 0, 0, 0, 0, 0, {}

    return (
        total_accuracy,
        total_precision / num_valid_classes,
        total_recall / num_valid_classes,
        total_f1 / num_valid_classes,
        total_iou / num_valid_classes,
    )
def calculate_segmentation_metrics_per_image(
    model,
    test_image_dir,
    test_mask_dir,
    num_images,
    num_classes,
    class_to_food,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    show_images=False
):
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_iou = 0
    valid_count = 0

    files = sorted(f for f in os.listdir(test_image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg')))

    for i in tqdm(range(num_images), desc="Evaluating"):
        try:
            fn = files[i]
            base = os.path.splitext(fn)[0]

            img_path = os.path.join(test_image_dir, fn)
            mask_path = os.path.join(test_mask_dir, base + ".png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(test_mask_dir, base + ".jpg")
            if not os.path.exists(mask_path):
                print(f"Mask not found for {fn}, skipping.")
                continue

            # === YOUR EXACT POST-PROCESSING PIPELINE STARTS HERE ===
            img = Image.open(img_path).convert("RGB")
            gt_mask = np.array(Image.open(mask_path))

            transform = get_transforms()
            inp = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                soft_pred = apply_tta(model, inp, device)  # shape: (1, C, 512, 512)

            img_np = np.array(img.resize(IMG_SIZE))
            soft_pred_np = soft_pred.squeeze(0).cpu().numpy()  # shape: (C, 512, 512)
            pred_mask_np = apply_crf(img_np, soft_pred_np, num_classes=104)
            pred_mask_np = apply_superpixel_majority_voting(pred_mask_np, img_np)

            gt_resized_np = resize_mask(gt_mask, IMG_SIZE)

            if show_images:
                visualize_results(img, class_to_food, pred_mask_np, gt_resized_np)

            acc, prec, rec, f1, iou = compute_metrics(pred_mask_np, gt_resized_np, num_classes)
            total_accuracy += acc
            total_precision += prec
            total_recall += rec
            total_f1 += f1
            total_iou += iou
            valid_count += 1

        except Exception as e:
            print(f"Error processing image {i} ({files[i]}): {e}")
            continue

    if valid_count == 0:
        print("No valid images processed.")
        return

    print(f"\n--- Segmentation Metrics over {valid_count} images ---")
    print(f"Pixel-wise Accuracy:  {total_accuracy / valid_count:.4f}")
    print(f"Precision:            {total_precision / valid_count:.4f}")
    print(f"Recall (Sensitivity): {total_recall / valid_count:.4f}")
    print(f"F1 Score:             {total_f1 / valid_count:.4f}")
    print(f"IoU:                  {total_iou / valid_count:.4f}")

    return {
        "accuracy": total_accuracy / valid_count,
        "precision": total_precision / valid_count,
        "recall": total_recall / valid_count,
        "f1": total_f1 / valid_count,
        "iou": total_iou / valid_count
    }

def compute_classwise_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int):
    epsilon = 1e-7
    class_metrics = {}

    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        gt_cls = (gt_mask == cls)

        if np.sum(gt_cls) == 0:
            continue  # Skip class if not in ground truth

        tp = np.sum(pred_cls & gt_cls)
        fp = np.sum(pred_cls & ~gt_cls)
        fn = np.sum(~pred_cls & gt_cls)

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        iou = tp / (tp + fp + fn + epsilon)

        class_metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou
        }

    return class_metrics
from collections import defaultdict

def calculate_segmentation_metrics_per_class(
    model,
    test_image_dir,
    test_mask_dir,
    num_images,
    num_classes,
    class_to_food,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    show_images=False
):
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_iou = 0
    valid_count = 0

    # Track class-wise metrics
    classwise_sums = defaultdict(lambda: {"precision": 0, "recall": 0, "f1": 0, "iou": 0, "count": 0})

    files = sorted(f for f in os.listdir(test_image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg')))

    for i in tqdm(range(num_images), desc="Evaluating"):
        try:
            fn = files[i]
            base = os.path.splitext(fn)[0]

            img_path = os.path.join(test_image_dir, fn)
            mask_path = os.path.join(test_mask_dir, base + ".png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(test_mask_dir, base + ".jpg")
            if not os.path.exists(mask_path):
                print(f"Mask not found for {fn}, skipping.")
                continue

            img = Image.open(img_path).convert("RGB")
            gt_mask = np.array(Image.open(mask_path))
            transform = get_transforms()
            inp = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                soft_pred = apply_tta(model, inp, device)

            img_np = np.array(img.resize(IMG_SIZE))
            soft_pred_np = soft_pred.squeeze(0).cpu().numpy()
            pred_mask_np = apply_crf(img_np, soft_pred_np, num_classes=num_classes)
            pred_mask_np = apply_superpixel_majority_voting(pred_mask_np, img_np)

            gt_resized_np = resize_mask(gt_mask, IMG_SIZE)

            if show_images:
                visualize_results(img, class_to_food, pred_mask_np, gt_resized_np)

            acc = np.mean(pred_mask_np == gt_resized_np)
            total_accuracy += acc

            class_metrics = compute_classwise_metrics(pred_mask_np, gt_resized_np, num_classes)

            for cls, metrics in class_metrics.items():
                classwise_sums[cls]["precision"] += metrics["precision"]
                classwise_sums[cls]["recall"] += metrics["recall"]
                classwise_sums[cls]["f1"] += metrics["f1"]
                classwise_sums[cls]["iou"] += metrics["iou"]
                classwise_sums[cls]["count"] += 1

            valid_count += 1

        except Exception as e:
            print(f"Error processing image {i} ({files[i]}): {e}")
            continue

    if valid_count == 0:
        print("No valid images processed.")
        return

    print(f"\n--- Segmentation Metrics over {valid_count} images ---")
    print(f"Pixel-wise Accuracy:  {total_accuracy / valid_count:.4f}")

    classwise_avg = {}
    for cls, metrics in classwise_sums.items():
        count = metrics["count"]
        if count == 0:
            continue
        classwise_avg[class_to_food[cls]] = {
            "precision": metrics["precision"] / count,
            "recall": metrics["recall"] / count,
            "f1": metrics["f1"] / count,
            "iou": metrics["iou"] / count
        }

    return {
        "pixel_accuracy": total_accuracy / valid_count,
        "per_class_metrics": classwise_avg
    }
def create_multichannel_allergen_mask_from_np(mask_np, allergens, food_allergens_df):
    """
    Convert a food label mask into a multi-channel binary mask per allergen.

    mask_np: (H, W) array of food IDs
    allergens: list of allergen names in desired order
    food_allergens_df: pandas DataFrame where index is food ID and columns are allergens (0/1 values)
    """
    h, w = mask_np.shape
    multi_mask = np.zeros((len(allergens), h, w), dtype=np.uint8)

    unique_ids = np.unique(mask_np)
    for food_id in unique_ids:
        if food_id not in food_allergens_df.index:
            continue
        for i, allergen in enumerate(allergens):
            if food_allergens_df.loc[food_id, allergen] == 1:
                multi_mask[i][mask_np == food_id] = 1

    return multi_mask


def calculate_allergen_metrics(
    model,
    test_image_dir,
    test_mask_dir,
    allergens,
    food_allergens_df,
    num_images,
    num_classes,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    show_images=False
):
    # Initialize totals
    total_accuracy = {a: 0.0 for a in allergens}
    total_precision = {a: 0.0 for a in allergens}
    total_recall = {a: 0.0 for a in allergens}
    total_f1 = {a: 0.0 for a in allergens}
    total_iou = {a: 0.0 for a in allergens}
    valid_count = {a: 0 for a in allergens}

    files = sorted(f for f in os.listdir(test_image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg')))

    for i in tqdm(range(num_images), desc="Evaluating allergens"):
        try:
            fn = files[i]
            base = os.path.splitext(fn)[0]

            img_path = os.path.join(test_image_dir, fn)
            mask_path = os.path.join(test_mask_dir, base + ".png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(test_mask_dir, base + ".jpg")
            if not os.path.exists(mask_path):
                print(f"Mask not found for {fn}, skipping.")
                continue

            # Load image & GT
            img = Image.open(img_path).convert("RGB")
            gt_mask = np.array(Image.open(mask_path))

            # Transform & predict
            transform = get_transforms()
            inp = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                soft_pred = apply_tta(model, inp, device)

            img_np = np.array(img.resize(IMG_SIZE))
            soft_pred_np = soft_pred.squeeze(0).cpu().numpy()
            pred_mask_np = apply_crf(img_np, soft_pred_np, num_classes=num_classes)
            pred_mask_np = apply_superpixel_majority_voting(pred_mask_np, img_np)

            # Resize GT mask
            gt_resized_np = resize_mask(gt_mask, IMG_SIZE)

            # Convert to binary masks per allergen
            pred_multi_mask = create_multichannel_allergen_mask_from_np(pred_mask_np, allergens, food_allergens_df)
            gt_multi_mask = create_multichannel_allergen_mask_from_np(gt_resized_np, allergens, food_allergens_df)

            # Compute metrics per allergen
            for idx, allergen in enumerate(allergens):
                pred_bin = pred_multi_mask[idx]
                gt_bin = gt_multi_mask[idx]

                if np.sum(gt_bin) == 0 and np.sum(pred_bin) == 0:
                    # No allergen present
                    continue

                tp = np.sum((pred_bin == 1) & (gt_bin == 1))
                fp = np.sum((pred_bin == 1) & (gt_bin == 0))
                fn = np.sum((pred_bin == 0) & (gt_bin == 1))
                tn = np.sum((pred_bin == 0) & (gt_bin == 0))

                accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                iou = tp / (tp + fp + fn + 1e-8)

                total_accuracy[allergen] += accuracy
                total_precision[allergen] += precision
                total_recall[allergen] += recall
                total_f1[allergen] += f1
                total_iou[allergen] += iou
                valid_count[allergen] += 1

        except Exception as e:
            print(f"Error processing {files[i]}: {e}")
            continue

    # Average metrics
    for allergen in allergens:
        if valid_count[allergen] > 0:
            total_accuracy[allergen] /= valid_count[allergen]
            total_precision[allergen] /= valid_count[allergen]
            total_recall[allergen] /= valid_count[allergen]
            total_f1[allergen] /= valid_count[allergen]
            total_iou[allergen] /= valid_count[allergen]

    return total_accuracy, total_precision, total_recall, total_f1, total_iou
def calculate_allergen_metrics_image_level(
    model,
    test_image_dir,
    test_mask_dir,
    allergens,
    food_allergens_df,
    num_images,
    num_classes,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    # Initialize totals
    total_accuracy = {a: 0.0 for a in allergens}
    total_precision = {a: 0.0 for a in allergens}
    total_recall = {a: 0.0 for a in allergens}
    total_f1 = {a: 0.0 for a in allergens}
    valid_count = {a: 0 for a in allergens}

    files = sorted(f for f in os.listdir(test_image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg')))

    for i in tqdm(range(num_images), desc="Evaluating allergens (image-level)"):
        try:
            fn = files[i]
            base = os.path.splitext(fn)[0]

            img_path = os.path.join(test_image_dir, fn)
            mask_path = os.path.join(test_mask_dir, base + ".png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(test_mask_dir, base + ".jpg")
            if not os.path.exists(mask_path):
                print(f"Mask not found for {fn}, skipping.")
                continue

            # Load image & GT
            img = Image.open(img_path).convert("RGB")
            gt_mask = np.array(Image.open(mask_path))

            # Transform & predict
            transform = get_transforms()
            inp = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                soft_pred = apply_tta(model, inp, device)

            img_np = np.array(img.resize(IMG_SIZE))
            soft_pred_np = soft_pred.squeeze(0).cpu().numpy()
            pred_mask_np = apply_crf(img_np, soft_pred_np, num_classes=num_classes)
            pred_mask_np = apply_superpixel_majority_voting(pred_mask_np, img_np)

            # Resize GT mask
            gt_resized_np = resize_mask(gt_mask, IMG_SIZE)

            # Convert to binary masks per allergen
            pred_multi_mask = create_multichannel_allergen_mask_from_np(pred_mask_np, allergens, food_allergens_df)
            gt_multi_mask = create_multichannel_allergen_mask_from_np(gt_resized_np, allergens, food_allergens_df)

            # Convert to "allergen present" flags
            pred_present = [int(np.any(pred_multi_mask[idx])) for idx in range(len(allergens))]
            gt_present = [int(np.any(gt_multi_mask[idx])) for idx in range(len(allergens))]

            # Compute metrics for each allergen (image-level)
            for idx, allergen in enumerate(allergens):
                pred_label = pred_present[idx]
                gt_label = gt_present[idx]

                tp = int(pred_label == 1 and gt_label == 1)
                fp = int(pred_label == 1 and gt_label == 0)
                fn = int(pred_label == 0 and gt_label == 1)
                tn = int(pred_label == 0 and gt_label == 0)

                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall / (precision + recall + 1e-8)
                      if (precision + recall) > 0 else 0.0)

                total_accuracy[allergen] += accuracy
                total_precision[allergen] += precision
                total_recall[allergen] += recall
                total_f1[allergen] += f1
                valid_count[allergen] += 1

        except Exception as e:
            print(f"Error processing {files[i]}: {e}")
            continue

    # Average metrics
    for allergen in allergens:
        if valid_count[allergen] > 0:
            total_accuracy[allergen] /= valid_count[allergen]
            total_precision[allergen] /= valid_count[allergen]
            total_recall[allergen] /= valid_count[allergen]
            total_f1[allergen] /= valid_count[allergen]

    return total_accuracy, total_precision, total_recall, total_f1


