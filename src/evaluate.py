import torch
import pandas as pd
from collections import defaultdict
import config
from utils.visualization import predict_and_show
from utils.metrics import *
from models.MLASwin import SwinTransformerWithMLA


if __name__ == "__main__": 

    train_im_dir, train_mask_dir = config.TRAIN_IM_DIR, config.TRAIN_MASK_DIR
    test_im_dir, test_mask_dir = config.TEST_IM_DIR, config.TEST_MASK_DIR
    target_save_dir = config.TARGET_SAVE_DIR
    labels_csv_path = config.LABELS_CSV_PATH
    food_allergens = config.FOOD_ALLERGENS

    food_allergens_df = pd.read_csv(food_allergens)
    class_to_food = dict(zip(food_allergens_df['id'], food_allergens_df['food'])) 

    model = SwinTransformerWithMLA(num_classes=104, decoder_dim=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    for i in range(0,20):
        predict_and_show(
            model, class_to_food,
            test_image_dir=test_im_dir,
            test_mask_dir =test_mask_dir,
            index = i)
    merged_df, gt_df, pred_df = process_all_images(test_im_dir, test_mask_dir, food_allergens, model, device)
    merged_df.to_csv(config.PIXEL_SUMMARY, index=False)

    pixel_summary_df = pd.read_csv(config.PIXEL_SUMMARY)
    
    all_gt_counts = defaultdict(int, dict(zip(pixel_summary_df["Food"], pixel_summary_df["GT Pixel Count"])))

    all_pred_counts = defaultdict(int, dict(zip(pixel_summary_df["Food"], pixel_summary_df["Pred Pixel Count"])))

    gt_counts_dict = dict(all_gt_counts)
    pred_counts_dict = dict(all_pred_counts)

    print(gt_counts_dict)
    print(pred_counts_dict )
    
    test_im_path = config.TEST_IM_DIR + "00000048.jpg"  
    test_mask_path = config.TEST_MASK_DIR  + "00000048.png"
    process_and_compare_masks(test_im_path, test_mask_path, food_allergens, model, device)
    
    avg_accuracy, avg_precision, avg_recall = calculate_class_set_metrics(model, test_im_dir, test_mask_dir, num_images=1000)

    calculate_segmentation_metrics_per_image(model, test_im_dir, test_mask_dir, 1000, 104, class_to_food)

    results = calculate_segmentation_metrics_per_class(model, test_im_dir,test_mask_dir, 1000, 104, class_to_food, show_images=False  
    )

    print("\n--- Overall Metrics ---")
    for metric, value in results.items():
        if metric != "per_class":
            if isinstance(value, (float, int)):
                print(f"{metric.capitalize()}: {value:.4f}")
            elif isinstance(value, dict):
                print(f"{metric.capitalize()}:")
                for sub_metric, sub_value in value.items():
                    if isinstance(sub_value, (float, int)):
                        print(f"  {sub_metric}: {sub_value:.4f}")
                    else:
                        print(f"  {sub_metric}: {sub_value}")
            else:
                print(f"{metric.capitalize()}: {value}")
    
    allergens = ["milk", "egg", "fish", "crustacean shellfish",
                "tree nuts", "peanut", "wheat", "soy"]
    num_images = len([f for f in os.listdir(test_im_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    food_allergens_df = pd.read_csv(food_allergens,  index_col="id")

    acc, prec, rec, f1, iou = calculate_allergen_metrics(
        model=model,
        test_image_dir=test_im_dir,
        test_mask_dir=test_mask_dir,
        allergens=allergens,
        food_allergens_df=food_allergens_df,
        num_images=num_images,
        num_classes=len(food_allergens_df),  
    )
    for a in allergens:
        print(f"{a}: Acc={acc[a]:.4f}  Prec={prec[a]:.4f}  Rec={rec[a]:.4f}  F1={f1[a]:.4f}  IoU={iou[a]:.4f}")

    num_classes = len(food_allergens_df)  
    num_images = len([f for f in os.listdir(test_im_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    total_acc, total_prec, total_rec, total_f1 = calculate_allergen_metrics_image_level(
        model=model,
        test_image_dir=test_im_dir,
        test_mask_dir=test_mask_dir,
        allergens=allergens,
        food_allergens_df=food_allergens_df,
        num_images=num_images,
        num_classes=num_classes
    )
    print("\nImage-level allergen metrics:")
    for allergen in allergens:
        print(f"{allergen:20s} Acc={total_acc[allergen]:.4f}  "
            f"Prec={total_prec[allergen]:.4f}  "
            f"Rec={total_rec[allergen]:.4f}  "
            f"F1={total_f1[allergen]:.4f}")