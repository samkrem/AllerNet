import pandas as pd
import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from src.utils.metrics import fast_hist,compute_mIoU_from_confmat, compute_pixel_accuracy
from src.utils.visualization import show_segmentation_result
from src.utils.augmentations import SimpleAugment

import src.config as config
from src.models.MLASwin import SwinTransformerWithMLA
from src.models.losses import combined_loss

from src.data.FoodSegDataset import FoodSegDataset

scaler = torch.cuda.amp.GradScaler()  # define once globally, before training loop

def train(m, ldr, crit, opt, dev, ep, tot): # Removed aux_weight
    m.train()
    total_loss = 0
    confmat = torch.zeros((104, 104), dtype=torch.int64, device=dev)

    bar = tqdm(ldr, desc=f"Epoch {ep}/{tot} ▶ train", leave=True)
    for imgs, msks in bar:
        imgs, msks = imgs.to(dev), msks.to(dev)

        # Mixed precision forward
        with torch.cuda.amp.autocast():
            # The model should now return a single tensor
            logits = m(imgs)
            loss = crit(logits, msks)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        confmat += fast_hist(preds, msks, 104)
        bar.set_postfix(loss=f"{loss.item():.4f}")

    ious, miou = compute_mIoU_from_confmat(confmat)
    acc = compute_pixel_accuracy(confmat)

    print(f"Training loss: {total_loss / len(ldr)}, Training mIoU: {miou:.4f}, Pixel Acc: {acc:.4f}")
    return total_loss / len(ldr), float(miou), float(acc)


@torch.no_grad()
def validate(m, ldr, crit, dev, ep, tot, num_classes=104, visualize_n=0): # Removed aux_weight
    m.eval()
    total = 0
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=dev)
    vis_count = 0

    bar = tqdm(ldr, desc="⭑ validate", leave=True)
    for imgs, msks in bar:
        imgs, msks = imgs.to(dev), msks.to(dev)

        # Mixed precision forward
        with torch.cuda.amp.autocast():
            # The model should now return a single tensor
            logits = m(imgs)
            loss = crit(logits, msks)

        total += loss.item()
        preds = torch.argmax(logits, dim=1)
        confmat += fast_hist(preds, msks, num_classes)
        bar.set_postfix(val_loss=f"{loss.item():.4f}")

        if vis_count < visualize_n:
            for i in range(min(imgs.shape[0], visualize_n - vis_count)):
                img_to_show = imgs[i]
                mean = torch.tensor([0.485, 0.456, 0.406], device=dev)
                std = torch.tensor([0.229, 0.224, 0.225], device=dev)
                img_to_show = img_to_show * std[:, None, None] + mean[:, None, None]
                img_to_show = torch.clamp(img_to_show, 0, 1)
                show_segmentation_result(img_to_show, preds[i], msks[i])
                vis_count += 1

    ious, miou = compute_mIoU_from_confmat(confmat)
    acc = compute_pixel_accuracy(confmat)

    print(f"Test loss: {total / len(ldr)}, Test mIoU: {miou:.4f}, Pixel Acc: {acc:.4f}")
    return total / len(ldr), float(miou), float(acc)

if __name__ == "__main__":
    train_im_dir, train_mask_dir = config.TRAIN_IM_DIR, config.TRAIN_MASK_DIR
    test_im_dir, test_mask_dir = config.TEST_IM_DIR, config.TEST_MASK_DIR
    target_save_dir = config.TARGET_SAVE_DIR
    labels_csv_path = config.LABELS_CSV_PATH
    
    joint_tf_train = SimpleAugment()
    train_ds = FoodSegDataset(train_im_dir, train_mask_dir, transform=None)#joint_tf_train) # image_dir first, then mask_dir
    train_ds = FoodSegDataset(train_im_dir, train_mask_dir, transform=joint_tf_train)#joint_tf_train) # image_dir first, then mask_dir
    test_ds  = FoodSegDataset(test_im_dir, test_mask_dir,  transform=None)  # image_dir first, then mask_dir

    labels_df = pd.read_csv(labels_csv_path)
    class_to_food = dict(zip(labels_df['id'], labels_df['food'])) 

    param_batch_size = config.PARAM_BATCH_SIZE
    param_num_workers = config.PARAM_NUM_WORKERS
    param_lr = config.PARAM_LR
    epochs = config.EPOCHS
    target_save_Dir = config.TARGET_SAVE_DIR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_ds, batch_size=param_batch_size, shuffle=True, #train_ds
                            num_workers=param_num_workers, pin_memory=True)
    val_loader   = DataLoader(test_ds,  batch_size=param_batch_size, shuffle=False,
                            num_workers=param_num_workers, pin_memory=True)
    model = SwinTransformerWithMLA(num_classes=104, decoder_dim=256)
    model.to(device)

    food_label_distribution = config.FOOD_LABEL_DISTRIBUTION
    food_dist_df = pd.read_csv(food_label_distribution)
    df = pd.read_csv(food_label_distribution)
    print(df)
    pixel_counts = df["Count"].values

    frequencies = pixel_counts / pixel_counts.sum()
    alpha = 0.1  # smoothing factor
    smoothed_freqs = alpha + (1 - alpha) * frequencies
    median = np.median(smoothed_freqs[smoothed_freqs > 0])
    weights = median / smoothed_freqs
    weights = np.clip(weights, 0, 10)

    criterion = combined_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay= 0.05)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50,   # Set to your total number of epochs
        eta_min=1e-6
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=5,  # Number of epochs before the first restart
    #     T_mult=2, # Factor by which T_0 is multiplied for each restart
    #     eta_min=1e-6
    # )
 
    train_losses, val_losses = [], []
    train_mious, val_mious = [], []
    train_accs, val_accs = []

    best_val_loss = float("inf")
    best_miou = -float("inf")
    epochs_no_improve = 0
    patience = config.PATIENCE
    early_stop = config.EARLY_STOP

    for epoch in range(1, epochs + 1):
        train_loss, miou, acc = train(model, train_loader, criterion, optimizer, device, ep=epoch, tot=epochs)

        visualize = (epoch % 5 == 0)
        val_loss, val_miou, val_acc = validate(
            model,
            val_loader,
            criterion,
            device,
            ep=epoch,
            tot=epochs,
            visualize_n=1 if visualize else 0
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_mious.append(miou)
        val_mious.append(val_miou)
        train_accs.append(acc)
        val_accs.append(val_acc)

        # Check for improvement (mIoU based)
        if val_miou > best_miou:
            best_miou = val_miou
            epochs_no_improve = 0
            filename = "super_updated_mla_swintransformerFoodSegCE2DL8512.pth"
            save_path = os.path.join(target_save_dir, filename)
            torch.save(model.state_dict(), save_path)
            print(f"✅ Epoch {epoch}: Improved validation mIoU to {val_miou:.4f}")
        else:
            epochs_no_improve += 1
            print(f"⚠️ Epoch {epoch}: No improvement ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print(f"⛔ Early stopping triggered after {epoch} epochs.")
            early_stop = True
            break

        # Cosine annealing step (per epoch)
        scheduler.step()  # no val_loss for CosineAnnealingLR

