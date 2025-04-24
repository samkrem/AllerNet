import cv2
import pandas as pd
import os
import json
from tqdm import tqdm


# Step 1: Load the CSV
df = pd.read_csv("food_allergens.csv")

# Step 2: Build a label → allergens mapping
label_to_allergens = {}
allergen_columns = df.columns[2:]
for _, row in df.iterrows():
    label_id = row["id"]
    allergens = set(allergen_columns[row[allergen_columns] == 1])
    label_to_allergens[label_id] = allergens
# Step 3: Process all mask images
def process_masks(mask_folder, start=0, end=100):
    image_to_allergens = {}
    all_files = [f for f in os.listdir(mask_folder)
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    # Filter by filenames that are numeric and in range
    valid_files = []
    for f in all_files:
        name = os.path.splitext(f)[0]
        if name.isdigit():
            idx = int(name)
            if start <= idx <= end:
                valid_files.append(f)

    for filename in tqdm(valid_files, desc=f"Processing images {start} to {end}"):
        path = os.path.join(mask_folder, filename)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            continue

        unique_ids = set(int(x) for x in set(mask.flatten()))
        allergens_in_image = set()

        for label_id in unique_ids:
            allergens_in_image.update(label_to_allergens.get(label_id, []))

        image_to_allergens[filename] = sorted(allergens_in_image)

    return image_to_allergens


# Example usage
mask_folder = "FoodSeg103/Images/ann_dir/train"
image_allergen_map = process_masks(mask_folder)

# Step 4: Save or display
with open("image_to_allergens.json", "w") as f:
    json.dump(image_allergen_map, f, indent=2)

print(image_allergen_map)