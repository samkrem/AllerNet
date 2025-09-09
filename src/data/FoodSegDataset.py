import torch
from torch.utils.data import Dataset
import re, os
from torchvision import transforms
from PIL import Image
import numpy as np
from ..config import IMG_SIZE
class FoodSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_size=IMG_SIZE):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size 
        bad = re.compile(r'.*\(\d+\)\..*') #regex handling
        self.images = sorted(f for f in os.listdir(image_dir) if not bad.match(f)) #mask image pairs have same number

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fn = self.images[idx]
        base = os.path.splitext(fn)[0]
        img_path = os.path.join(self.image_dir, fn)
        mask_path = os.path.join(self.mask_dir, base + ".png")

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  #greyscale

        if self.transform:
            img, mask = self.transform(img, mask)

        img = transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.BICUBIC)(img)
        mask = transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.NEAREST)(mask)

        img_tensor = transforms.ToTensor()(img)
        img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(img_tensor) #normalize

        mask_array = np.array(mask, dtype=np.uint8)  
        mask_tensor = torch.from_numpy(mask_array).long()

        return img_tensor, mask_tensor 

