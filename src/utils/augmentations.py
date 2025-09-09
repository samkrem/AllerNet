from torchvision import transforms
import random
from PIL import Image
from ..config import IMG_SIZE
class SimpleAugment:
    def __call__(self, img, mask):
        img = transforms.Resize((int(1.25*IMG_SIZE[0]), int(1.25*IMG_SIZE[1])))(img) #increase image size for croping purposes
        mask = transforms.Resize((int(1.25*IMG_SIZE[0]), int(1.25*IMG_SIZE[1])), interpolation=transforms.InterpolationMode.NEAREST)(mask)

        if random.random() > 0.5:
            scale = random.uniform(0.6, 1.0)
            ratio = random.uniform(0.75, 1.33)  # typical aspect ratio range
            crop = transforms.RandomResizedCrop(size=IMG_SIZE, scale=(scale, scale), ratio=(ratio, ratio))

            params = crop.get_params(img, crop.scale, crop.ratio) #need to apply crop transform to img/mask
            img = transforms.functional.resized_crop(img, *params, size=IMG_SIZE, interpolation=Image.BICUBIC)
            mask = transforms.functional.resized_crop(mask, *params, size=IMG_SIZE, interpolation=Image.NEAREST)
        else:
            img = transforms.Resize(IMG_SIZE)(img)
            mask = transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.NEAREST)(mask)

        if random.random() > 0.5: 
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        angle = random.randint(-30, 30)
        img = img.rotate(angle, resample=Image.BICUBIC)
        mask = mask.rotate(angle, resample=Image.NEAREST)

        # Color Jitter, 
        # if random.random() > 0.5:
        #     color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        #     img = color_jitter(img)
        # if random.random() > 0.5:
        #     img_np = np.array(img).astype(np.float32)  # Convert PIL to NumPy array (float32 for precision)
        #     noise = np.random.normal(0, 10, img_np.shape)  # Generate Gaussian noise with std=10
        #     img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)  # Add noise and clip to valid pixel range
        #     img = Image.fromarray(img_np)  # Convert back to PIL

        return img, mask
