DATA_DIR = "AllerNet/data/FoodSeg103/"
IMAGES_DIR = DATA_DIR +  "Images/img_dir/" #train and test images (inputs)
TRAIN_IM_DIR = IMAGES_DIR + 'train/'
TEST_IM_DIR = IMAGES_DIR + 'test/'

MASK_DIR = DATA_DIR + "Images/ann_dir/" #train and test masks (outputs)
TRAIN_MASK_DIR = MASK_DIR + 'train/'
LABELS_DIR =  "AllerNet/data/Labels/"
TEST_MASK_DIR = MASK_DIR + 'test/'

LABELS_CSV_PATH = LABELS_DIR + "food_allergens.csv"

FOOD_ALLERGENS = LABELS_DIR + "food_allergens.csv"
FOOD_LABEL_DISTRIBUTION = LABELS_DIR + "food_label_distribution.csv"

BACKBONE_NAME = "beitv2_large_patch16_224"
PARAM_BATCH_SIZE = 4
PARAM_NUM_WORKERS = 6
PARAM_LR = 1e-4
EPOCHS = 50
PATIENCE = 10
EARLY_STOP = True
IMG_SIZE = (512,512)
TARGET_SAVE_DIR = "/content/drive/MyDrive/AllerNet/Saved_Models"

