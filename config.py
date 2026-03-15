import os

# Dataset paths (absolute to Desktop location)
DATASET_ROOT = r"C:\Users\user\Desktop\derma_ai\dataset\SkinDisease\SkinDisease"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
TEST_DIR = os.path.join(DATASET_ROOT, "test")

# 22 classes from dataset
CLASSES = [
    "Acne",
    "Actinic_Keratosis", 
    "Benign_tumors",
    "Bullous",
    "Candidiasis",
    "DrugEruption",
    "Eczema",
    "Infestations_Bites",
    "Lichen",
    "Lupus",
    "Moles",
    "Psoriasis",
    "Rosacea",
    "Seborrh_Keratoses",
    "SkinCancer",
    "Sun_Sunlight_Damage",
    "Tinea",
    "Unknown_Normal",
    "Vascular_Tumors",
    "Vasculitis",
    "Vitiligo",
    "Warts"
]

NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}

# Model paths
MODEL_PATH = "derma_model.pth"

print(f"Dataset train: {TRAIN_DIR}")
print(f"Dataset test: {TEST_DIR}")
print(f"Num classes: {NUM_CLASSES}")
