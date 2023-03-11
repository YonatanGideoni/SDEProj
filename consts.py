import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {DEVICE}")

SIGMA = 25.0
IMG_LEN = 28
BS = 1
IMG_TENS_SHAPE = (BS, 1, IMG_LEN, IMG_LEN)
SMALL_NUMBER = 1e-4

float_dtype = torch.float32
