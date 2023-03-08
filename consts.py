import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIGMA = 25.0
IMG_LEN = 28
BS = 1
img_tens_shape = (BS, 1, IMG_LEN, IMG_LEN)
