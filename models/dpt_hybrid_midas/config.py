import torch

# CONFIG
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-6
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = (384, 384)
NUM_WORKERS = 2
PIN_MEMORY = True
