import os
import torch

# Config
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = (426, 560)
NUM_WORKERS = 2
PIN_MEMORY = True

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = "/cluster/courses/cil/monocular_depth/data"
train_dir = os.path.join(data_dir, "train/")
test_dir = os.path.join(data_dir, "test/")
train_list_file = os.path.join(data_dir, "train_list.txt")
test_list_file = os.path.join(data_dir, "test_list.txt")
output_dir = os.path.join(current_dir, "data/monocular_depth_output")
results_dir = os.path.join(output_dir, "results")
predictions_dir = os.path.join(output_dir, "predictions")
