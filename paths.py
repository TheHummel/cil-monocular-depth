import os

# PATHS
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
DATA_DIR = "/cluster/courses/cil/monocular_depth/data"

train_dir = os.path.join(DATA_DIR, "train/")
test_dir = os.path.join(DATA_DIR, "test/")
train_list_file = os.path.join(DATA_DIR, "train_list.txt")
test_list_file = os.path.join(DATA_DIR, "test_list.txt")
output_dir = OUTPUT_DIR
results_dir = os.path.join(output_dir, "results")
predictions_dir = os.path.join(output_dir, "predictions")
