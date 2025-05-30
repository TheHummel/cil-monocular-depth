import os
import sys

import torch

from tqdm import tqdm


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def custom_collate_fn(batch):
    # handle None values in a batch
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def print_tqdm(message):
    tqdm.write(message, file=sys.stderr)

