import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DepthDataset(Dataset):
    def __init__(
        self, data_dir, list_file, transform=None, target_transform=None, has_gt=True
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.has_gt = has_gt
        self.skipped_files = 0
        with open(list_file, "r") as f:
            if has_gt:
                self.file_pairs = [line.strip().split() for line in f]
            else:
                self.file_list = [line.strip() for line in f]

    def __getitem__(self, idx):
        try:
            if self.has_gt:
                rgb_path = os.path.join(self.data_dir, self.file_pairs[idx][0])
                depth_path = os.path.join(self.data_dir, self.file_pairs[idx][1])
                if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
                    self.skipped_files += 1
                    return None
                rgb = Image.open(rgb_path).convert("RGB")
                depth = np.load(depth_path)
                if isinstance(depth, list):
                    self.skipped_files += 1
                    return None
                depth = depth.astype(np.float32)
                depth = torch.from_numpy(depth)
                if self.transform:
                    rgb = self.transform(rgb)
                if self.target_transform:
                    depth = self.target_transform(depth)
                else:
                    depth = depth.unsqueeze(0)
                return rgb, depth, self.file_pairs[idx][0]
            else:
                rgb_path = os.path.join(
                    self.data_dir, self.file_list[idx].split(" ")[0]
                )
                if not os.path.exists(rgb_path):
                    self.skipped_files += 1
                    return None
                rgb = Image.open(rgb_path).convert("RGB")
                if self.transform:
                    rgb = self.transform(rgb)
                return rgb, self.file_list[idx]
        except Exception as e:
            self.skipped_files += 1
            return None

    def get_skipped_count(self):
        return self.skipped_files
