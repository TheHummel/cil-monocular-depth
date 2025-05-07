import os
import torch
from transformers import AutoModelForDepthEstimation

from dotenv import load_dotenv

from config import DEVICE


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def custom_collate_fn(batch):
    # handle None values in a batch
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def load_model_from_hf(model_name, device=DEVICE):
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file. Please add it.")

    model = AutoModelForDepthEstimation.from_pretrained(model_name, token=hf_token)
    model.to(device)

    print("Model loaded successfully!")
    print(model)

    return model
