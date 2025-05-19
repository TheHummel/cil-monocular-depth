import torch
import torch.nn as nn

class SILogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        valid_mask = (target > 1e-6).float()
        pred = torch.clamp(pred, min=1e-6)
        target = torch.clamp(target, min=1e-6)
        diff_log = torch.log(pred) - torch.log(target)
        diff_log = diff_log * valid_mask
        count = torch.sum(valid_mask) + 1e-6
        log_mean = torch.sum(diff_log) / count
        squared_term = torch.sum(diff_log**2) / count
        mean_term = log_mean**2
        loss = torch.sqrt(squared_term + mean_term)
        return loss
