import torch
import torch.nn as nn

class SILogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=1e-6)
        target = torch.clamp(target, min=1e-6)

        valid_mask = (target > 1e-6).float()

        diff_log = torch.log(pred) - torch.log(target)
        diff_log = diff_log * valid_mask

        batch_size = pred.shape[0]
        diff_log_flat = diff_log.view(batch_size, -1)
        valid_mask_flat = valid_mask.view(batch_size, -1)
        count = torch.sum(valid_mask_flat, dim=1) + 1e-6

        sum_diff_log = torch.sum(diff_log_flat, dim=1)
        alpha = -1 * sum_diff_log / count

        diff_log_with_alpha = diff_log_flat + alpha.unsqueeze(1)
        diff_log_with_alpha = diff_log_with_alpha * valid_mask_flat

        squared_term = torch.sum(diff_log_with_alpha**2, dim=1) / count
        per_image_loss = torch.sqrt(squared_term)

        loss = torch.mean(per_image_loss)

        return loss
