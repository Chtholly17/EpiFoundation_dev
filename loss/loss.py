import torch
from   torch import nn
import torch.nn.functional as F



class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def masked_mse_loss(
        self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the masked MSE loss between input and target.
        """
        mask = mask.float()
        loss = F.mse_loss(input * mask, target * mask, reduction="sum")
        return loss / mask.sum()

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.masked_mse_loss(input, target, mask)


class ZeroInflatedLoss(nn.Module):
    """
    Loss for zero-inflated regression model.
    Combines BCE for zero classification and MSE for conditional regression.
    """
    def __init__(self, bce_weight=1.0, mse_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        
    def forward(self, regression_pred, zero_logits, target, mask):
        """
        Args:
            regression_pred: (batch, seq_len) - predicted non-zero values
            zero_logits: (batch, seq_len) - logits for zero classification
            target: (batch, seq_len) - ground truth values
            mask: (batch, seq_len) - valid positions (non-padding)
        """
        mask = mask.float()
        
        # Binary labels: 1 if target is zero, 0 otherwise
        is_zero = (target == 0).float()
        
        # BCE loss for zero classification (on all valid positions)
        bce_loss = F.binary_cross_entropy_with_logits(
            zero_logits * mask, 
            is_zero * mask, 
            reduction='sum'
        ) / mask.sum()
        
        # MSE loss for regression (only on non-zero targets)
        non_zero_mask = mask * (1 - is_zero)  # Valid AND non-zero
        if non_zero_mask.sum() > 0:
            mse_loss = F.mse_loss(
                regression_pred * non_zero_mask,
                target * non_zero_mask,
                reduction='sum'
            ) / non_zero_mask.sum()
        else:
            mse_loss = torch.tensor(0.0, device=target.device)
        
        total_loss = self.bce_weight * bce_loss + self.mse_weight * mse_loss
        
        return total_loss, bce_loss, mse_loss