from torch import nn

class DiceLoss(nn.Module):
  """
  Dice loss implementation in PyTorch for single-channel input.
  """
  def __init__(self, eps=1e-7):
    super(DiceLoss, self).__init__()
    self.eps = eps

  def forward(self, input, target):
    """
    Calculates the Dice loss.

    Args:
      input: Predicted probabilities from the model (B, 1, H, W).
      target: Ground truth labels (B, 1, H, W).

    Returns:
      Dice loss (1 - Dice score).
    """
    # Ensure single channel format for both input and target
    if input.size(1) > 1:
      raise ValueError("Input should have only one channel (single-channel probability).")

    # Flatten input and target
    input = input.flatten()
    target = target.flatten()

    # Smooth intersection and union
    intersection = (input * target).sum()
    union = input.sum() + target.sum()

    # Dice loss calculation
    dice_loss = 1 - (2 * intersection + self.eps) / (union + self.eps)

    return dice_loss