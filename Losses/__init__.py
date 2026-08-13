"""
Loss functions package.

包含语义分割任务常用损失函数:
    - CrossEntropyLoss
    - DiceLoss
"""


from .cross_entropy import CrossEntropyLoss
from .dice_loss import DiceLoss


__all__ = [
    "CrossEntropyLoss",
    "DiceLoss",
]