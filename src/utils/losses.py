import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.base.base_loss import BaseLoss
from src.utils.lovasz_losses import lovasz_softmax


def make_one_hot(labels, classes):
    one_hot = (
        torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3])
        .zero_()
        .to(labels.device)
    )
    target = one_hot.scatter_(1, labels.data, 1)
    return target


def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    # cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()


class CrossEntropyLoss2d(BaseLoss):
    def __init__(
        self,
        weight: Tensor | None = None,
        ignore_index: int = 255,
        reduction: str = "mean",
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.CE = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def _forward(self, output: Tensor, target: Tensor) -> Tensor:
        output = F.softmax(output, dim=1)
        not_empty_images = torch.nonzero(target != self.ignore_index, as_tuple=False)
        if not_empty_images.numel() == 0:
            zero_tensor = output * 0
            zero_tensor = zero_tensor.sum()
            return zero_tensor
        loss = self.CE(output, target)
        return loss


class SampleWeightedBCELoss(BaseLoss):
    def __init__(
        self,
        reduction: str = "mean",
        balance: float = 1
    ):
        super().__init__()
        self.reduction = reduction
        self.balance = balance

    def _forward(self, output: Tensor, target: Tensor) -> Tensor:  # pylint: disable=W0622
        return F.binary_cross_entropy_with_logits(
            input=output,
            target=target.float(),
            weight=_compute_sample_weights(output=output, targets=target, balance=self.balance),
            reduction=self.reduction,
        )


def _compute_sample_weights(output: Tensor, targets: Tensor, balance: float) -> Tensor:
    with torch.no_grad():
        weights = torch.zeros_like(output)
        samples_num = output.size()[0]
        for sample_idx in range(samples_num):
            t = targets[sample_idx, :, :]
            pos = (t == 1).sum()
            neg = (t == 0).sum()
            total = pos + neg

            weights[sample_idx, t == 1] = neg / total
            weights[sample_idx, t == 0] = pos * balance / total

    return weights


class DiceLoss(BaseLoss):
    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def _forward(self, output: Tensor, target: Tensor) -> Tensor:
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - (
            (2.0 * intersection + self.smooth)
            / (output_flat.sum() + target_flat.sum() + self.smooth)
        )
        return loss


class FocalLoss(BaseLoss):
    def __init__(
        self,
        gamma: float = 2,
        alpha: Tensor = None,
        ignore_index: int = 255,
        size_average: bool = True,
    ):
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def _forward(self, output: Tensor, target: Tensor) -> Tensor:
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class CE_DiceLoss(BaseLoss):
    def __init__(self, smooth=1, reduction="mean", ignore_index=255, weight=None):
        super().__init__()
        self.dice = DiceLoss(smooth=smooth, ignore_index=ignore_index)
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=weight, reduction=reduction, ignore_index=ignore_index
        )

    def _forward(self, output: Tensor, target: Tensor) -> Tensor:
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss


class LovaszSoftmax(BaseLoss):
    def __init__(self, classes: str = "present", per_image: bool = False, ignore_index: int = 255):
        super().__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def _forward(self, output: Tensor, target: Tensor) -> Tensor:
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(
            logits,
            target,
            classes=self.classes,
            per_image=self.per_image,
            ignore=self.ignore_index,
        )
        return loss
