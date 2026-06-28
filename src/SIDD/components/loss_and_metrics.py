from SIDD.config.configuration import ConfigurationManager
from SIDD.entity.config_entity import LossMetricsConfig
import torch
import torch.nn as nn


class LossAndMetrics:
    def __init__(self, config: LossMetricsConfig):
        self.config = config

    def get_dice_loss(self):
        return DiceLoss(
            smooth=float(self.config.smooth)
        )

    def get_bce_dice_loss(self):
        return BCEDiceLoss(
            smooth=float(self.config.smooth)
        )

class DiceLoss(nn.Module):
    def __init__(self, smooth):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, targets):
        pred = torch.sigmoid(pred)

        pred = pred.flatten(start_dim=2)
        targets = targets.flatten(start_dim=2)

        intersection = (pred * targets).sum(dim=2)
        union = pred.sum(dim=2) + targets.sum(dim=2)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()



class BCEDiceLoss(nn.Module):
    def __init__(self, smooth):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)

    def forward(self, pred, targets):
        bce_loss = self.bce(pred, targets)
        dice_loss = self.dice(pred, targets)
        total_loss = bce_loss + dice_loss
        return total_loss
    

try:
    config = ConfigurationManager()

    loss_config = config.get_loss_metrics_config()

    metrics_builder = LossAndMetrics(loss_config)

    criterion = metrics_builder.get_bce_dice_loss()

    dummy_pred = torch.randn(
        2,
        4,
        256,
        1600,
        requires_grad=True
    )

    dummy_target = torch.randint(
        0,
        2,
        (2,4,256,1600)
    ).float()

    loss = criterion(
        dummy_pred,
        dummy_target
    )

    print(loss)
    print(loss.requires_grad)
except Exception as e:
    raise e