from typing import Tuple, Optional, List, Dict
import torch.nn as nn
import torch
from torch.nn.utils.weight_norm import WeightNorm
from dalib.modules.grl import WarmStartGradientReverseLayer
import torch.nn.functional as F

def shift_log(x: torch.Tensor, offset: Optional[float] = 1e-6) -> torch.Tensor:
    return torch.log(torch.clamp(x + offset, max=1.))

class ImageClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, finetune=True):
        super(ImageClassifier, self).__init__()
        self.backbone = nn.Sequential(backbone,nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten())
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        self.num_classes = num_classes
        if bottleneck is None:
            self.bottleneck = nn.Sequential(
            )
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim
        # cls head, eqinv head, cluster head
        if head is None:
            self.head = nn.Linear(self._features_dim, num_classes)
        else:
            self.head = head
        self.eqinv_head = nn.Linear(bottleneck_dim, num_classes)
        self.cluster_head = nn.Linear(bottleneck_dim, num_classes)
        self.finetune = finetune

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor, freeze_feature=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        if freeze_feature:
            with torch.no_grad():
                f = self.backbone(x)
        else:
            f = self.backbone(x)
        f1 = self.bottleneck(f)
        predictions = self.head(f1)
        preds_nograd = self.head(f1.detach())
        eqinv_preds = self.eqinv_head(f1)
        eqinv_preds_nograd = self.eqinv_head(f1.detach())
        outputs = {
            "y": predictions,
            "y_cluster_all": eqinv_preds,
            "feature": f,
            "bottleneck_feature": f1,
            "y_nograd": preds_nograd,
            "y_cluster_all_nograd": eqinv_preds_nograd
        }
        outputs["y_cluster_u"] = self.cluster_head(f1)
        outputs["y_cluster_u_nograd"] = self.cluster_head(f1.detach())
        return outputs

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
            {"params": self.eqinv_head.parameters(), "lr": 1.0 * base_lr},
            {"params": self.cluster_head.parameters(), "lr": 1.0 * base_lr},
        ]
        return params