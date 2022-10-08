from torch import nn
from torchvision.models import mobilenet_v3_large
from torchvision.ops.misc import FrozenBatchNorm2d

from .utils import pooling

class RCNN(nn.Module):
    def __init__(self, roi_res=100, pooling_type='square') -> None:
        super().__init__()
        # backbone
        self.backbone = mobilenet_v3_large(weights="IMAGENET1K_V1", norm_layer=FrozenBatchNorm2d)
        self.backbone.fc = nn.Linear(in_features=960, out_features=2)

        # freeze bottom layers
        start_count = 1
        end_count = 4
        for count, child in enumerate(self.backbone.children()):
            if count >= start_count and count < end_count:
                child.requires_grad_(False)
        
        # Pooling
        self.roi_res = roi_res
        self.pooling_type = pooling_type

    def forward(self, image, rois):
        warps = pooling.roi_pool(image, rois, self.roi_res, self.pooling_type)
        class_logits = self.backbone(warps)
        return class_logits