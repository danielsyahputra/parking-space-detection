from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.hub import load_state_dict_from_url

from .utils import pooling
from .utils.head import ClassificationHead

class FasterRCNN_FPN(nn.Module):
    def __init__(self, rois_res=7, pooling_type='square') -> None:
        super().__init__()
        self.backbone = resnet_fpn_backbone('resnet50', pretrained=False)
        hidden_dim = 256
        self.roi_res = rois_res
        self.pooling_type = pooling_type

        in_features = hidden_dim * self.roi_res ** 2
        self.head = ClassificationHead(in_features=in_features)
        weights_url = 'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
        state_dict = load_state_dict_from_url(weights_url, progress=False)
        self.load_state_dict(state_dict, strict=False)

    def forward(self, image, rois):
        features = self.backbone(image[None])
        features = list(features.values())
        features = pooling.pool_FPN_features(features, rois, self.roi_res, self.pooling_type)
        features = features.flatten(1)
        class_logits = self.head(features)
        return class_logits