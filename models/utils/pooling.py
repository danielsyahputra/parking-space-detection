import torch
from torch import nn, Tensor
from typing import Optional, List
from torch.nn.functional import grid_sample

@torch.jit.script
def linspace(start: Tensor, stop: Tensor, num: int):
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    for i in range(start.dim):
        steps = steps.unsqueeze(-1)
    out = start[None] + steps*(stop - start)[None]
    return out

@torch.jit.script 
def roi_grid(rois: Tensor, size: int = 3):
    idx_edge_1 = linspace(start=rois[:, 1], stop=rois[:, 2], num=size)
    idx_edge_2 = linspace(start=rois[:, 0], stop=rois[:, 3], num=size)
    rois_interpolated = linspace(start=idx_edge_1, stop=idx_edge_2, num=size)
    rois_interpolated = rois_interpolated.permute([2, 1, 0, 3])
    return rois_interpolated

@torch.jit.script 
def index_by(tensor: Tensor, idx: Tensor):
    assert idx.min() >= 0
    assert idx.max() <= 1
    _, h, w = tensor.shape
    idx_abs = idx.clone()
    idx_abs[:, 0] *= w - 1
    idx_abs[:, 1] *= h-1
    idx_abs = idx_abs.round().to(torch.long)
    tensor = tensor[:, idx_abs[..., 1], idx[..., 0]]
    tensor = tensor.transpose(1, 0)
    return tensor

@torch.jit.script 
def roi_pool_square(tensor: Tensor, rois: Tensor, size: int = 3):
    w = torch.amax(rois[:, :, 0], 1) - torch.amin(rois[:, :, 0], 1)
    h = torch.amax(rois[:, :, 1], 1) - torch.amin(rois[:, :, 1], 1)
    c = torch.mean(rois, 1, keepdim=True)
    c = c.repeat(1, 4, 1)
    c[:, 0, 0] += w/2
    c[:, 0, 1] += h/2
    c[:, 1, 0] -= w/2
    c[:, 1, 1] += h/2
    c[:, 2, 0] -= w/2
    c[:, 2, 1] -= h/2
    c[:, 3, 0] += w/2
    c[:, 3, 1] -= h/2
    rois_interpolated = roi_grid(c, size)
    rois_interpolated = (rois_interpolated * 2) - 1
    warps = torch.stack([grid_sample(tensor[None], r[None], align_corners=True)[0] for r in rois_interpolated])
    return warps

@torch.jit.script 
def roi_pool_qdrl(tensor: Tensor, rois: Tensor, size: int = 3):
    rois_interpolated = roi_grid(rois, size)
    rois_interpolated = (rois_interpolated * 2) - 1
    warps = torch.stack([grid_sample(tensor[None], r[None], align_corners=True)[0] for r in rois_interpolated])
    return warps

@torch.jit.script 
def roi_pool(tensor: Tensor, rois: Tensor, size: int = 3, pooling_type: str = "square"):
    if pooling_type == "square":
        return roi_pool_square(tensor, rois, size)
    elif pooling_type == "qdrl":
        return roi_pool_qdrl(tensor, rois, size)
    else: 
        raise Exception(f"Unknown pooling method: {pooling_type}")

@torch.jit.script 
def get_level_idx(features: List[Tensor], rois: Tensor, size: int):
    scale_factor = 4
    image_h = scale_factor * features[0].shape[2]
    image_w = scale_factor * features[0].shape[3]

    w = rois[:, :, 0].amax(1) - rois[:, :, 0].amin(1)
    h = rois[:, :, 1].amax(1) - rois[:, :, 1].amin(1)

    w = w * image_w
    h = h * image_h

    k_min = 2
    k_max = 5
    lvl_0 = 4
    k = lvl_0 + torch.log2(torch.sqrt(w * h) / 224.)
    k = k.floor().int()
    k = k.clamp(k_min, k_max)
    roi_pooling_level = k - k_min
    return roi_pooling_level

@torch.jit.script 
def pool_FPN_features(features: List[Tensor], rois: Tensor, size: int, pooling_type: str = "square"):
    device = features[0].device
    roi_pooling_level = get_level_idx(features, rois, size)
    c = features[0].shape[1]
    pooled_rois = torch.zeros((len(rois,), c, size, size), device=device)
    for level, level_features in enumerate(features):
        idx_in_level = torch.nonzero(level == roi_pooling_level).squeeze(1)
        if len(idx_in_level) > 0:
            rois_per_level = rois[idx_in_level]
            pooled_rois[idx_in_level] = roi_pool(level_features[0], rois_per_level, size, pooling_type)

    return pooled_rois