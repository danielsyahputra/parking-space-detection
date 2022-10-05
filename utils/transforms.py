import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

def preprocess(image, res=None):
    if res is not None:
        image = F.resize(image, res)
    image = image.to(torch.float32) / 255
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    return image

def random_image_rotation(image, points, max_angles=30.0):
    device = image.device
    assert points.min() >= 0, points.min()
    assert points.max() <= 1, points.max()

    angle_deg = (2 * torch.rand(1).item() - 1) * max_angles
    _, H1, W1 = image.shape
    image = F.rotate(image, angle_deg, expand=True)
    _, H2, W2 = image.shape

    angle_rad = torch.tensor((angle_deg / 180.0) * 3.141592)
    RM = torch.tensor([[torch.cos(angle_rad), -torch.sin(angle_rad)],
                        [torch.sin(angle_rad),  torch.cos(angle_rad)]], dtype=torch.float, device=device)

    points = points.clone()
    points -= 0.5
    points[..., 0] *= (W1 - 1)
    points[..., 1] *= (H1 - 1)
        
    # rotate the points
    points = points @ RM
        
    # move points back to the relative coordinate system
    points[..., 0] /= (W2 - 1)
    points[..., 1] /= (H2 - 1)
    points += 0.5
        
    # check that the points remain within [0, 1]
    assert points.min() >= 0, points.min()
    assert points.max() <= 1, points.max()
    return image, points

def augment(image, rois):
    if torch.rand(1).item() > 0.5:
        image = F.hflip(image)
        rois = rois.clone()
        rois[:,:, 0] = 1 - rois[:, :, 0]

    image = transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.8, hue=0.1)(image)
    image, rois = random_image_rotation(image, rois, 15)
    return image, rois