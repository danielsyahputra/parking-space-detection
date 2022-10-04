import json
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from torch.utils.data import DataLoader
from functools import lru_cache

class ACPDS():
    def __init__(self, dataset_path: str, type: str = 'train', res=None) -> None:
        self.dataset_path = dataset_path
        self.type = type
        self.res = res

        # load annotations
        with open(f"{self.dataset_path}/annotations.json", "r") as f:
            all_annotations = json.load(f)

        # Split train, val, and test
        if type in ['train', 'valid', 'test']:
            annotations = all_annotations[type]
        else:
            assert type == "all"
            annotations = {key:[] for key in all_annotations['train'].keys()}
            for t in ['train', 'valid', 'test']:
                for key, value in all_annotations[t].items():
                    annotations[key] += value

        self.file_names = annotations['file_names']
        self.rois = annotations['rois_list']
        self.occupancies = annotations['occupancy_list']

    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        image_path = f"{self.dataset_path}/images/{self.file_names[idx]}"
        image = torchvision.io.read_image(image_path)
        if self.res is not None:
            image = F.resize(image, self.res)

        # Load occupancy
        occupancy = self.occupancies[idx]
        occupancy = torch.tensor(occupancy, dtype=torch.int64)

        # Load rois
        rois = self.rois[idx]
        rois = torch.tensor(rois)

        return image, rois, occupancy

    def __len__(self):
        return len(self.file_names)

    
def collate_fn(batch):
    images = [item[0] for item in batch]
    rois = [item[1] for item in batch]
    occupancy = [item[2] for item in batch]
    return [images, rois, occupancy]

def get_loaders(dataset_path: str = "dataset", *args, **kwargs):
    train = ACPDS(dataset_path, 'train', *args, **kwargs)
    valid = ACPDS(dataset_path, 'valid', *args, **kwargs)
    test = ACPDS(dataset_path, 'test', *args, **kwargs)
    train_loader = DataLoader(train, batch_size=1, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid, batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader