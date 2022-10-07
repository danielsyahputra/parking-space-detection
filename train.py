import argparse
import torch
import os
from pathlib import Path
from data import dataset
from utils.engine import train_model
from models.rcnn import RCNN

def main(args):
    device = torch.device("cpu")
    # load dataloader
    os.makedirs("output", exist_ok=True)
    wd = Path(os.getcwd()) / "output"
    train_loader, valid_loader, test_loader = dataset.get_loaders(dataset_path="data/dataset")
    print(len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset))
    train_model(RCNN(roi_res=64, pooling_type="qdrl"), 
                train_loader, valid_loader, test_loader,
                f"{wd}/RCNN_64_qdrl", device, verbose=True)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parking Space Occupancy")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="Input batch size for training (default: 32)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=16,
        metavar="N",
        help="Input batch size for testing (default: 16)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="Number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="Learning rate (default: 1e-4)"
    )
    args = parser.parse_args()
    main(args)