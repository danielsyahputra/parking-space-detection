import argparse
import torch
import os
from pathlib import Path
from data import dataset
from utils.engine import train_model
from models.rcnn import RCNN

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

def main(args):
    device = get_device()
    os.makedirs("output", exist_ok=True)
    wd = Path(os.getcwd()) / "output"

    experiment_name = args.experiment_name
    epochs = args.epochs

    train_loader, valid_loader, test_loader = dataset.get_loaders(dataset_path="data/dataset")
    print(len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset))
    train_model(RCNN(roi_res=64, pooling_type="qdrl"), 
                train_loader, valid_loader, test_loader,
                f"{wd}/RCNN_64_qdrl", device, experiment_name, epochs=epochs, verbose=True)    

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
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="experiment",
        metavar="EN",
        help="Experiment Name for Tracking in MLFlow"
    )
    args = parser.parse_args()
    main(args)