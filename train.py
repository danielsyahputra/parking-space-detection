import argparse
import torch
from data import dataset

def main(args):
    # load dataloader
    train_loader, valid_loader, test_loader = dataset.get_loaders(dataset_path="data/dataset")
    print(len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset))
    del train_loader
    del valid_loader
    del test_loader

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