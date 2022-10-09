import argparse
from ast import parse
import torch
import os
from pathlib import Path
from data import dataset
from utils.engine import train_model
from models.rcnn import RCNN
from models.faster_rcnn_fpn import FasterRCNN_FPN

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def train_all(model_name: str, args):
    device = get_device()
    os.makedirs("output", exist_ok=True)
    wd = Path(os.getcwd()) / "output"

    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    epochs = args.epochs
    lr = args.lr
    experiment_name = args.experiment_name

    train_loader, valid_loader, test_loader = dataset.get_loaders(batch_size=batch_size, 
                                                            test_batch_size=test_batch_size,
                                                            dataset_path="data/dataset")

    roi_res_list = [64, 128, 256]
    faster_rcnn_res_list = [800, 1100, 1440]
    pooling_types = ["qdrl", "square"]

    for pooling_type in pooling_types:
        for roi_res, res in zip(roi_res_list, faster_rcnn_res_list):
            params = {
                "batch_size": batch_size,
                "test_batch_size": test_batch_size,
                "model_name": model_name,
                "pooling_type": pooling_type,
            }
            if model_name == "RCNN":
                print(f"================================ Training {model_name}_{pooling_type}_{roi_res} ================================")
                params["rois_res"] = roi_res
                model = RCNN(roi_res=roi_res, pooling_type=pooling_type)
                train_model(model, train_loader, valid_loader, test_loader, 
                            f"{wd}/{experiment_name}", device, experiment_name, 
                            lr=lr, epochs=epochs, params_dict=params)
            else:
                print(f"================================ Training {model_name}_{pooling_type}_{res} ================================")
                params["rois_res"] = res
                model = FasterRCNN_FPN(pooling_type=pooling_type)
                train_model(model, train_loader, valid_loader, test_loader, 
                            f"{wd}/{experiment_name}", device, experiment_name,
                            lr=lr, epochs=epochs, res=res, params_dict=params)


def main(args):
    train_all(model_name="RCNN", args=args)
    train_all(model_name="FasterRCNN_FPN", args=args)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parking Space Occupancy")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help="Input batch size for training (default: 32)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
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
        default="Paper_Experiment",
        metavar="EN",
        help="Name of Experiment in MLFLow"
    )
    args = parser.parse_args()
    main(args)