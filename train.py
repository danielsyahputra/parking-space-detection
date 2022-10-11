import argparse
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

def main(args):
    device = get_device()
    os.makedirs("output", exist_ok=True)
    wd = Path(os.getcwd()) / "output"

    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    epochs = args.epochs
    lr = args.lr
    model_name = args.model_name
    pooling_type = args.pooling_type
    roi_res = args.roi_res
    experiment_name = f"{model_name}_{pooling_type}_{roi_res}"

    train_loader, valid_loader, test_loader = dataset.get_loaders(batch_size=batch_size, 
                                                                test_batch_size=test_batch_size,
                                                                dataset_path="data/dataset")

    print(f"================================ Training {experiment_name} ================================")
    params = {
        "batch_size": batch_size,
        "test_batch_size": test_batch_size,
        "model_name": model_name,
        "pooling_type": pooling_type,
        "roi_res": roi_res,
    }
    if model_name == "RCNN":
        model = RCNN(roi_res=roi_res, pooling_type=pooling_type)
        train_model(model, train_loader, valid_loader, test_loader, 
                    f"{wd}/{experiment_name}", device, experiment_name, 
                    lr=lr, epochs=epochs, params_dict=params)
    else:
        model = FasterRCNN_FPN(pooling_type=pooling_type)
        train_model(model, train_loader, valid_loader, test_loader, 
                    f"{wd}/{experiment_name}", device, experiment_name,
                    lr=lr, epochs=epochs, res=roi_res, params_dict=params)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parking Space Occupancy")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help="Batch size for training data (default: 1)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        metavar="N",
        help="Batch size for validation and testing data (default: 1)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="Number of epochs for training (default: 10)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="Learning rate for training (default: 1e-4)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="RCNN",
        metavar="MN",
        help="Model that is used for training. Value: RCNN or FasterRCNN_FPN"
    )
    parser.add_argument(
        "--pooling-type",
        type=str,
        default="qdrl",
        metavar="PT",
        help="Pooling type used for training. Value: qdrl or square"
    )
    parser.add_argument(
        "--roi-res",
        type=int,
        default=64,
        metavar="RR",
        help="""ROI Resolution Used for Training. The possible value if using RCNN as the baseline are 64, 128, and 256. 
        But, you can use 800, 1100, and 1440 if you want to use FasterRCNN_FPN."""
    )
    args = parser.parse_args()
    main(args)