# Parking Space Occupancy Detection

Reproduce the Experiment for the Image-Based Parking Space Occupancy Classification: Dataset and Baseline Paper Using MLFlow

_**Accompanying paper: [Image-Based Parking Space Occupancy Classification: Dataset and Baseline](https://arxiv.org/abs/2107.12207)**_

## About Experiments

Two methods are provided by this paper i.e. RCNN and Faster RCNN with FPN. All of these method are customed based on ACPDS dataset. This repository aims to reproduce the experiments in this paper, then track the result using MLFlow Tracking.

### Configurations:

| Models | Pooling Type | ROI Resolution |
|:----:|:-------------|:-------------|
| RCNN | qdrl | 64 |
| RCNN | qdrl | 128 |
| RCNN | qdrl | 256 |
| RCNN | square | 64 |
| RCNN | square | 128 |
| RCNN | square | 256 |

| Models | Pooling Type | Resolution |
|:----:|:-------------|:-------------|
| FasterRCNN_FPN | qdrl | 800 |
| FasterRCNN_FPN | qdrl | 1100 |
| FasterRCNN_FPN | qdrl | 1440 |
| FasterRCNN_FPN | square | 800 |
| FasterRCNN_FPN | square | 1100 |
| FasterRCNN_FPN | square | 1440 |


Each of these experiment are trained using some configuration such as:


- `Epoch`: 10
- `Train batch size`: 1
- `Val and Test batch size`: 1
- `Optimizer`: AdamW
- `Learning rate`: 1e-4
- etc.


## Run Experiments

Clone this repository
```
git clone https://github.com/danielsyahputra/parking-space-detection.git
cd parking-space-detection
```

Download data that is used for training
```
python3 main.py
```

### Training All Experiment at Once

If you want to run all experiment at once.

```
python3 train_all.py \
      
```

### Training one experiment
If you just want to know one of these experiments above

```
python3 train.py \ 
      --batch-size <train_batch_size> \
      --test-batch-size <test_batch_size> \
      --epochs <your_epoch> \
      --model-name <model_baseline> \
      --pooling-type <pooling_type> \
      --roi-res <roir_res>      
```

Docs
```
--batch-size: Batch size for training data (default: 1).

--test-batch-size: Batch size for testing and validation data (default: 1).

--epochs: Number of epochs for experiment (default: 10).

--lr: Learning rate used for experiment (default 1e-4).

--model-name: The baseline model used for experiment (default: RCNN).
List of possibles value: [RCNN, FasterRCNN_FPN]

--pooling-type: Pooling type used for experiment (default: qdrl)
List of possibles value: [qdrl, square]

--roi-res: This arguments based on --model-name before. 
Possibles value for RCNN baseline: [64, 128, 256]
Possibles value for FasterRCNN_FPN baseline: [800, 1100, 1440]
```

Example:
```
python3 train.py \ 
      --epochs 10 \
      --model-name RCNN \
      --pooling-type square \
      --roi-res 128    
```

## Result


## MLFlow


## Acknowledgments

```
@misc{marek2021imagebased,
      title={Image-Based Parking Space Occupancy Classification: Dataset and Baseline}, 
      author={Martin Marek},
      year={2021},
      eprint={2107.12207},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
