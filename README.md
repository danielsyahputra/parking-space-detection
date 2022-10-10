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
python3 train_all.py
```

### Training one experiment
If you just want to know one of these experiments above

```
python3 train.py
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
