# Parking Space Occupancy Detection

**Reproduce the Experiment for the Image-Based Parking Space Occupancy Classification: Dataset and Baseline Paper Using MLFlow. Please refers to [martin-marek/parking-space-occupancy](https://github.com/martin-marek/parking-space-occupancy) for the original implementation.**

_**Accompanying paper: [Image-Based Parking Space Occupancy Classification: Dataset and Baseline](https://arxiv.org/abs/2107.12207)**_

In this repository, I provide:
- Code to reproduce all of my result.
- Download link for [training logs with MLFlow](https://drive.google.com/uc?id=1yMkr0ABnUK3yNT3u5TNMvMQwYeK7iBwN).
- Colab notebooks to [explore training experiments with MLFlow](https://colab.research.google.com/drive/16IaPSdUdTAesIf6JZnsCu_vdiRlT6wrp?usp=sharing), [explore my experiment result using MLFlow UI](https://colab.research.google.com/drive/1GMHvqljWwrUDEfhTqNYoxHiRSIwoMq4q?usp=sharing).

## About Experiments

Two methods are provided by this paper i.e. RCNN and Faster RCNN with FPN. All of these method are customed based on ACPDS dataset. This repository aims to reproduce the experiments in this paper, then track the result using MLFlow Tracking.

### Experiments

<table>
<tr><th>RCNN </th><th>FasterRCNN with FPN</th></tr>
<tr><td>

| Models | Pooling Type | ROI Resolution |
|:----:|:-------------|:-------------|
| RCNN | qdrl | 64 |
| RCNN | qdrl | 128 |
| RCNN | qdrl | 256 |
| RCNN | square | 64 |
| RCNN | square | 128 |
| RCNN | square | 256 |

</td><td>

| Models | Pooling Type | Resolution |
|:----:|:-------------|:-------------|
| FasterRCNN_FPN | qdrl | 800 |
| FasterRCNN_FPN | qdrl | 1100 |
| FasterRCNN_FPN | qdrl | 1440 |
| FasterRCNN_FPN | square | 800 |
| FasterRCNN_FPN | square | 1100 |
| FasterRCNN_FPN | square | 1440 |

</td></tr> </table>


Each of these experiment are trained using some configuration such as:


- `Epoch`: 10
- `Train batch size`: 1
- `Val and Test batch size`: 1
- `Optimizer`: AdamW
- `Learning rate`: 1e-4
- etc.

Note: Because of my limited resources, I couldn't try the epochs the same as the one in the paper. 

## Run Experiments

MLFlow Run:

```
mlflow run https://github.com/danielsyahputra/parking-space-detection.git -P batch_size=1 test_batch_size=1 epochs=<YOUR_EPOCH> experiment_name=<YOUR_EXPERIMENT_NAME> --env-manager=local
```

Withour MLFlow Run:

Clone this repository

```
git clone https://github.com/danielsyahputra/parking-space-detection.git
cd parking-space-detection
```

Download data that is used for training
```
python3 download.py --download-data True
```

### Training All Experiment at Once

If you want to run all experiment at once.

```
python3 train_all.py \
      --batch-size <BATCH_SIZE> \ 
      --test-batch-size <YOUR_TEXT_BATCH_SIZE> \ 
      --epochs <YOUR_EPOCH> \ 
      --lr <YOUR_LEARNING_RATE>
      --experiment-name <YOUR_EXPERIMENT_NAME>      
```

Docs:
```
--batch-size: Batch size for training data (default: 1).

--test-batch-size: Batch size for testing and validation data (default: 1).

--epochs: Number of epochs for experiment (default: 10).

--lr: Learning rate used for experiment (default 1e-4).

--experiment-name: The name of experiment that will be passed to MLFlow Tracking.
```

### Training One Experiment
If you just want to know the result of one of the experiment:

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

## Results

<table>
<tr><th>RCNN </th><th>FasterRCNN with FPN</th></tr>
<tr><td>

Pooling Type | ROI Resolution | Time | Test Acc | Test Loss |
|:-------------|:-------------|:-------------|:-------------|:-------------|
| qdrl | 64 |  257.7 | 0.965 | 0.098
| qdrl | 128 | 237.4 | 0.945 | 0.116 
| qdrl | 256 | 271.7 | 0.962 | 0.082 
| **square** | **64** | **234.2** | **0.972** | **0.075** 
| square | 128 | 235 | 0.944 | 0.13 
| square | 256 | 271.6 | 0.954 | 0.103 

</td><td>

| Pooling Type | Resolution | Time | Test Acc | Test Loss | 
|:-------------|:-------------|:-------------|:-------------|:-------------|
| qdrl | 800 | 305.9 | 0.967 | 0.085 
| qdrl | 1100 | 305.4 | 0.972 | 0.075 
| qdrl | 1440 | 362.6 | 0.973 | 0.084 
| **square** | **800** | **275.6** | **0.974** | 0.078 
| square | 1100 | 302.8 | 0.972 | 0.071 
| square | 1440 | 361.5 | **0.974** | **0.073** 

</td></tr> </table>

All these experimental results have been tracked using MLFlow and can be accessed with the following step.

## MLFlow

There are two ways for accessing my tracking result.

1. Local
```
git clone https://github.com/danielsyahputra/parking-space-detection.git
cd parking-space-detection
python3 download.py --download-mlruns True
mlflow ui
```
Then, open your [http://127.0.0.1:5000](http://127.0.0.1:5000)

2. [Colab Notebook](https://colab.research.google.com/drive/1GMHvqljWwrUDEfhTqNYoxHiRSIwoMq4q?usp=sharing)

To access MLFlow UI that is run in Colab, we need a third-party to forward the localhost in the collab machine so it can be accesed outside the machine. In this solution, I use [Ngrok](https://dashboard.ngrok.com), a programmable network edge that adds connectivity,
security, and observability to your apps with no code changes. For more information, you can check the collab link that I have given before.

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
