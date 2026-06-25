# DL-CDI: Deep Learning Phase Retrieval for Coherent Diffractive Imaging

A 3D encoder–decoder CNN framework for phase retrieval from X-ray diffraction patterns. Given a reciprocal-space intensity measurement, the network reconstructs the real-space complex object (amplitude + phase).

**Authors:** Ahmed H. Mokhtar, Marcus Newton
**Derived from:** work by Longlong Wu
**License:** GNU GPL v3

---

## Overview

The network is a 3D encoder–decoder CNN with a dual-branch decoder. It is trained on synthetic diffraction data and can then be applied to experimental measurements via iterative refinement with support constraints (ShrinkWrap).

### Key features
- 3D encoder–decoder architecture (no skip connections)
- Bottleneck channels split into two independent decoder branches — one for amplitude, one for phase
- Separable 1D convolutions to reduce parameter count
- Mixed-optimiser training schedule (Adadelta + Adam with varying epsilon)
- StepLR learning rate scheduling
- ShrinkWrap support constraint for experimental data refinement
- Automatic Kaiming weight initialisation for convolutional layers

---

## File structure

| File | Description |
|------|-------------|
| `model.py` | Network architecture: building blocks and `NNModel` |
| `train.py` | `CNNTrain`: data loading, loss functions, training loop |
| `predict.py` | `ShrinkWrap`, `CNNPredict`: iterative refinement with support constraint |
| `gendata.py` | `GenData`: synthetic data generator (hexagonal prism, octahedron, monoclinic) |
| `runTrain.py` | Example script: generate data and train the network |
| `runYMO.py` | Example script: run phase retrieval on experimental data |

---

## Installation

```bash
git clone https://github.com/<your-username>/DL_CDI.git
cd DL_CDI
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA following the [official instructions](https://pytorch.org/get-started/locally/).

---

## Usage

### 1. Generate training data

```python
from gendata import GenData

d = GenData()
d.SetShape([32, 32, 32])
d.SetN(12000)
d.SetMorphology("hexprism")  # or "octahedron", "monoclinic"
d.GenShapeData()
d.SaveData()
# Produces: fs_amps.npy  (n, 1, x, y, z)  reciprocal-space intensities
#           rs_objs.npy  (n, 2, x, y, z)  real-space amplitude + phase
```

### 2. Train the network

See `runTrain.py` for a full example. Key steps:

```python
from cnnphase import NNModel, CNNTrain
import torch.optim as optim
import torch.optim.lr_scheduler as ss
import torch.nn as nn

cnn = CNNTrain()
cnn.SetDevice('cuda')
cnn.SetInputData('fs_amps.npy')
cnn.SetTargetData('rs_objs.npy')
cnn.SetModel(NNModel)
cnn.SetBatchSize(32)          # max 32 per GPU
cnn.InitialiseWeights(nn.init.kaiming_normal_, mode='fan_in', nonlinearity='leaky_relu')
cnn.AddLR(1e-3)
cnn.AddOptimiser(optim.Adam, eps=1e-8)
cnn.AddScheduler(ss.StepLR)
cnn.SetNEpochs(150)
cnn.AddOpStep(150)
cnn.TrainNN()
cnn.SaveParameters()
cnn.PlotLoss()
```

### 3. Phase retrieval on experimental data

See `runYMO.py` for a full example. The `CNNPredict` class loads a trained checkpoint and refines the reconstruction iteratively.

```python
from cnnphase import NNModel, CNNPredict

predict = CNNPredict()
predict.SetDevice('cuda')
predict.SetModel(NNModel)
predict.SetExpData('expdata_ML.npy', mask=190, square_root=True)
predict.SetSupport('support.npy')
predict.SetTrainedNN('CP150_<timestamp>.pth')
# ... configure optimisers, schedulers, epochs ...
predict.TransferPredict()
predict.SaveParameters(training=False)
```

---

## Training data format

| Array | Shape | Description |
|-------|-------|-------------|
| `fs_amps.npy` | `(n, 1, x, y, z)` | Reciprocal-space intensities (input) |
| `rs_objs.npy` | `(n, 2, x, y, z)` | Real-space object — channel 0: amplitude, channel 1: phase (target) |

---

## Notes

- **Batch size:** memory-limited to 32 per GPU. Use `DataParallel` for larger batches.
- **Weight initialisation:** Kaiming for conv layers; ones/zeros for batch norm — this significantly stabilises training.
- **Learning rate scheduler:** `StepLR` scales LR by gamma every N steps.
- **Mixed precision:** `GradScaler` is available in `cnnphase.py` for AMP training.
