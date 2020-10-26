# DeepHeartBeat
by Fabian Laumer, Gabriel Fringeli, Alina Dubatovka, Laura Manduchi

## Introduction



## Available models

We provide the pre-trained TensorFlow models used in the experiments of our paper, which includes:
- Echocardiogram models trained on the [EchoNet-Dynamic](https://echonet.github.io/dynamic/) dataset
- Single Lead ECG model trained on the [PhysioNet Computing in Cardiology Challenge 2017](https://physionet.org/content/challenge-2017/1.0.0/) dataset

The models can be loaded in the following way:

```
from utils import *

# EchoNet-Dynmaic
model = load_echonet_dynamic_model(i) # with i in [0, 1, 2, 3, 4]

# PhysioNet
model = load_physionet_model()
```

