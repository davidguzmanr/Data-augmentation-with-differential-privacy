# Image classifier with differential privacy and data augmentation

To train the model with differential privacy and data augmentation, for instance, run

```
python train.py --config=config.yaml --model.differential_privacy=True --model.data_augmentation=True
```

You can also change the configuration in `config.yaml`.

[train.py](https://github.com/davidguzmanr/CSC2516/blob/main/cifar10/train.py) contains everything related to train the model, [Training-CIFAR10.ipynb](https://github.com/davidguzmanr/CSC2516/blob/main/cifar10/Training-CIFAR10.ipynb) uses this to train the different models we tried and [Membership-inference-CIFAR10.ipynb](https://github.com/davidguzmanr/CSC2516/blob/main/cifar10/Membership-inference-CIFAR10.ipynb) contains the membership inference attacks on the trained models.

The logs of all the experiments can be found [here](https://csc2516-neural-networks-and-deep-learning.s3.amazonaws.com/cifar10/lightning_logs.zip).
