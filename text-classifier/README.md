# Text classifier with differential privacy and data augmentation

To train the model with differential privacy and data augmentation, for instance, run

```
python train.py --config=config.yaml --model.differential_privacy=True --model.data_augmentation=True
```

You can also change the configuration in `config.yaml`.

[train.py](https://github.com/davidguzmanr/CSC2516/blob/main/text-classifier/train.py) contains everything related to train the model, [Training-BERT.ipynb](https://github.com/davidguzmanr/CSC2516/blob/main/text-classifier/Training-BERT.ipynb) uses this to train the different models we tried and [Membership-inference-BERT.ipynb](https://github.com/davidguzmanr/CSC2516/blob/main/text-classifier/Membership-inference-BERT.ipynb) contains the membership inference attacks on the trained models.

The logs of all the experiments can be found [here](https://csc2516-neural-networks-and-deep-learning.s3.amazonaws.com/text-classifier/lightning_logs.zip).
