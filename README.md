# CSC2516: How Does Data Augmentation Affect Differential Privacy in Deep Learning?
Final project for CSC2516 Winter 2023 Neural Networks and Deep Learning

> **How Does Data Augmentation Affect Differential Privacy in Deep Learning?**<br>
> David Guzmán<sup>1</sup>, Abraham Morales<sup>2</sup>, Rui Xian<sup>2</sup><br>
> <sup>1</sup>Department of Computer Science, <sup>2</sup>Department of Statistical Sciences<br>
>
> <p align="justify"><b>Abstract:</b> <i>Deep learning often adopts data augmentation as an essential and efficient technique to generate new training examples from existing data to improve the model robustness and generalization. Differential privacy is a technique used to preserve the privacy of individual data points while releasing statistical information about a dataset. In this work, we study the relationship between data augmentation and differential privacy in deep learning for image and text classification. We found that although data augmentation has a negative effect on the performance of models trained with differential privacy, it improves the model robustness against membership inference attacks.</i></p>

## Experiments
1. In [cifar10](https://github.com/davidguzmanr/CSC2516/tree/main/cifar10) are the experiments for the image classifier with differential privacy and membership inference attacks.
2. In [text-classifier](https://github.com/davidguzmanr/CSC2516/tree/main/text-classifier) are the experiments for the text classifier with differential privacy and membership inference attacks.
