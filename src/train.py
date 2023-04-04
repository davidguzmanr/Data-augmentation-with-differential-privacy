"""
Utils to train the model in CIFAR10 using DP and/or data augmentation.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from torchvision import transforms
from torchvision import models
from torchvision.datasets import CIFAR10

from pytorch_lightning import LightningModule
from pytorch_lightning.cli import LightningCLI

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from torchmetrics import Accuracy

PATH_DATASETS = 'data'
# These values are assumed to be known, but they can also be computed with modest privacy budgets.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli_expert.html#argument-linking
        parser.link_arguments("trainer.max_epochs", "model.epochs", apply_on="parse")


class CIFAR10DP(LightningModule):
    def __init__(
        self,
        epochs: int = 50,
        batch_size: int = 128,
        lr: float = 0.01,
        optimizer_name: str = 'SGD',
        weight_decay: float = 5e-3,
        differential_privacy: bool = False,
        data_augmentation: bool = False,
        epsilon: float = 10.0,
        delta: float = 2e-5,
        max_grad_norm: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Set our init args as class attributes
        self.differential_privacy = differential_privacy
        self.data_augmentation = data_augmentation
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (3, 32, 32)
        channels, width, height = self.dims

        # Transformations to use in data augmentation
        augmentations = [
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]
        normalize = [
            transforms.ToTensor(),
            # If necessary, they can be computed with modest privacy budgets.
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]

        if self.data_augmentation:
            self.transform_train = transforms.Compose(augmentations + normalize)
        else:
            self.transform_train = transforms.Compose(normalize)

        self.transform_eval = transforms.Compose(normalize)

        # For the noise in the DP Gaussian mechanism
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator = torch.Generator(device=device).manual_seed(2**42)

        # Setup data
        self.prepare_data()
        self.setup()

        # Define PyTorch model, in order to compare we should always use the same model regardless
        # we are training with DP or not, so even in the vanilla model we should validate/fix the layers
        self.model = ModuleValidator.fix(models.resnet18(num_classes=10))

        if self.differential_privacy:
            (
                self.privacy_engine,
                self.private_model,
                self.private_optimizer,
                self.private_train_loader,
            ) = self.make_private()

        # Define the metrics to evaluate
        self.train_accuracy = Accuracy(num_classes=self.num_classes)
        self.val_accuracy = Accuracy(num_classes=self.num_classes)
        self.test_accuracy = Accuracy(num_classes=self.num_classes)

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        # log step metric
        self.train_accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "train_acc",
            self.train_accuracy,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        if self.differential_privacy and self.trainer.is_last_batch:
            epsilon = self.privacy_engine.get_epsilon(self.delta)
            self.log('epsilon', epsilon, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        # log step metric
        self.val_accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.test_accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        if self.optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        if self.differential_privacy:
            optimizer = self.private_optimizer

        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lambda epoch: 0.1
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", 'frequency': 20}]

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # Download data
        CIFAR10(PATH_DATASETS, train=True, download=True)
        CIFAR10(PATH_DATASETS, train=False, download=True)

    def setup(self, stage=None):
        self.cifar10_train = CIFAR10(PATH_DATASETS, train=True, transform=self.transform_train)

        # We will use test as validation to get plots like in https://arxiv.org/pdf/1607.00133.pdf
        self.cifar10_val = CIFAR10(PATH_DATASETS, train=False, transform=self.transform_eval)
        self.cifar10_test = CIFAR10(PATH_DATASETS, train=False, transform=self.transform_eval)

        # self.cifar10_val, self.cifar10_test = random_split(
        #     cifar10_test, [5000, 5000], generator=torch.Generator().manual_seed(42)
        # )

    def train_dataloader(self):
        if self.differential_privacy:
            return self.private_train_loader
        else:
            return DataLoader(self.cifar10_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, num_workers=4)

    ####################
    # DP RELATED STUFF
    ####################
    def make_private(self):
        """
        Add privacy-related responsibilities to the main PyTorch training
        objects: model, optimizer, and the train dataloader.
        """
        privacy_engine = PrivacyEngine()
        train_loader = DataLoader(self.cifar10_train, batch_size=self.batch_size, num_workers=4)

        if self.optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        (
            private_model,
            private_optimizer,
            private_train_loader,
        ) = privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=self.epochs,
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            max_grad_norm=self.max_grad_norm,
            noise_generator=self.generator,
        )

        return privacy_engine, private_model, private_optimizer, private_train_loader


def cli_main():
    # The LightningCLI removes all the boilerplate associated with arguments parsing. This is purely optional.
    cli = MyLightningCLI(
        CIFAR10DP, seed_everything_default=42, save_config_overwrite=True, run=False
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)


if __name__ == "__main__":
    cli_main()
