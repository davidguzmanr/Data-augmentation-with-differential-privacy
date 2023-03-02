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
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# class MyLightningCLI(LightningCLI):
#     def add_arguments_to_parser(self, parser):
#         parser.add_argument("--differential_privacy.disable", default=False)
#         parser.add_argument("--differential_privacy.delta", default=1e-5)


class CIFAR10DP(LightningModule):
    def __init__(
        self,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        differential_privacy: bool = False,
        data_augmentation: bool = False,
        epsilon: float = 50.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.2,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Set our init args as class attributes
        self.differential_privacy = differential_privacy
        self.data_augmentation = data_augmentation
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (3, 32, 32)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # If necessary, they can be computed with modest privacy budgets.
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
                # TODO: add data augmentation when data_augmentation=True
            ]
        )
        # For the noise in the DP Gaussian mechanism and splitting train/val
        self.generator = torch.Generator(device=device).manual_seed(2**42)

        # Setup data
        self.prepare_data()
        self.setup()

        # Define PyTorch model, in order to compare we should always use the same model regardless
        # we are training with DP or not, so even in the vanilla model we should validate/fix the layers
        self.model = ModuleValidator.fix(models.resnet18(num_classes=10))

        if self.differential_privacy:
            private_model, private_optimizer, private_train_loader = self.make_private()

            self.model = private_model
            self.private_optimizer = private_optimizer
            self.private_train_loader = private_train_loader

        # Define the metrics to evaluate
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.val_accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.differential_privacy:
            return self.private_optimizer

        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # Download data
        CIFAR10(PATH_DATASETS, train=True, download=True)
        CIFAR10(PATH_DATASETS, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val/tests datasets for use in dataloaders
        cifar10_full = CIFAR10(PATH_DATASETS, train=True, transform=self.transform)
        self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [48000, 2000], generator=self.generator)
        self.cifar10_test = CIFAR10(PATH_DATASETS, train=False, transform=self.transform)

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
        train_loader = DataLoader(
            self.cifar10_train, batch_size=self.batch_size, num_workers=4
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        private_model, private_optimizer, private_train_loader = privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=self.epochs,
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            max_grad_norm=self.max_grad_norm,
            noise_generator=self.generator,
        )

        return private_model, private_optimizer, private_train_loader


def cli_main():
    # The LightningCLI removes all the boilerplate associated with arguments parsing. This is purely optional.
    cli = LightningCLI(
        CIFAR10DP,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)


if __name__ == "__main__":
    CIFAR10(PATH_DATASETS, train=True, download=True)
    CIFAR10(PATH_DATASETS, train=False, download=True)
    cli_main()
