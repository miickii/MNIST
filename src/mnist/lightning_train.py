from mnist.lightning_model import MyAwesomeModel
from pytorch_lightning import Trainer, loggers
from torch.utils.data import DataLoader
from mnist.dataset import MnistDataset
import typer
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(batch_size: int = 32, epochs: int = 5) -> None:
    model = MyAwesomeModel()
    train_dataset = MnistDataset(train=True)
    test_dataset = MnistDataset(train=False)
    trainer = Trainer(max_epochs=epochs, limit_train_batches=0.2, logger=loggers.WandbLogger(project="my_awesome_project"))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    trainer.fit(model, train_dataloader, test_dataloader)
    trainer.test(model, test_dataloader)

def main():
    typer.run(train)
