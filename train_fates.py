import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from tqdm import tqdm

from encoder import Encoder
from load_data import load_data
from MLP import MLP
from pathlib import Path


class NAGAN(nn.Module):
    """Not a GAN."""
    def __init__(self, image_size, n_classes, n_confounders, n_latents):
        super().__init__()
        # Add encoder
        self.encoder = Encoder(image_size, output_classes=n_latents)
        # Add classifier
        self.classifier = MLP(n_latents, n_classes, b_dropout=True)
        # Add discriminator
        self.discriminator = MLP(n_latents, n_confounders)

    def forward(self, x):
        z = self.encoder(x)
        y = self.classifier(z)
        u = self.discriminator(z)
        return (u, y)


def set_requires_grad(models, values):
    """
    Parameters
    ----------
    models: List
        List of nn.Modules for which to set `requires_grad`
    values: List
        List of values to which to set `requires_grad`.

    Raises
    ------
    AssertionError: if `len(models) != len(values)`
    """
    assert len(models) == len(values), "Should be as many models as values"
    for model, value in zip(models, values):
        for param in model.parameters():
            param.requires_grad = value


class Trainer:
    def __init__(self, nagan, train_dl, val_dl, test_dl, lr_c=1e-4, lr_d=1e-4):
        """
        Parameters
        ----------
        nagan: NAGAN
            The Not-a-GAN model. Has the attributes `classifier` and
            `discriminator` which are both of type `torch.nn.Module`
        train_dl: torch.utils.data.DataLoader
            Train dataset
        val_dl: torch.utils.data.DataLoader
            Validation dataset
        test_dl: torch.utils.data.DataLoader
            Test dataset
        lr_c: float
            Learning rate for the `nagan.classifier`
        lr_d: float
            Learnig rate for the `nagan.discriminator`
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(self.device)
        self.nagan = nagan.to(self.device)
        class_parameters = (list(nagan.classifier.parameters()) +
                            list(nagan.encoder.parameters()))
        self.optimizer_c = optim.Adam(class_parameters, lr_c)
        self.optimizer_d = optim.Adam(nagan.discriminator.parameters(), lr_d)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.writer = SummaryWriter()
        self.global_iter = 0
        self.global_epoch = 0

    def train(self):
        """Runs adversarial optimization on all bacthes in the dataloader

        Returns
        -------
        loss_c: the cross entropy on the class, not on the confounders.
        """
        self.nagan.train()
        epoch_loss = 0
        epoch_loss_d = 0
        n_iter = 0
        for x, y_u in tqdm(self.train_dl, 'training'):
            x = x.to(self.device)
            y = y_u["y"].to(self.device)
            u = y_u["u"].to(self.device)
            # Classifier step
            self.optimizer_c.zero_grad()
            set_requires_grad([self.nagan.encoder,
                               self.nagan.classifier,
                               self.nagan.discriminator],
                              [True, True, False])
            # Get the latent and output from the classifier
            z = self.nagan.encoder(x)
            y_pred = self.nagan.classifier(z)
            # Get the discriminator's classification of the confounder
            u_pred = self.nagan.discriminator(z)
            # Get losses
            loss_c = self.loss_fn(y_pred, y)
            loss_d = self.loss_fn(u_pred, u)
            # Total loss = Good cross entropy - evil cross entropy
            loss = loss_c - loss_d
            loss.backward()
            self.optimizer_c.step()
            # Discriminator step
            self.optimizer_d.zero_grad()
            set_requires_grad([self.nagan.encoder,
                               self.nagan.classifier,
                               self.nagan.discriminator],
                              [False, False, True])
            # Get the discriminator output
            u_pred_d = self.nagan.discriminator(z.detach())
            # Get the cross entropy
            loss = self.loss_fn(u_pred_d, u)
            loss.backward()
            self.optimizer_d.step()
            epoch_loss += loss_c.item()
            epoch_loss_d += loss.item()
            self.global_iter += 1
            n_iter += 1
            self.writer.add_scalar("Loss", loss_c.item(), self.global_iter)
            self.writer.add_scalar("Discrim. Loss", loss.item(), self.global_iter)
        self.global_epoch += 1
        return epoch_loss / n_iter, epoch_loss_d / n_iter

    @torch.no_grad()
    def evaluate(self, dataloader, name=""):
        """Gets a classification accuracy on the dataloader

        Returns
        -------
        accuracy: the model accuracy on the class.
        """
        self.nagan.eval()
        # Get an accuracy
        u_groundtruths = []
        y_groundtruths = []
        u_predictions = []
        y_predictions = []
        for x, y_u in tqdm(dataloader, desc=name):
            x = x.to(self.device)
            y = y_u["y"].to(self.device)
            u = y_u["u"].to(self.device)
            z = self.nagan.encoder(x)
            y_logits = self.nagan.classifier(z)
            y_pred = torch.argmax(y_logits, dim=1)
            u_logits = self.nagan.discriminator(z)
            u_pred = torch.argmax(u_logits, dim=1)
            y_predictions.append(y_pred)
            y_groundtruths.append(y)
            u_predictions.append(u_pred)
            u_groundtruths.append(u)
        u_predictions = torch.cat(u_predictions).detach().cpu().numpy()
        u_groundtruths = torch.cat(u_groundtruths).detach().cpu().numpy()
        y_predictions = torch.cat(y_predictions).detach().cpu().numpy()
        y_groundtruths = torch.cat(y_groundtruths).detach().cpu().numpy()
        print(u_predictions.shape)
        print(y_predictions.shape)
        print(u_groundtruths.shape)
        print(y_groundtruths.shape)
        # Get the accuracy
        u_confusion = confusion_matrix(u_predictions, u_groundtruths)
        y_confusion = confusion_matrix(y_predictions, y_groundtruths)
        u_accuracy = u_confusion.trace() / u_confusion.sum()
        y_accuracy = y_confusion.trace() / y_confusion.sum()
        print("U", u_confusion)
        print("Y", y_confusion)
        self.writer.add_scalar(f"U_Accuracy/{name}", u_accuracy, self.global_epoch)
        self.writer.add_scalar(f"Y_Accuracy/{name}", y_accuracy, self.global_epoch)
        return u_accuracy, y_accuracy

    def validation(self):
        accuracy = self.evaluate(self.val_dl, name="Validation")
        return accuracy

    def test(self):
        accuracy = self.evaluate(self.test_dl, name="Test")
        return accuracy

    def save(self, path):
        state_dict = {"optimizer_c": self.optimizer_c.state_dict(),
                      "optimizer_d": self.optimizer_d.state_dict(),
                      "nagan": self.nagan.state_dict(),
                      "iterations": self.global_iter,
                      "epoch": self.global_epoch}
        torch.save(state_dict, path)

    def load(self, path):
        state_dict = torch.load(path)
        self.optimizer_c.load_state_dict(state_dict["optimizer_c"])
        self.optimizer_d.load_state_dict(state_dict["optimizer_d"])
        self.nagan.load_state_dict(state_dict["nagan"])
        self.global_iter = state_dict["iterations"]
        self.global_epoch = state_dict["epoch"]


def get_checkpoint(directory, latest=True, best=False):
    """
    Parameters
    ----------
    paths: list
        Directory of checkpoints, named as `epochs_accuracy.pth`"
    """
    directory = Path(directory)
    paths = [path for path in directory.iterdir() if str(path).endswith('pth')]
    if len(paths) == 0:
        return None, 0
    if latest:
        epochs = [int(name.stem.split('_')[0]) for name in paths]
        idx = np.argmax(epochs)
        return paths[idx], epochs[idx]
    elif best:
        accuracies = [int(name.stem.split('_')[1]) for name in paths]
        idx = np.argmax(accuracies), epochs[idx]
        return paths[idx]


def Parser():
    parser = argparse.ArgumentParser("Not A GAN Classifier")
    parser.add_argument("--input_size", nargs=3, default=(1, 108, 108))
    parser.add_argument("--confounders", type=int, default=6,
                        help="Number of classes in the confounding features")
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--latents", type=int, default=512)
    parser.add_argument("--epochs", default=100, type=int,
                        help="Number of epochs to train")
    parser.add_argument("--directory", default='./checkpoints',
                        help="Root directory for checkpoints")
    parser.add_argument("--save_every", default=100, 
                        help="How often (in epochs) to checkpoint the model")
    return parser


if __name__ == "__main__":
    parser = Parser()
    args = parser.parse_args()
    # Data
    train_dl, val_dl, test_dl = load_data()

    epochs = args.epochs
    input_size = args.input_size
    model = NAGAN(input_size, n_classes=args.classes,
                  n_confounders=args.confounders,
                  n_latents=args.latents)
    print("Testing model")
    dummy = torch.zeros((1, 1, input_size[1], input_size[2]))
    u, y = model(dummy)
    print(u.shape)
    print(y.shape)
    assert(tuple(y.shape) == (1, args.classes))
    assert(tuple(u.shape) == (1, args.confounders))

    print("Creating trainer")
    trainer = Trainer(model, train_dl, val_dl, test_dl)

    path, start = get_checkpoint(args.directory, latest=True)
    if path is not None:
        print(f"Loading checkpoint at epoch {start}.")
        trainer.load(path)
    print("Training")
    # Training loop
    for epoch in range(start, epochs):
        # One full epoch
        loss_c, loss_d = trainer.train()
        # Validation
        u_accuracy, y_accuracy = trainer.validation()
        print(f"Epoch {epoch}: loss {loss_c}, d. loss: {loss_d},\nVal. U accuracy: {u_accuracy},\nVal. Y accuracy: {y_accuracy}")
        # Checkpoint
        if epoch % args.save_every == 0:
            trainer.save(args.directory + f"/{epoch}.pth")
