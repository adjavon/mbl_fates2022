import argparse
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from train_fates import NAGAN
from balance_accuracy import balance_accuracy
from load_data import load_data


class Tester:
    def __init__(self, nagan, test_dl):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.nagan = nagan.to(self.device)
        self.test_dl = test_dl

    def load(self, path):
        """Load the NAGAN and some metadata from the path."""
        state_dict = torch.load(path)
        self.nagan.load_state_dict(state_dict["nagan"])
        global_iter = state_dict["iterations"]
        epoch = state_dict["epoch"]
        return global_iter, epoch

    @torch.no_grad()
    def evaluate(self, name):
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
        for x, y_u in tqdm(self.test_dl, name):
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
        # Get the accuracy
        u_confusion = confusion_matrix(u_predictions, u_groundtruths)
        y_confusion = confusion_matrix(y_predictions, y_groundtruths)
        results = {"u": (u_groundtruths, u_predictions),
                   "y": (y_groundtruths, y_predictions),
                   "confusion":  (u_confusion, y_confusion)}
        return results

    def test(self, directory, output_file="test_results.csv"):
        directory = Path(directory)
        # Load model weights
        with open(output_file, 'w') as fd:
            fd.write("path,iterations,epoch,u_accuracy,y_accuracy\n")
        for path in directory.iterdir():
            global_iter, epoch = self.load(path)
            results = self.evaluate(str(path))
            # Get balanced accuracy
            u_accuracy = balance_accuracy(*results["u"])
            y_accuracy = balance_accuracy(*results["y"])
            u_conf, y_conf = results["confusion"]
            print("U:", u_accuracy, u_conf)
            print("Y:", y_accuracy, y_conf)
            with open(output_file, 'a') as fd:
                fd.write(f"{str(path)},{global_iter},{epoch},{u_accuracy},{y_accuracy}\n")


def Parser():
    parser = argparse.ArgumentParser("Not A GAN Classifier")
    parser.add_argument("--input_size", nargs=3, default=(1, 108, 108))
    parser.add_argument("--confounders", type=int, default=6,
                        help="Number of classes in the confounding features")
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--latents", type=int, default=512)
    parser.add_argument("--directory", default='./checkpoints',
                        help="Root directory for checkpoints")
    parser.add_argument("-o", "--output", type=str,
                        default="test_results.csv",
                        help="File to save resulting accuracies in.")
    return parser


if __name__ == "__main__":
    parser = Parser()
    args = parser.parse_args()
    # Data
    _, _, test_dl = load_data()

    input_size = args.input_size
    model = NAGAN(input_size, n_classes=args.classes,
                  n_confounders=args.confounders,
                  n_latents=args.latents)
    print("Checking model.")
    dummy = torch.zeros((1, 1, input_size[1], input_size[2]))
    u, y = model(dummy)
    assert(tuple(y.shape) == (1, args.classes))
    assert(tuple(u.shape) == (1, args.confounders))
    
    print("Running test")
    tester = Tester(model, test_dl)
    tester.test(args.directory, output_file=args.output)
