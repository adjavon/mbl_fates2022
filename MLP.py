import numpy as np
import torch


class MLP(torch.nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
        n_latent=4096,
        b_dropout=False,
        depth=4):

        super(MLP, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.b_dropout = b_dropout

        current_size = n_input
        self.layers = []
        for i in range(depth):
            if self.b_dropout == True:
                self.layers += [
                    torch.nn.Linear(current_size,n_latent),
                    torch.nn.ReLU(),
                    torch.nn.Dropout()
                ]
            else:
                self.layers += [
                    torch.nn.Linear(current_size,n_latent),
                    torch.nn.ReLU(),
                ]
            current_size = n_latent

        self.layers += [torch.nn.Linear(current_size, n_output)]
        # Build Model
        self.model_seq = torch.nn.Sequential(*self.layers)

    def forward(self, z):
        y = self.model_seq(z)
        return y
