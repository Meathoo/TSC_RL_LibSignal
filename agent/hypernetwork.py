import torch.nn as nn


class HyperNetwork(nn.Module):
    """
    Generate flattened actor parameters from meta input.
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
