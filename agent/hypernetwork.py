import torch.nn as nn


class LinearHyperNetwork(nn.Module):
    """
    Linear HyperMARL-style generator for flattened target-network parameters.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.net(x)


class MLPHyperNetwork(nn.Module):
    """
    MLP generator for flattened target-network parameters.
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


class HyperNetwork(MLPHyperNetwork):
    """
    Backward-compatible name for the original MLP hypernetwork.
    """


def build_hypernetwork(hypernet_type, input_dim, hidden_dims, output_dim, dropout=0.0):
    """
    Build a linear or MLP hypernetwork from config.
    """

    kind = str(hypernet_type or 'mlp').lower()
    if kind in ('linear', 'lin'):
        return LinearHyperNetwork(input_dim, output_dim)
    if kind in ('mlp', 'nonlinear'):
        return MLPHyperNetwork(input_dim, hidden_dims, output_dim, dropout=dropout)
    raise ValueError(f"Unknown hypernetwork type: {hypernet_type}")
