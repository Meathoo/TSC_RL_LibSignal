import math

import torch
import torch.nn as nn


class GeneratedParamScaler:
    """
    Reset fan-in/out scaling for hypernetwork-generated target parameters.

    Hypernetworks emit unconstrained flattened parameters. This scaler maps each
    generated target layer back to a standard initialization scale before the
    generated weight/bias is used in the target network forward pass.
    """

    def __init__(
        self,
        enabled=False,
        mode='fan_in',
        hidden_gain=math.sqrt(2.0),
        output_gain=1.0,
        bias_scale=1.0,
    ):
        self.enabled = bool(enabled)
        self.mode = str(mode or 'fan_in').lower()
        self.hidden_gain = float(hidden_gain)
        self.output_gain = float(output_gain)
        self.bias_scale = float(bias_scale)

    def _gain(self, layer_idx, layer_count):
        return self.output_gain if layer_idx == layer_count - 1 else self.hidden_gain

    def _weight_scale(self, out_dim, in_dim, layer_idx, layer_count):
        if not self.enabled:
            return 1.0

        fan_in = max(1.0, float(in_dim))
        fan_out = max(1.0, float(out_dim))
        gain = self._gain(layer_idx, layer_count)

        if self.mode in ('fan_in', 'kaiming', 'kaiming_normal'):
            return gain / math.sqrt(fan_in)
        if self.mode in ('fan_out',):
            return gain / math.sqrt(fan_out)
        if self.mode in ('fan_avg', 'fan_in_out', 'xavier', 'xavier_normal'):
            return gain * math.sqrt(2.0 / (fan_in + fan_out))
        if self.mode in ('none', 'identity', 'off'):
            return 1.0
        raise ValueError(f"Unknown generated parameter RF mode: {self.mode}")

    def _bias_scale(self, in_dim, layer_idx, layer_count):
        if not self.enabled:
            return 1.0
        fan_in = max(1.0, float(in_dim))
        return self.bias_scale * self._gain(layer_idx, layer_count) / math.sqrt(fan_in)

    def scale_weight(self, weight, out_dim, in_dim, layer_idx, layer_count):
        return weight * self._weight_scale(out_dim, in_dim, layer_idx, layer_count)

    def scale_bias(self, bias, in_dim, layer_idx, layer_count):
        return bias * self._bias_scale(in_dim, layer_idx, layer_count)


def build_generated_param_scaler(
    cfg,
    output_gain_key=None,
    default_output_gain=1.0,
):
    """
    Build RF scaler from model config.
    """

    output_gain = cfg.get('hyper_rf_output_gain', default_output_gain)
    if output_gain_key is not None:
        output_gain = cfg.get(output_gain_key, output_gain)

    return GeneratedParamScaler(
        enabled=bool(cfg.get('hyper_rf_scaling', False)),
        mode=cfg.get('hyper_rf_mode', 'fan_in'),
        hidden_gain=float(cfg.get('hyper_rf_hidden_gain', math.sqrt(2.0))),
        output_gain=float(output_gain),
        bias_scale=float(cfg.get('hyper_rf_bias_scale', 1.0)),
    )


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


class LayerwiseHyperNetwork(nn.Module):
    """
    HyperMARL-style generator with one output head per target-network layer.

    The public HyperMARL code generates target weights and biases with separate
    heads for each actor/critic layer. The returned tensor is still flattened so
    existing LibSignal forward code can slice it without knowing which generator
    implementation was used.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        target_layout,
        hypernet_type='mlp',
        dropout=0.0,
        use_bias=True,
        head_init_gain=1.0,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.target_layout = self._normalize_layout(target_layout)
        if len(self.target_layout) == 0:
            raise ValueError('LayerwiseHyperNetwork requires a target_layout')

        kind = str(hypernet_type or 'mlp').lower()
        if kind not in ('linear', 'lin', 'mlp', 'nonlinear'):
            raise ValueError(f"Unknown hypernetwork type: {hypernet_type}")
        self.hypernet_type = kind
        self.weight_heads = nn.ModuleList()
        self.bias_heads = nn.ModuleList()

        for _, (out_dim, in_dim), _, weight_end, bias_end in self.target_layout:
            weight_dim = int(out_dim) * int(in_dim)
            bias_dim = int(out_dim)
            if kind in ('linear', 'lin'):
                weight_head = nn.Linear(input_dim, weight_dim, bias=use_bias)
                bias_head = nn.Linear(input_dim, bias_dim, bias=use_bias)
            else:
                weight_head = self._build_mlp_head(
                    input_dim,
                    hidden_dims,
                    weight_dim,
                    dropout,
                    use_bias,
                )
                bias_head = self._build_mlp_head(
                    input_dim,
                    hidden_dims,
                    bias_dim,
                    dropout,
                    use_bias,
                )
            self._init_weight_head(weight_head, head_init_gain)
            self._init_bias_head(bias_head)
            self.weight_heads.append(weight_head)
            self.bias_heads.append(bias_head)

        expected_dim = int(self.target_layout[-1][-1])
        if expected_dim != self.output_dim:
            raise ValueError(
                f"Layerwise layout output_dim mismatch: layout={expected_dim}, "
                f"output_dim={self.output_dim}"
            )

    @staticmethod
    def _normalize_layout(target_layout):
        layout = list(target_layout or [])
        if not layout:
            return []

        if len(layout[0]) == 4:
            layer_layout = []
            if len(layout) % 2 != 0:
                raise ValueError('Named parameter layout must contain weight/bias pairs')
            for idx in range(0, len(layout), 2):
                weight_name, weight_shape, weight_start, weight_end = layout[idx]
                bias_name, bias_shape, bias_start, bias_end = layout[idx + 1]
                if not str(weight_name).endswith('.weight') or not str(bias_name).endswith('.bias'):
                    raise ValueError('Named parameter layout must alternate weight then bias')
                if int(weight_end) != int(bias_start):
                    raise ValueError('Named parameter layout must be contiguous')
                out_dim, in_dim = tuple(weight_shape)
                layer_name = str(weight_name).rsplit('.', 1)[0]
                layer_layout.append(
                    (
                        layer_name,
                        (int(out_dim), int(in_dim)),
                        int(weight_start),
                        int(weight_end),
                        int(bias_end),
                    )
                )
            return layer_layout

        layer_layout = []
        for idx, item in enumerate(layout):
            if len(item) != 5:
                raise ValueError('Target layout entries must have 4 or 5 fields')

            if isinstance(item[1], (tuple, list)):
                name, shape, weight_start, bias_start, bias_end = item
                out_dim, in_dim = tuple(shape)
            else:
                out_dim, in_dim, weight_start, bias_start, bias_end = item
                name = f'layer{idx}'
            layer_layout.append(
                (
                    str(name),
                    (int(out_dim), int(in_dim)),
                    int(weight_start),
                    int(bias_start),
                    int(bias_end),
                )
            )
        return layer_layout

    @staticmethod
    def _build_mlp_head(input_dim, hidden_dims, output_dim, dropout, use_bias):
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for idx in range(len(dims) - 1):
            linear = nn.Linear(dims[idx], dims[idx + 1], bias=use_bias)
            nn.init.orthogonal_(linear.weight, gain=math.sqrt(2.0))
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim, bias=use_bias))
        return nn.Sequential(*layers)

    @staticmethod
    def _last_linear(module):
        if isinstance(module, nn.Linear):
            return module
        for layer in reversed(module):
            if isinstance(layer, nn.Linear):
                return layer
        raise TypeError('Hypernetwork head does not contain a Linear layer')

    def _init_weight_head(self, module, gain):
        layer = self._last_linear(module)
        nn.init.orthogonal_(layer.weight, gain=float(gain))
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def _init_bias_head(self, module):
        layer = self._last_linear(module)
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        chunks = []
        for weight_head, bias_head in zip(self.weight_heads, self.bias_heads):
            chunks.append(weight_head(x))
            chunks.append(bias_head(x))
        return torch.cat(chunks, dim=-1)


class HyperNetwork(MLPHyperNetwork):
    """
    Backward-compatible name for the original MLP hypernetwork.
    """


def build_hypernetwork(
    hypernet_type,
    input_dim,
    hidden_dims,
    output_dim,
    dropout=0.0,
    target_layout=None,
    head_mode='flat',
    use_bias=True,
    head_init_gain=1.0,
):
    """
    Build a linear or MLP hypernetwork from config.
    """

    kind = str(hypernet_type or 'mlp').lower()
    mode = str(head_mode or 'flat').lower()
    if mode in ('layerwise', 'headed', 'official'):
        return LayerwiseHyperNetwork(
            input_dim,
            hidden_dims,
            output_dim,
            target_layout=target_layout,
            hypernet_type=kind,
            dropout=dropout,
            use_bias=use_bias,
            head_init_gain=head_init_gain,
        )
    if kind in ('linear', 'lin'):
        return LinearHyperNetwork(input_dim, output_dim)
    if kind in ('mlp', 'nonlinear'):
        return MLPHyperNetwork(input_dim, hidden_dims, output_dim, dropout=dropout)
    raise ValueError(f"Unknown hypernetwork type: {hypernet_type}")
