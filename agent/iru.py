import torch
import torch.nn as nn


class InterpolationRecurrentUnit(nn.Module):
    """One IRU cell used for within-decision iterative computation."""

    def __init__(self, hidden_dim, layer_norm=True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.layer_norm_enabled = bool(layer_norm)
        self.recurrence_norm = (
            nn.LayerNorm(self.hidden_dim * 2)
            if self.layer_norm_enabled
            else nn.Identity()
        )
        self.forget_gate = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.input_gate = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # Match the paper's public implementation: initially bias the cell
        # toward retaining its current computation.
        nn.init.ones_(self.forget_gate.bias)

    def forward(self, cell_state, hidden_state, context):
        combined = torch.cat([context, hidden_state], dim=-1)
        combined = self.recurrence_norm(combined)

        forget = torch.sigmoid(self.forget_gate(combined))
        candidate = torch.tanh(self.input_gate(combined))
        next_cell = forget * cell_state + (1.0 - forget) * candidate
        next_hidden = torch.tanh(next_cell)
        return next_cell, next_hidden


class IRUNetwork(nn.Module):
    """
    Feed-forward policy/value network with recurrent depth.

    The recurrent state is initialized to zero on every forward pass.  The
    same IRU block parameters are reused for ``thinking_steps`` iterations;
    no state is carried between environment timesteps.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        thinking_steps=5,
        num_blocks=1,
        layer_norm=True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.thinking_steps = int(thinking_steps)
        self.num_blocks = int(num_blocks)
        self.layer_norm_enabled = bool(layer_norm)

        if self.hidden_dim <= 0:
            raise ValueError('IRU hidden_dim must be positive')
        if self.thinking_steps <= 0:
            raise ValueError('IRU thinking_steps must be positive')
        if self.num_blocks <= 0:
            raise ValueError('IRU num_blocks must be positive')

        self.input_embedding = nn.Linear(self.input_dim, self.hidden_dim)
        self.input_norm = (
            nn.LayerNorm(self.hidden_dim)
            if self.layer_norm_enabled
            else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [
                InterpolationRecurrentUnit(
                    self.hidden_dim,
                    layer_norm=self.layer_norm_enabled,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.output_head = nn.Linear(self.hidden_dim, self.output_dim)

    def forward_features(self, inputs, film_params=None, film_scale=0.1):
        """Run within-decision recurrent computation up to the output head.

        ``film_params`` is an optional per-sample tensor containing
        ``(gamma_context, beta_context, gamma_hidden, beta_hidden)``.  The
        context adapter is applied once and the hidden adapter is reused after
        every thinking step.  Reusing the same adapter keeps the number of
        HyperLight parameters independent of ``thinking_steps`` and preserves
        the IRU's shared-computation interpretation.

        Zero FiLM parameters are exactly the unmodulated IRU path.
        """
        context = self.input_norm(self.input_embedding(inputs))
        hidden_gamma = None
        hidden_beta = None
        if film_params is not None:
            expected_dim = 4 * self.hidden_dim
            if film_params.shape[:-1] != inputs.shape[:-1]:
                raise ValueError(
                    'IRU FiLM batch shape must match inputs: '
                    f'film={tuple(film_params.shape[:-1])}, '
                    f'inputs={tuple(inputs.shape[:-1])}'
                )
            if film_params.shape[-1] != expected_dim:
                raise ValueError(
                    f'IRU FiLM parameter mismatch: expected={expected_dim}, '
                    f'actual={film_params.shape[-1]}'
                )
            film_scale = float(film_scale)
            if film_scale < 0.0:
                raise ValueError('IRU FiLM scale must be non-negative')
            context_gamma, context_beta, hidden_gamma, hidden_beta = torch.split(
                film_params,
                self.hidden_dim,
                dim=-1,
            )
            context = context * (
                1.0 + film_scale * torch.tanh(context_gamma)
            )
            context = context + film_scale * torch.tanh(context_beta)

        state_shape = (*context.shape[:-1], self.hidden_dim)
        cell_state = context.new_zeros(state_shape)
        hidden_state = context.new_zeros(state_shape)

        for _ in range(self.thinking_steps):
            residual = hidden_state
            for block in self.blocks:
                cell_state, hidden_state = block(
                    cell_state,
                    hidden_state,
                    context,
                )
            hidden_state = residual + hidden_state
            if hidden_gamma is not None:
                hidden_state = hidden_state * (
                    1.0 + film_scale * torch.tanh(hidden_gamma)
                )
                hidden_state = hidden_state + film_scale * torch.tanh(hidden_beta)

        return hidden_state

    def forward(self, inputs, film_params=None, film_scale=0.1):
        """Run within-decision recurrent computation and the output head."""
        hidden_state = self.forward_features(
            inputs,
            film_params=film_params,
            film_scale=film_scale,
        )
        return self.output_head(hidden_state)
