"""Neural building blocks for topology-aware HyperLight policies.

These modules deliberately have no CityFlow dependency.  The agent is
responsible for turning simulator-specific lane/link data into padded movement
tokens and for supplying the static directed edge list.
"""

import math

import torch
import torch.nn as nn


class _MovementAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, feedforward_dim, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
        )

    def forward(self, tokens, padding_mask):
        attended, _ = self.attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        tokens = self.norm1(tokens + self.dropout(attended))
        tokens = self.norm2(tokens + self.dropout(self.feedforward(tokens)))
        return tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class MovementTokenEncoder(nn.Module):
    """Encode a variable-size set of lane-link movements into one node state.

    Dynamic lane features are paired with a static phase-to-movement mask, the
    current signal phase, and two normalized positional scalars.  Attention is
    local to one intersection; no neighboring observation enters the actor.
    """

    def __init__(
        self,
        dynamic_feature_dim,
        action_dim,
        token_dim=64,
        output_dim=64,
        num_heads=4,
        num_layers=1,
        feedforward_dim=None,
        dropout=0.0,
        static_feature_dim=0,
    ):
        super().__init__()
        dynamic_feature_dim = int(dynamic_feature_dim)
        action_dim = int(action_dim)
        token_dim = int(token_dim)
        output_dim = int(output_dim)
        static_feature_dim = int(static_feature_dim)
        num_heads = int(num_heads)
        num_layers = int(num_layers)
        if dynamic_feature_dim <= 0 or action_dim <= 0:
            raise ValueError('MovementTokenEncoder dimensions must be positive')
        if token_dim <= 0 or output_dim <= 0 or num_layers < 0 or static_feature_dim < 0:
            raise ValueError('MovementTokenEncoder hidden dimensions are invalid')
        if num_heads <= 0 or token_dim % num_heads != 0:
            raise ValueError('movement token_dim must be divisible by num_heads')

        # dynamic + phase availability + current phase + current-green +
        # normalized source-lane position + normalized movement position
        input_dim = dynamic_feature_dim + static_feature_dim + 2 * action_dim + 3
        self.static_feature_dim = static_feature_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, token_dim),
            nn.GELU(),
            nn.LayerNorm(token_dim),
        )
        ff_dim = int(feedforward_dim or (2 * token_dim))
        self.blocks = nn.ModuleList(
            [
                _MovementAttentionBlock(token_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_projection = nn.Sequential(
            nn.Linear(2 * token_dim + action_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )
        self.output_dim = output_dim

    def forward(
        self,
        dynamic_features,
        token_mask,
        phase_availability,
        current_phase,
        source_position=None,
        movement_position=None,
        static_features=None,
    ):
        """
        Args:
            dynamic_features: ``[B, N, M, F]`` lane-derived features.
            token_mask: ``[N, M]`` or ``[B, N, M]`` valid-token mask.
            phase_availability: ``[N, M, A]`` or batched equivalent.
            current_phase: one-hot/soft phase features ``[B, N, A]``.
        Returns:
            Encoded local node states with shape ``[B, N, output_dim]``.
        """
        if dynamic_features.dim() != 4:
            raise ValueError('dynamic_features must have shape [B, N, M, F]')
        batch_size, node_count, movement_count, _ = dynamic_features.shape
        device = dynamic_features.device
        dtype = dynamic_features.dtype

        if token_mask.dim() == 2:
            token_mask = token_mask.unsqueeze(0).expand(batch_size, -1, -1)
        if token_mask.shape != (batch_size, node_count, movement_count):
            raise ValueError('token_mask shape does not match dynamic_features')
        token_mask = token_mask.to(device=device, dtype=torch.bool)
        if not token_mask.any(dim=-1).all():
            raise ValueError('every intersection needs at least one valid movement token')

        if phase_availability.dim() == 3:
            phase_availability = phase_availability.unsqueeze(0).expand(batch_size, -1, -1, -1)
        phase_availability = phase_availability.to(device=device, dtype=dtype)
        if phase_availability.shape[:3] != (batch_size, node_count, movement_count):
            raise ValueError('phase_availability shape does not match movement tokens')
        if current_phase.shape[:2] != (batch_size, node_count):
            raise ValueError('current_phase shape does not match movement tokens')
        current_phase = current_phase.to(device=device, dtype=dtype)
        if phase_availability.shape[-1] != current_phase.shape[-1]:
            raise ValueError('phase feature dimensions do not match')

        def _position_feature(value):
            if value is None:
                return dynamic_features.new_zeros(batch_size, node_count, movement_count, 1)
            if value.dim() == 2:
                value = value.unsqueeze(0).expand(batch_size, -1, -1)
            if value.shape != (batch_size, node_count, movement_count):
                raise ValueError('movement position shape does not match token mask')
            return value.to(device=device, dtype=dtype).unsqueeze(-1)

        if self.static_feature_dim == 0:
            static_features = dynamic_features.new_zeros(
                batch_size,
                node_count,
                movement_count,
                0,
            )
        else:
            if static_features is None:
                raise ValueError('static movement features are required by this encoder')
            if static_features.dim() == 3:
                static_features = static_features.unsqueeze(0).expand(batch_size, -1, -1, -1)
            expected_static_shape = (
                batch_size,
                node_count,
                movement_count,
                self.static_feature_dim,
            )
            if static_features.shape != expected_static_shape:
                raise ValueError('static_features shape does not match movement tokens')
            static_features = static_features.to(device=device, dtype=dtype)

        expanded_phase = current_phase.unsqueeze(2).expand(-1, -1, movement_count, -1)
        current_green = (phase_availability * expanded_phase).sum(dim=-1, keepdim=True)
        token_input = torch.cat(
            [
                dynamic_features,
                static_features,
                phase_availability,
                expanded_phase,
                current_green,
                _position_feature(source_position),
                _position_feature(movement_position),
            ],
            dim=-1,
        )
        tokens = self.input_projection(token_input)
        flat_tokens = tokens.reshape(batch_size * node_count, movement_count, -1)
        padding_mask = (~token_mask).reshape(batch_size * node_count, movement_count)
        flat_tokens = flat_tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            flat_tokens = block(flat_tokens, padding_mask)
        tokens = flat_tokens.reshape(batch_size, node_count, movement_count, -1)

        mask = token_mask.unsqueeze(-1)
        count = mask.sum(dim=2).clamp_min(1).to(dtype=dtype)
        mean_pool = (tokens * mask).sum(dim=2) / count
        max_pool = tokens.masked_fill(~mask, torch.finfo(dtype).min).max(dim=2).values
        pooled = torch.cat([mean_pool, max_pool, current_phase], dim=-1)
        return self.output_projection(pooled)


class FiLMConditioner(nn.Module):
    """Small hypernetwork that emits bounded per-layer FiLM parameters."""

    def __init__(
        self,
        meta_dim,
        feature_dim,
        num_layers,
        hidden_dims=(64,),
        scale=0.1,
        zero_init=True,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_layers = int(num_layers)
        self.scale = float(scale)
        if self.feature_dim <= 0 or self.num_layers <= 0 or self.scale < 0.0:
            raise ValueError('invalid FiLM conditioner dimensions or scale')

        dims = [int(meta_dim)] + [int(item) for item in hidden_dims]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        layers.append(nn.Linear(dims[-1], 2 * self.num_layers * self.feature_dim))
        self.net = nn.Sequential(*layers)
        if zero_init:
            final = self.net[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, meta):
        shape = meta.shape[:-1] + (self.num_layers, 2, self.feature_dim)
        return self.net(meta).view(shape)

    def apply(self, features, conditioning, layer_index):
        gamma = torch.tanh(conditioning[..., layer_index, 0, :])
        beta = torch.tanh(conditioning[..., layer_index, 1, :])
        return features * (1.0 + self.scale * gamma) + self.scale * beta


class DirectedGraphAttentionLayer(nn.Module):
    """Multi-head message passing that preserves the supplied edge direction.

    An edge ``src -> dst`` means that ``dst`` receives a message from ``src``.
    Self loops are inserted internally, including for isolated intersections.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads=4,
        dropout=0.0,
        use_edge_weight=False,
        edge_weight_scale=1.0,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_heads = int(num_heads)
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError('graph hidden_dim must be positive and divisible by num_heads')
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_edge_weight = bool(use_edge_weight)
        self.edge_weight_scale = float(edge_weight_scale)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(self, node_features, edge_index, edge_weight=None):
        if node_features.dim() != 3:
            raise ValueError('node_features must have shape [B, N, D]')
        batch_size, node_count, _ = node_features.shape
        device = node_features.device
        dtype = node_features.dtype
        edge_index = edge_index.to(device=device, dtype=torch.long)
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError('edge_index must have shape [2, E]')
        if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= node_count):
            raise ValueError('edge_index contains an invalid node index')

        self_nodes = torch.arange(node_count, device=device, dtype=torch.long)
        self_edges = torch.stack([self_nodes, self_nodes], dim=0)
        original_edge_count = edge_index.shape[1]
        edge_index = torch.cat([edge_index, self_edges], dim=1)
        src, dst = edge_index[0], edge_index[1]

        q = self.query(node_features).view(batch_size, node_count, self.num_heads, self.head_dim)
        k = self.key(node_features).view(batch_size, node_count, self.num_heads, self.head_dim)
        v = self.value(node_features).view(batch_size, node_count, self.num_heads, self.head_dim)
        scores = (q[:, dst] * k[:, src]).sum(dim=-1) / math.sqrt(float(self.head_dim))

        if self.use_edge_weight:
            if edge_weight is None:
                original_weights = torch.ones(original_edge_count, device=device, dtype=dtype)
            else:
                original_weights = edge_weight.to(device=device, dtype=dtype)
                if original_weights.numel() != original_edge_count:
                    raise ValueError('edge_weight length does not match edge_index')
            if original_weights.numel():
                original_weights = original_weights / original_weights.mean().clamp_min(1e-8)
            all_weights = torch.cat(
                [original_weights, torch.ones(node_count, device=device, dtype=dtype)],
                dim=0,
            ).clamp_min(1e-8)
            scores = scores + self.edge_weight_scale * all_weights.log().view(1, -1, 1)

        scatter_index = dst.view(1, -1, 1).expand(batch_size, -1, self.num_heads)
        max_score = scores.new_full((batch_size, node_count, self.num_heads), -torch.inf)
        max_score.scatter_reduce_(1, scatter_index, scores, reduce='amax', include_self=True)
        exp_scores = torch.exp(scores - max_score.gather(1, scatter_index))
        denominator = scores.new_zeros((batch_size, node_count, self.num_heads))
        denominator.scatter_add_(1, scatter_index, exp_scores)
        attention = exp_scores / denominator.gather(1, scatter_index).clamp_min(1e-8)
        attention = self.dropout(attention)

        messages = attention.unsqueeze(-1) * v[:, src]
        aggregate = messages.new_zeros(batch_size, node_count, self.num_heads, self.head_dim)
        message_index = dst.view(1, -1, 1, 1).expand_as(messages)
        aggregate.scatter_add_(1, message_index, messages)
        aggregate = aggregate.reshape(batch_size, node_count, self.hidden_dim)

        node_features = self.norm1(node_features + self.dropout(self.output(aggregate)))
        return self.norm2(node_features + self.dropout(self.feedforward(node_features)))


class DirectedGraphCritic(nn.Module):
    """Directed graph centralized critic with optional meta-conditioned FiLM."""

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        use_edge_weight=False,
        edge_weight_scale=1.0,
        global_pool=True,
        meta_dim=None,
        film_hidden_dims=(64,),
        film_scale=0.1,
        film_zero_init=True,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_layers = int(num_layers)
        if hidden_dim <= 0 or num_layers <= 0:
            raise ValueError('graph critic dimensions must be positive')
        self.input_projection = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            [
                DirectedGraphAttentionLayer(
                    hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_edge_weight=use_edge_weight,
                    edge_weight_scale=edge_weight_scale,
                )
                for _ in range(num_layers)
            ]
        )
        self.film = None
        if meta_dim is not None:
            self.film = FiLMConditioner(
                meta_dim,
                hidden_dim,
                num_layers,
                hidden_dims=film_hidden_dims,
                scale=film_scale,
                zero_init=film_zero_init,
            )
        self.global_pool = bool(global_pool)
        value_input_dim = hidden_dim * (3 if self.global_pool else 1)
        self.value_head = nn.Sequential(
            nn.Linear(value_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features, edge_index, edge_weight=None, meta=None):
        hidden = self.input_projection(node_features)
        conditioning = None
        if self.film is not None:
            if meta is None:
                raise ValueError('graph critic FiLM requires agent meta features')
            conditioning = self.film(meta)
        for layer_index, layer in enumerate(self.layers):
            hidden = layer(hidden, edge_index, edge_weight=edge_weight)
            if conditioning is not None:
                hidden = self.film.apply(hidden, conditioning, layer_index)

        if self.global_pool:
            global_mean = hidden.mean(dim=1, keepdim=True).expand_as(hidden)
            global_max = hidden.max(dim=1, keepdim=True).values.expand_as(hidden)
            hidden = torch.cat([hidden, global_mean, global_max], dim=-1)
        return self.value_head(hidden).squeeze(-1)
