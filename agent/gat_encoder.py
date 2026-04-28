import math
import torch
import torch.nn as nn


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.0):
        super().__init__()
        if out_dim % heads != 0:
            raise ValueError("out_dim must be divisible by heads")

        self.heads = heads
        self.head_dim = out_dim // heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: [B, N, D], adj: [N, N] or [B, N, N]
        bsz, n_nodes, _ = x.shape

        q = self.q_proj(x).view(bsz, n_nodes, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, n_nodes, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, n_nodes, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.scale + 1e-8)

        if adj.dim() == 2:
            adj = adj.unsqueeze(0).unsqueeze(0)
        else:
            adj = adj.unsqueeze(1)

        mask = adj > 0
        weight_bias = torch.log(adj.clamp_min(1e-6))
        scores = scores + weight_bias
        scores = scores.masked_fill(~mask, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, n_nodes, -1)
        out = self.out_proj(out)

        out = self.dropout(out)
        return self.norm(out + self.residual(x))


class GATEncoder(nn.Module):
    """
    Two-layer GAT encoder with residual connections and layer norm.
    """

    def __init__(self, input_dim, hidden_dim, heads=4, layers=2, dropout=0.0):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")

        self.layers = nn.ModuleList()
        for i in range(layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(GraphAttentionLayer(in_dim, hidden_dim, heads=heads, dropout=dropout))

    def forward(self, x, adj):
        h = x
        for layer in self.layers:
            h = layer(h, adj)
        return h
