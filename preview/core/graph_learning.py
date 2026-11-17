# SPDX-License-Identifier: Apache-2.0
"""Shared and context-dependent graph learning from the frozen PEMS implementation.

The active graph constructor and its input preparation are extracted from
graph_distribution.py and model_distribution.py. The preview uses standalone
module inputs and a device-aware, nonpersistent diagonal buffer.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DualGraphLearner(nn.Module):
    """Build shared and context-dependent graphs for routed propagation."""

    def __init__(
        self,
        in_dim: int,
        num_nodes: int,
        embed_dim: int,
        *,
        patch_len: int = 12,
        smooth_depth: int = 1,
    ) -> None:
        super().__init__()
        if min(in_dim, num_nodes, embed_dim, patch_len) < 1 or smooth_depth < 0:
            raise ValueError("Dimensions and patch_len must be positive, and smooth_depth nonnegative.")
        self.in_dim = in_dim
        self.num_nodes = num_nodes
        self.patch_len = patch_len
        self.smooth_depth = smooth_depth
        self.embed1 = nn.Embedding(num_nodes, embed_dim)
        self.short_conv = nn.Conv2d(in_dim, 16, (1, 5), padding="same")
        self.short_line = nn.Parameter(torch.randn(16, patch_len, embed_dim))
        self.diff_conv = nn.Conv2d(in_dim, 16, (1, 5), padding="same")
        self.diff_line = nn.Parameter(torch.randn(16, patch_len, embed_dim))
        self.long_line = nn.Linear(embed_dim, embed_dim)
        self.norm_dynamic1 = nn.LayerNorm(embed_dim)
        self.register_buffer("diag", torch.eye(num_nodes, dtype=torch.float32) * 1e-5, persistent=False)

    def get_static(self) -> Tensor:
        """Return the shared adjacency matrix with shape [N, N]."""
        node_index = torch.arange(self.num_nodes, device=self.embed1.weight.device)
        node_embedding = self.embed1(node_index)
        adjacency = F.relu(torch.mm(node_embedding, node_embedding.transpose(0, 1)))
        return adjacency / adjacency.sum(dim=-1, keepdim=True)

    def get_dynamic(self, deviation: Tensor, differences: Tensor) -> Tensor:
        """Return patch adjacencies [B, P, N, N] from matching [B, C, N, T] inputs."""
        if deviation.ndim != 4 or differences.shape != deviation.shape:
            raise ValueError("deviation and differences must have matching [B, C, N, T] shapes.")
        batch, channels, nodes, length = deviation.shape
        if (channels, nodes) != (self.in_dim, self.num_nodes):
            raise ValueError("Input channel or node count does not match the graph learner.")
        patch_count, remainder = divmod(length, self.patch_len)
        if patch_count == 0 or remainder:
            raise ValueError("The graph-construction length must be a positive multiple of patch_len.")

        node_index = torch.arange(nodes, device=self.embed1.weight.device)
        node_guidance = self.long_line(self.embed1(node_index))
        deviation_features = self.short_conv(deviation)
        difference_features = self.diff_conv(differences)
        patch_shape = (batch, deviation_features.size(1), nodes, patch_count, self.patch_len)
        deviation_features = deviation_features.view(patch_shape)
        difference_features = difference_features.view(patch_shape)

        deviation_embedding = torch.einsum("bcnpl,cld->bpnd", deviation_features, self.short_line)
        difference_embedding = torch.einsum("bcnpl,cld->bpnd", difference_features, self.diff_line)
        context = self.norm_dynamic1(deviation_embedding + torch.sigmoid(difference_embedding) * node_guidance)
        similarities = torch.einsum("bpik,bpjk->bpij", context, context)
        adjacency = F.relu(similarities) + self.diag
        return adjacency / adjacency.sum(dim=-1, keepdim=True)

    def forward(self, history: Tensor, *, history_length: int | None = None) -> tuple[Tensor, Tensor]:
        """Return shared [N, N] and context [B, N, N] propagation graphs.

        history is the backbone input after preprocessing and any left padding.
        history_length selects the trailing observed time steps used for graph
        construction. Differences are computed before this selection.
        """
        if history.ndim != 4 or history.shape[1:3] != (self.in_dim, self.num_nodes):
            raise ValueError("history must have shape [B, in_dim, num_nodes, T].")
        length = history.size(-1) if history_length is None else history_length
        if not 1 <= length <= history.size(-1):
            raise ValueError("history_length must be positive and at most history.size(-1).")

        shared_graph = self.get_static()
        propagated = torch.einsum(
            "bckj,ik->bcij", history, torch.matrix_power(shared_graph, self.smooth_depth)
        )
        deviation = history - propagated
        differences = F.pad(history[..., 1:] - history[..., :-1], (1, 0))
        context_graphs = self.get_dynamic(deviation[..., -length:], differences[..., -length:])
        return shared_graph, context_graphs.mean(dim=1)
