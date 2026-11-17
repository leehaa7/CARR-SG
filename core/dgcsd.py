#SPDX-License-Identifier: Apache-2.0
"""Routed dual-graph convolution extracted from the frozen PEMS gcn_module.

The standalone module exposes the routed path and its projection directly.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _propagate(features: Tensor, graph: Tensor) -> Tensor:
    """Apply the adjacency matrix along the node axis."""
    if graph.ndim == 2:
        graph = graph.unsqueeze(0).repeat(features.size(0), 1, 1)
    return torch.einsum("nvw, ncwl->ncvl", graph, features).contiguous()


class DGCSD(nn.Module):
    """Dual Graph Convolution with Spatial Decomposition."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        order: int = 2,
        smooth_depth: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, order) < 1 or smooth_depth < 0:
            raise ValueError("Channel counts and order must be positive, and smooth_depth nonnegative.")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1.")
        self.order = order
        self.smooth_depth = smooth_depth
        self.dropout = dropout
        self.projection = nn.Conv2d((order + 1) * in_channels, out_channels, kernel_size=1)

    def forward(self, hidden: Tensor, shared_graph: Tensor, context_graph: Tensor) -> Tensor:
        """Route [B, C_in, N, T] features and return [B, C_out, N, T]."""
        if hidden.ndim != 4:
            raise ValueError("hidden must have shape [B, C, N, T].")
        batch, _, nodes, _ = hidden.shape
        if shared_graph.shape != (nodes, nodes) or context_graph.shape != (batch, nodes, nodes):
            raise ValueError("Expected shared_graph [N, N] and context_graph [B, N, N].")

        #Split hidden features using the shared graph as a reference
        reference = torch.matrix_power(shared_graph, self.smooth_depth)
        shared = torch.einsum("bckj,ik->bcij", hidden, reference)
        complementary = hidden - shared
        depth_features = [hidden]
        #Propagate each component with its own graph
        for _ in range(self.order):
            shared = _propagate(shared, shared_graph)
            complementary = _propagate(complementary, context_graph)
            depth_features.append(shared + complementary)
        output = self.projection(torch.cat(depth_features, dim=1))
        return F.dropout(output, self.dropout, training=self.training)
