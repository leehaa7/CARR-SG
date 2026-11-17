"""Run the CARR-SG modules on synthetic tensors with `python demo.py`."""

import torch
from torch.nn import functional as F

from core import DGCSD, DualGraphLearner


def main() -> None:
    torch.manual_seed(19)
    batch_size, num_nodes, history_length = 2, 4, 12
    channels, hidden_steps = 8, 7
    observed = torch.randn(batch_size, 1, num_nodes, history_length)
    history = F.pad(observed, (1, 0))
    hidden = torch.randn(batch_size, channels, num_nodes, hidden_steps)
    learner = DualGraphLearner(in_dim=1, num_nodes=num_nodes, embed_dim=10, patch_len=history_length).eval()
    convolution = DGCSD(in_channels=channels, out_channels=channels).eval()

    with torch.inference_mode():
        shared_graph, context_graph = learner(history, history_length=history_length)
        output = convolution(hidden, shared_graph, context_graph)

    if shared_graph.shape != (num_nodes, num_nodes) or context_graph.shape != (batch_size, num_nodes, num_nodes):
        raise RuntimeError("Unexpected shape for a propagation graph.")
    if output.shape != hidden.shape or not torch.isfinite(output).all():
        raise RuntimeError("Unexpected output shape or nonfinite output.")
    torch.testing.assert_close(shared_graph.sum(dim=-1), torch.ones(num_nodes))
    torch.testing.assert_close(context_graph.sum(dim=-1), torch.ones(batch_size, num_nodes))
    print(f"Shared graph: {tuple(shared_graph.shape)}")
    print(f"Context graph: {tuple(context_graph.shape)}")
    print(f"DGCSD output: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
