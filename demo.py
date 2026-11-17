"""Show graph construction and propagation with synthetic data or PEMS04."""

import argparse
from pathlib import Path
import zipfile

import torch
from torch import Tensor
from torch.nn import functional as F

from core import DGCSD, DualGraphLearner


def load_pems04_windows(path: Path, batch_size: int, history_length: int) -> Tensor:
    """Read two observed flow windows from the start of the PEMS04 array."""
    import numpy as np

    length = batch_size * history_length
    with zipfile.ZipFile(path) as archive, archive.open("data.npy") as stream:
        version = np.lib.format.read_magic(stream)
        readers = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
        }
        if version not in readers:
            raise ValueError(f"Unsupported PEMS04 array version: {version}")
        shape, fortran_order, dtype = readers[version](stream)
        if (fortran_order or len(shape) != 3 or shape[0] < length
                or shape[1] != 307 or shape[2] < 1 or dtype.kind not in "fiu"):
            raise ValueError("Expected a row-major PEMS04 array with 307 sensors and a flow channel.")
        #Read only the rows needed for the input windows
        byte_count = length * shape[1] * shape[2] * dtype.itemsize
        payload = stream.read(byte_count)
        if len(payload) != byte_count:
            raise ValueError("The PEMS04 array ends before the two input windows.")
        windows = np.frombuffer(payload, dtype=dtype).reshape(batch_size, history_length, *shape[1:])
        flow = windows[..., 0].astype(np.float32)
    return torch.from_numpy(flow).transpose(1, 2).unsqueeze(1).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pems04", type=Path, metavar="PEMS04.npz", help="Use two real traffic windows.")
    args = parser.parse_args()

    torch.manual_seed(19)
    batch_size, history_length = 2, 12
    if args.pems04 is None:
        observed = torch.randn(batch_size, 1, 4, history_length)
        hidden = torch.randn(batch_size, 8, 4, 7)
    else:
        observed = load_pems04_windows(args.pems04, batch_size, history_length)
        if not torch.isfinite(observed).all():
            raise ValueError("The selected PEMS04 windows contain nonfinite values.")
        mean = observed.mean(dim=-1, keepdim=True)
        variance = observed.var(dim=-1, unbiased=False, keepdim=True)
        observed = (observed - mean) / torch.sqrt(variance + 1e-5)
        #Use the observed flow as hidden features for this demo
        hidden = observed

    num_nodes, channels = observed.size(2), hidden.size(1)
    history = F.pad(observed, (1, 0))
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
    if args.pems04 is not None:
        print(f"Input flow windows: {tuple(observed.shape)}")
    print(f"Shared graph: {tuple(shared_graph.shape)}")
    print(f"Context graph: {tuple(context_graph.shape)}")
    if args.pems04 is not None:
        context_change = (context_graph[0] - context_graph[1]).abs().sum(dim=-1).mean()
        #Inspect the first propagation step in each branch
        reference = torch.matrix_power(shared_graph, convolution.smooth_depth)
        shared_features = torch.einsum("ij,bcjt->bcit", reference, hidden)
        complementary_features = hidden - shared_features
        shared_hop = torch.einsum("ij,bcjt->bcit", shared_graph, shared_features)
        context_hop = torch.einsum("bij,bcjt->bcit", context_graph, complementary_features)
        print(f"Context graph change across windows (mean row L1): {context_change.item():.4f}")
        print(f"Shared-branch features after one hop: {tuple(shared_hop.shape)}")
        print(f"Context-branch features after one hop: {tuple(context_hop.shape)}")
        point = (0, 0, 0, 0)
        print(f"Window 1, sensor 0, step 0: shared hop={shared_hop[point].item():.4f}, "
              f"context hop={context_hop[point].item():.4f}")
    print(f"DGCSD output: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
