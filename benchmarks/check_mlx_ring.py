"""Verify that two MLX ring ranks can exchange data without loading a model."""

import argparse
import socket


def check_gpu_handoff(mx, group):
    rank = group.rank()
    shape = (1, 4, 2048)
    if rank == 1:
        activations = mx.ones(shape, dtype=mx.bfloat16, stream=mx.gpu)
        activations = mx.add(activations, 1, stream=mx.gpu)
        activations = mx.distributed.send(activations, 0, group=group, stream=mx.cpu)
    else:
        activations = mx.zeros(shape, dtype=mx.bfloat16, stream=mx.cpu)
        activations = mx.distributed.recv_like(activations, 1, group=group, stream=mx.cpu)
        activations = mx.add(activations, 3, stream=mx.gpu)

    gathered = mx.distributed.all_gather(activations, group=group, stream=mx.cpu)
    checksum = mx.sum(gathered.astype(mx.float32), stream=mx.gpu).item()
    if checksum != 57_344:
        raise RuntimeError(f"Rank {rank} received incorrect GPU pipeline checksum {checksum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="Also test a GPU activation handoff.")
    args = parser.parse_args()

    import mlx.core as mx

    host = socket.gethostname()
    print(f"{host}: initializing MLX ring", flush=True)
    group = mx.distributed.init(backend="ring", strict=True)
    if group.size() != 2:
        raise RuntimeError(f"Expected two ranks, got {group.size()}")

    rank = group.rank()
    print(f"{host}: rank {rank} joined, checking data exchange", flush=True)
    total = mx.distributed.all_sum(mx.array(rank + 1), group=group, stream=mx.cpu)
    if total.item() != 3:
        raise RuntimeError(f"{host}: rank {rank} received an incorrect sum")
    print(f"{host}: rank {rank} ring OK (sum=3)", flush=True)
    if args.gpu:
        print(f"{host}: rank {rank} checking GPU activation handoff", flush=True)
        check_gpu_handoff(mx, group)
        print(f"{host}: rank {rank} GPU activation handoff OK", flush=True)


if __name__ == "__main__":
    main()
