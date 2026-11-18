"""Verify that two MLX ring ranks can exchange data without loading a model."""

import socket


def main():
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


if __name__ == "__main__":
    main()
