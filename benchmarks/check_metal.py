"""Check Metal visibility and run a small MLX GPU operation."""

import ctypes
import sys


def main() -> int:
    if sys.platform != "darwin":
        print("This benchmark requires macOS with Metal.", file=sys.stderr)
        return 1

    metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
    get_device = metal.MTLCreateSystemDefaultDevice
    get_device.argtypes = []
    get_device.restype = ctypes.c_void_p
    if not get_device():
        print(
            "Metal exposes no default GPU to this process. "
            "Run this check in macOS Terminal.app with the same Python executable.",
            file=sys.stderr,
        )
        return 1

    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    result = (mx.array([1, 2]) + 1).tolist()
    if result != [2, 3]:
        raise RuntimeError(f"Unexpected MLX GPU result: {result}")
    print("Metal is visible and MLX GPU computation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
