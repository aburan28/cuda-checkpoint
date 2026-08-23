"""Hold real CUDA state without PyTorch.

A rank has to be a genuine CUDA process for cuda-checkpoint to act on it at
all. Torch would do, but on a node where the GPU is nearly full of somebody's
inference server, and where the point of the test is the checkpoint machinery
rather than the framework, a few hundred lines of ctypes over libcuda is the
smaller and more honest instrument.

Allocates a device buffer, fills it with a known byte, and can verify it later -
which is what turns "the process came back" into "the process came back with its
memory".
"""

import ctypes

CUDA_SUCCESS = 0
PATTERN = 0xAB


class CudaError(RuntimeError):
    pass


class DeviceBuffer:
    """A device allocation that outlives a checkpoint, and knows its own value."""

    def __init__(self, size=8 << 20, device=0):
        self.size = int(size)
        self.lib = ctypes.CDLL("libcuda.so.1")
        self._check(self.lib.cuInit(0), "cuInit")

        dev = ctypes.c_int()
        self._check(self.lib.cuDeviceGet(ctypes.byref(dev), device), "cuDeviceGet")

        ctx = ctypes.c_void_p()
        self._check(
            self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev),
            "cuDevicePrimaryCtxRetain",
        )
        self._check(self.lib.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
        self.ctx = ctx

        ptr = ctypes.c_ulonglong()
        self._check(
            self.lib.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(self.size)),
            "cuMemAlloc",
        )
        self.ptr = ptr
        self._check(
            self.lib.cuMemsetD8_v2(ptr, ctypes.c_ubyte(PATTERN),
                                   ctypes.c_size_t(self.size)),
            "cuMemsetD8",
        )
        self._check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")

    def _check(self, rc, what):
        if rc != CUDA_SUCCESS:
            raise CudaError(f"{what} returned {rc}")

    def intact(self, samples=4096):
        """Read the buffer back and confirm every sampled byte is the pattern.

        Samples rather than reads it whole: the question is whether the memory
        survived, and a corrupted restore does not corrupt one byte in eight
        million.
        """
        count = min(samples, self.size)
        host = (ctypes.c_ubyte * count)()
        self._check(
            self.lib.cuMemcpyDtoH_v2(host, self.ptr, ctypes.c_size_t(count)),
            "cuMemcpyDtoH(head)",
        )
        if any(byte != PATTERN for byte in host):
            return False

        tail = (ctypes.c_ubyte * count)()
        offset = ctypes.c_ulonglong(self.ptr.value + self.size - count)
        self._check(
            self.lib.cuMemcpyDtoH_v2(tail, offset, ctypes.c_size_t(count)),
            "cuMemcpyDtoH(tail)",
        )
        return all(byte == PATTERN for byte in tail)
