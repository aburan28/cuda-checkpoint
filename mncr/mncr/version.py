__version__ = "0.1.0"

# Minimum external versions this system is written against. The agent refuses to
# operate on a node that reports less, because restore behaviour is not defined
# below these points.
MIN_DRIVER = 610          # 610 for job-file IPC; 580 is the floor for --device-map
MIN_CRIU = (4, 0)         # process-tree support with the CUDA plugin
MIN_NCCL = (2, 29, 7)     # ncclCommSuspend / ncclCommResume exist from here
