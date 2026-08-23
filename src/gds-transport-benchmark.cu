#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cuda_runtime.h>
#include <cufile.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr size_t kDirectIoAlignment = 4096;
constexpr unsigned char kPattern = 0xa5;
constexpr size_t kVerifyChunkBytes = 4 * 1024 * 1024;

enum class Mode {
    Staged,
    Gds,
    Both,
};

struct Options {
    std::string directory;
    size_t bytes = 1024ULL * 1024 * 1024;
    int iterations = 5;
    int warmup = 1;
    int device = 0;
    Mode mode = Mode::Both;
    bool durable = true;
    bool verify = true;
    bool keepFiles = false;
};

[[noreturn]] void fail(const std::string &message)
{
    throw std::runtime_error(message);
}

void checkCuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess) {
        fail(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::string cuFileStatusError(CUfileError_t status)
{
    std::string message(CUFILE_ERRSTR(status.err));
    if (IS_CUDA_ERR(status)) {
        const char *cudaError = nullptr;
        const CUresult lookup =
            cuGetErrorString(CU_FILE_CUDA_ERR(status), &cudaError);
        if (lookup == CUDA_SUCCESS && cudaError != nullptr) {
            message += ": ";
            message += cudaError;
        }
    }
    return message;
}

void checkCuFile(CUfileError_t status, const char *operation)
{
    if (status.err != CU_FILE_SUCCESS) {
        fail(std::string(operation) + ": " + cuFileStatusError(status));
    }
}

size_t parseBytes(const char *text)
{
    if (text == nullptr || *text == '\0' || *text == '-') {
        fail("invalid byte count");
    }

    errno = 0;
    char *end = nullptr;
    unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || value == 0) {
        fail(std::string("invalid byte count: ") + text);
    }

    unsigned long long multiplier = 1;
    if (*end != '\0') {
        if (end[1] != '\0') {
            fail(std::string("invalid byte suffix: ") + end);
        }
        switch (*end) {
        case 'k':
        case 'K':
            multiplier = 1024ULL;
            break;
        case 'm':
        case 'M':
            multiplier = 1024ULL * 1024;
            break;
        case 'g':
        case 'G':
            multiplier = 1024ULL * 1024 * 1024;
            break;
        case 't':
        case 'T':
            multiplier = 1024ULL * 1024 * 1024 * 1024;
            break;
        default:
            fail(std::string("invalid byte suffix: ") + end);
        }
    }

    if (value > std::numeric_limits<size_t>::max() / multiplier) {
        fail("byte count overflows size_t");
    }
    size_t bytes = static_cast<size_t>(value * multiplier);
    if (bytes > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
        fail("byte count exceeds the platform's file-offset range");
    }
    if (bytes % kDirectIoAlignment != 0) {
        fail("byte count must be a multiple of 4096 for direct I/O");
    }
    return bytes;
}

int parsePositiveInt(const char *text, const char *name, bool allowZero = false)
{
    errno = 0;
    char *end = nullptr;
    long value = std::strtol(text, &end, 10);
    const long minimum = allowZero ? 0 : 1;
    if (errno != 0 || end == text || *end != '\0' || value < minimum ||
        value > std::numeric_limits<int>::max()) {
        fail(std::string("invalid ") + name + ": " + text);
    }
    return static_cast<int>(value);
}

void usage(FILE *stream, const char *program)
{
    std::fprintf(
        stream,
        "Usage: %s --directory <GDS mount> [options]\n"
        "\n"
        "Compare checkpoint-shaped GPU I/O paths:\n"
        "  staged: cudaMemcpy(device->host) + pwrite, then pread + cudaMemcpy(host->device)\n"
        "  gds:    cuFileWrite(device->file), then cuFileRead(file->device)\n"
        "\n"
        "Options:\n"
        "  --bytes <N[K|M|G|T]>  Device payload size (default: 1G)\n"
        "  --iterations <N>      Measured cycles per mode (default: 5)\n"
        "  --warmup <N>          Unreported warm-up cycles (default: 1)\n"
        "  --device <index>      CUDA device index (default: 0)\n"
        "  --mode staged|gds|both (default: both)\n"
        "  --no-sync             Exclude fdatasync from checkpoint timing\n"
        "  --no-verify           Skip post-restore device validation\n"
        "  --keep-files          Retain generated image files\n"
        "  --help                Show this message\n",
        program);
}

Options parseOptions(int argc, char **argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto requireValue = [&](const char *name) -> const char * {
            if (++i >= argc) {
                fail(std::string(name) + " requires a value");
            }
            return argv[i];
        };

        if (arg == "--directory") {
            options.directory = requireValue("--directory");
        } else if (arg == "--bytes") {
            options.bytes = parseBytes(requireValue("--bytes"));
        } else if (arg == "--iterations") {
            options.iterations = parsePositiveInt(
                requireValue("--iterations"), "iteration count");
        } else if (arg == "--warmup") {
            options.warmup = parsePositiveInt(
                requireValue("--warmup"), "warm-up count", true);
        } else if (arg == "--device") {
            options.device = parsePositiveInt(
                requireValue("--device"), "device index", true);
        } else if (arg == "--mode") {
            const std::string value(requireValue("--mode"));
            if (value == "staged") {
                options.mode = Mode::Staged;
            } else if (value == "gds") {
                options.mode = Mode::Gds;
            } else if (value == "both") {
                options.mode = Mode::Both;
            } else {
                fail("--mode must be staged, gds, or both");
            }
        } else if (arg == "--no-sync") {
            options.durable = false;
        } else if (arg == "--no-verify") {
            options.verify = false;
        } else if (arg == "--keep-files") {
            options.keepFiles = true;
        } else if (arg == "--help" || arg == "-h") {
            usage(stdout, argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            fail("unknown argument: " + arg);
        }
    }

    if (options.directory.empty()) {
        fail("--directory is required");
    }
    return options;
}

void validateDirectory(const std::string &path)
{
    struct stat info = {};
    if (stat(path.c_str(), &info) != 0) {
        fail("cannot stat output directory " + path + ": " +
             std::strerror(errno));
    }
    if (!S_ISDIR(info.st_mode)) {
        fail("output path is not a directory: " + path);
    }
}

std::string joinPath(const std::string &directory, const std::string &name)
{
    if (!directory.empty() && directory.back() == '/') {
        return directory + name;
    }
    return directory + "/" + name;
}

class OutputFiles {
  public:
    explicit OutputFiles(bool keep) : keep_(keep) {}

    ~OutputFiles()
    {
        if (keep_) {
            return;
        }
        for (const std::string &path : paths_) {
            if (unlink(path.c_str()) != 0 && errno != ENOENT) {
                std::fprintf(stderr, "warning: could not remove %s: %s\n",
                             path.c_str(), std::strerror(errno));
            }
        }
    }

    void track(const std::string &path) { paths_.push_back(path); }

  private:
    bool keep_;
    std::vector<std::string> paths_;
};

class DeviceBuffer {
  public:
    explicit DeviceBuffer(size_t bytes) : bytes_(bytes)
    {
        checkCuda(cudaMalloc(&pointer_, bytes_), "cudaMalloc");
    }

    ~DeviceBuffer()
    {
        if (pointer_ != nullptr) {
            cudaError_t status = cudaFree(pointer_);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "warning: cudaFree: %s\n",
                             cudaGetErrorString(status));
            }
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    void *get() const { return pointer_; }
    size_t size() const { return bytes_; }

  private:
    void *pointer_ = nullptr;
    size_t bytes_;
};

class PinnedAlignedBuffer {
  public:
    explicit PinnedAlignedBuffer(size_t bytes) : bytes_(bytes)
    {
        int status = posix_memalign(&pointer_, kDirectIoAlignment, bytes_);
        if (status != 0) {
            fail(std::string("posix_memalign: ") + std::strerror(status));
        }
        cudaError_t cudaStatus =
            cudaHostRegister(pointer_, bytes_, cudaHostRegisterDefault);
        if (cudaStatus != cudaSuccess) {
            std::free(pointer_);
            pointer_ = nullptr;
            fail(std::string("cudaHostRegister: ") +
                 cudaGetErrorString(cudaStatus));
        }
        registered_ = true;
    }

    ~PinnedAlignedBuffer()
    {
        if (registered_) {
            cudaError_t status = cudaHostUnregister(pointer_);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "warning: cudaHostUnregister: %s\n",
                             cudaGetErrorString(status));
            }
        }
        std::free(pointer_);
    }

    PinnedAlignedBuffer(const PinnedAlignedBuffer &) = delete;
    PinnedAlignedBuffer &operator=(const PinnedAlignedBuffer &) = delete;

    void *get() const { return pointer_; }

  private:
    void *pointer_ = nullptr;
    size_t bytes_;
    bool registered_ = false;
};

class DirectFile {
  public:
    DirectFile(const std::string &path, size_t bytes) : path_(path)
    {
        fd_ = open(path_.c_str(), O_CREAT | O_EXCL | O_RDWR | O_DIRECT, 0600);
        if (fd_ < 0) {
            fail("open " + path_ + ": " + std::strerror(errno));
        }
        if (ftruncate(fd_, static_cast<off_t>(bytes)) != 0) {
            int savedErrno = errno;
            close(fd_);
            fd_ = -1;
            fail("ftruncate " + path_ + ": " + std::strerror(savedErrno));
        }
    }

    ~DirectFile()
    {
        if (fd_ >= 0 && close(fd_) != 0) {
            std::fprintf(stderr, "warning: close %s: %s\n", path_.c_str(),
                         std::strerror(errno));
        }
    }

    DirectFile(const DirectFile &) = delete;
    DirectFile &operator=(const DirectFile &) = delete;

    int get() const { return fd_; }

  private:
    std::string path_;
    int fd_ = -1;
};

class CuFileDriver {
  public:
    CuFileDriver()
    {
        checkCuFile(cuFileDriverOpen(), "cuFileDriverOpen");
        open_ = true;
    }

    ~CuFileDriver()
    {
        if (open_) {
            CUfileError_t status = cuFileDriverClose();
            if (status.err != CU_FILE_SUCCESS) {
                std::fprintf(stderr, "warning: cuFileDriverClose: %s\n",
                             cuFileStatusError(status).c_str());
            }
        }
    }

    CuFileDriver(const CuFileDriver &) = delete;
    CuFileDriver &operator=(const CuFileDriver &) = delete;

  private:
    bool open_ = false;
};

class CuFileBuffer {
  public:
    CuFileBuffer(const void *pointer, size_t bytes) : pointer_(pointer)
    {
        checkCuFile(cuFileBufRegister(pointer_, bytes, 0),
                    "cuFileBufRegister");
        registered_ = true;
    }

    ~CuFileBuffer()
    {
        if (registered_) {
            CUfileError_t status = cuFileBufDeregister(pointer_);
            if (status.err != CU_FILE_SUCCESS) {
                std::fprintf(stderr, "warning: cuFileBufDeregister: %s\n",
                             cuFileStatusError(status).c_str());
            }
        }
    }

    CuFileBuffer(const CuFileBuffer &) = delete;
    CuFileBuffer &operator=(const CuFileBuffer &) = delete;

  private:
    const void *pointer_;
    bool registered_ = false;
};

class CuFileHandle {
  public:
    explicit CuFileHandle(int fd)
    {
        CUfileDescr_t descriptor = {};
        descriptor.handle.fd = fd;
        descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        checkCuFile(cuFileHandleRegister(&handle_, &descriptor),
                    "cuFileHandleRegister");
    }

    ~CuFileHandle()
    {
        if (handle_ != nullptr) {
            cuFileHandleDeregister(handle_);
        }
    }

    CuFileHandle(const CuFileHandle &) = delete;
    CuFileHandle &operator=(const CuFileHandle &) = delete;

    CUfileHandle_t get() const { return handle_; }

  private:
    CUfileHandle_t handle_ = nullptr;
};

void writeAll(int fd, const void *buffer, size_t bytes)
{
    const auto *cursor = static_cast<const unsigned char *>(buffer);
    size_t completed = 0;
    while (completed < bytes) {
        ssize_t result = pwrite(fd, cursor + completed, bytes - completed,
                                static_cast<off_t>(completed));
        if (result < 0 && errno == EINTR) {
            continue;
        }
        if (result <= 0) {
            fail(std::string("pwrite: ") +
                 (result == 0 ? "unexpected zero-byte write"
                              : std::strerror(errno)));
        }
        completed += static_cast<size_t>(result);
    }
}

void readAll(int fd, void *buffer, size_t bytes)
{
    auto *cursor = static_cast<unsigned char *>(buffer);
    size_t completed = 0;
    while (completed < bytes) {
        ssize_t result = pread(fd, cursor + completed, bytes - completed,
                               static_cast<off_t>(completed));
        if (result < 0 && errno == EINTR) {
            continue;
        }
        if (result <= 0) {
            fail(std::string("pread: ") +
                 (result == 0 ? "unexpected end of file"
                              : std::strerror(errno)));
        }
        completed += static_cast<size_t>(result);
    }
}

std::string cuFileIoError(ssize_t result)
{
    if (IS_CUFILE_ERR(result)) {
        return CUFILE_ERRSTR(result);
    }
    if (result == 0) {
        return "unexpected zero-byte I/O";
    }
    return std::strerror(errno);
}

void cuFileWriteAll(CUfileHandle_t handle, const void *device, size_t bytes)
{
    size_t completed = 0;
    while (completed < bytes) {
        ssize_t result = cuFileWrite(handle, device, bytes - completed,
                                     static_cast<off_t>(completed),
                                     static_cast<off_t>(completed));
        if (result <= 0) {
            fail("cuFileWrite: " + cuFileIoError(result));
        }
        completed += static_cast<size_t>(result);
    }
}

void cuFileReadAll(CUfileHandle_t handle, void *device, size_t bytes)
{
    size_t completed = 0;
    while (completed < bytes) {
        ssize_t result = cuFileRead(handle, device, bytes - completed,
                                    static_cast<off_t>(completed),
                                    static_cast<off_t>(completed));
        if (result <= 0) {
            fail("cuFileRead: " + cuFileIoError(result));
        }
        completed += static_cast<size_t>(result);
    }
}

void syncFile(int fd)
{
    if (fdatasync(fd) != 0) {
        fail(std::string("fdatasync: ") + std::strerror(errno));
    }
}

double elapsedSeconds(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start)
        .count();
}

void verifyPattern(const DeviceBuffer &device)
{
    const size_t chunkBytes = std::min(device.size(), kVerifyChunkBytes);
    std::vector<unsigned char> host(chunkBytes);
    const auto *base = static_cast<const unsigned char *>(device.get());

    for (size_t offset = 0; offset < device.size(); offset += chunkBytes) {
        size_t length = std::min(chunkBytes, device.size() - offset);
        checkCuda(cudaMemcpy(host.data(), base + offset, length,
                             cudaMemcpyDeviceToHost),
                  "verification cudaMemcpy");
        auto mismatch = std::find_if(
            host.begin(), host.begin() + static_cast<ptrdiff_t>(length),
            [](unsigned char byte) { return byte != kPattern; });
        if (mismatch != host.begin() + static_cast<ptrdiff_t>(length)) {
            const size_t badOffset =
                offset + static_cast<size_t>(mismatch - host.begin());
            fail("verification failed at device byte " +
                 std::to_string(badOffset));
        }
    }
}

void printResult(const char *mode, const char *operation, int iteration,
                 size_t bytes, double seconds, bool durable)
{
    const double gib = static_cast<double>(bytes) /
                       static_cast<double>(1024ULL * 1024 * 1024);
    std::printf("%s,%s,%d,%zu,%.9f,%.6f,%s\n", mode, operation,
                iteration, bytes, seconds, gib / seconds,
                durable ? "true" : "false");
}

void runStaged(const Options &options, DeviceBuffer &device,
               const std::string &path)
{
    std::fprintf(stderr, "running host-staged path using %s\n", path.c_str());
    PinnedAlignedBuffer host(options.bytes);
    DirectFile file(path, options.bytes);

    const int total = options.warmup + options.iterations;
    for (int cycle = 0; cycle < total; ++cycle) {
        auto start = std::chrono::steady_clock::now();
        checkCuda(cudaMemcpy(host.get(), device.get(), options.bytes,
                             cudaMemcpyDeviceToHost),
                  "checkpoint cudaMemcpy device-to-host");
        writeAll(file.get(), host.get(), options.bytes);
        if (options.durable) {
            syncFile(file.get());
        }
        double checkpointSeconds = elapsedSeconds(start);

        checkCuda(cudaMemset(device.get(), 0, options.bytes),
                  "pre-restore cudaMemset");
        checkCuda(cudaDeviceSynchronize(), "pre-restore synchronize");

        start = std::chrono::steady_clock::now();
        readAll(file.get(), host.get(), options.bytes);
        checkCuda(cudaMemcpy(device.get(), host.get(), options.bytes,
                             cudaMemcpyHostToDevice),
                  "restore cudaMemcpy host-to-device");
        double restoreSeconds = elapsedSeconds(start);

        if (options.verify) {
            verifyPattern(device);
        }
        if (cycle >= options.warmup) {
            const int iteration = cycle - options.warmup;
            printResult("host_staged", "checkpoint", iteration, options.bytes,
                        checkpointSeconds, options.durable);
            printResult("host_staged", "restore", iteration, options.bytes,
                        restoreSeconds, options.durable);
        }
    }
}

void runGds(const Options &options, DeviceBuffer &device,
            const std::string &path)
{
    std::fprintf(stderr, "running GDS path using %s\n", path.c_str());
    CuFileDriver driver;
    CuFileBuffer registeredBuffer(device.get(), device.size());
    DirectFile file(path, options.bytes);
    CuFileHandle handle(file.get());

    const int total = options.warmup + options.iterations;
    for (int cycle = 0; cycle < total; ++cycle) {
        auto start = std::chrono::steady_clock::now();
        cuFileWriteAll(handle.get(), device.get(), options.bytes);
        if (options.durable) {
            syncFile(file.get());
        }
        double checkpointSeconds = elapsedSeconds(start);

        checkCuda(cudaMemset(device.get(), 0, options.bytes),
                  "pre-restore cudaMemset");
        checkCuda(cudaDeviceSynchronize(), "pre-restore synchronize");

        start = std::chrono::steady_clock::now();
        cuFileReadAll(handle.get(), device.get(), options.bytes);
        double restoreSeconds = elapsedSeconds(start);

        if (options.verify) {
            verifyPattern(device);
        }
        if (cycle >= options.warmup) {
            const int iteration = cycle - options.warmup;
            printResult("gds", "checkpoint", iteration, options.bytes,
                        checkpointSeconds, options.durable);
            printResult("gds", "restore", iteration, options.bytes,
                        restoreSeconds, options.durable);
        }
    }
}

} // namespace

int main(int argc, char **argv)
{
    try {
        Options options = parseOptions(argc, argv);
        validateDirectory(options.directory);
        checkCuda(cudaSetDevice(options.device), "cudaSetDevice");

        DeviceBuffer device(options.bytes);
        checkCuda(cudaMemset(device.get(), kPattern, device.size()),
                  "initial cudaMemset");
        checkCuda(cudaDeviceSynchronize(), "initial synchronize");

        const std::string prefix =
            "cuda-checkpoint-transport-" + std::to_string(getpid());
        const std::string stagedPath =
            joinPath(options.directory, prefix + "-staged.img");
        const std::string gdsPath =
            joinPath(options.directory, prefix + "-gds.img");

        OutputFiles outputFiles(options.keepFiles);
        if (options.mode == Mode::Staged || options.mode == Mode::Both) {
            outputFiles.track(stagedPath);
        }
        if (options.mode == Mode::Gds || options.mode == Mode::Both) {
            outputFiles.track(gdsPath);
        }

        std::puts(
            "mode,operation,iteration,bytes,seconds,gib_per_second,durable");
        if (options.mode == Mode::Staged || options.mode == Mode::Both) {
            runStaged(options, device, stagedPath);
        }
        if (options.mode == Mode::Gds || options.mode == Mode::Both) {
            runGds(options, device, gdsPath);
        }
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
