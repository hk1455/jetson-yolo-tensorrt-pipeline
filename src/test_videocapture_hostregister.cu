#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <unistd.h>

#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
        std::cerr << "CUDA error: " << cudaGetErrorString(err__)             \
                  << " (" << static_cast<int>(err__) << ")"                  \
                  << " at " << __FILE__ << ":" << __LINE__ << '\n';          \
        std::exit(EXIT_FAILURE);                                             \
    }                                                                        \
} while (0)

__global__ void checksumKernel(
    const unsigned char* data,
    size_t bytes,
    unsigned long long* result)
{
    unsigned long long local = 0;

    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < bytes;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        local += static_cast<unsigned long long>(data[i]);
    }

    atomicAdd(result, local);
}

static unsigned long long cpuChecksum(
    const unsigned char* data,
    size_t bytes)
{
    unsigned long long sum = 0;

    for (size_t i = 0; i < bytes; ++i)
        sum += static_cast<unsigned long long>(data[i]);

    return sum;
}

static size_t roundUp(size_t n, size_t alignment)
{
    return ((n + alignment - 1) / alignment) * alignment;
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr
            << "Usage: " << argv[0]
            << " <video_path> [max_frames]\n";
        return 1;
    }

    const std::string video_path = argv[1];
    const int max_frames =
        (argc >= 3) ? std::stoi(argv[2]) : 300;

    // Call before other CUDA Runtime operations.
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    // ------------------------------------------------------------
    // 1. Open video and read one normal frame to discover the real
    //    decoded frame layout.
    // ------------------------------------------------------------
    cv::VideoCapture cap(video_path);

    if (!cap.isOpened())
    {
        std::cerr << "Failed to open video: "
                  << video_path << '\n';
        return 1;
    }

    cv::Mat probe;

    if (!cap.read(probe) || probe.empty())
    {
        std::cerr << "Failed to read probe frame\n";
        return 1;
    }

    if (!probe.isContinuous())
    {
        std::cerr << "Probe frame is not continuous\n";
        return 1;
    }

    const int width  = probe.cols;
    const int height = probe.rows;
    const int type   = probe.type();

    if (type != CV_8UC3)
    {
        std::cerr
            << "This test expects CV_8UC3 BGR frames.\n"
            << "Actual type = " << type << '\n';
        return 1;
    }

    const size_t row_bytes =
        static_cast<size_t>(width) * probe.elemSize();

    const size_t image_bytes =
        row_bytes * static_cast<size_t>(height);

    std::cout
        << "Video frame layout\n"
        << "  width       : " << width << '\n'
        << "  height      : " << height << '\n'
        << "  channels    : " << probe.channels() << '\n'
        << "  image bytes : " << image_bytes << '\n';

    try {
        std::cout << "  backend     : " << cap.getBackendName() << "\n\n";
    } catch (...) {
        std::cout << "  backend     : <unknown>\n\n";
    }

    // ------------------------------------------------------------
    // 2. Allocate ordinary CPU memory ourselves.
    //    This is NOT cudaHostAlloc().
    // ------------------------------------------------------------
    const long page_size_long = sysconf(_SC_PAGESIZE);

    if (page_size_long <= 0)
    {
        std::cerr << "Failed to query system page size\n";
        return 1;
    }

    const size_t page_size =
        static_cast<size_t>(page_size_long);

    const size_t allocation_bytes =
        roundUp(image_bytes, page_size);

    void* host_memory = nullptr;

    const int memalign_result =
        posix_memalign(
            &host_memory,
            page_size,
            allocation_bytes
        );

    if (memalign_result != 0 || host_memory == nullptr)
    {
        std::cerr << "posix_memalign failed\n";
        return 1;
    }

    std::memset(host_memory, 0, allocation_bytes);

    auto* registered_host_ptr =
        static_cast<unsigned char*>(host_memory);

    // ------------------------------------------------------------
    // 3. Register that EXISTING CPU memory once.
    // ------------------------------------------------------------
    CUDA_CHECK(
        cudaHostRegister(
            registered_host_ptr,
            allocation_bytes,
            cudaHostRegisterMapped));

    unsigned char* mapped_device_ptr = nullptr;

    CUDA_CHECK(
        cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&mapped_device_ptr),
            registered_host_ptr,
            0));

    std::cout
        << "Registered host ptr : "
        << static_cast<void*>(registered_host_ptr) << '\n'
        << "Mapped device ptr   : "
        << static_cast<void*>(mapped_device_ptr) << "\n\n";

    // ------------------------------------------------------------
    // 4. Wrap the registered memory with cv::Mat.
    //    cv::Mat does NOT own this external buffer.
    // ------------------------------------------------------------
    cv::Mat registered_frame(
        height,
        width,
        type,
        registered_host_ptr,
        row_bytes);

    unsigned long long* device_checksum = nullptr;
    CUDA_CHECK(cudaMalloc(&device_checksum, sizeof(unsigned long long)));

    int frames_tested = 0;
    int pointer_changes = 0;
    int layout_changes = 0;
    int checksum_failures = 0;

    constexpr int kChecksumEvery = 30;

    // ------------------------------------------------------------
    // 5. Test loop
    // ------------------------------------------------------------
    for (int i = 0; i < max_frames; ++i)
    {
        // Restore the Mat header to our registered external buffer
        // before every read. If VideoCapture replaces the Mat header,
        // we can safely detect it without losing the original pointer.
        registered_frame = cv::Mat(
            height,
            width,
            type,
            registered_host_ptr,
            row_bytes);

        unsigned char* before =
            registered_frame.data;

        if (!cap.     (registered_frame))
        {
            std::cout << "EOF after "
                      << frames_tested
                      << " tested frames\n";
            break;
        }

        if (registered_frame.empty())
        {
            std::cerr << "Frame became empty\n";
            break;
        }

        ++frames_tested;

        unsigned char* after =
            registered_frame.data;

        const bool pointer_same =
            (after == before) &&
            (after == registered_host_ptr);

        const bool layout_same =
            registered_frame.cols == width &&
            registered_frame.rows == height &&
            registered_frame.type() == type &&
            registered_frame.step == row_bytes;

        if (!pointer_same)
        {
            ++pointer_changes;

            std::cerr
                << "[FAIL] frame " << frames_tested
                << ": VideoCapture replaced the buffer\n"
                << "       expected ptr = "
                << static_cast<void*>(registered_host_ptr) << '\n'
                << "       before ptr   = "
                << static_cast<void*>(before) << '\n'
                << "       after ptr    = "
                << static_cast<void*>(after) << '\n';

            // mapped_device_ptr still refers to registered_host_ptr,
            // not to any replacement allocation returned by OpenCV.
            continue;
        }

        if (!layout_same)
        {
            ++layout_changes;

            std::cerr
                << "[FAIL] frame " << frames_tested
                << ": frame layout changed\n";
            continue;
        }

        // Verify occasionally that CPU and GPU see the same bytes.
        // This is for correctness testing, NOT benchmarking.
        if ((frames_tested % kChecksumEvery) == 0 ||
            frames_tested == 1)
        {
            const unsigned long long cpu_sum =
                cpuChecksum(
                    registered_host_ptr,
                    image_bytes);

            CUDA_CHECK(
                cudaMemset(
                    device_checksum,
                    0,
                    sizeof(unsigned long long)));

            constexpr int threads = 256;
            constexpr int blocks  = 120;

            checksumKernel<<<blocks, threads>>>(
                mapped_device_ptr,
                image_bytes,
                device_checksum);

            CUDA_CHECK(cudaGetLastError());

            unsigned long long gpu_sum = 0;

            CUDA_CHECK(
                cudaMemcpy(
                    &gpu_sum,
                    device_checksum,
                    sizeof(unsigned long long),
                    cudaMemcpyDeviceToHost));

            if (cpu_sum != gpu_sum)
            {
                ++checksum_failures;

                std::cerr
                    << "[FAIL] frame " << frames_tested
                    << ": CPU/GPU checksum mismatch\n"
                    << "       CPU = " << cpu_sum << '\n'
                    << "       GPU = " << gpu_sum << '\n';
            }
            else
            {
                std::cout
                    << "[OK] frame " << frames_tested
                    << " pointer stable, checksum = "
                    << cpu_sum << '\n';
            }
        }
    }

    // ------------------------------------------------------------
    // 6. Cleanup
    //    DO NOT cudaFree(mapped_device_ptr).
    // ------------------------------------------------------------
    CUDA_CHECK(cudaFree(device_checksum));

    CUDA_CHECK(
        cudaHostUnregister(
            registered_host_ptr));

    std::free(host_memory);

    std::cout
        << "\n========== RESULT ==========\n"
        << "Frames tested     : " << frames_tested << '\n'
        << "Pointer changes   : " << pointer_changes << '\n'
        << "Layout changes    : " << layout_changes << '\n'
        << "Checksum failures : " << checksum_failures << '\n';

    if (frames_tested > 0 &&
        pointer_changes == 0 &&
        layout_changes == 0 &&
        checksum_failures == 0)
    {
        std::cout
            << "\nPASS:\n"
            << "For this video/backend, VideoCapture reused the "
               "preallocated cudaHostRegister() buffer in this test.\n";

        return 0;
    }

    std::cout
        << "\nFAIL / NOT SAFE TO ASSUME:\n"
        << "This VideoCapture path did not reliably preserve the "
           "registered buffer.\n";

    return 2;
}
