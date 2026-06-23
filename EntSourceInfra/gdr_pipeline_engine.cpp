#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sched.h>
#include <numa.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

constexpr size_t CANVAS_WIDTH       = 7680;
constexpr size_t CANVAS_HEIGHT      = 4320;
constexpr size_t PIXEL_STRIDE       = 2; // 10-bit raw canvas content unpacked
constexpr size_t TOTAL_PIXELS       = CANVAS_WIDTH * CANVAS_HEIGHT;
constexpr size_t FRAME_BUFFER_SIZE  = TOTAL_PIXELS * PIXEL_STRIDE;
constexpr int    RING_DEPTH         = 256; 

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "Fatal CUDA Crash: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

struct PipelineBuffer {
    uint8_t* d_vram_ptr;
    struct ibv_mr* rdma_mr;
    cudaEvent_t    gpu_fence;
    uint32_t       drm_fb_id;
    uint32_t       drm_handle;
};

__global__ void FastVideoCanvasHDR(uint16_t* frameData, size_t numPixels) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels) {
        uint32_t pixel = frameData[idx];
        // Fast fixed-point mapping approximating standard gamma expansion curves
        uint32_t accelerated = (pixel * pixel) >> 10; 
        frameData[idx] = static_cast<uint16_t>(accelerated > 1023 ? 1023 : accelerated);
    }
}

class ProductionVideoWallEngine {
private:
    int m_drm_fd;
    uint32_t m_drm_crtc_id;
    struct ibv_context* m_ib_ctx;
    struct ibv_pd* m_ib_pd;
    struct ibv_cq* m_ib_cq;
    struct ibv_qp* m_ib_qp;

    std::vector<PipelineBuffer> m_rx_ring;
    std::vector<PipelineBuffer> m_process_ring;
    std::vector<PipelineBuffer> m_display_ring;

    std::atomic<bool> m_running;
    std::thread       m_network_thread;
    std::thread       m_display_thread;

    void SetThreadAffinity(int core_id, int numa_node) {
        numa_run_on_node(numa_node);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }

    void SetupLinuxDRMKMS() {
        m_drm_fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
        if (m_drm_fd < 0) throw std::runtime_error("DRM/KMS Device access error.");

        drmModeRes *resources = drmModeGetResources(m_drm_fd);
        if (!resources) throw std::runtime_error("Failed to query DRM/KMS display resources.");
        
        m_drm_crtc_id = resources->crtcs[0];
        drmModeFreeResources(resources);
    }

    void InitializeNativeRDMA() {
        int dev_count = 0;
        struct ibv_device** dev_list = ibv_get_device_list(&dev_count);
        if (!dev_list || dev_count == 0) throw std::runtime_error("No 400GbE RDMA adapters discovered.");
        
        m_ib_ctx = ibv_open_device(dev_list[0]);
        ibv_free_device_list(dev_list);
        if (!m_ib_ctx) throw std::runtime_error("Failed to acquire device context.");

        m_ib_pd = ibv_alloc_pd(m_ib_ctx);
        m_ib_cq = ibv_create_cq(m_ib_ctx, 8192, NULL, NULL, 0);

        struct ibv_qp_init_attr qp_init_attr = {
            .send_cq = m_ib_cq, .recv_cq = m_ib_cq,
            .cap = { .max_send_wr = 4096, .max_recv_wr = 4096, .max_send_sge = 1, .max_recv_sge = 1 },
            .qp_type = IBV_QPT_RC
        };
        m_ib_qp = ibv_create_qp(m_ib_pd, &qp_init_attr);

        struct ibv_qp_attr qp_attr = {
            .qp_state = IBV_QPS_INIT, .port_num = 1, .pkey_index = 0,
            .qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE
        };
        ibv_modify_qp(m_ib_qp, &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);

        struct ibv_qp_attr rtr_attr = { .qp_state = IBV_QPS_RTR, .path_mtu = IBV_MTU_4096, .dest_qp_num = 1, .ah_attr = {.port_num = 1} };
        ibv_modify_qp(m_ib_qp, &rtr_attr, IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_AV);
        
        struct ibv_qp_attr rts_attr = { .qp_state = IBV_QPS_RTS, .sq_psn = 0 };
        ibv_modify_qp(m_ib_qp, &rts_attr, IBV_QP_STATE | IBV_QP_SQ_PSN);
    }

    void ProvisionRingLayers(std::vector<PipelineBuffer>& ring) {
        int access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        ring.resize(RING_DEPTH);

        for (int i = 0; i < RING_DEPTH; ++i) {
            CHECK_CUDA(cudaMalloc(&(ring[i].d_vram_ptr), FRAME_BUFFER_SIZE));
            CHECK_CUDA(cudaEventCreateWithFlags(&(ring[i].gpu_fence), cudaEventDisableTiming));

            /* The nvidia_peermem driver intercepts this allocation and resolves 
               the physical VRAM boundaries seamlessly within ib_core */
            ring[i].rdma_mr = ibv_reg_mr(m_ib_pd, ring[i].d_vram_ptr, FRAME_BUFFER_SIZE, access_flags);
            if (!ring[i].rdma_mr) throw std::runtime_error("Direct native VRAM MR registration rejected.");

            struct drm_mode_create_dumb create_req = {
                .height = static_cast<uint32_t>(CANVAS_HEIGHT), .width = static_cast<uint32_t>(CANVAS_WIDTH),
                .bpp = 16, .flags = 0, .handle = 0, .pitch = 0, .size = 0
            };
            if (ioctl(m_drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &create_req) == 0) {
                ring[i].drm_handle = create_req.handle;
                drmModeAddFB(m_drm_fd, CANVAS_WIDTH, CANVAS_HEIGHT, 16, 16, create_req.pitch, ring[i].drm_handle, &ring[i].drm_fb_id);
            }

            struct ibv_sge sge = { .addr = reinterpret_cast<uint64_t>(ring[i].d_vram_ptr), .length = (uint32_t)FRAME_BUFFER_SIZE, .lkey = ring[i].rdma_mr->lkey };
            struct ibv_recv_wr wr = { .wr_id = (uint64_t)i, .next = NULL, .sg_list = &sge, .num_sge = 1 }, *bad_wr;
            ibv_post_recv(m_ib_qp, &wr, &bad_wr);
        }
    }

public:
    ProductionVideoWallEngine() : m_drm_fd(-1), m_running(false) {
        SetupLinuxDRMKMS();
        InitializeNativeRDMA();
    }

    void BuildEngineTopology() {
        ProvisionRingLayers(m_rx_ring);
        ProvisionRingLayers(m_process_ring);
        ProvisionRingLayers(m_display_ring);
    }

    void StartProcessingLoop() {
        m_running = true;

        // Ingestion & Capture Worker (NUMA Node 0, Core 2)
        m_network_thread = std::thread([this]() {
            this->SetThreadAffinity(2, 0);
            struct ibv_wc wc[32];
            cudaStream_t compute_stream;
            CHECK_CUDA(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
            
            size_t threads = 256;
            size_t blocks = (TOTAL_PIXELS + threads - 1) / threads;
            int current_slot = 0;

            while (m_running) {
                int comps = ibv_poll_cq(m_ib_cq, 32, wc);
                for (int i = 0; i < comps; ++i) {
                    if (wc[i].status == IBV_WC_SUCCESS) {
                        FastVideoCanvasHDR<<<blocks, threads, 0, compute_stream>>>(
                            reinterpret_cast<uint16_t*>(m_rx_ring[current_slot].d_vram_ptr), TOTAL_PIXELS
                        );
                        CHECK_CUDA(cudaEventRecord(m_rx_ring[current_slot].gpu_fence, compute_stream));
                        current_slot = (current_slot + 1) % RING_DEPTH;
                    }
                }
                std::this_thread::yield();
            }
            CHECK_CUDA(cudaStreamDestroy(compute_stream));
        });

        // DRM/KMS Page-Flip Presentation Worker (NUMA Node 0, Core 4)
        m_display_thread = std::thread([this]() {
            this->SetThreadAffinity(4, 0);
            int display_slot = 0;

            while (m_running) {
                // Device synchronization ensures the GPU compute pass finishes before the page-flip
                CHECK_CUDA(cudaEventSynchronize(m_rx_ring[display_slot].gpu_fence));

                if (m_drm_fd >= 0) {
                    drmModePageFlip(m_drm_fd, m_drm_crtc_id, m_display_ring[display_slot].drm_fb_id, DRM_MODE_PAGE_FLIP_EVENT, this);
                }
                display_slot = (display_slot + 1) % RING_DEPTH;
                std::this_thread::yield();
            }
        });
    }

    void ShutdownEngine() {
        m_running = false;
        if (m_network_thread.joinable()) m_network_thread.join();
        if (m_display_thread.joinable()) m_display_thread.join();

        auto FreeRing = [this](std::vector<PipelineBuffer>& ring) {
            for (int i = 0; i < RING_DEPTH; ++i) {
                ibv_dereg_mr(ring[i].rdma_mr);
                CHECK_CUDA(cudaEventDestroy(ring[i].gpu_fence));
                CHECK_CUDA(cudaFree(ring[i].d_vram_ptr));
                if (m_drm_fd >= 0) {
                    drmModeRmFB(m_drm_fd, ring[i].drm_fb_id);
                    struct drm_mode_destroy_dumb destroy_req = { .handle = ring[i].drm_handle };
                    ioctl(m_drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy_req);
                }
            }
        };

        FreeRing(m_rx_ring);
        FreeRing(m_process_ring);
        FreeRing(m_display_ring);

        ibv_destroy_qp(m_ib_qp);
        ibv_destroy_cq(m_ib_cq);
        ibv_dealloc_pd(m_ib_pd);
        ibv_close_device(m_ib_ctx);
        if (m_drm_fd >= 0) close(m_drm_fd);
    }
};

int main() {
    try {
        ProductionVideoWallEngine engine;
        engine.BuildEngineTopology();
        engine.StartProcessingLoop();

        std::cout << "[SYSTEM] Ingestion loop active. Press Enter to cleanly terminate pipeline." << std::endl;
        std::cin.get();

        engine.ShutdownEngine();
    } catch (const std::exception& e) {
        std::cerr << "Engine Panic: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
