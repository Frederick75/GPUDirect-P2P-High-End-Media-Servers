#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/pci.h>
#include <rdma/ib_verbs.h>
#include <rdma/peer_mem.h>
#include "nvidia-p2p.h"

#define MODULE_NAME "gdr_peer_driver"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("High Performance Architecture Specialist");
MODULE_DESCRIPTION("Enterprise Peer-Memory Client Registration Engine");
MODULE_VERSION("5.0");

struct gdr_peer_context {
    struct nvidia_p2p_page_table *page_table;
    void *client_context;
};

static void *gdr_peer_acquire(unsigned long addr, size_t size, void *peer_private_data) {
    struct gdr_peer_context *ctx;
    int ret;

    ctx = kzalloc(sizeof(*ctx), GFP_KERNEL);
    if (!ctx) return ERR_PTR(-ENOMEM);

    /* Pin non-pageable GPU VRAM directly via the proprietary NVIDIA sub-architecture */
    ret = nvidia_p2p_get_pages(0, 0, addr, size, &ctx->page_table, NULL, NULL);
    if (ret < 0) {
        kfree(ctx);
        return ERR_PTR(ret);
    }

    return ctx;
}

static size_t gdr_peer_get_pages(void *context, u64 offset, size_t size, 
                                 struct page **pages, size_t dma_mapped) {
    struct gdr_peer_context *ctx = context;
    size_t page_count = size >> PAGE_SHIFT;
    size_t i;

    if (!ctx->page_table) return 0;

    /* Feed direct GPU physical-backed tracking entries natively to the IB core layout */
    for (i = 0; i < page_count; i++) {
        pages[i] = (struct page *)(ctx->page_table->entries[i]->physical_address);
    }

    return page_count;
}

static void gdr_peer_release(void *context) {
    struct gdr_peer_context *ctx = context;
    if (ctx) {
        if (ctx->page_table) {
            nvidia_p2p_free_page_table(ctx->page_table);
        }
        kfree(ctx);
    }
}

static struct peer_memory_client gdr_peer_client = {
    .name          = MODULE_NAME,
    .acquire       = gdr_peer_acquire,
    .get_pages     = gdr_peer_get_pages,
    .release       = gdr_peer_release,
};

static int __init gdr_peer_init(void) {
    int ret = ib_register_peer_memory_client(&gdr_peer_client, &gdr_peer_client.client_context);
    if (ret) {
        pr_err("%s: Failed to register native peer memory client: %d\n", MODULE_NAME, ret);
        return ret;
    }
    pr_info("%s: Native Peer-Memory client registered successfully.\n", MODULE_NAME);
    return 0;
}

static void __exit gdr_peer_exit(void) {
    ib_unregister_peer_memory_client(gdr_peer_client.client_context);
    pr_info("%s: Native Peer-Memory client deregistered cleanly.\n", MODULE_NAME);
}

module_init(gdr_peer_init);
module_exit(gdr_peer_exit);
