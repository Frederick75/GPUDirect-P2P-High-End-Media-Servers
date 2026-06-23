#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/pci.h>

/* Path to NVIDIA proprietary driver headers (typically via nvidia-p2p.h) */
#include "nvidia-p2p.h"

#define MODULE_NAME        "gdr_nv_bridge"
#define DEVICE_NAME        "gdr_nv_bridge"
#define CLASS_NAME         "gdr"
#define GDR_IOCTL_MAGIC    'G'

struct gdr_pin_mapping {
    uint64_t user_va;
    uint64_t size;
    uint64_t page_table_ptr;
    uint64_t dma_bus_addr;
};

#define GDR_IOCTL_PIN_GPU_MEM   _IOWR(GDR_IOCTL_MAGIC, 1, struct gdr_pin_mapping)
#define GDR_IOCTL_UNPIN_GPU_MEM _IOW(GDR_IOCTL_MAGIC, 2, struct gdr_pin_mapping)

MODULE_LICENSE("GPL");
MODULE_AUTHOR("High Performance Architecture Specialist");
MODULE_DESCRIPTION("Enterprise GPUDirect RDMA Memory Mirroring Bridge");
MODULE_VERSION("1.1");

static int major_number;
static struct class* gdr_class = NULL;
static struct device* gdr_device = NULL;
static struct cdev gdr_cdev;

struct gdr_kernel_session {
    struct nvidia_p2p_page_table *page_table;
    bool is_pinned;
};

static void nvidia_p2p_free_callback(void *data) {
    struct gdr_kernel_session *session = (struct gdr_kernel_session *)data;
    if (session) {
        pr_warn("%s: Hardware memory backing revoked dynamically by NVIDIA driver.\n", MODULE_NAME);
        session->is_pinned = false;
        session->page_table = NULL;
    }
}

static long gdr_ioctl(struct file *file, unsigned int cmd, unsigned long arg) {
    struct gdr_pin_mapping map;
    struct gdr_kernel_session *session = file->private_data;
    int ret = 0;

    if (copy_from_user(&map, (void __user *)arg, sizeof(struct gdr_pin_mapping))) {
        return -EFAULT;
    }

    switch (cmd) {
        case GDR_IOCTL_PIN_GPU_MEM:
            if (session->is_pinned) {
                return -EEXIST;
            }

            /* Pin the physical GPU VRAM pages utilizing the NVIDIA P2P sub-architecture */
            ret = nvidia_p2p_get_pages(0, 0, map.user_va, map.size, 
                                       &session->page_table, 
                                       nvidia_p2p_free_callback, session);
            if (ret < 0) {
                pr_err("%s: Failed pinning GPU VRAM pages. Error Code: %d\n", MODULE_NAME, ret);
                return ret;
            }

            if (!session->page_table || session->page_table->entries[0]->physical_address == 0) {
                nvidia_p2p_free_page_table(session->page_table);
                return -EINVAL;
            }

            session->is_pinned = true;
            map.page_table_ptr = (uint64_t)session->page_table;
            map.dma_bus_addr = session->page_table->entries[0]->physical_address;

            if (copy_to_user((void __user *)arg, &map, sizeof(struct gdr_pin_mapping))) {
                nvidia_p2p_free_page_table(session->page_table);
                session->is_pinned = false;
                return -EFAULT;
            }
            break;

        case GDR_IOCTL_UNPIN_GPU_MEM:
            if (!session->is_pinned || !session->page_table) {
                return -EINVAL;
            }
            nvidia_p2p_free_page_table(session->page_table);
            session->is_pinned = false;
            session->page_table = NULL;
            pr_info("%s: Unpinned memory mappings systematically.\n", MODULE_NAME);
            break;

        default:
            return -ENOTTY;
    }
    return 0;
}

static int gdr_open(struct inode *inode, struct file *file) {
    struct gdr_kernel_session *session = kzalloc(sizeof(struct gdr_kernel_session), GFP_KERNEL);
    if (!session) return -ENOMEM;
    file->private_data = session;
    return 0;
}

static int gdr_release(struct inode *inode, struct file *file) {
    struct gdr_kernel_session *session = file->private_data;
    if (session) {
        if (session->is_pinned && session->page_table) {
            nvidia_p2p_free_page_table(session->page_table);
        }
        kfree(session);
    }
    return 0;
}

static const struct file_operations gdr_fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = gdr_ioctl,
    .open = gdr_open,
    .release = gdr_release,
};

static int __init gdr_bridge_init(void) {
    dev_t dev;
    if (alloc_chrdev_region(&dev, 0, 1, DEVICE_NAME) < 0) return -1;
    major_number = MAJOR(dev);

    cdev_init(&gdr_cdev, &gdr_fops);
    gdr_cdev.owner = THIS_MODULE;
    if (cdev_add(&gdr_cdev, dev, 1) < 0) goto unregister_device;

    gdr_class = class_create(CLASS_NAME);
    if (IS_ERR(gdr_class)) goto cdev_del;

    gdr_device = device_create(gdr_class, NULL, dev, NULL, DEVICE_NAME);
    if (IS_ERR(gdr_device)) goto destroy_class;

    pr_info("%s: Production kernel bridge listening securely on major %d.\n", MODULE_NAME, major_number);
    return 0;

destroy_class:
    class_destroy(gdr_class);
cdev_del:
    cdev_del(&gdr_cdev);
unregister_device:
    unregister_chrdev_region(dev, 1);
    return -1;
}

static void __exit gdr_bridge_exit(void) {
    dev_t dev = MKDEV(major_number, 0);
    device_destroy(gdr_class, dev);
    class_destroy(gdr_class);
    cdev_del(&gdr_cdev);
    unregister_chrdev_region(dev, 1);
    pr_info("%s: Modular cleanup sequence finished.\n", MODULE_NAME);
}

module_init(gdr_bridge_init);
module_exit(gdr_bridge_exit);
