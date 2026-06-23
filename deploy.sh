#!/usr/bin/env bash
set -euo pipefail

echo "[BUILD] Compiling kernel module and hardened engine infrastructure..."
make -j$(nproc)

echo "[PROVISIONING] Tweaking system kernels to handle high-performance 400GbE media streams..."
# Lock real-time processing threads into memory to eliminate swapping latencies
ulimit -l unlimited

# Enforce explicit IOMMU pass-through optimizations for ATS translation layers
# Ensure 'intel_iommu=on iommu=pt' is active within your system /etc/default/grub configurations

# Isolate CPU affinity paths away from OS scheduling disruptions
# Assign dedicated processing workers directly to matching NUMA paths
sudo sysctl -w vm.zone_reclaim_mode=0

echo "[DRIVERS] Injecting infrastructure modules..."
sudo modprobe nvidia
sudo modprobe nvidia_peermem
sudo insmod gdr_nv_bridge.ko || echo "Module already active."
sudo chmod 666 /dev/gdr_nv_bridge

echo "[RUNNING] Executing Engine under strict NUMA node constraint boundaries..."
# Execute the processing loop aligned with node 0 to optimize PCIe traffic paths
exec numactl --cpunodebind=0 --membind=0 ./gdr_pipeline_engine
