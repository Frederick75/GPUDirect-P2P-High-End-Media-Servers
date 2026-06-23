#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] Building kernel modules and user-space engines..."
make -j$(nproc)

echo "[2/4] Loading core drivers and establishing peer memory hooks..."
sudo modprobe ib_core
sudo modprobe nvidia
# Crucial: This bridges InfiniBand verbs and CUDA memory registrations
sudo modprobe nvidia_peermem 
sudo insmod gdr_peer_driver.ko || echo "Module already loaded."

echo "[3/4] Tuning system memory limits for line-rate streaming..."
# Eliminate page swap penalties for processing worker pools
ulimit -l unlimited
sudo sysctl -w vm.zone_reclaim_mode=0

echo "[4/4] Starting processing loop on isolated PCIe root complex..."
# Run the pipeline engine aligned with NUMA node 0 to optimize PCIe bus traffic
exec numactl --cpunodebind=0 --membind=0 ./gdr_pipeline_engine
