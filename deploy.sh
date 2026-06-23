# 1. Build kernel driver modules and user applications
make

# 2. Inject core kernel dependencies (Must be executed with root authority)
sudo modprobe nvidia
sudo modprobe nvidia_peermem
sudo insmod gdr_nv_bridge.ko

# 3. Create structural access permissions for device interfaces
sudo chmod 666 /dev/gdr_nv_bridge

# 4. Bind thread pools directly onto isolated NUMA cores flanking the PCIe Network Card
numactl --cpunodebind=0 --membind=0 ./gdr_pipeline_engine
