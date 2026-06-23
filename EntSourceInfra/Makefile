obj-m     += gdr_peer_driver.o
KDIR      := /lib/modules/$(shell uname -r)/build
PWD       := $(shell pwd)

# Target Paths for Dependencies
CUDA_PATH ?= /usr/local/cuda
CXX       := g++
CXXFLAGS  := -O3 -std=c++17 -Wall -Wextra -pthread -D_GNU_SOURCE
INCLUDES  := -I$(CUDA_PATH)/include -I/usr/include/libdrm -I$(KDIR)/drivers/infiniband/core
LIBS      := -L$(CUDA_PATH)/lib64 -lcudart -libverbs -lnuma -ldrm

all: modules application

modules:
	make -C $(KDIR) M=$(PWD) modules

application: gdr_pipeline_engine.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) gdr_pipeline_engine.cpp $(LIBS) -o gdr_pipeline_engine

clean:
	make -C $(KDIR) M=$(PWD) clean
	rm -f gdr_pipeline_engine
