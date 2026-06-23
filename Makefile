obj-m += gdr_nv_bridge.o
KDIR := /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

# Target Paths for CUDA Installation and System Verbs Layouts
CUDA_PATH ?= /usr/local/cuda
CXX       := g++
CXXFLAGS  := -O3 -std=c++17 -Wall -Wextra -pthread
INCLUDES  := -I$(CUDA_PATH)/include
LIBS      := -L$(CUDA_PATH)/lib64 -lcudart -libverbs -lEGL

all: modules application

modules:
	make -C $(KDIR) M=$(PWD) modules

application: gdr_pipeline_engine.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) gdr_pipeline_engine.cpp $(LIBS) -o gdr_pipeline_engine

clean:
	make -C $(KDIR) M=$(PWD) clean
	rm -f gdr_pipeline_engine
