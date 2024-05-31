DPU_DIR := dpu
HOST_DIR := host
BUILDDIR ?= bin
NR_TASKLETS ?= 16
NR_DPUS ?= 1

define conf_filename
	${BUILDDIR}/.NR_DPUS_$(1)_NR_TASKLETS_$(2).conf
endef
CONF := $(call conf_filename,${NR_DPUS},${NR_TASKLETS})

COMMON_INCLUDES := support
HOST_TARGET := ${BUILDDIR}/lex_host
DPU_TARGET := ${BUILDDIR}/lex_dpu

# HOST_SOURCES := $(wildcard ${HOST_DIR}/app.cpp)
HOST_SOURCES := $(wildcard ${HOST_DIR}/app-hot-change.cpp)
# HOST_SOURCES := $(wildcard ${HOST_DIR}/app-quick-hot-change.cpp)
DPU_SOURCES := $(wildcard ${DPU_DIR}/task.c)

.PHONY: all clean test

__dirs := $(shell mkdir -p ${BUILDDIR})

COMMON_FLAGS := -Wall -Wextra  -g -I${COMMON_INCLUDES}
HOST_FLAGS := ${COMMON_FLAGS} -std=c++17 -fopenmp -ltbb -ljemalloc -O3 -march=native `dpu-pkg-config --cflags --libs dpu` -DNR_TASKLETS=${NR_TASKLETS} -DNR_DPUS=${NR_DPUS}
DPU_FLAGS := ${COMMON_FLAGS} -O3 -DNR_TASKLETS=${NR_TASKLETS}

all: ${HOST_TARGET} ${DPU_TARGET}

${CONF}:
	$(RM) $(call conf_filename,*,*)
	touch ${CONF}

${HOST_TARGET}: ${HOST_SOURCES} ${COMMON_INCLUDES} ${CONF}
	g++ -o $@ ${HOST_SOURCES} ${HOST_FLAGS}

${DPU_TARGET}: ${DPU_SOURCES} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -o $@ ${DPU_SOURCES}

clean:
	$(RM) -r $(BUILDDIR)
