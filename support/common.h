#ifndef _COMMON_H_
#define _COMMON_H_

#ifdef TL
#define TASKLETS_INITIALIZER TASKLETS(TL, main, 2048, 2)
#define NB_OF_TASKLETS_PER_DPU TL
#else
#define TASKLETS_INITIALIZER TASKLETS(16, main, 2048, 2)
#define NB_OF_TASKLETS_PER_DPU 16
#endif

// Data type
#define DTYPE uint64_t

// Vector size
// #define INPUT_SIZE 2048 // 2048576

// max PGM level number
#define MAX_LEVEL 5

// max model number in DPU
#define MAX_MODEL_SIZE 2560

// max buffer size
#define MAX_BUFFER_SIZE 1 << 15

// buffer struct
typedef struct {
	int n_tasks;
	int op_type;
	DTYPE sbuffer[MAX_BUFFER_SIZE];
}send_buffer_t;

typedef struct{
	int n_tasks;
	int padding;
	uint32_t rbuffer[MAX_BUFFER_SIZE];
}recv_buffer_t;

// learned parameter
#define EpsilonRecursive_dpu 4
#define DataEpsilon 64 // 512

#define INVAILD_POS 1000000000U

typedef struct {
	uint64_t input_size;
	enum kernels {
		kernel1 = 0,
		nr_kernels = 1,
	} kernel;
	int max_levels; // level_offset[max_level] is the end of top level offset
	int level_offset[MAX_LEVEL];
	uint32_t start_pos;
	size_t setEpsilon;
} dpu_arguments_t;

// Structures used by dpu to store model
typedef struct {
    DTYPE key;             ///< The first key that the segment indexes.
    float slope;    ///< The slope of the segment.
    int32_t intercept; ///< The intercept of the segment.
} pgm_model_t;

#ifndef ENERGY
#define ENERGY 0
#endif
#define PRINT 0

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#endif
