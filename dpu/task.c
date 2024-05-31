/*
* learned index with multiple tasklets
*
*/
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <perfcounter.h>
#include "common.h"
#include <string.h>
// #include <mutex.h>
// #include <vmutex.h>

#define DEBUG 0
#define PRINT 0
#define BLOCK_SIZE 8
#define USE_BLOCK 0
#define FETCH_QUERY 32
#define MOV_NUM 3  // 2^MOV_NUM = INTERLEAVE

#define WORD_MASK 0xfffffff8
__host dpu_arguments_t DPU_INPUT_ARGUMENTS;
__mram_noinit send_buffer_t dpu_sbuffer;
__mram_noinit recv_buffer_t dpu_recv_buffer;
__host pgm_model_t model_array[MAX_MODEL_SIZE];

// learned index struct and addr, init at start
uint32_t start_mram_input_addr; // key

BARRIER_INIT(my_barrier, NR_TASKLETS);

extern int main_kernel1(void);

int(*kernels[nr_kernels])(void) = {main_kernel1};

int main(void){
  // Kernel
  return kernels[DPU_INPUT_ARGUMENTS.kernel]();
}

// main_kernel1
int main_kernel1() {
  unsigned int tasklet_id = me();
  #if PRINT
  printf("tasklet_id = %u\n", tasklet_id);
  #endif
  // if(tasklet_id == 0){
  //   mem_reset(); // Reset the heap
  // }
  // Barrier
  barrier_wait(&my_barrier);
  

  if(dpu_sbuffer.op_type == 0){
    // INIT
    // init index struct at create
    if(tasklet_id == 0){
      mem_reset(); // Reset the heap
      // max_level = DPU_INPUT_ARGUMENTS.max_levels;
      // model_size = DPU_INPUT_ARGUMENTS.level_offset[max_level];
      start_mram_input_addr = (uint32_t) DPU_MRAM_HEAP_POINTER;
    }
    // Barrier
    barrier_wait(&my_barrier);
  } 
  else if(dpu_sbuffer.op_type == 1){
    // LOOKUP
    DTYPE searching_for;
    DTYPE searching_block[FETCH_QUERY];
    uint32_t current_query_in_block = 0;
    uint32_t num_task = dpu_sbuffer.n_tasks;
    uint32_t current_mram_query = tasklet_id * (num_task / NR_TASKLETS);
    uint32_t total_ops = num_task / NR_TASKLETS;
    if(current_mram_query % 2 != 0){
      current_mram_query--;
      total_ops++;
    }
    // if(total_ops %2 != 0){
    //   total_ops++;
    // }
    if(tasklet_id == NR_TASKLETS - 1){
      total_ops = num_task - current_mram_query;
    }

    if(total_ops % 2 != 0){
      total_ops++;
    }
    dpu_recv_buffer.n_tasks = num_task;
    // uint32_t task_number = num_task / NR_TASKLETS;
    // if(tasklet_id == NR_TASKLETS - 1)
    //   task_number = num_task - current_mram_query - 1;

    // init cache for learned search
    DTYPE fetch_value;
    DTYPE check_block[2];
    uint32_t temp_pos[2];
    uint32_t current_temp_op_pos = 0;
    #if USE_BLOCK
    DTYPE fetch_block[BLOCK_SIZE];
    #endif

    uint32_t l,r,mid;
    #if USE_BLOCK
    uint32_t bl,br,bmid;
    #endif
    // predict pos
    int predicted_pos;

    // int search_num = 0;

    for(uint32_t targets = 0; targets < total_ops; targets++)
    {
      if(current_query_in_block == 0){
        mram_read((__mram_ptr void const *)&(dpu_sbuffer.sbuffer[current_mram_query]), &searching_block, 8 * FETCH_QUERY);
      }
      searching_for = searching_block[current_query_in_block];

      // search single model level to get model
      l = DPU_INPUT_ARGUMENTS.level_offset[0];
      r = DPU_INPUT_ARGUMENTS.level_offset[1] - 1;
      mid = l;
      while(l < r){
        mid = l + (r - l) / 2;
        if(model_array[mid].key <= searching_for){
          l = mid + 1;
        }else{
          r = mid;
        }
      }

      // prediction, current_model is model_array[l-1]
      predicted_pos = (int)(model_array[l-1].slope * (searching_for - model_array[l-1].key)) + model_array[l-1].intercept;
      
      if((l < (DPU_INPUT_ARGUMENTS.level_offset[1] - 1)) && (model_array[l].intercept < predicted_pos)){
        predicted_pos = model_array[l].intercept;
      }

      if(predicted_pos > (int)(DPU_INPUT_ARGUMENTS.setEpsilon)){ //
        l = predicted_pos - DPU_INPUT_ARGUMENTS.setEpsilon; //DPU_INPUT_ARGUMENTS.setEpsilon;
      }else{
        l = 0;
      }
      r = predicted_pos + DPU_INPUT_ARGUMENTS.setEpsilon + 2; //DPU_INPUT_ARGUMENTS.setEpsilon
      if(r >= DPU_INPUT_ARGUMENTS.input_size){
        r = DPU_INPUT_ARGUMENTS.input_size;
      }

      // l = 0;
      // r = DPU_INPUT_ARGUMENTS.input_size;
      // 二分查找预测区域
      while(l < r){
        // search_num++;
        #if USE_BLOCK
        if((r - l) <= BLOCK_SIZE){
          mram_read((__mram_ptr void const *) (DPU_MRAM_HEAP_POINTER + 8 * l), &fetch_block, sizeof(DTYPE) * BLOCK_SIZE);
          break;
        }
        #endif
        mid = l + (r - l) / 2;
        mram_read((__mram_ptr void const *) (DPU_MRAM_HEAP_POINTER + 8 * mid), &fetch_value, sizeof(DTYPE));
        if(fetch_value <= searching_for){
          l = mid + 1;
        }else{
          r = mid;
        }
      }

      // search block
      #if USE_BLOCK
      bl = 0;
      br = BLOCK_SIZE;
      while(bl < br){
        bmid = bl + (br - bl) / 2;
        if(fetch_block[bmid] <= searching_for){
          bl = bmid + 1;
        }else{
          br = bmid;
        }
      }
      l = l + bl;
      #endif

      if(l > (DPU_INPUT_ARGUMENTS.input_size - 1)){
        temp_pos[current_temp_op_pos] = INVAILD_POS;
      }else{
      mram_read((__mram_ptr void const *) (DPU_MRAM_HEAP_POINTER + 8 * (l - 1)), &check_block, sizeof(DTYPE) * 2);
      if(check_block[0] <= searching_for && check_block[1] > searching_for){
        temp_pos[current_temp_op_pos] = (uint32_t)(l - 1 + DPU_INPUT_ARGUMENTS.start_pos) << MOV_NUM;// << 2; 
      }else{
        if(l == (DPU_INPUT_ARGUMENTS.input_size - 1)){
          // 边界处理
          // temp_pos[current_temp_op_pos] = INVAILD_POS;
          temp_pos[current_temp_op_pos] = (uint32_t)(l - 1 + DPU_INPUT_ARGUMENTS.start_pos) << MOV_NUM;
        }else{
          temp_pos[current_temp_op_pos] = INVAILD_POS;
        }
      }
      }

      // mram_read((__mram_ptr void const *) (start_mram_input_addr + 8 * (l - 1)), &check_block, sizeof(DTYPE) * 2);
      // if(check_block[0] <= searching_for && check_block[1] > searching_for){
      //   dpu_recv_buffer.rbuffer[current_mram_query] = (uint64_t)(l - 1 + DPU_INPUT_ARGUMENTS.start_pos) << 2;// << 2; 
      // }else{
      //   if(l == (DPU_INPUT_ARGUMENTS.input_size - 1)){
      //     // 边界处理
      //     dpu_recv_buffer.rbuffer[current_mram_query] = (uint64_t)(l - 1 + DPU_INPUT_ARGUMENTS.start_pos) << 2;
      //   }else{
      //     dpu_recv_buffer.rbuffer[current_mram_query] = INVAILD_POS;
      //   }
      // }
      
      // #if DEBUG
      //   search_num++;
      // #endif
        // write results to reveive buffer
        // printf("pos %u , key %lu \n", l - 1, searching_for);

      current_mram_query ++;

      current_temp_op_pos++;
      // if((targets == (total_ops - 1)) && current_temp_op_pos < 2){
      //   l = temp_pos[0];
      //   mram_read((__mram_ptr void const *)(&(dpu_recv_buffer.rbuffer[current_mram_query - 2])), &temp_pos, 8);
      //   temp_pos[1] = l;
      //   mram_write((void*)temp_pos, &(dpu_recv_buffer.rbuffer[current_mram_query - 2]), 8);
      // }
      if(current_temp_op_pos == 2){
        mram_write((void*)temp_pos, &(dpu_recv_buffer.rbuffer[current_mram_query - 2]), 8);
        current_temp_op_pos = 0;
      }

      current_query_in_block++;
      if(current_query_in_block == FETCH_QUERY)
        current_query_in_block = 0;

    }
    #if DEBUG
      dpu_recv_buffer.n_tasks = search_num;
    #endif
    // printf("total_ops %u current_temp_op_pos % u \n", total_ops, current_temp_op_pos);
  } 

  // Barrier
  barrier_wait(&my_barrier);
  return 0;
}
