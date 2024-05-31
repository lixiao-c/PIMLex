/**
* app-new.c
*
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
extern "C" {
#include <dpu.h>
#include <dpu_log.h>
}
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <time.h>

#if ENERGY
extern "C" {
#include <dpu_probe.h>
}
#endif

#include "timer.h"
#include "model.h"
#include <omp.h>
#include <thread>
#include <utility>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <tbb/parallel_sort.h>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <string>
#include <chrono>
#include <jemalloc/jemalloc.h>
#include "flags.h" 
#include "utils.h"
#include "pgm_index.hpp"
#include "tscns.h"
#include "dram_index.h"
#include <atomic>

std::atomic<int> unequal(0);

// Define the DPU Binary path as DPU_BINARY here
#define DPU_BINARY "./bin/lex_dpu"

std::string output_path;

#define DEBUG 0
#define BATCH_NUM 10000000
#define THREAD_NUM 32 // 32线程性能更差？
#define PRINT 0
#define INTERLEAVE 4
#define NR_PARTITION 128

DRAM_index dindex;

struct dpu_set_t dpu_set, dpu;
// DTYPE* upper_dram_level; // 存储在DRAM上的上层索引，用于查找DPU。需要建立
send_buffer_t *host_send_buffer[THREAD_NUM]; // read only, 存储query 
recv_buffer_t *host_recv_buffer[THREAD_NUM]; // read only, 存储返回值, 所有线程共享一个recv buffer
std::shared_mutex lock;
dpu_arguments_t input_arguments[NR_DPUS];

DTYPE *ret_merge_key[NR_DPUS]; //read only, 存储merge之后的key
#define MAX_MERGE_KEY_SIZE 500000
int mem_addr_flag_host; // 0: new is DPU_MRAM_HEAP_POINTER + MRAM_MID_SIZE, 1 is DPU_MRAM_HEAP_POINTER

class partition_access{
public:
	class access_ratio{
	public:
		int partition_id;
		double partition_ratio;
		uint64_t num_dpu_per_partition;
		double per_dpu_ratio;
		
		access_ratio(){
		}
		access_ratio& operator =(const access_ratio& aa)//赋值运算符 
		{
			this->partition_id = aa.partition_id;
			this->partition_ratio = aa.partition_ratio;
			this->num_dpu_per_partition = aa.num_dpu_per_partition;
			this->per_dpu_ratio = aa.per_dpu_ratio;
			return *this;
		}
	};
	uint64_t access_num[THREAD_NUM][NR_PARTITION];
	access_ratio record_ratio[THREAD_NUM][NR_PARTITION];

	partition_access() {
		for(int i = 0; i < THREAD_NUM; i++){
			for(int j = 0; j < NR_PARTITION; j++){
				access_num[i][j] = 0;
				record_ratio[i][j].partition_id = j;
				record_ratio[i][j].partition_ratio = 0.0;
				record_ratio[i][j].num_dpu_per_partition = 0;
			}
		}
	};

	void reset(){
		for(int i = 0; i < THREAD_NUM; i++){
			for(int j = 0; j < NR_PARTITION; j++){
				access_num[i][j] = 0;
				record_ratio[i][j].partition_id = j;
				record_ratio[i][j].partition_ratio = 0.0;
				record_ratio[i][j].num_dpu_per_partition = 0;
			}
		}
	}
	void reset_by_thread(int tid){
		for(int j = 0; j < NR_PARTITION; j++){
			access_num[tid][j] = 0;
			record_ratio[tid][j].partition_id  = j;
			record_ratio[tid][j].partition_ratio = 0.0;
			record_ratio[tid][j].num_dpu_per_partition = 0;
		}
	}

	static bool cmp_p_ratio(access_ratio a, access_ratio b){// 从小到大排序
		return a.partition_ratio < b.partition_ratio;
	}
	static bool cmp_dpu_ratio(access_ratio a, access_ratio b){ // 从大到小排序
		return a.per_dpu_ratio > b.per_dpu_ratio;
	}
	static bool cmp_pid(access_ratio a, access_ratio b){ // 从小到大排序
		return a.partition_id < b.partition_id;
	}

	void calculate_partition_skew(int partition_ratio_to_dpu, int total_access_num, int tid){
		int avg_access_num = total_access_num / NR_PARTITION;
		double max_access_ratio = 0.0;
		double total_ratio = 0.0;
		uint64_t total_alloc_dpu_num = 0;
		for(int i = 0; i < NR_PARTITION; i++){
			record_ratio[tid][i].partition_ratio = (double)access_num[tid][i] / avg_access_num;
			// 分配的副本数, 最少分配1个
			record_ratio[tid][i].num_dpu_per_partition = std::max(static_cast<uint64_t>(record_ratio[tid][i].partition_ratio * partition_ratio_to_dpu), (uint64_t)1);
			// 每个副本承担的请求比例
			record_ratio[tid][i].per_dpu_ratio = record_ratio[tid][i].partition_ratio / record_ratio[tid][i].num_dpu_per_partition;
			// 分配的总副本数量
			total_alloc_dpu_num += record_ratio[tid][i].num_dpu_per_partition;
			// debug
			total_ratio += record_ratio[tid][i].partition_ratio;
		}

		// 按照每个dpu需要处理的请求ratio从大到小排序
		std::sort(record_ratio[tid], record_ratio[tid] + NR_PARTITION, cmp_dpu_ratio);
		if(total_alloc_dpu_num > NR_DPUS){
			/***分配数量多于DPU数***/
			uint64_t overflow_dpu_num = total_alloc_dpu_num - NR_DPUS;
			// 1) 从负载最轻的partition剥夺DPU
			int min_available_off = NR_PARTITION - 1;
			for(uint64_t i = 0; i < overflow_dpu_num; i++){
				while(record_ratio[tid][min_available_off].num_dpu_per_partition == 1){
					min_available_off--;
				}
				if(min_available_off < 0){
					std::cout << "min_available_off less than 0 " << std::endl;
					assert(0);
				}
				record_ratio[tid][min_available_off].num_dpu_per_partition--;
				// 新的dpu ratio
				record_ratio[tid][min_available_off].per_dpu_ratio = record_ratio[tid][min_available_off].partition_ratio 
					/ record_ratio[tid][min_available_off].num_dpu_per_partition;
				// dpu ratio增大，改变排序
				int k;
				for(k = 0; k < min_available_off; k++){
					// 找到第一个比自己小的数
					if(record_ratio[tid][k].per_dpu_ratio < record_ratio[tid][min_available_off].per_dpu_ratio){
						break;
					}
				}
				// 插入数据
				access_ratio temp = record_ratio[tid][min_available_off];
				for(int ll = min_available_off; ll > k; ll --){
					record_ratio[tid][ll] = record_ratio[tid][ll - 1];
				}
				record_ratio[tid][k] = temp;
				
				// 开始下一轮
			}
			// debug
			uint64_t test_total_dpu = 0;
			for(int j = 0; j < NR_PARTITION; j++){
				test_total_dpu += record_ratio[tid][j].num_dpu_per_partition;
			}
			std::cout << "test dpu total number (less adjust)" << test_total_dpu << std::endl;

		}else if(total_alloc_dpu_num < NR_DPUS){
			/***分配数量少于DPU***/
			// 2) 给负载最重的partition分配更多DPU
			uint64_t less_dpu_num =  NR_DPUS - total_alloc_dpu_num;
			int max_available_off = 0;
			for(uint64_t i = 0; i < less_dpu_num; i++){
				while(record_ratio[tid][max_available_off].num_dpu_per_partition == 1){
					max_available_off++;
				}
				if(max_available_off >= NR_PARTITION){
					std::cout << "max_available_off larger than NR_PARTITION " << std::endl;
					assert(0);
				}
				record_ratio[tid][max_available_off].num_dpu_per_partition++;
				// 新的dpu ratio
				record_ratio[tid][max_available_off].per_dpu_ratio = record_ratio[tid][max_available_off].partition_ratio 
					/ record_ratio[tid][max_available_off].num_dpu_per_partition;
				//dpu ratio变小, 调整顺序
				int k;
				for(k = max_available_off; k < NR_PARTITION; k++){
					// 第一个比自己小的数
					if(record_ratio[tid][k].per_dpu_ratio < record_ratio[tid][max_available_off].per_dpu_ratio){
						break;
					}	
				}
				if(k == NR_PARTITION)
					k--;
				// 插入数据
				access_ratio temp = record_ratio[tid][max_available_off];
				for(int ll = max_available_off; ll < k; ll ++){
					record_ratio[tid][ll] = record_ratio[tid][ll + 1];
				}
				record_ratio[tid][k] = temp;
				// 开始下一轮
			}
			// debug
			uint64_t test_total_dpu = 0;
			for(int j = 0; j < NR_PARTITION; j++){
				test_total_dpu += record_ratio[tid][j].num_dpu_per_partition;
			}
			std::cout << "test dpu total number (large adjust) " << test_total_dpu << std::endl;

		}
		// 按照partiton id排序
		std::sort(record_ratio[tid], record_ratio[tid] + NR_PARTITION, cmp_pid);

		//debug		
		// for(int i = 0; i < NR_PARTITION; i++){
		// 	std::cout << "part id " << record_ratio[tid][i].partition_id;
		// 	std::cout << " part ratio " << record_ratio[tid][i].partition_ratio;
		// 	std::cout << " dpu ratio " << record_ratio[tid][i].per_dpu_ratio;
		// 	std::cout << " dpu number " << record_ratio[tid][i].num_dpu_per_partition << std::endl;
		// }
		// std::cout << "total ratio " << total_ratio << std::endl;

	}
};
partition_access partition_access_level; // 记录partition上访问次数。需要建立

class dram_level{
public:
	DTYPE low_bound_key[NR_PARTITION];
	uint64_t start_dpu_id[NR_PARTITION];
	uint64_t nr_replicas_per_partition[NR_PARTITION];

	dram_level() {};
	void get_start_dpu_id(){
		uint64_t cumulative_down = 0;
		for(int i = 0; i < NR_PARTITION; i++){
			start_dpu_id[i] = cumulative_down;
			cumulative_down += nr_replicas_per_partition[i];
			// std::cout << "partition " << i << " start dpu " << start_dpu_id[i] << " replicas " << nr_replicas_per_partition[i] << std::endl;
		}
	}

	uint64_t inline search_upper_level(DTYPE key, int rd, int tid, bool sample = false){
		int l, r, m;
		l = 0; 
		r = NR_PARTITION;
		while(l < r){
			m = l + (r - l) / 2;
			if(low_bound_key[m] <= key)
				l = m + 1;
			else
				r = m;
		}
		if(sample){
			// record access count for partition
			partition_access_level.access_num[tid][l - 1]++;
		}
		// std::cout << "nr_replicas_per_partition[l - 1] " << nr_replicas_per_partition[l - 1] << " ramdom " << key % nr_replicas_per_partition[l - 1] << std::endl;
		return start_dpu_id[l - 1] + (rd % nr_replicas_per_partition[l - 1]); // 随机选取dpu
	}

	int inline search_upper_level_get_partition(DTYPE key){
		int l, r, m;
		l = 0; 
		r = NR_PARTITION;
		while(l < r){
			m = l + (r - l) / 2;
			if(low_bound_key[m] <= key)
				l = m + 1;
			else
				r = m;
		}
		return l - 1; // 返回partition
	}
};
dram_level upper_dram_level; // 存储在DRAM上的上层索引，用于查找DPU。需要建立

class replicas_info_t{
public:
	uint64_t cur_dpu;
	uint64_t cur_ret_partition;
	uint64_t nr_replicas_per_partition[NR_PARTITION];
	uint64_t nr_replicas_cumulative_partition[NR_PARTITION];
	replicas_info_t(){
		for(int i = 0; i < NR_PARTITION; i++){
			nr_replicas_per_partition[i] = 1;
		}
		cur_dpu = 0;
		cur_ret_partition = 0;
	}
	replicas_info_t(uint64_t set_replicas){
		for(int i = 0; i < NR_PARTITION; i++){
			nr_replicas_per_partition[i] = set_replicas;
		}
		cur_dpu = 0;
		cur_ret_partition = 0;
	}
	void get_cumulative_info(){
		uint64_t cumulative = 0;
		for(int i = 0; i < NR_PARTITION; i++){
			cumulative += nr_replicas_per_partition[i];
			nr_replicas_cumulative_partition[i] = cumulative;
		}
	}
	void reset(){
		cur_dpu = 0;
		cur_ret_partition = 0;
	}
	uint64_t next_partition(){
		uint64_t ret = 0;
		if(cur_dpu < nr_replicas_cumulative_partition[cur_ret_partition]){
			ret = cur_ret_partition;
		}else{
			cur_ret_partition++;
			while(cur_dpu >= nr_replicas_cumulative_partition[cur_ret_partition]){
				cur_ret_partition++;
			}
			ret = cur_ret_partition;
		}
		cur_dpu++;
		return ret;
	}
	void load_hot_replicas_info(partition_access* partition_access_info, int tid){
		for(int i = 0; i < NR_PARTITION; i++){
			nr_replicas_per_partition[i] = partition_access_info->record_ratio[tid][i].num_dpu_per_partition;
		}
		cur_dpu = 0;
		cur_ret_partition = 0;
	}
};


class dpu_cost_model{
	public:
	uint32_t wram_search_num;
	uint32_t mram_search_num;
	float cost;
	dpu_cost_model(){
	}
};

// Create input arrays
void create_query(DTYPE * input, DTYPE * querys, uint64_t  nr_elements, uint64_t nr_querys) {
	std::mt19937_64 gen(std::random_device{}());
	std::uniform_int_distribution<int> dis(0, nr_elements - 1);
	for (int i = 0; i < nr_querys; i++) {
		int pos = dis(gen);
		querys[i] = input[pos];
	}
}

void create_query_zipf(DTYPE * input, DTYPE * querys, uint64_t  nr_elements, uint64_t nr_querys) {
	ScrambledZipfianGenerator zipf_gen(nr_elements);
	for (int i = 0; i < nr_querys; i++) {
		int pos = zipf_gen.nextValue();
		querys[i] = input[pos];
	}
}

void create_insert_op(DTYPE * input, DTYPE * querys, uint64_t  nr_init_keys, uint64_t nr_querys, uint64_t nr_total_keys) {
	// 插入不重复的key
	if(nr_querys > (nr_total_keys - nr_init_keys))
		std::cout << "ERROR, insert keys larger than total keys" << std::endl;
	int pos = nr_init_keys;
	for (int i = 0; i < nr_querys; i++) {
		querys[i] = input[pos];
		pos++;
	}
}

void init_host(){
	for(int j = 0; j < THREAD_NUM; j++){
		host_send_buffer[j] = new send_buffer_t[NR_DPUS];
		for(int i = 0; i < NR_DPUS; i++){
			host_send_buffer[j][i].n_tasks = 0;
		}
	}
	for(int j = 0; j < THREAD_NUM; j++){
		host_recv_buffer[j] = new recv_buffer_t[NR_DPUS];
		for(int i = 0; i < NR_DPUS; i++){
			host_recv_buffer[j][i].n_tasks = 0;
		}
	}

	// create merge buf
	for(int i = 0; i < NR_DPUS; i++){
		ret_merge_key[i] = new DTYPE[MAX_MERGE_KEY_SIZE];
	}
	mem_addr_flag_host = 0;
}

// bulk_load for dpu
void bulk_load_for_dpu(DTYPE* keys, uint64_t total_input_size, int nr_partition){

	// std::sort(keys, keys + total_input_size);
	tbb::parallel_sort(keys, keys + total_input_size);

	// generated key and payloads for index
	dindex.total_size = total_input_size;
	dindex.keys = new DTYPE[total_input_size];
	memcpy(dindex.keys, keys, total_input_size*sizeof(DTYPE));
	dindex.payloads = new string_payload[total_input_size];
	for (long long i = 0; i < total_input_size; i++) {
        dindex.payloads[i] = string_payload('c');
    }

	// partial key
	dindex.partial_total_size = dindex.total_size / INTERLEAVE;
	dindex.partial_keys = new DTYPE[dindex.partial_total_size];
	int pi = 0;
	for(int i = 0; i < total_input_size; i += INTERLEAVE){
		dindex.partial_keys[pi] = dindex.keys[i];
		pi++;
	}

	// 平均分配数据，并进行相应的bulkload

	// Construct and bulk-load the Dynamic PGM-index
    std::cout << "----- start bulk load ----- " << std::endl;
    auto build_start_time = std::chrono::high_resolution_clock::now();
	unsigned long per_dpu_input_size = dindex.partial_total_size / nr_partition;


	// 建立并装填索引
	int transfer_model_size = MAX_MODEL_SIZE;
	int start = 0;

	size_t init_Epsilon = DataEpsilon;
	double cur_cost, prev_cost, next_cost;
	double wram_search_num, mram_search_num, wram_search_num_w_min_cost;
	int prev_model_size, cur_model_size, next_model_size, model_size;
	// 随机挑选一个段, 默认挑选第一个段
	pgm::PGMIndex<DTYPE>* test_pgm;
	// cur cost
	test_pgm = new pgm::PGMIndex<DTYPE> (dindex.partial_keys + start, dindex.partial_keys + start + per_dpu_input_size, init_Epsilon);
	cur_model_size = test_pgm->levels_offsets[test_pgm->height()];
	wram_search_num = std::log2((double)cur_model_size);
	mram_search_num = std::log2((double)init_Epsilon);
	cur_cost = wram_search_num + 2 * mram_search_num;
	wram_search_num_w_min_cost = wram_search_num;
	// prev cost
	test_pgm->rebuild(dindex.partial_keys + start, dindex.partial_keys + start + per_dpu_input_size, init_Epsilon / 2);
	prev_model_size = test_pgm->levels_offsets[test_pgm->height()];
	wram_search_num = std::log2((double)prev_model_size);
	mram_search_num = std::log2((double)init_Epsilon / 2);
	prev_cost = wram_search_num + 2 * mram_search_num;
	// next cost
	test_pgm->rebuild(dindex.partial_keys + start, dindex.partial_keys + start + per_dpu_input_size, init_Epsilon * 2);
	next_model_size = test_pgm->levels_offsets[test_pgm->height()];
	wram_search_num = std::log2((double)next_model_size);
	mram_search_num = std::log2((double)init_Epsilon * 2);
	next_cost = wram_search_num + 2 * mram_search_num;

	std::cout << "cur cost " << cur_cost << " prev cost" << prev_cost << " next cost " << next_cost << std::endl;
	int loop_num = 0;
	if(cur_cost > prev_cost && prev_model_size < MAX_MODEL_SIZE){
		// 减少epsilon
		while (init_Epsilon > 1)
		{
			loop_num++;
			init_Epsilon = init_Epsilon / 2;
			next_cost = cur_cost;
			cur_cost = prev_cost;
			test_pgm->rebuild(dindex.partial_keys + start, dindex.partial_keys + start + per_dpu_input_size, init_Epsilon / 2);
			model_size = test_pgm->levels_offsets[test_pgm->height()];
			wram_search_num = std::log2((double)model_size);
			mram_search_num = std::log2((double)init_Epsilon / 2);
			prev_cost = wram_search_num + 2 * mram_search_num;
			std::cout << "cur cost " << cur_cost  << std::endl;
			if(cur_cost <= prev_cost || loop_num > 2 || model_size >= MAX_MODEL_SIZE){
				break;
			}
			wram_search_num_w_min_cost = wram_search_num;
		}
	}else if(cur_cost > next_cost && next_model_size < MAX_MODEL_SIZE){
		// 增大epsilon
		while (init_Epsilon < 2048)
		{
			loop_num++;
			init_Epsilon = init_Epsilon * 2;
			prev_cost = cur_cost;
			cur_cost = next_cost;
			test_pgm->rebuild(dindex.partial_keys + start, dindex.partial_keys + start + per_dpu_input_size, init_Epsilon * 2);
			model_size = test_pgm->levels_offsets[test_pgm->height()];
			wram_search_num = std::log2((double)model_size);
			mram_search_num = std::log2((double)init_Epsilon * 2);
			next_cost = wram_search_num + 2 * mram_search_num;
			std::cout << "cur cost " << cur_cost << std::endl;
			if(cur_cost <= next_cost || loop_num > 2 || model_size >= MAX_MODEL_SIZE){
				break;
			}
			wram_search_num_w_min_cost = wram_search_num;
		}
	}else{
		// done
	}
	std::cout << "set Epsilon " << init_Epsilon << std::endl;
	// init_Epsilon = DataEpsilon;

	double cur_p_wram_search_num, cur_p_mram_search_num, cur_p_cost, min_p_cost;
	for(int kk = 0; kk < nr_partition; kk++){
		// 建立DRAM上的PGM, 初始化误差范围为64
		dindex.pgm_dram_index[kk] = new pgm::PGMIndex<DTYPE> (dindex.partial_keys + start, dindex.partial_keys + start + per_dpu_input_size, init_Epsilon);
		int cur_model_size = dindex.pgm_dram_index[kk]->levels_offsets[dindex.pgm_dram_index[kk]->height()];
		size_t cur_Epsilon = init_Epsilon;
		cur_p_wram_search_num = std::log2((double)cur_model_size);
		cur_p_mram_search_num = std::log2((double)cur_model_size);
		cur_p_cost = cur_p_wram_search_num + 2 * cur_p_mram_search_num;
		if(cur_p_wram_search_num > wram_search_num_w_min_cost + 2){
			// 尝试调整误差限界一次
			min_p_cost = cur_p_cost;
			dindex.pgm_dram_index[kk]->rebuild(dindex.partial_keys + start, dindex.partial_keys + start + per_dpu_input_size, cur_Epsilon * 2);
			cur_model_size = dindex.pgm_dram_index[kk]->levels_offsets[dindex.pgm_dram_index[kk]->height()];
			cur_p_wram_search_num = std::log2((double)cur_model_size);
			cur_p_mram_search_num = std::log2((double)cur_model_size);
			cur_p_cost = cur_p_wram_search_num + 2 * cur_p_mram_search_num;
			if(cur_p_cost < min_p_cost){
				//调整误差区间
				cur_Epsilon = cur_Epsilon * 2;
			}else{
				// 恢复原有区间
				dindex.pgm_dram_index[kk]->rebuild(dindex.partial_keys + start, dindex.partial_keys + start + per_dpu_input_size, cur_Epsilon);
			}
		}
		while(cur_model_size > MAX_MODEL_SIZE){
			cur_Epsilon = cur_Epsilon / 2;
			dindex.pgm_dram_index[kk]->rebuild(dindex.partial_keys + start, dindex.partial_keys + start + per_dpu_input_size, cur_Epsilon);
			cur_model_size = dindex.pgm_dram_index[kk]->levels_offsets[dindex.pgm_dram_index[kk]->height()];
		}
		dindex.partiton_epsilon[kk] = cur_Epsilon;
		//填充DRAM level 设置边界key
		upper_dram_level.low_bound_key[kk] = dindex.partial_keys[start];

		// 将模型和数据装填到DPU
		// 1. Create kernel arguments
		input_arguments[kk] = {per_dpu_input_size, dpu_arguments_t::kernels(0)};
		input_arguments[kk].start_pos = static_cast<uint32_t>(start);
		input_arguments[kk].max_levels = dindex.pgm_dram_index[kk]->height();
		input_arguments[kk].setEpsilon = cur_Epsilon;
		for(int i = 0; i < dindex.pgm_dram_index[kk]->levels_offsets.size(); i++){
			input_arguments[kk].level_offset[i] = dindex.pgm_dram_index[kk]->levels_offsets[i];
			std::cout << "level offset " << i << " " << input_arguments[kk].level_offset[i] << std::endl;
		}

		dindex.dram_start_pos[kk] = static_cast<uint32_t>(start);
		start += per_dpu_input_size;
		
		// 2. package pgm index
		if(dindex.pgm_dram_index[kk]->segments.size() > MAX_MODEL_SIZE){
			std::cout << "PGM model size is larger than MAX_MODEL_SIZE" << std::endl; 
			assert(0);
		}

		dindex.transfer_model[kk] = new pgm_model_t[MAX_MODEL_SIZE];
		for(int i = 0; i < dindex.pgm_dram_index[kk]->segments.size(); i++){
			dindex.transfer_model[kk][i].key = dindex.pgm_dram_index[kk]->segments[i].key;
			dindex.transfer_model[kk][i].slope = dindex.pgm_dram_index[kk]->segments[i].slope;
			dindex.transfer_model[kk][i].intercept = dindex.pgm_dram_index[kk]->segments[i].intercept;
			if(dindex.transfer_model[kk][i].intercept < 0){
				std::cout << "------- intercept less than 0 ---------" << std::endl;
			}
		}
		
		// debug
		// std::cout << "into max level offset " << input_arguments[kk].level_offset[input_arguments[kk].max_levels] << std::endl;  

	}

	// 3. transfer to dpu
	// 装填参数
	uint64_t nr_dpu_per_partition = NR_DPUS / NR_PARTITION;
	replicas_info_t* replicas_info= new replicas_info_t(nr_dpu_per_partition);
	replicas_info->get_cumulative_info();

	// 填充DRAM level的dpu信息
	memcpy(upper_dram_level.nr_replicas_per_partition, replicas_info->nr_replicas_per_partition, sizeof(uint64_t) * NR_PARTITION);
	upper_dram_level.get_start_dpu_id();

	uint64_t i = 0;
	uint64_t cur_p = 0;
	DPU_FOREACH(dpu_set, dpu, i)
	{
		cur_p = replicas_info->next_partition();
		DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments[cur_p]));
	}
	DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(input_arguments[0]), DPU_XFER_DEFAULT));

	// 装填数据 key
	auto push_start_time = std::chrono::high_resolution_clock::now();
	i = 0;
	replicas_info->reset();
	DPU_FOREACH(dpu_set, dpu, i)
	{
		cur_p = replicas_info->next_partition();
		DPU_ASSERT(dpu_prepare_xfer(dpu, dindex.partial_keys + cur_p * per_dpu_input_size));
	}
	DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, per_dpu_input_size * sizeof(DTYPE), DPU_XFER_DEFAULT));
	auto push_end_time = std::chrono::high_resolution_clock::now();
    double total_push_time =std::chrono::duration_cast<std::chrono::nanoseconds>(push_end_time -
                                                             push_start_time).count();
    std::cout << "push key time " << total_push_time / 1e9 << " s" << std::endl;
	
	// 装填模型
	i = 0;
	replicas_info->reset();
	DPU_FOREACH(dpu_set, dpu, i)
	{
		cur_p = replicas_info->next_partition();
		DPU_ASSERT(dpu_prepare_xfer(dpu, dindex.transfer_model[cur_p]));
	}
	DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "model_array", 0, MAX_MODEL_SIZE * sizeof(pgm_model_t), DPU_XFER_DEFAULT));
	std::cout << "end transfer model " << std::endl;

    auto build_end_time = std::chrono::high_resolution_clock::now();
    double total_build_time =std::chrono::duration_cast<std::chrono::nanoseconds>(build_end_time -
                                                             build_start_time).count();
    std::cout << "bulk load index build time " << total_build_time / 1e9 << " s" << std::endl;
	std::cout << "train model ops " << train_ops << std::endl;

}

void load_hot_replica(int tid){
	unsigned long per_dpu_input_size = dindex.partial_total_size / NR_PARTITION;
	// 计算replicas
	replicas_info_t* hot_replicas_info= new replicas_info_t();
	hot_replicas_info->load_hot_replicas_info(&partition_access_level, tid);
	hot_replicas_info->get_cumulative_info();

	// 填充DRAM level的dpu信息, TODO 加锁！！
	memcpy(upper_dram_level.nr_replicas_per_partition, hot_replicas_info->nr_replicas_per_partition, sizeof(uint64_t) * NR_PARTITION);
	upper_dram_level.get_start_dpu_id();

	// Load to DPU
	auto push_start_time = std::chrono::high_resolution_clock::now();
	uint64_t i = 0;
	uint64_t cur_p = 0;
	DPU_FOREACH(dpu_set, dpu, i)
	{
		cur_p = hot_replicas_info->next_partition();
		DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments[cur_p]));
	}
	DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(input_arguments[0]), DPU_XFER_DEFAULT));

	// 装填数据 key
	i = 0;
	hot_replicas_info->reset();
	DPU_FOREACH(dpu_set, dpu, i)
	{
		cur_p = hot_replicas_info->next_partition();
		DPU_ASSERT(dpu_prepare_xfer(dpu, dindex.partial_keys + cur_p * per_dpu_input_size));
	}
	DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, per_dpu_input_size * sizeof(DTYPE), DPU_XFER_DEFAULT));
	
	// 装填模型
	i = 0;
	hot_replicas_info->reset();
	DPU_FOREACH(dpu_set, dpu, i)
	{
		cur_p = hot_replicas_info->next_partition();
		DPU_ASSERT(dpu_prepare_xfer(dpu, dindex.transfer_model[cur_p]));
	}
	DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "model_array", 0, MAX_MODEL_SIZE * sizeof(pgm_model_t), DPU_XFER_DEFAULT));

	auto push_end_time = std::chrono::high_resolution_clock::now();
    double total_push_time =std::chrono::duration_cast<std::chrono::nanoseconds>(push_end_time -
                                                             push_start_time).count();
    std::cout << "load hot replicas time " << total_push_time / 1e9 << " s" << std::endl;

}

// void inline index_lookup_concurrent(DTYPE* keys, int nr_requests){
// 	for(int i = 0; i < NR_DPUS; i++){
// 		host_send_buffer[i].n_tasks = 0;
// 	}

// 	int dpu_idx, slot_idx;
// 	#pragma omp parallel for schedule(dynamic, 10000) private(dpu_idx, slot_idx), num_threads(32) // 32线程并行的填充buffer
// 	for(int req = 0; req < nr_requests; req++){
// 		dpu_idx = search_upper_level(keys[req]);
// 		slot_idx = atomic_inc_return(&(host_send_buffer[dpu_idx].n_tasks));
// 		host_send_buffer[dpu_idx].sbuffer[slot_idx] = keys[req];
// 	}
// }

std::pair<int, int> inline max_buffer_size_search(int tid){
	int max = 0;
	for(int i = 0; i < NR_DPUS; i++){
		if(host_send_buffer[tid][i].n_tasks > max)
			max = host_send_buffer[tid][i].n_tasks;
	}
	//note: return {send_buffer_size, recv_buffer_size}
	int sbuffer_size = max * sizeof(DTYPE) + 2 * sizeof(int);
	int rbuffer_size = max * sizeof(DTYPE) + 2 * sizeof(int);
	return {sbuffer_size, rbuffer_size};
}

std::pair<int, int> inline max_buffer_size_insert(int tid){
	int max = 0;
	for(int i = 0; i < NR_DPUS; i++){
		if(host_send_buffer[tid][i].n_tasks > max)
			max = host_send_buffer[tid][i].n_tasks;
	}
	//note: return {send_buffer_size, recv_buffer_size}
	int sbuffer_size = max * sizeof(DTYPE) * 2 + 2 * sizeof(int);
	int rbuffer_size = max * sizeof(DTYPE) + 2 * sizeof(int);
	if(sbuffer_size % 8 != 0){
		sbuffer_size = sbuffer_size + 8 - (sbuffer_size % 8);
	}
	return {sbuffer_size, rbuffer_size};
}

void inline dpu_sync(int thread_id, int buffer_size){
	// send query
	uint64_t i = 0;
	DPU_FOREACH(dpu_set, dpu, i)
	{
		DPU_ASSERT(dpu_prepare_xfer(dpu, &host_send_buffer[thread_id][i]));
	}

	DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "dpu_sbuffer", 0, buffer_size, DPU_XFER_DEFAULT)); // buffer_size  sizeof(send_buffer_t)

	// launch dpu
	// auto get_start_time = std::chrono::high_resolution_clock::now();
	DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
	// auto get_end_time = std::chrono::high_resolution_clock::now();
	// double dram_time =std::chrono::duration_cast<std::chrono::nanoseconds>(get_end_time -
	//                                              get_start_time).count();
	// std::cout << "dpu search time " << dram_time / 1e9 << " s" << std::endl;
}

void inline get_index_lookup_result(int tid, int buffer_size){
	uint64_t i = 0;
	DPU_FOREACH(dpu_set, dpu, i)
	{
		DPU_ASSERT(dpu_prepare_xfer(dpu, &host_recv_buffer[tid][i]));
	}

	DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "dpu_recv_buffer", 0, buffer_size, DPU_XFER_DEFAULT)); // buffer_size sizeof(recv_buffer_t)

#if DEBUG
	// debug, check result
	i = 0;
	DPU_FOREACH(dpu_set, dpu, i)
	{
		std::cout << "recv access number " <<  host_recv_buffer[i].n_tasks << std::endl;
		std::cout << "send task number " <<  host_send_buffer[tid][i].n_tasks << std::endl;
	}
	int found_error = 0;
	DPU_FOREACH(dpu_set, dpu, i)
	{
		int cmp_count = host_send_buffer[tid][i].n_tasks; // - host_send_buffer[i].n_tasks % NR_TASKLETS - 1;
		for(unsigned int h = 0; h < cmp_count; h++)
		{
			if(host_recv_buffer[i].rbuffer[h] != host_send_buffer[tid][i].sbuffer[h].key){
				found_error++;
			}
			// std::cout << "send " << host_send_buffer[i].sbuffer[h] << " recv " << host_recv_buffer[i].rbuffer[h] << std::endl;
		}
	}
	if(!found_error)
		std::cout << "FOUND OK " << std::endl;
	else
		std::cout << "FOUND ERROR, error num " << found_error << std::endl;
#endif
}

void inline get_real_payload(int tid){
	string_payload ret;
	int task_num;
	// int total_task_num = 0;
	bool not_found;
	for(int i = 0; i < NR_DPUS; i++){
		task_num = host_recv_buffer[tid][i].n_tasks;
		// total_task_num+= task_num;
		for(int j = 0; j < task_num; j++){
			not_found = true;
			// __builtin_prefetch(&(host_recv_buffer[tid][i].rbuffer[j + 4]), 0);
			for(int k = 0; k < INTERLEAVE; k++){
				__builtin_prefetch(&(dindex.keys[host_recv_buffer[tid][i].rbuffer[j + 4]]), 0);
				// __builtin_prefetch(&(dindex.payloads[host_recv_buffer[tid][i].rbuffer[j + 2]]), 0);
				if((host_recv_buffer[tid][i].rbuffer[j] + k < dindex.total_size) && (dindex.keys[host_recv_buffer[tid][i].rbuffer[j] + k] == host_send_buffer[tid][i].sbuffer[j])){ //(host_recv_buffer[tid][i].rbuffer[j] + k < dindex.total_size) &&
					ret = dindex.payloads[host_recv_buffer[tid][i].rbuffer[j] + k];
					not_found = false;
					break;
				}
			}
			if(not_found){
				// unequal++;
				DTYPE search_key = host_send_buffer[tid][i].sbuffer[j];
				int p = upper_dram_level.search_upper_level_get_partition(search_key);
				if(p >= NR_PARTITION)
					p = NR_PARTITION - 1;
				if(p < 0)
					p = 0;
				auto range_ret = dindex.pgm_dram_index[p]->search(search_key);
				unsigned long per_dpu_input_size = dindex.partial_total_size / NR_DPUS;
				size_t l = range_ret.lo + (per_dpu_input_size * p);
				size_t r = range_ret.hi + (per_dpu_input_size * p);
				if(r >= dindex.partial_total_size){
					r = dindex.partial_total_size - 1;
				}
				size_t m;
				while(l < r){
					m = l + (r - l) / 2;
					if(dindex.partial_keys[m] <= search_key)
						l = m + 1;
					else
						r = m;
				}
				size_t cur_key_off = (l - 1) * INTERLEAVE;
				for(int k = 0; k < INTERLEAVE; k++){
					if((cur_key_off + k < dindex.total_size) && dindex.keys[cur_key_off + k] == host_send_buffer[tid][i].sbuffer[j]){ //(cur_key_off + k < dindex.total_size) && 
						ret = dindex.payloads[cur_key_off + k];
						break;
					}
				}
				
			}
			// std::cout << "dpu  " << i << std::endl;
			// std::cout << " key pos " <<  (uint64_t)(host_recv_buffer[tid][i].rbuffer[j]) << std::endl; 
			// std::cout << "recv key " << (uint64_t)(dindex.keys[host_recv_buffer[tid][i].rbuffer[j]]) << std::endl;
			// std::cout << "send key " << (uint64_t)(host_send_buffer[tid][i].sbuffer[j]) << std::endl;
			// std::cout << "prev send key " << (uint64_t)(host_send_buffer[tid][i].sbuffer[j - 1]) << std::endl;
		}
	}
}

// int inline get_index_insert_result(){
// 	uint64_t i = 0;
// 	DPU_FOREACH(dpu_set, dpu, i)
// 	{
// 		DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments[i]));
// 	}
// 	DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(input_arguments[i]), DPU_XFER_DEFAULT));
// 	int need_retrain = 0;
// 	for(int k = 0; k < NR_DPUS; k++){
// 		if(input_arguments[k].ret == 1){
// 			need_retrain = 1;
// 		}
// 	}
// 	return need_retrain;
// }

uint64_t inline get_index_merge_size(){
	uint64_t i = 0;
	DPU_FOREACH(dpu_set, dpu, i)
	{
		DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments[i]));
	}
	DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(input_arguments[i]), DPU_XFER_DEFAULT));
	//return max size
	uint64_t max_size = 0;
	for(int k = 0; k < NR_DPUS; k++){
		if(input_arguments[k].input_size > max_size){
			max_size = input_arguments[k].input_size;
		}
	}
	return max_size;
}

void inline get_index_merge_key(uint64_t max_size){
	uint64_t i = 0;
	DPU_FOREACH(dpu_set, dpu, i)
	{
		DPU_ASSERT(dpu_prepare_xfer(dpu, ret_merge_key[i]));
	}
	if(mem_addr_flag_host == 0){
		uint32_t MRAM_MID_SIZE = 30 * 1024 * 1024; 
		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, MRAM_MID_SIZE, max_size * sizeof(DTYPE), DPU_XFER_DEFAULT)); // 
	}else if(mem_addr_flag_host == 1){
		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, max_size * sizeof(DTYPE), DPU_XFER_DEFAULT)); // 
	}
}


// Main of the Host Application
int main(int argc, char **argv) {
    auto flags = parse_flags(argc, argv);
    std::string keys_file_path = get_required(flags, "keys_file");
    auto init_num_keys = stoi(get_required(flags, "init_num_keys"));
	uint64_t num_querys = stoi(get_required(flags, "query_num"));
	bool is_insert = get_boolean_flag(flags, "insert"); // 默认为search only
	bool is_search = get_boolean_flag(flags, "search"); // 默认为search only
	auto total_num_keys = stoi(get_required(flags, "total_num_keys"));
	std::string sample_distribution = get_with_default(flags, "sample_distribution", "uniform");
	output_path = get_with_default(flags, "output_path", "./out.csv");
	
	// Read keys from binary file
    std::cout << "start load keys from files " <<  keys_file_path << std::endl;
    DTYPE* keys = new DTYPE[total_num_keys];
    load_binary_data(keys, total_num_keys, keys_file_path);
	std::random_shuffle(keys, keys + total_num_keys);

	uint32_t nr_of_dpus;
	uint64_t input_size = init_num_keys; 

	// Allocate DPUs and load binary
	DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
	DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
	DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));

	init_host();

	bulk_load_for_dpu(keys, input_size, NR_PARTITION);
	// init DPU
	int j = 0;
	DPU_FOREACH(dpu_set, dpu, j)
	{
		host_send_buffer[0][j].op_type = 0; // INIT
	}
	dpu_sync(0, 2 * sizeof(int));
#if PRINT
	unsigned int each_dpu_bulk_load= 0;
	printf("Display DPU Logs\n");
	DPU_FOREACH(dpu_set, dpu)
	{
		printf("DPU#%d:\n", each_dpu_bulk_load);
		DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
		each_dpu_bulk_load++;
	}
#endif


	// DTYPE * input  = (DTYPE*)malloc((input_size) * sizeof(DTYPE));
	DTYPE * querys = (DTYPE*)malloc((num_querys) * sizeof(DTYPE));

	std::cout << "input size " << input_size << " query num " << num_querys;

	// Create an input file with arbitrary data
	if(is_insert){
		// INSERT
		create_insert_op(keys, querys, input_size, num_querys, total_num_keys);
	}else{
		// SEARCH
		if (sample_distribution == "uniform") {
			uint64_t elements_num = (input_size / INTERLEAVE / NR_PARTITION) * NR_PARTITION * INTERLEAVE;
			std::cout << "element number " << elements_num << std::endl;
            create_query(keys, querys, elements_num, num_querys);
        } else if (sample_distribution == "zipf") {
            create_query_zipf(keys, querys, input_size, num_querys);
        }
	}

	// prepare retrain model 
	pgm_model_t* retrain_transfer_model[NR_DPUS];
	for(int kk = 0; kk < NR_DPUS; kk++)
		retrain_transfer_model[kk] = new pgm_model_t[MAX_MODEL_SIZE];
	

	/*****Warm UP*****/ 
	// int wtid = 0;	
	// int wpd = 32;
	// int wdpu_idx;
	// for(int req = 0; req < 1000000; req++){
	// 	__builtin_prefetch(&querys[req + wpd], 0);
	// 	wdpu_idx = upper_dram_level.search_upper_level(querys[req], req, wtid, true);		
	// }
	// partition_access_level.calculate_partition_skew(NR_DPUS / NR_PARTITION, 1000000, wtid);
	// load_hot_replica(wtid);
	//clear wtid 


	// thread 
	std::vector<std::thread> thread_array;
	thread_array.reserve(THREAD_NUM);
	int per_thread_ops = num_querys / THREAD_NUM;

	
	TSCNS tn;
	tn.init();
	printf("Begin running\n");
	auto start_time = tn.rdtsc();
	auto end_time = tn.rdtsc();

	if(is_insert){
		// INSERT
		// for(int i = 0; i < THREAD_NUM; i++){
		// 	thread_array.emplace_back(
		// 	[&](size_t thread_id){
		// 		// thread content
		// 		int tid = Coremeta::threadID();
		// 		int start_ops = per_thread_ops * tid;
		// 		int end_ops = per_thread_ops * (tid + 1);
				
		// 		int pd = 32;
		// 		int dpu_idx, slot_idx;
		// 		for(int req = start_ops; req < end_ops; req++){
		// 			__builtin_prefetch(&querys[req + pd], 0);
		// 			dpu_idx = search_upper_level(querys[req]);
		// 			slot_idx = host_send_buffer[tid][dpu_idx].n_tasks;
		// 			host_send_buffer[tid][dpu_idx].n_tasks++;
		// 			host_send_buffer[tid][dpu_idx].sbuffer[2 * slot_idx] = querys[req]; // key
		// 			host_send_buffer[tid][dpu_idx].sbuffer[2 * slot_idx + 1] = querys[req]; // value
		// 		}
		// 		// set op type
		// 		int h = 0;
		// 		DPU_FOREACH(dpu_set, dpu, h)
		// 		{
		// 			host_send_buffer[tid][h].op_type = 2; //INSERT
		// 		}
		// 		std::pair<int, int> buffer_size = max_buffer_size_insert(tid);

		// 		// execuse
		// 		lock.lock();
		// 		dpu_sync(tid, buffer_size.first);
		// 		// get_index_lookup_result(tid, buffer_size.second);
		// 		int rr = get_index_insert_result();
		// 		std::cout << "retrain flag " << rr << " tid " << tid << std::endl; 
		// 		// unsigned int each_dpu = 0;
		// 		// printf("Display DPU Logs\n");
		// 		// DPU_FOREACH(dpu_set, dpu)
		// 		// {
		// 		// 	printf("DPU#%d:\n", each_dpu);
		// 		// 	DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
		// 		// 	each_dpu++;
		// 		// }
		// 		if(rr){
		// 			// merge data
		// 			auto retrain_start_time = tn.rdtsc();
		// 			auto retrain_end_time = tn.rdtsc();
		// 			h = 0;
		// 			DPU_FOREACH(dpu_set, dpu, h)
		// 			{
		// 				host_send_buffer[tid][h].op_type = 3; //MERGE
		// 			}
		// 			dpu_sync(tid, 2 * sizeof(int));
		// 			//get merge data
		// 			uint64_t max_merge_arr_size = get_index_merge_size();
		// 			get_index_merge_key(max_merge_arr_size);
		// 			if(mem_addr_flag_host == 0){
		// 				mem_addr_flag_host = 1;
		// 			}else{
		// 				mem_addr_flag_host = 0;
		// 			}
		// 			// retrain
		// 			int transfer_model_size = MAX_MODEL_SIZE;
		// 			#pragma omp parallel for num_threads(32) 
		// 			for(int kk = 0; kk < NR_DPUS; kk++){
		// 				// 建立DRAM上的PGM
		// 				pgm::PGMIndex<DTYPE, DataEpsilon> per_slice_pgm(ret_merge_key[kk], ret_merge_key[kk] + input_arguments[kk].input_size);
		// 				// 1. Create kernel arguments
		// 				input_arguments[kk].max_levels = per_slice_pgm.height();
		// 				for(int i = 0; i < per_slice_pgm.levels_offsets.size(); i++){
		// 					input_arguments[kk].level_offset[i] = per_slice_pgm.levels_offsets[i];
		// 					// std::cout << "level offset " << i << " " << input_arguments[kk].level_offset[i] << std::endl;
		// 				}
						
		// 				// 2. package pgm index
		// 				if(per_slice_pgm.segments.size() > MAX_MODEL_SIZE){
		// 					std::cout << "PGM model size is larger than MAX_MODEL_SIZE" << std::endl; 
		// 					assert(0);
		// 				}
		// 				for(int i = 0; i < per_slice_pgm.segments.size(); i++){
		// 					retrain_transfer_model[kk][i].key = per_slice_pgm.segments[i].key;
		// 					retrain_transfer_model[kk][i].slope = per_slice_pgm.segments[i].slope;
		// 					retrain_transfer_model[kk][i].intercept = per_slice_pgm.segments[i].intercept;
		// 				}
		// 			}
		// 			uint64_t i = 0;
		// 			DPU_FOREACH(dpu_set, dpu, i)
		// 			{
		// 				DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments[i]));
		// 			}
		// 			DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(input_arguments[i]), DPU_XFER_DEFAULT));
		// 			i = 0;
		// 			DPU_FOREACH(dpu_set, dpu, i)
		// 			{
		// 				DPU_ASSERT(dpu_prepare_xfer(dpu, retrain_transfer_model[i]));
		// 			}
		// 			DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "model_array", 0, MAX_MODEL_SIZE * sizeof(pgm_model_t), DPU_XFER_DEFAULT));

		// 			retrain_end_time = tn.rdtsc();
		// 			auto retrain_diff = tn.tsc2ns(retrain_end_time) - tn.tsc2ns(retrain_start_time);
		// 			std::cout << "retrain time (second) " << (retrain_diff/(double) 1000000000) << std::endl;

		// 			// for(int kk = 0; kk < NR_DPUS; kk++){
		// 			// 	// print model info
		// 			// 	// int model_size = per_slice_pgm.segments.size(); 
		// 			// 	std::cout << "into max level offset " << input_arguments[kk].level_offset[input_arguments[kk].max_levels] << std::endl;  
		// 			// }
		// 		}
		// 		lock.unlock();
		// 	}
		// 	,i);
		// }

		// for (auto &t : thread_array) {
		// 		t.join();
		// }

	}else{
		// SEARCH
		for(int i = 0; i < THREAD_NUM; i++){
			thread_array.emplace_back(
			[&](size_t thread_id){
				// thread content
				int tid = Coremeta::threadID();
				int start_ops = per_thread_ops * tid;
				int end_ops = per_thread_ops * (tid + 1);
				
				int pd = 32;
				int dpu_idx, slot_idx;
				for(int req = start_ops; req < end_ops; req++){
					__builtin_prefetch(&querys[req + pd], 0);
					dpu_idx = upper_dram_level.search_upper_level(querys[req], req, tid, false);
					slot_idx = host_send_buffer[tid][dpu_idx].n_tasks;
					host_send_buffer[tid][dpu_idx].n_tasks++;
					host_send_buffer[tid][dpu_idx].sbuffer[slot_idx] = querys[req];
					
				}
				// set op type
				int h = 0;
				DPU_FOREACH(dpu_set, dpu, h)
				{
					host_send_buffer[tid][h].op_type = 1; //LOOKUP
				}
				std::pair<int, int> buffer_size = max_buffer_size_search(tid);

				// std::cout << "max buffer size " << buffer_size.first / 8 << " expected buffer size " << (end_ops - start_ops) / NR_DPUS << std::endl;

				// execuse
				lock.lock();
				dpu_sync(tid, buffer_size.first);
				get_index_lookup_result(tid, buffer_size.second);
				lock.unlock();

				// auto get_start_time = std::chrono::high_resolution_clock::now();
				get_real_payload(tid);
				// auto get_end_time = std::chrono::high_resolution_clock::now();
    			// double dram_time =std::chrono::duration_cast<std::chrono::nanoseconds>(get_end_time -
                //                                              get_start_time).count();
    			// std::cout << "dram search time " << dram_time / 1e9 << " s" << std::endl;

				
#if PRINT
				unsigned int each_dpu = 0;
				printf("Display DPU Logs\n");
				DPU_FOREACH(dpu_set, dpu)
				{
					printf("DPU#%d:\n", each_dpu);
					DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
					each_dpu++;
				}
#endif
			}
			,i);
		}

		for (auto &t : thread_array) {
				t.join();
		}
	}

	std::cout << "uequal number " << unequal << std::endl;


	end_time = tn.rdtsc();
	auto diff = tn.tsc2ns(end_time) - tn.tsc2ns(start_time);
	std::cout << "run time (second) " << (diff/(double) 1000000000) << std::endl;
	std::cout << "throughput (second) " << num_querys / (diff/(double) 1000000000) << std::endl;

	free(keys);
	DPU_ASSERT(dpu_free(dpu_set));

	/*****print stat ***/
	// time id
	std::time_t t = std::time(nullptr);
	char time_str[100];
	if (!file_exists(output_path)) {
		std::ofstream ofile;
		ofile.open(output_path, std::ios::app);
		ofile << "id" << ",";
		ofile << "key_path" << ",";
		ofile << "throughput" << ",";
		ofile << "init_table_size" << ",";
		ofile << "operation_num" << ",";
		ofile << "thread_num" << std::endl;
	}

	std::ofstream ofile;
	ofile.open(output_path, std::ios::app);
	if (std::strftime(time_str, sizeof(time_str), "%Y%m%d%H%M%S", std::localtime(&t))) {
		ofile << time_str << ',';
	}
    ofile << keys_file_path << ",";
    ofile << num_querys / (diff/(double) 1000000000) << ",";
	ofile << init_num_keys << ",";
	ofile << num_querys << ",";
	ofile << THREAD_NUM << std::endl;
    ofile.close();

	return 0;
}