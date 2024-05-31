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
#include <unistd.h>
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
#include <pthread.h>
#include "concurrent.h"

std::atomic<int> unequal(0);
std::atomic<int> invaild_pos(0);

// Define the DPU Binary path as DPU_BINARY here
#define DPU_BINARY "./bin/lex_dpu"

std::string output_path;

#define DEBUG 0
#define BATCH_NUM 10000000
#define THREAD_NUM 32 // 32线程性能更差？
#define PRINT 0
#define INTERLEAVE 8
#define NR_PARTITION 128
#define PAGE_RATIO 8
#define MERGE_THREAD_NUM 32

DRAM_index dindex;
ol_lock upper_level_lock;
ol_lock dindex_lock;

ol_lock* lock_arr;

ol_lock merge_lock;
std::atomic<int> thread_in_write(0);
std::atomic<int> current_write_num(0);
std::shared_mutex merge_shard_lock;

struct dpu_set_t dpu_set, dpu;
// DTYPE* upper_dram_level; // 存储在DRAM上的上层索引，用于查找DPU。需要建立
send_buffer_t *host_send_buffer[THREAD_NUM]; // read only, 存储query 
recv_buffer_t *host_recv_buffer[THREAD_NUM]; // read only, 存储返回值, 所有线程共享一个recv buffer
std::shared_mutex lock;
dpu_arguments_t input_arguments[NR_DPUS];

int mix_order[NR_DPUS];
bool is_pure_read;

struct type_array_t{
	char ops[MAX_BUFFER_SIZE];
};
type_array_t *type_array[THREAD_NUM];


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

	int max_dpu_num_off(int tid){
		uint64_t max = 0;
		int off = 0;
		for(int i = 0; i < NR_PARTITION; i++){
			if(record_ratio[tid][i].num_dpu_per_partition > max){
				max = record_ratio[tid][i].num_dpu_per_partition;
				off = i;
			}
		}
		return off;
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
		// 微调分配比例，尝试从负载最轻的parititon中剥夺副本分给负载最重的
		double after_adjust_per_dpu_ratio;
		int min_available_off = NR_PARTITION - 1;
		int max_available_off = 0;

		for(int adjust_loop = 0; adjust_loop < 4; adjust_loop++){
			bool enter_adjust = false;
			// min cost
			after_adjust_per_dpu_ratio = record_ratio[tid][min_available_off].partition_ratio 
				/ (record_ratio[tid][min_available_off].num_dpu_per_partition - 1);
			if(after_adjust_per_dpu_ratio < record_ratio[tid][max_available_off].per_dpu_ratio){
				enter_adjust = true;
			}else{
				min_available_off = max_dpu_num_off(tid);
				after_adjust_per_dpu_ratio = record_ratio[tid][min_available_off].partition_ratio 
				/ (record_ratio[tid][min_available_off].num_dpu_per_partition - 1);
				if(after_adjust_per_dpu_ratio < record_ratio[tid][max_available_off].per_dpu_ratio){
					enter_adjust = true;
				}
			}

			if(enter_adjust){
				std::cout << "adjust partition " << std::endl;
				// 调整可以降低整体开销，开始调整！
				// 1）剥夺负载最轻的partition
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

				// 2) 给予负载最重的partition
				record_ratio[tid][max_available_off].num_dpu_per_partition++;
				// 新的dpu ratio
				record_ratio[tid][max_available_off].per_dpu_ratio = record_ratio[tid][max_available_off].partition_ratio 
					/ record_ratio[tid][max_available_off].num_dpu_per_partition;
				//dpu ratio变小, 调整顺序
				for(k = max_available_off; k < NR_PARTITION; k++){
					// 第一个比自己小的数
					if(record_ratio[tid][k].per_dpu_ratio < record_ratio[tid][max_available_off].per_dpu_ratio){
						break;
					}	
				}
				if(k == NR_PARTITION)
					k--;
				// 插入数据
				temp = record_ratio[tid][max_available_off];
				for(int ll = max_available_off; ll < k; ll ++){
					record_ratio[tid][ll] = record_ratio[tid][ll + 1];
				}
				record_ratio[tid][k] = temp;

			}else{
				break;
			}
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

void gather_partition_access_from_buffer(int tid){
	int cur_partition = 0;
	partition_access_level.access_num[tid][cur_partition] = 0;
	upper_dram_level.start_dpu_id[cur_partition];
	for(int i = 0; i < NR_DPUS; i++){
		if((cur_partition == (NR_PARTITION - 1)) || (i < upper_dram_level.start_dpu_id[cur_partition + 1])){
			partition_access_level.access_num[tid][cur_partition] +=  host_send_buffer[tid][i].n_tasks;
		}
		if(i == upper_dram_level.start_dpu_id[cur_partition + 1]){
			cur_partition++;
			partition_access_level.access_num[tid][cur_partition] = 0;
			partition_access_level.access_num[tid][cur_partition] +=  host_send_buffer[tid][i].n_tasks;
		}
	}
	//debug
	// for(int j = 0; j < NR_PARTITION; j++){
	// 	std::cout << "partition id " << j << " access num " << partition_access_level.access_num[tid][j] << std::endl;
	// }
}

// Create input arrays
void create_query(DTYPE * input, DTYPE * querys, uint64_t  nr_elements, uint64_t nr_querys) {
	std::mt19937_64 gen(std::random_device{}());
	std::uniform_int_distribution<int> dis(0, nr_elements - 1);
	for (int i = 0; i < nr_querys; i++) {
		int pos = dis(gen);
		querys[i] = input[pos];
	}
}

void create_query_zipf(DTYPE * input, DTYPE * querys, uint64_t  nr_elements, uint64_t nr_querys, size_t* random_seed = nullptr) {
	ScrambledZipfianGenerator zipf_gen(nr_elements, random_seed);
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

	for(int j = 0; j < THREAD_NUM; j++){
		type_array[j] = new type_array_t[NR_DPUS];
	}

	for(int j = 0; j < NR_DPUS; j++){
		mix_order[j] = j;
	}

	lock_arr = new ol_lock[500000];
	is_pure_read = true;
}

inline void clear_send_buffer(int tid){
	for(int i = 0; i < NR_DPUS; i++){
		host_send_buffer[tid][i].n_tasks = 0;
	}
}

inline void clear_all_buffer(){
	for(int j = 0; j < THREAD_NUM; j++){
		for(int i = 0; i < NR_DPUS; i++){
			host_send_buffer[j][i].n_tasks = 0;
		}
	}
	for(int j = 0; j < THREAD_NUM; j++){
		for(int i = 0; i < NR_DPUS; i++){
			host_recv_buffer[j][i].n_tasks = 0;
		}
	}
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
	dindex.min_key = dindex.keys[0];

	// create buf pages
	dindex.num_buf_pages = dindex.total_size / PAGE_RATIO + 1;
	dindex.buffer_pages = new buf_page[dindex.num_buf_pages];
	for(int i = 0; i < dindex.num_buf_pages; i++){
		dindex.buffer_pages[i].init();
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
	// int rbuffer_size = max * sizeof(DTYPE) + 2 * sizeof(int);
	int rbuffer_size = max * sizeof(uint32_t) + 2 * sizeof(int);
	if(rbuffer_size % 8 != 0){
		rbuffer_size = rbuffer_size + 4; // 8 - (rbuffer_size % 8);
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
	string_payload* pret = new string_payload('0');
	uint32_t interleave_page_offset = dindex.num_buf_pages / MERGE_THREAD_NUM;
	int task_num;
	bool not_found;;
	for(int i = 0; i < NR_DPUS; i++){
		task_num = host_recv_buffer[tid][i].n_tasks - 1;
		// total_task_num+= task_num;
		for(int j = 0; j < task_num; j++){
			not_found = true;
			// __builtin_prefetch(&(host_recv_buffer[tid][i].rbuffer[j + 4]), 0);
			if(host_recv_buffer[tid][i].rbuffer[j] != INVAILD_POS){
				for(int k = 0; k < INTERLEAVE; k++){
					__builtin_prefetch(&(dindex.keys[host_recv_buffer[tid][i].rbuffer[j + 4]]), 0);
					// __builtin_prefetch(&(dindex.payloads[host_recv_buffer[tid][i].rbuffer[j + 2]]), 0);
					if((host_recv_buffer[tid][i].rbuffer[j] + k < dindex.total_size) && (dindex.keys[host_recv_buffer[tid][i].rbuffer[j] + k] == host_send_buffer[tid][i].sbuffer[j])){ //(host_recv_buffer[tid][i].rbuffer[j] + k < dindex.total_size) &&
						ret = dindex.payloads[host_recv_buffer[tid][i].rbuffer[j] + k];
						not_found = false;
						break;
					}
				}
			}else{
				// invaild_pos++;
			}
			if(not_found){
				bool dram_check = true;
				if(!is_pure_read && host_recv_buffer[tid][i].rbuffer[j] < dindex.total_size){
					uint32_t ipos = host_recv_buffer[tid][i].rbuffer[j] / PAGE_RATIO;
					if(dindex.buffer_pages[ipos].find_key_page(host_send_buffer[tid][i].sbuffer[j], pret)){
						ret = *pret;
						dram_check = false;
					}else{
						dindex.overflow_trees[std::min(ipos / interleave_page_offset, (uint32_t)MERGE_THREAD_NUM - 1)].lookup(host_send_buffer[tid][i].sbuffer[j], ret);
					}
				}
				if(dram_check){
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
						if((cur_key_off + k < dindex.total_size) && dindex.keys[cur_key_off + k] == search_key){ //(cur_key_off + k < dindex.total_size) && 
							ret = dindex.payloads[cur_key_off + k];
							break;
						}
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

void inline predecessor_real_payload(int tid){
	DTYPE retk;
	string_payload ret;
	string_payload* pret = new string_payload('0');
	DTYPE* page_pkey = new DTYPE(0);
	int task_num;
	bool not_found;;
	for(int i = 0; i < NR_DPUS; i++){
		task_num = host_recv_buffer[tid][i].n_tasks - 1;
		// total_task_num+= task_num;
		for(int j = 0; j < task_num; j++){
			not_found = true;
			// __builtin_prefetch(&(host_recv_buffer[tid][i].rbuffer[j + 4]), 0);
			if(host_recv_buffer[tid][i].rbuffer[j] != INVAILD_POS){
				for(int k = 0; k < INTERLEAVE; k++){
					__builtin_prefetch(&(dindex.keys[host_recv_buffer[tid][i].rbuffer[j + 4]]), 0);
					// __builtin_prefetch(&(dindex.payloads[host_recv_buffer[tid][i].rbuffer[j + 2]]), 0);
					if((host_recv_buffer[tid][i].rbuffer[j] + k < dindex.total_size) && (dindex.keys[host_recv_buffer[tid][i].rbuffer[j] + k] == host_send_buffer[tid][i].sbuffer[j])){ //(host_recv_buffer[tid][i].rbuffer[j] + k < dindex.total_size) &&
						if(host_recv_buffer[tid][i].rbuffer[j] + k > 1){
							retk = dindex.keys[host_recv_buffer[tid][i].rbuffer[j] + k - 1];
							ret = dindex.payloads[host_recv_buffer[tid][i].rbuffer[j] + k - 1];
						}
						not_found = false;
						break;
					}
				}
				uint32_t ipos = host_recv_buffer[tid][i].rbuffer[j] / PAGE_RATIO;
				if(dindex.buffer_pages[ipos].predecessor_key_page(host_send_buffer[tid][i].sbuffer[j], pret, page_pkey)){
					if(*page_pkey < retk){
						retk = *page_pkey;
						ret = *pret;
					}
				}
			}else{
				// invaild_pos++;
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
					if((cur_key_off + k < dindex.total_size) && dindex.keys[cur_key_off + k] == search_key){ //(cur_key_off + k < dindex.total_size) && 
						if(cur_key_off + k > 1){
							retk = dindex.keys[cur_key_off + k - 1];
							ret = dindex.payloads[cur_key_off + k - 1];
						}
						break;
					}
				}
				uint32_t ipos = cur_key_off / PAGE_RATIO;
				if(dindex.buffer_pages[ipos].predecessor_key_page(search_key, pret, page_pkey)){
					if(*page_pkey < retk){
						retk = *page_pkey;
						ret = *pret;
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

void inline mix_payload(int tid){ // mix ops
	string_payload ret;
	int task_num;
	uint32_t version;
	bool not_found;
	string_payload new_val('u');
	string_payload* val = new string_payload('c');
	string_payload oval('c');
	string_payload* pret = new string_payload('0');
	uint32_t interleave_page_offset = dindex.num_buf_pages / MERGE_THREAD_NUM;
	uint32_t pos;
	
	int cur_mix_order[NR_DPUS];
	memcpy(cur_mix_order, mix_order, sizeof(int) * NR_DPUS);
	std::random_shuffle(cur_mix_order, cur_mix_order + NR_DPUS);

	int i;

	for(int count = 0; count < NR_DPUS; count++){
		i = cur_mix_order[count];
		task_num = host_recv_buffer[tid][i].n_tasks - 1;
		for(int j = 0; j < task_num; j++){
			not_found = true;
			__builtin_prefetch(&(type_array[tid][i].ops[j + 4]), 0);
			if(host_recv_buffer[tid][i].rbuffer[j] != INVAILD_POS){
				if(type_array[tid][i].ops[j] == 2){
					// INSERT
					uint32_t ipos = host_recv_buffer[tid][i].rbuffer[j] / PAGE_RATIO;
					if(ipos < dindex.partial_total_size - 1 && dindex.partial_keys[ipos] <= host_send_buffer[tid][i].sbuffer[j] 
					&& dindex.partial_keys[ipos + 1] > host_send_buffer[tid][i].sbuffer[j]){ // 确认插入位置正确
						if(!dindex.buffer_pages[ipos].insert_page(host_send_buffer[tid][i].sbuffer[j], val)){
							dindex.overflow_trees[std::min(ipos / interleave_page_offset, (uint32_t)MERGE_THREAD_NUM - 1)].insert(host_send_buffer[tid][i].sbuffer[j], oval);
						}
					}
					not_found = false;
				}else{
					for(int k = 0; k < INTERLEAVE; k++){
						__builtin_prefetch(&(dindex.keys[host_recv_buffer[tid][i].rbuffer[j + 4]]), 0);
						if((host_recv_buffer[tid][i].rbuffer[j] + k < dindex.total_size) && (dindex.keys[host_recv_buffer[tid][i].rbuffer[j] + k] == host_send_buffer[tid][i].sbuffer[j])){ //(host_recv_buffer[tid][i].rbuffer[j] + k < dindex.total_size) &&
							
							pos = host_recv_buffer[tid][i].rbuffer[j] + k;
							if(type_array[tid][i].ops[j] == 0){
								// GET
								REGET:
								if (lock_arr[pos >> 9].test_lock_set(version)){ // !!!! 9在load 150M时表现好，10在load 200M时表现好。锁的大小导致性能急速下降？
									while(lock_arr[pos >> 9].test_lock_set(version)){
									};
								}
								ret = dindex.payloads[host_recv_buffer[tid][i].rbuffer[j] + k];
								if (lock_arr[pos >> 9].test_lock_version_change(version)){
									goto REGET;
								}
							}else if(type_array[tid][i].ops[j] == 1){
								// UPDATE
								lock_arr[pos >> 9].get_lock();
								dindex.payloads[host_recv_buffer[tid][i].rbuffer[j] + k] = new_val;
								lock_arr[pos >> 9].release_lock();
							}
							not_found = false;
							break;

						}
					}
				}
			}

			if(not_found){
				// unequal++;
				bool dram_check = true;
				if(host_recv_buffer[tid][i].rbuffer[j] < dindex.total_size){
					uint32_t ipos = host_recv_buffer[tid][i].rbuffer[j] / PAGE_RATIO;
					if(type_array[tid][i].ops[j] == 0){
						if(dindex.buffer_pages[ipos].find_key_page(host_send_buffer[tid][i].sbuffer[j], pret)){
							ret = *pret;
							dram_check = false;
						}
					}else if(type_array[tid][i].ops[j] == 1){
						if(dindex.buffer_pages[ipos].update_key_page(host_send_buffer[tid][i].sbuffer[j], pret)){
							dram_check = false;
						}
					}
				}

				if(dram_check){
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
						if((cur_key_off + k < dindex.total_size) && dindex.keys[cur_key_off + k] == search_key){ //(cur_key_off + k < dindex.total_size) && 
							pos = cur_key_off + k;
							if(type_array[tid][i].ops[j] == 0){
								// GET
								REGET2:
								if (lock_arr[pos >> 9].test_lock_set(version)){
									while(lock_arr[pos >> 9].test_lock_set(version)){
									};
								}
								ret = dindex.payloads[host_recv_buffer[tid][i].rbuffer[j] + k];
								if (lock_arr[pos >> 9].test_lock_version_change(version)){
									goto REGET2;
								}
							}else if(type_array[tid][i].ops[j] == 1){
								// UPDATE
								lock_arr[pos >> 9].get_lock();
								dindex.payloads[host_recv_buffer[tid][i].rbuffer[j] + k] = new_val;
								lock_arr[pos >> 9].release_lock();
							}
							break;
						}
					}
				}
				
			}
		}
	}
}

void inline insert_payload(int tid){
	string_payload* val = new string_payload('c');
	string_payload oval('c');
	int task_num;
	uint32_t interleave_page_offset = dindex.num_buf_pages / MERGE_THREAD_NUM;
	bool not_found;
	for(int i = 0; i < NR_DPUS; i++){
		task_num = host_recv_buffer[tid][i].n_tasks - 1;
		for(int j = 0; j < task_num; j++){
			not_found = true;
			// __builtin_prefetch(&(host_recv_buffer[tid][i].rbuffer[j + 4]), 0);
			if(host_recv_buffer[tid][i].rbuffer[j] != INVAILD_POS){
				// 有PIM算错的情况，需要再check一下partial key，但这样会多读一下cacheline
				__builtin_prefetch(&(dindex.buffer_pages[host_recv_buffer[tid][i].rbuffer[j + 4] / PAGE_RATIO]), 0);
				__builtin_prefetch(&(dindex.partial_keys[host_recv_buffer[tid][i].rbuffer[j + 4] / PAGE_RATIO]), 0);
				uint32_t ipos = host_recv_buffer[tid][i].rbuffer[j] / PAGE_RATIO;
				if(ipos < dindex.partial_total_size - 1 && dindex.partial_keys[ipos] <= host_send_buffer[tid][i].sbuffer[j] 
				&& dindex.partial_keys[ipos + 1] > host_send_buffer[tid][i].sbuffer[j]){ // 确认插入位置正确
					// dindex.buffer_pages[ipos].insert_page(host_send_buffer[tid][i].sbuffer[j], val);
					if(!dindex.buffer_pages[ipos].insert_page(host_send_buffer[tid][i].sbuffer[j], val)){
						// unequal++;
						dindex.overflow_trees[std::min(ipos / interleave_page_offset, (uint32_t)MERGE_THREAD_NUM - 1)].insert(host_send_buffer[tid][i].sbuffer[j], oval);
					};
				}
				// else{
				// 	unequal++;
				// }

			}else{
				// invaild_pos++;
				// unequal++;
				DTYPE insert_key = host_send_buffer[tid][i].sbuffer[j];
				int p = upper_dram_level.search_upper_level_get_partition(insert_key);
				if(p >= NR_PARTITION)
					p = NR_PARTITION - 1;
				if(p < 0)
					p = 0;
				auto range_ret = dindex.pgm_dram_index[p]->search(insert_key);
				unsigned long per_dpu_input_size = dindex.partial_total_size / NR_DPUS;
				size_t l = range_ret.lo + (per_dpu_input_size * p);
				size_t r = range_ret.hi + (per_dpu_input_size * p);
				if(r >= dindex.partial_total_size){
					r = dindex.partial_total_size - 1;
				}
				size_t m;
				while(l < r){
					m = l + (r - l) / 2;
					if(dindex.partial_keys[m] <= insert_key)
						l = m + 1;
					else
						r = m;
				}
				size_t cur_key_off = (l - 1) * INTERLEAVE / PAGE_RATIO;
				if(cur_key_off < dindex.partial_total_size - 1 && dindex.partial_keys[cur_key_off] <= insert_key 
				&& dindex.partial_keys[cur_key_off + 1] > insert_key){ // 确认插入位置正确
					if(!dindex.buffer_pages[cur_key_off].insert_page(insert_key, val)){
						// unequal++;
						dindex.overflow_trees[std::min(cur_key_off / interleave_page_offset, (uint64_t)MERGE_THREAD_NUM - 1)].insert(insert_key, oval);
					};
				}
				
			}
		}
	}
}

void inline scan_payload(int tid){
	int scan_length = 50;
	std::vector<std::pair<DTYPE, string_payload>> results;
	results.resize(scan_length);
	int task_num;
	uint32_t cur_pos;
	int cur_page_id, cur_result_off;
	DTYPE min_key;
	bool not_found;
	for(int i = 0; i < NR_DPUS; i++){
		task_num = host_recv_buffer[tid][i].n_tasks - 1;
		// total_task_num+= task_num;
		for(int j = 0; j < task_num; j++){
			not_found = true;
			// __builtin_prefetch(&(host_recv_buffer[tid][i].rbuffer[j + 4]), 0);
			if(host_recv_buffer[tid][i].rbuffer[j] != INVAILD_POS){
				for(int k = 0; k < INTERLEAVE; k++){
					__builtin_prefetch(&(dindex.keys[host_recv_buffer[tid][i].rbuffer[j + 4]]), 0);
					if((host_recv_buffer[tid][i].rbuffer[j] + k < dindex.total_size) && (dindex.keys[host_recv_buffer[tid][i].rbuffer[j] + k] == host_send_buffer[tid][i].sbuffer[j])){
						cur_pos = host_recv_buffer[tid][i].rbuffer[j] + k;
						not_found = false;
					}
				}

				if(!not_found){
					cur_page_id = cur_pos / PAGE_RATIO;
					min_key =  host_send_buffer[tid][i].sbuffer[j];
					cur_result_off = 0;
					int loop = 0;
					
					// 探查第一个page
					if(dindex.buffer_pages[cur_page_id].page_count != 0){
						// check page
						for(int pp = 0; pp < dindex.buffer_pages[cur_page_id].page_count; pp++){
							if(dindex.buffer_pages[cur_page_id].buf_key[pp] > min_key){
								// sort insert
								int pl = cur_result_off - INTERLEAVE;
								for(; pl < cur_result_off; pl++){
									if(results[pl].first > dindex.buffer_pages[cur_page_id].buf_key[pp]){
										break;
									}
								}
								// move data
								for(int mm = cur_result_off; mm > pl; mm--){
									results[mm] = results[mm - 1];
								}
								results[pl] = {dindex.buffer_pages[cur_page_id].buf_key[pp], *(dindex.buffer_pages[cur_page_id].buf_payload[pp])};
								loop++;
								cur_result_off++;
							}
						}
					}


					while(loop < scan_length && cur_pos < dindex.total_size){
						if(dindex.keys[cur_pos] > min_key){
							results[cur_result_off].first = dindex.keys[cur_pos];
							results[cur_result_off].second = dindex.payloads[cur_pos];
							loop++;
							cur_result_off++;
							
							// page has key?
							if(cur_pos % INTERLEAVE == 0){
								cur_page_id = cur_pos / PAGE_RATIO;
								if(dindex.buffer_pages[cur_page_id].page_count != 0){
									// check page
									for(int pp = 0; pp < dindex.buffer_pages[cur_page_id].page_count; pp++){
										if(dindex.buffer_pages[cur_page_id].buf_key[pp] > min_key){
											// sort insert
											int pl = cur_result_off - INTERLEAVE;
											for(; pl < cur_result_off; pl++){
												if(results[pl].first > dindex.buffer_pages[cur_page_id].buf_key[pp]){
													break;
												}
											}
											// move data
											for(int mm = cur_result_off; mm > pl; mm--){
												results[mm] = results[mm - 1];
											}
											results[pl] = {dindex.buffer_pages[cur_page_id].buf_key[pp], *(dindex.buffer_pages[cur_page_id].buf_payload[pp])};
											loop++;
											cur_result_off++;
										}
									}
								}
							}


						}
						cur_pos++;
					}
				}


			}else{
				// invaild_pos++;
				// unequal++;
			}
		}
	}
}

void inline search_payload_from_dram(DTYPE search_key, string_payload* ret){
	int p = upper_dram_level.search_upper_level_get_partition(search_key);
	if(p >= NR_PARTITION)
		p = NR_PARTITION - 1;
	if(p < 0)
		p = 0;
	auto range_ret = dindex.pgm_dram_index[p]->search(search_key);
	unsigned long per_dpu_input_size = dindex.partial_total_size / NR_PARTITION;
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
		if((cur_key_off + k < dindex.total_size) && dindex.keys[cur_key_off + k] == search_key){ //(cur_key_off + k < dindex.total_size) && 
			*ret = dindex.payloads[cur_key_off + k];
			break;
		}
	}
}


void inline merge_and_retrain(){
	std::vector<std::thread> merge_thread_array;
	merge_thread_array.reserve(MERGE_THREAD_NUM);
	size_t key_count[MERGE_THREAD_NUM];
	size_t merge_offset_array[MERGE_THREAD_NUM];

	uint64_t new_total_size = 0;
	uint64_t new_partial_total_size = 0;
	DTYPE* new_keys;
    string_payload* new_payloads;
	DTYPE* new_partial_keys;
	pthread_barrier_t bar1, bar2, bar3;
	pthread_barrier_init(&bar1, NULL, MERGE_THREAD_NUM); // MERGE_THREAD_NUM 个等待
	pthread_barrier_init(&bar2, NULL, MERGE_THREAD_NUM); 
	pthread_barrier_init(&bar3, NULL, MERGE_THREAD_NUM); 

	std::atomic<int> inc_id(0);

	TSCNS tn;
	tn.init();
	auto start_time = tn.rdtsc();
	auto end_time = tn.rdtsc();

	// 不要 cur_old_off = (dindex.total_size / MERGE_THREAD_NUM) * tid, 可能出现page不对齐的情况，这时候并行排序会出错

	for(int i = 0; i < MERGE_THREAD_NUM; i++){
	merge_thread_array.emplace_back(
	[&](size_t thread_id){
		int tid = inc_id.fetch_add(1);
		key_count[tid] = 0;
		// 1) check total buf size
		uint64_t fetch_key_count;
		uint32_t interleave_page_offset = dindex.num_buf_pages / MERGE_THREAD_NUM;
		uint32_t start_page_offset = interleave_page_offset * tid;
		uint32_t end_page_offset = interleave_page_offset * (tid + 1);
		if(tid == (MERGE_THREAD_NUM - 1))
		 end_page_offset = (uint32_t)(dindex.num_buf_pages);
		for(uint32_t j = start_page_offset; j < end_page_offset; j++){
			key_count[tid] += dindex.buffer_pages[j].page_count;
		}

		// std::cout << "thread id  " << tid << std::endl;
		pthread_barrier_wait(&bar1);

		if(tid == 0){
			size_t temp_total_size = 0;
			for(int jj = 0; jj < MERGE_THREAD_NUM; jj++){
				new_total_size += key_count[jj];
				merge_offset_array[jj] = temp_total_size; // 累加的offset
				temp_total_size += key_count[jj];
			}
			new_total_size += dindex.total_size;
			new_keys = new DTYPE[new_total_size];
			new_payloads = new string_payload[new_total_size];
			new_partial_total_size = new_total_size / INTERLEAVE;
			new_partial_keys = new DTYPE[new_partial_total_size];
		}

		pthread_barrier_wait(&bar2);

		// 2) merge start
		DTYPE old_key, buf_key;
		string_payload* old_value;
		string_payload* buf_value;
		// offset para
		// uint32_t cur_old_off = (dindex.total_size / MERGE_THREAD_NUM) * tid;
		// uint32_t cur_new_off = (dindex.total_size / MERGE_THREAD_NUM) * tid + merge_offset_array[tid];
		uint32_t page_id = (dindex.num_buf_pages / MERGE_THREAD_NUM) * tid;
		uint64_t in_page_off = 0;
		uint32_t max_page_id = (dindex.num_buf_pages / MERGE_THREAD_NUM) * (tid + 1);
		if(tid == (MERGE_THREAD_NUM - 1)){
			max_page_id = dindex.num_buf_pages;
		}
		// bound para
		uint32_t cur_old_off = page_id * PAGE_RATIO;
		uint32_t cur_new_off = cur_old_off + merge_offset_array[tid];
		uint32_t max_old_input_size = (max_page_id) * PAGE_RATIO; // TODO, last thread
		if(tid == (MERGE_THREAD_NUM - 1)){
			max_old_input_size = dindex.total_size;
		}

		uint64_t total_ops = max_old_input_size - cur_old_off + key_count[tid];
		if(tid == (MERGE_THREAD_NUM - 1)){
			total_ops = key_count[tid] + max_old_input_size - cur_old_off;
		}

		// uint32_t max_old_input_size = (dindex.total_size / MERGE_THREAD_NUM) * (tid + 1); // TODO, last thread
		// if(tid == (MERGE_THREAD_NUM - 1)){
		// 	max_old_input_size = dindex.total_size;
		// }
		// end flag
		uint32_t old_end_flag = 0;
		uint32_t buf_end_flag = 0;

		// get old_key and buf_key
		old_key = dindex.keys[cur_old_off];
		old_value = &(dindex.payloads[cur_old_off]);
		cur_old_off++;

		while(dindex.buffer_pages[page_id].page_count == 0){
			page_id++;
		}
		if(page_id < max_page_id){
			buf_key = dindex.buffer_pages[page_id].buf_key[in_page_off];
			// buf_value = &(dindex.buffer_pages[page_id].buf_payload[in_page_off]);
			buf_value = dindex.buffer_pages[page_id].buf_payload[in_page_off];
			in_page_off++;
		}else{
			buf_end_flag = 1;
		}

		// uint64_t total_ops = dindex.total_size / MERGE_THREAD_NUM + key_count[tid];
		

		// 3) start merge
		for(uint64_t targets = 0; targets < total_ops; targets++){
			__builtin_prefetch(&(dindex.keys[cur_old_off + 4]), 0);
			if(old_end_flag == 0 && buf_end_flag == 0){
				// normal compare
				if(old_key < buf_key){
					// inset old key
					new_keys[cur_new_off] = old_key;
					new_payloads[cur_new_off] = *old_value;
					cur_new_off++;
					// fetch new old_key
					if(cur_old_off < max_old_input_size){
						old_key = dindex.keys[cur_old_off];
						old_value = &(dindex.payloads[cur_old_off]);
						cur_old_off++;
					}else{
						old_end_flag = 1;
					}
				}else{
					// insert buf key
					if(old_key == buf_key){
						// update
						// fetch new old_key
						if(cur_old_off < max_old_input_size){
							old_key = dindex.keys[cur_old_off];
							old_value = &(dindex.payloads[cur_old_off]);
							cur_old_off++;
						}else{
							old_end_flag = 1;
						}
					}
					new_keys[cur_new_off] = buf_key;
					new_payloads[cur_new_off] = *buf_value;
					cur_new_off++;
					// fetch new buf_key
					if(in_page_off < dindex.buffer_pages[page_id].page_count){
						// still on current page
						buf_key = dindex.buffer_pages[page_id].buf_key[in_page_off];
						// buf_value = &(dindex.buffer_pages[page_id].buf_payload[in_page_off]);
						buf_value = dindex.buffer_pages[page_id].buf_payload[in_page_off];
						in_page_off++;
					}else{
						// get next page
						page_id++;
						in_page_off = 0;
						while(dindex.buffer_pages[page_id].page_count == 0){
							page_id++;
						}
						if(page_id < max_page_id){
							buf_key = dindex.buffer_pages[page_id].buf_key[in_page_off];
							buf_value = dindex.buffer_pages[page_id].buf_payload[in_page_off];
							// buf_value = &(dindex.buffer_pages[page_id].buf_payload[in_page_off]);
							in_page_off++;
						}else{
							buf_end_flag = 1;
						}
					}
				}
			}else if(old_end_flag == 0 && buf_end_flag == 1){
				// only left old key
				new_keys[cur_new_off] = old_key;
				new_payloads[cur_new_off] = *old_value;
				cur_new_off++;

				if(cur_old_off < dindex.total_size){
					old_key = dindex.keys[cur_old_off];
					old_value = &(dindex.payloads[cur_old_off]);
					cur_old_off++;
				}else{
					old_end_flag = 1;
				}


			}else if(old_end_flag == 1 && buf_end_flag == 0){
				// only left buf key
				new_keys[cur_new_off] = buf_key;
				new_payloads[cur_new_off] = *buf_value;
				cur_new_off++;
				// fetch new buf_key
				if(in_page_off < dindex.buffer_pages[page_id].page_count){
					// still on current page
					buf_key = dindex.buffer_pages[page_id].buf_key[in_page_off];
					// buf_value = &(dindex.buffer_pages[page_id].buf_payload[in_page_off]);
					buf_value = dindex.buffer_pages[page_id].buf_payload[in_page_off];
					in_page_off++;
				}else{
					// get next page
					page_id++;
					in_page_off = 0;
					while(dindex.buffer_pages[page_id].page_count == 0){
						page_id++;
					}
					if(page_id < max_page_id){
						buf_key = dindex.buffer_pages[page_id].buf_key[in_page_off];
						// buf_value = &(dindex.buffer_pages[page_id].buf_payload[in_page_off]);
						buf_value = dindex.buffer_pages[page_id].buf_payload[in_page_off];
						in_page_off++;
					}else{
						buf_end_flag = 1;
					}
				}

			}else{
				//error
				printf("error in merge \n");
			}
			// end loop
		}

		pthread_barrier_wait(&bar3);

		// 4) create partial keys
		uint64_t processing_num_keys = new_total_size / MERGE_THREAD_NUM - ((new_total_size / MERGE_THREAD_NUM) % INTERLEAVE);
		if(processing_num_keys % INTERLEAVE != 0){
			std::cout << "processing_num_keys is not align with INTERLEAVE" << std::endl;
		}
		uint64_t partital_keys_array_offset = (processing_num_keys * tid) / INTERLEAVE;
		uint64_t start_processing_key_offset = processing_num_keys * tid;
		uint64_t end_processing_key_offset = processing_num_keys * (tid + 1);
		if(tid == MERGE_THREAD_NUM - 1){
			end_processing_key_offset = new_total_size;
		}
		for(int p = start_processing_key_offset; p < end_processing_key_offset; p+= INTERLEAVE){
			__builtin_prefetch(&(new_keys[p + 8]), 0);
			new_partial_keys[partital_keys_array_offset] = new_keys[p];
			partital_keys_array_offset++;
		}

	}, i);

	}

	for (auto &t : merge_thread_array) {
			t.join();
	}

	// end_time = tn.rdtsc();
	// auto diff = tn.tsc2ns(end_time) - tn.tsc2ns(start_time);
	// std::cout << "merge time (second) " << (diff/(double) 1000000000) << std::endl;

	// std::cout << "new_total_size " << new_total_size << std::endl;
	// if(std::is_sorted(new_keys, new_keys + new_total_size - 10))
	// 	std::cout << "is sort " << std::endl;

	// for(int kk = 0; kk < MERGE_THREAD_NUM; kk++){
	// 	std::cout << kk << " " << begin_pos[kk] << " " << end_pos[kk] << std::endl;
	// 	if(std::is_sorted(new_keys+ begin_pos[kk], new_keys + end_pos[kk]))
	// 		std::cout << kk << " is sort " << std::endl;
	// 	else
	// 		std::cout << kk << " not sort !!" << std::endl;
		
	// 	if(kk != MERGE_THREAD_NUM - 1)
	// 	std::cout << "key1 " << new_keys[end_pos[kk] - 1] << " key 2 " << new_keys[end_pos[kk]] << std::endl;
	// }

	// for(int kk = 0; kk < MERGE_THREAD_NUM; kk++){
	// 	std::cout << kk << " page " << begin_page[kk] << " " << end_page[kk] << std::endl;
	// }

	// for(int kk = 0; kk < MERGE_THREAD_NUM; kk++){
	// 	std::cout << kk << " old " << begin_old[kk] << " " << end_old[kk] << std::endl;
	// }
	
	// // debug for partial key
	// DTYPE* test_partial_keys = new DTYPE[new_partial_total_size];
	// int tp = 0;
	// for(int h = 0; h < new_total_size; h+=INTERLEAVE){
	// 	test_partial_keys[tp] = new_keys[h];
	// 	tp++;
	// }
	// for(int h = 0; h < new_partial_total_size; h++){
	// 	if(test_partial_keys[h] != new_partial_keys[h]){
	// 		std::cout << "partial key is not equal " << h << std::endl;
	// 	}
	// }
	

	// Set dindex
	/***** first lock ******/
	// dindex_lock.get_lock();
	// upper_level_lock.get_lock();
	/***** set dindex *****/
	// delete[] dindex.partial_keys; // delete background?
	dindex.partial_keys = new_partial_keys;
	// delete[] dindex.keys;
	dindex.keys = new_keys;
	// delete[] dindex.payloads;
	dindex.payloads = new_payloads;
	dindex.total_size = new_total_size;
	dindex.partial_total_size = new_partial_total_size;
	dindex.min_key = dindex.keys[0];
	// delete[] dindex.buffer_pages;
	dindex.num_buf_pages = dindex.total_size / PAGE_RATIO + 1;
	// dindex.buffer_pages = new buf_page[dindex.num_buf_pages];
	// start_time = tn.rdtsc();
	dindex.buffer_pages = new buf_page[dindex.num_buf_pages];
	#pragma omp parallel for schedule(dynamic, 1000) num_threads(32) 
	for(int i = 0; i < dindex.num_buf_pages; i++){
		__builtin_prefetch(&(dindex.buffer_pages[i + 2]), 0);
		dindex.buffer_pages[i].page_count = 0;
    	dindex.buffer_pages[i].lock_ = 0;
	}
	// end_time = tn.rdtsc();
	// diff = tn.tsc2ns(end_time) - tn.tsc2ns(start_time);
	// std::cout << "tranin time (second) " << (diff/(double) 1000000000) << std::endl;

	
	/**** train ******/
	int push_model_size = 0;
	int start = 0;
	size_t init_Epsilon = 128;// DataEpsilon;
	unsigned long per_dpu_input_size = dindex.partial_total_size / NR_PARTITION;
	for(int kk = 0; kk < NR_PARTITION; kk++){
		// 建立DRAM上的PGM, 初始化误差范围为64
		// delete dindex.pgm_dram_index[kk];
		dindex.pgm_dram_index[kk] = new pgm::PGMIndex<DTYPE> (dindex.partial_keys + start, dindex.partial_keys + start + per_dpu_input_size, init_Epsilon);

		dindex.partiton_epsilon[kk] = init_Epsilon;
		//填充DRAM level 设置边界key
		upper_dram_level.low_bound_key[kk] = dindex.partial_keys[start];

		// 将模型和数据装填到DPU
		// 1. Create kernel arguments
		input_arguments[kk] = {per_dpu_input_size, dpu_arguments_t::kernels(0)};
		input_arguments[kk].start_pos = static_cast<uint32_t>(start);
		input_arguments[kk].max_levels = dindex.pgm_dram_index[kk]->height();
		input_arguments[kk].setEpsilon = init_Epsilon;
		for(int i = 0; i < dindex.pgm_dram_index[kk]->levels_offsets.size(); i++){
			input_arguments[kk].level_offset[i] = dindex.pgm_dram_index[kk]->levels_offsets[i];
			// std::cout << "level offset " << i << " " << input_arguments[kk].level_offset[i] << std::endl;
		}

		if(dindex.pgm_dram_index[kk]->levels_offsets.size() > push_model_size){
			push_model_size = dindex.pgm_dram_index[kk]->levels_offsets.size();
		}

		dindex.dram_start_pos[kk] = static_cast<uint32_t>(start);
		start += per_dpu_input_size;

		// 2. package pgm index
		if(dindex.pgm_dram_index[kk]->segments.size() > MAX_MODEL_SIZE){
			std::cout << "PGM model size is larger than MAX_MODEL_SIZE" << std::endl; 
			assert(0);
		}

		// delete[] dindex.transfer_model[kk];
		// dindex.transfer_model[kk] = new pgm_model_t[MAX_MODEL_SIZE];
		// for(int i = 0; i < dindex.pgm_dram_index[kk]->segments.size(); i++){
		// 	dindex.transfer_model[kk][i].key = dindex.pgm_dram_index[kk]->segments[i].key;
		// 	dindex.transfer_model[kk][i].slope = dindex.pgm_dram_index[kk]->segments[i].slope;
		// 	dindex.transfer_model[kk][i].intercept = dindex.pgm_dram_index[kk]->segments[i].intercept;
		// }
	}

	// start_time = tn.rdtsc();

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
	i = 0;
	replicas_info->reset();
	DPU_FOREACH(dpu_set, dpu, i)
	{
		cur_p = replicas_info->next_partition();
		DPU_ASSERT(dpu_prepare_xfer(dpu, dindex.partial_keys + cur_p * per_dpu_input_size));
	}
	DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, per_dpu_input_size * sizeof(DTYPE), DPU_XFER_DEFAULT));
	
	// 装填模型
	i = 0;
	replicas_info->reset();
	DPU_FOREACH(dpu_set, dpu, i)
	{
		cur_p = replicas_info->next_partition();
		DPU_ASSERT(dpu_prepare_xfer(dpu, dindex.pgm_dram_index[cur_p]->segments.data())); // dindex.transfer_model[cur_p]
	}
	DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "model_array", 0,  push_model_size * sizeof(pgm_model_t), DPU_XFER_DEFAULT)); //MAX_MODEL_SIZE


	// dindex_lock.release_lock();
	// upper_level_lock.release_lock();

	end_time = tn.rdtsc();
	auto diff = tn.tsc2ns(end_time) - tn.tsc2ns(start_time);
	std::cout << "transfer time (second) " << (diff/(double) 1000000000) << std::endl;
}


// Main of the Host Application
int main(int argc, char **argv) {
    auto flags = parse_flags(argc, argv);
    std::string keys_file_path = get_required(flags, "keys_file");
    auto init_num_keys = stoi(get_required(flags, "init_num_keys"));
	uint64_t num_querys = stoi(get_required(flags, "query_num"));
	bool is_insert = get_boolean_flag(flags, "insert"); // 默认为search only
	bool is_mix = get_boolean_flag(flags, "mix"); // 默认为search only
	bool is_search = get_boolean_flag(flags, "search"); // 默认为search only
	bool is_scan = get_boolean_flag(flags, "scan"); // 默认为search only
	bool is_predecessor = get_boolean_flag(flags, "predecessor"); // 默认为search only
	auto total_num_keys = stoi(get_required(flags, "total_num_keys"));
	std::string sample_distribution = get_with_default(flags, "sample_distribution", "uniform");
	output_path = get_with_default(flags, "output_path", "./out.csv");

	if(is_search){
		is_pure_read = true;
	}else{
		is_pure_read = false;
	}
	
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
	char* ops = (char*) malloc((num_querys) * sizeof(char));

	std::cout << "input size " << input_size << " query num " << num_querys;

	// Create an input file with arbitrary data
	if(is_insert){
		// INSERT
		create_insert_op(keys, querys, input_size, num_querys, total_num_keys);
	}else if(is_mix){
		// MIXED
		if (sample_distribution == "uniform") {
			// generate ops
			std::mt19937_64 gen(std::random_device{}());
			std::uniform_int_distribution<int> dis(0, input_size - 1);
			int insert_pos = init_num_keys;
			std::mt19937 gen1;
			std::uniform_real_distribution<> ratio_dis(0, 1);
			double prob;
			for(int i = 0; i < num_querys; i++){
				prob = ratio_dis(gen1);
				if(prob < 0.5){ // GET ratio
					int pos = dis(gen);
					querys[i] = keys[pos];
					ops[i] = 0;
				}
				else if (prob < 0.75){ //UPDATE 
					int pos = dis(gen);
					querys[i] = keys[pos];
					ops[i] = 1;
				}else{
					//INSERT
					if(insert_pos < total_num_keys  && keys[insert_pos] > dindex.min_key){
						querys[i] = keys[insert_pos];
						insert_pos++;
						ops[i] = 2;
					}else{
						//填充空位
						while(insert_pos < total_num_keys && keys[insert_pos] < dindex.min_key){
							insert_pos++;
						}
						if(insert_pos < total_num_keys){
							querys[i] = keys[insert_pos];
							insert_pos++;
							ops[i] = 2;
						}else{
							querys[i] = dindex.min_key + 1;
							ops[i] = 0;
						}
					}
				}
			}
        } else if (sample_distribution == "zipf") {
			std::cout << "enter mix zipf " << std::endl;
			// generate ops
			ScrambledZipfianGenerator zipf_gen(input_size, nullptr);
			int insert_pos = init_num_keys;
			std::mt19937 gen1;
			std::uniform_real_distribution<> ratio_dis(0, 1);
			double prob;
			for(int i = 0; i < num_querys; i++){
				prob = ratio_dis(gen1);
				if(prob < 0.5){ // GET ratio
					int pos = zipf_gen.nextValue();
					querys[i] = keys[pos];
					ops[i] = 0;
				}
				else if (prob < 0.75){ //UPDATE 0.55(0.5 get, 0.05 update, 0.45insert, >>9 lock, 性能较好)
					int pos = zipf_gen.nextValue();
					querys[i] = keys[pos];
					ops[i] = 1;
				}else{
					//INSERT
					if(insert_pos < total_num_keys  && keys[insert_pos] > dindex.min_key){
						querys[i] = keys[insert_pos];
						insert_pos++;
						ops[i] = 2;
					}else{
						//填充空位
						while(insert_pos < total_num_keys && keys[insert_pos] < dindex.min_key){
							insert_pos++;
						}
						if(insert_pos < total_num_keys){
							querys[i] = keys[insert_pos];
							insert_pos++;
							ops[i] = 2;
						}else{
							querys[i] = dindex.min_key + 1;
							ops[i] = 0;
						}
					}
				}
			}
        }
	}else if(is_scan){
		// SCAN
		if (sample_distribution == "uniform") {
			uint64_t elements_num = (input_size / INTERLEAVE / NR_PARTITION) * NR_PARTITION * INTERLEAVE;
			std::cout << "element number " << elements_num << std::endl;
            create_query(keys, querys, elements_num, num_querys);
        } else if (sample_distribution == "zipf") {
            create_query_zipf(keys, querys, input_size, num_querys);
        }
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

	delete[] keys;
	

	/*****Warm UP*****/ 
	int wtid = 0;	
	int wpd = 32;
	int wdpu_idx;
	for(int req = 0; req < 1000000; req++){
		__builtin_prefetch(&querys[req + wpd], 0);
		wdpu_idx = upper_dram_level.search_upper_level(querys[req], req, wtid, true);		
	}
	partition_access_level.calculate_partition_skew(NR_DPUS / NR_PARTITION, 1000000, wtid);
	load_hot_replica(wtid);
	/* clear wtid */


	// thread 
	std::vector<std::thread> thread_array;
	thread_array.reserve(THREAD_NUM);
	int per_thread_ops = num_querys / THREAD_NUM;
	int per_round_ops = per_thread_ops / 2;

	
	TSCNS tn;
	tn.init();
	printf("Begin running\n");
	auto start_time = tn.rdtsc();
	auto end_time = tn.rdtsc();

	if(is_insert){
		// INSERT
		for(int i = 0; i < THREAD_NUM; i++){
			thread_array.emplace_back(
			[&](size_t thread_id){
				// thread content
				int tid = Coremeta::threadID();
				int start_ops = per_thread_ops * tid;
				int end_ops = per_thread_ops * (tid + 1);
				
				int h;
				int fetch_write_num, fetch_write_thread_num;
				std::pair<int, int> buffer_size;
				int pd = 32;
				int dpu_idx, slot_idx;
				uint32_t version;

				int round_start_ops, round_end_ops;
			
				for(int round = 0; round < 2; round++){ // round = 2, 100M ops这里影响性能
					round_start_ops = start_ops + per_round_ops * round;
					round_end_ops = start_ops + per_round_ops * (round + 1);
			IRETRY:
				merge_shard_lock.lock();
				merge_shard_lock.unlock();
				if (merge_lock.test_lock_set(version)){
					while(merge_lock.test_lock_set(version)){
						// usleep(20);
					};
				} // Test whether the lock is set and record the version

				for(int req = round_start_ops; req < round_end_ops; req++){
					__builtin_prefetch(&querys[req + pd], 0);
					if(querys[req] > dindex.min_key){
						dpu_idx = upper_dram_level.search_upper_level(querys[req], req, tid, false);
						slot_idx = host_send_buffer[tid][dpu_idx].n_tasks;
						host_send_buffer[tid][dpu_idx].n_tasks++;
						host_send_buffer[tid][dpu_idx].sbuffer[slot_idx] = querys[req];
					}
				}

				if (merge_lock.test_lock_version_change(version)){
					clear_send_buffer(tid);
					goto IRETRY;
				} // Test whether the version is changed or not

				// set op type
				h = 0;
				DPU_FOREACH(dpu_set, dpu, h)
				{
					host_send_buffer[tid][h].op_type = 1; //LOOKUP
				}
				buffer_size = max_buffer_size_search(tid);

				// std::cout << "max buffer size " << buffer_size.first / 8 << " expected buffer size " << (end_ops - start_ops) / NR_DPUS << std::endl;

				// execuse
				lock.lock();
				dpu_sync(tid, buffer_size.first);
				get_index_lookup_result(tid, buffer_size.second);
				lock.unlock();

				fetch_write_num = current_write_num.fetch_add(per_round_ops);
				// std::cout << "fetch_write_num " << fetch_write_num << std::endl;
				if((double)fetch_write_num > (double)dindex.total_size * 0.45){
					// enter merge
					if(!merge_lock.try_get_lock()){
						// some thread has obtain lock
						// current_write_num -= per_round_ops;
						clear_send_buffer(tid);
						goto IRETRY;
					}
					merge_shard_lock.lock();

					while(thread_in_write.load() != 0);

					// std::cout << "enter merge" << std::endl;
					merge_and_retrain();
					// std::cout << "enter end" << std::endl;
					current_write_num.store(0);

					merge_shard_lock.unlock();
					merge_lock.release_lock();
					clear_send_buffer(tid);
					goto IRETRY;				
				}

				// auto get_start_time = std::chrono::high_resolution_clock::now();
				thread_in_write++;
				insert_payload(tid);
				thread_in_write--;
				// auto get_end_time = std::chrono::high_resolution_clock::now();
				// double dram_time =std::chrono::duration_cast<std::chrono::nanoseconds>(get_end_time -
				//                                              get_start_time).count();
				// std::cout << "dram search time " << dram_time / 1e9 << " s" << std::endl;

				}
			}
			,i);
		}

		for (auto &t : thread_array) {
				t.join();
		}

		// merge_and_retrain();
		// std::cout << "retrain end" << std::endl;

		//  2) prepare thread array and buffer
		// thread_array.clear();
		// per_thread_ops = num_querys / THREAD_NUM / 2;
		// Coremeta::reset();
		// clear_all_buffer();
		// for(int i = 0; i < THREAD_NUM; i++){
		// 	thread_array.emplace_back(
		// 	[&](size_t thread_id){
		// 		// thread content
		// 		int tid = Coremeta::threadID();
		// 		int start_ops = per_thread_ops * tid + num_querys / 2;
		// 		int end_ops = per_thread_ops * (tid + 1) + num_querys / 2;
				
		// 		int pd = 32;
		// 		int dpu_idx, slot_idx;
		// 		for(int req = start_ops; req < end_ops; req++){
		// 			__builtin_prefetch(&querys[req + pd], 0);
		// 			if(querys[req] > dindex.min_key){
		// 				dpu_idx = upper_dram_level.search_upper_level(querys[req], req, tid, false);
		// 				slot_idx = host_send_buffer[tid][dpu_idx].n_tasks;
		// 				host_send_buffer[tid][dpu_idx].n_tasks++;
		// 				host_send_buffer[tid][dpu_idx].sbuffer[slot_idx] = querys[req];
		// 			}
		// 		}
		// 		// set op type
		// 		int h = 0;
		// 		DPU_FOREACH(dpu_set, dpu, h)
		// 		{
		// 			host_send_buffer[tid][h].op_type = 1; //LOOKUP
		// 		}
		// 		std::pair<int, int> buffer_size = max_buffer_size_search(tid);

		// 		// std::cout << "max buffer size " << buffer_size.first / 8 << " expected buffer size " << (end_ops - start_ops) / NR_DPUS << std::endl;

		// 		// execuse
		// 		lock.lock();
		// 		dpu_sync(tid, buffer_size.first);
		// 		get_index_lookup_result(tid, buffer_size.second);
		// 		lock.unlock();

		// 		// auto get_start_time = std::chrono::high_resolution_clock::now();
		// 		insert_payload(tid);
		// 		// auto get_end_time = std::chrono::high_resolution_clock::now();
		// 		// double dram_time =std::chrono::duration_cast<std::chrono::nanoseconds>(get_end_time -
		// 		//                                              get_start_time).count();
		// 		// std::cout << "dram search time " << dram_time / 1e9 << " s" << std::endl;

			
		// 	}
		// 	,i);
		// }

		// for (auto &t : thread_array) {
		// 		t.join();
		// }


	}else if(is_mix){
		// MIX
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
					__builtin_prefetch(&ops[req + pd], 0);
					if(querys[req] > dindex.min_key){
						dpu_idx = upper_dram_level.search_upper_level(querys[req], req, tid, false);
						slot_idx = host_send_buffer[tid][dpu_idx].n_tasks;
						host_send_buffer[tid][dpu_idx].n_tasks++;
						host_send_buffer[tid][dpu_idx].sbuffer[slot_idx] = querys[req];
						type_array[tid][dpu_idx].ops[slot_idx] = ops[req];
					}
				}
				// set op type
				int h = 0;
				DPU_FOREACH(dpu_set, dpu, h)
				{
					host_send_buffer[tid][h].op_type = 1; //LOOKUP
				}
				std::pair<int, int> buffer_size = max_buffer_size_search(tid);

				// execuse
				lock.lock();
				dpu_sync(tid, buffer_size.first);
				get_index_lookup_result(tid, buffer_size.second);
				lock.unlock();

				// auto get_start_time = std::chrono::high_resolution_clock::now();
				mix_payload(tid);
				// auto get_end_time = std::chrono::high_resolution_clock::now();
    			// double dram_time =std::chrono::duration_cast<std::chrono::nanoseconds>(get_end_time -
                //                                              get_start_time).count();
    			// std::cout << "dram search time " << dram_time / 1e9 << " s" << std::endl;

			}
			,i);
		}

		for (auto &t : thread_array) {
				t.join();
		}

	}else if(is_scan){
		// SCAN range query
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

				// execuse
				lock.lock();
				dpu_sync(tid, buffer_size.first);
				get_index_lookup_result(tid, buffer_size.second);
				lock.unlock();

				scan_payload(tid);

			}
			,i);
		}

		for (auto &t : thread_array) {
				t.join();
		}
	}
	else if(is_predecessor){
		// PREDECESSOR
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

				// execuse
				lock.lock();
				dpu_sync(tid, buffer_size.first);
				get_index_lookup_result(tid, buffer_size.second);
				lock.unlock();

				predecessor_real_payload(tid);

			}
			,i);
		}

		for (auto &t : thread_array) {
				t.join();
		}
	}
	else{
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
	std::cout << "invaild pos number " << invaild_pos << std::endl;

	end_time = tn.rdtsc();
	auto diff = tn.tsc2ns(end_time) - tn.tsc2ns(start_time);
	std::cout << "before change run time (second) " << (diff/(double) 1000000000) << std::endl;
	std::cout << "befor change throughput (second) " << num_querys / (diff/(double) 1000000000) << std::endl;


	/******* hotspot change ******/
// 	// 1) create change workload. zipf --> uniform
// 	uint64_t elements_num = (input_size / INTERLEAVE / NR_PARTITION) * NR_PARTITION * INTERLEAVE;
// 	std::cout << "element number " << elements_num << std::endl;
// 	create_query(keys, querys, elements_num, num_querys);
// 	// size_t newseed = 1866;
// 	// create_query_zipf(keys, querys, input_size, num_querys, &newseed);
// 	// 2) prepare thread array and buffer
// 	thread_array.clear();
// 	per_thread_ops = num_querys / THREAD_NUM;
// 	Coremeta::reset();
// 	clear_all_buffer();
// 	// 3) run
// 	printf("Begin running after hotspot change\n");
// 	start_time = tn.rdtsc();
// 	end_time = tn.rdtsc();

// 	if(is_insert){
// 		// INSERT
// 	}else{
// 		// SEARCH
// 		for(int i = 0; i < THREAD_NUM; i++){
// 			thread_array.emplace_back(
// 			[&](size_t thread_id){
// 				// thread content
// 				int tid = Coremeta::threadID();
// 				int start_ops = per_thread_ops * tid;
// 				int end_ops = per_thread_ops * (tid + 1);

// 				int h;
// 				std::pair<int, int> buffer_size;
				
// 				int pd = 32;
// 				int dpu_idx, slot_idx;

// 				uint32_t version;
// 				string_payload payload;
// 				int dram_end_ops;
// 			RETRY:
// 				if (ol_lock.test_lock_set(version)){
// 					while(ol_lock.test_lock_set(version)){
// 						dram_end_ops = std::min(start_ops + 100, end_ops);
// 						for(int k = start_ops; k < dram_end_ops; k++){
// 							search_payload_from_dram(querys[k], &payload);
// 						}
// 						start_ops = std::min(start_ops + 100, end_ops);
// 						// usleep(5);
// 					};
// 				} // Test whether the lock is set and record the version

// 				for(int req = start_ops; req < end_ops; req++){
// 					__builtin_prefetch(&querys[req + pd], 0);
// 					dpu_idx = upper_dram_level.search_upper_level(querys[req], req, tid, false);
// 					slot_idx = host_send_buffer[tid][dpu_idx].n_tasks;
// 					host_send_buffer[tid][dpu_idx].n_tasks++;
// 					host_send_buffer[tid][dpu_idx].sbuffer[slot_idx] = querys[req];
// 				}

// 				if (ol_lock.test_lock_version_change(version)){
// 					clear_send_buffer(tid);
// 					goto RETRY;
// 				} // Test whether the version is changed or not

// 				// set op type
// 				h = 0;
// 				DPU_FOREACH(dpu_set, dpu, h)
// 				{
// 					host_send_buffer[tid][h].op_type = 1; //LOOKUP
// 				}
// 				buffer_size = max_buffer_size_search(tid);

// 				std::cout << "max buffer size " << buffer_size.first / 8 << std::endl;

// 				/** judge Load Balance !!! **/
// 				if((buffer_size.first / 8) > (1.3 * (end_ops - start_ops) / NR_DPUS)){
// 					//Find load unbalance
// 					//(1) lock the upper dram level
// 					if (!ol_lock.try_get_lock()) {
// 						// some thread has obtained lock and change hotspot
// 						clear_send_buffer(tid);
// 						goto RETRY;
// 					}

// 					// (2) get new hotspot info
// 					partition_access_level.reset_by_thread(tid);
// 					gather_partition_access_from_buffer(tid);
// 					partition_access_level.calculate_partition_skew(NR_DPUS / NR_PARTITION, end_ops - start_ops, tid);
// 					load_hot_replica(tid);


// 					// (4)release lock
// 					ol_lock.release_lock();
// 					clear_send_buffer(tid);
// 					goto RETRY;
// 				}

// 				// std::cout << "max buffer size " << buffer_size.first / 8 << " expected buffer size " << (end_ops - start_ops) / NR_DPUS << std::endl;

// 				// execuse
// 				lock.lock();
// 				dpu_sync(tid, buffer_size.first);
// 				get_index_lookup_result(tid, buffer_size.second);
// 				lock.unlock();

// 				// auto get_start_time = std::chrono::high_resolution_clock::now();
// 				get_real_payload(tid);
// 				// auto get_end_time = std::chrono::high_resolution_clock::now();
//     			// double dram_time =std::chrono::duration_cast<std::chrono::nanoseconds>(get_end_time -
//                 //                                              get_start_time).count();
//     			// std::cout << "dram search time " << dram_time / 1e9 << " s" << std::endl;

				
// #if PRINT
// 				unsigned int each_dpu = 0;
// 				printf("Display DPU Logs\n");
// 				DPU_FOREACH(dpu_set, dpu)
// 				{
// 					printf("DPU#%d:\n", each_dpu);
// 					DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
// 					each_dpu++;
// 				}
// #endif
// 			}
// 			,i);
// 		}

// 		for (auto &t : thread_array) {
// 				t.join();
// 		}
// 	}

// 	std::cout << "uequal number " << unequal << std::endl;

// 	end_time = tn.rdtsc();
// 	diff = tn.tsc2ns(end_time) - tn.tsc2ns(start_time);
// 	std::cout << "after change run time (second) " << (diff/(double) 1000000000) << std::endl;
// 	std::cout << "after change throughput (second) " << num_querys / (diff/(double) 1000000000) << std::endl;



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