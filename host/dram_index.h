#ifndef _DRAM_INDEX_H_
#define _DRAM_INDEX_H_

#include "./pgm_index.hpp"
#include <iostream>
#include "../support/common.h"
#include "concurrent.h"
#include <mutex>
#include <shared_mutex>
#include "../support/overflow_tree.h"

class string_payload{
    public:
    char real_payload[32];
    string_payload(){
        // memset(real_payload, '0', 32);
    }
    string_payload(int i){
        memset(real_payload, i, 32);
    }
    string_payload& operator =(const string_payload& str)//赋值运算符 
    {
        memcpy(this->real_payload, str.real_payload, 32);
        return *this;
    }
};

struct entry_t {
    DTYPE buf_key;
    string_payload* buf_payload;
};

#define BUF_SIZE 16
class buf_page{
    public:
    uint32_t page_count;
    uint32_t lock_;
    // ol_lock plock;
    DTYPE buf_key[BUF_SIZE];
    string_payload* buf_payload[BUF_SIZE];
    // entry_t kv[BUF_SIZE];
    buf_page(int i){
        page_count = 0;
        lock_ = 0;
    }

    inline void init(){
       page_count = 0;
        lock_ = 0; 
    }

    buf_page(){
    }

    /*** concurrency management **/
    inline void get_lock() {
    uint32_t new_value = 0;
    uint32_t old_value = 0;
    do {
        while (true) {
        old_value = __atomic_load_n(&lock_, __ATOMIC_ACQUIRE);
        if (!(old_value & lockSet)) {
            old_value &= lockMask;
            break;
        }
        }
        new_value = old_value | lockSet;
    } while (!CAS(&lock_, &old_value, new_value));
    }

    inline void release_lock() {
    uint32_t v = lock_;
    __atomic_store_n(&lock_, v + 1 - lockSet, __ATOMIC_RELEASE);
    }

    /*if the lock is set, return true*/
    inline bool test_lock_set(uint32_t &version) {
    version = __atomic_load_n(&lock_, __ATOMIC_ACQUIRE);
    return (version & lockSet) != 0;
    }

    // test whether the version has change, if change, return true
    inline bool test_lock_version_change(uint32_t old_version) {
    auto value = __atomic_load_n(&lock_, __ATOMIC_ACQUIRE);
    return (old_version != value);
    }

    bool insert_page(DTYPE& key, string_payload* val){
        get_lock();
        if(page_count >= BUF_SIZE){
            release_lock();
            return false;
        }
        if(page_count == 0){
            buf_key[0] = key;
            buf_payload[0] = new string_payload(); 
            *buf_payload[0] = *val;
            // kv[0].buf_key = key;
            // kv[0].buf_payload = new string_payload(); 
            // *(kv[0].buf_payload) = *val;
            page_count++;
            release_lock();
            return true;
        }
        __builtin_prefetch(buf_payload, 0);

        // linear search
        // int pl = 0;
        // for(; pl < page_count; pl++){
        //     if(buf_key[pl] > key){
        //         break;
        //     }
        // }
        // binary search
        int pl = 0, pr = page_count, pmid;
        while(pl < pr){
            pmid = pl + (pr - pl) / 2;
            if(buf_key[pmid] <= key){
                pl = pmid + 1;
            }else{
                pr = pmid;
            }
        }

        // // insert to pl
        memmove(&(buf_key[pl + 1]), &(buf_key[pl]), sizeof(DTYPE) * (page_count - pl));
        memmove(&(buf_payload[pl + 1]), &(buf_payload[pl]), sizeof(string_payload*) * (page_count - pl));
        // memmove(&(kv[pl + 1]), &(kv[pl]), sizeof(entry_t) * (page_count - pl));
        buf_key[pl] = key;
        buf_payload[pl] = new string_payload(); 
        *buf_payload[pl] = *val;
        // kv[pl].buf_key = key;
        // kv[pl].buf_payload = new string_payload(); 
        // *(kv[pl].buf_payload) = *val;
        page_count++;
        release_lock();
        return true;
    }

    bool find_key_page(DTYPE& key, string_payload* val){
        int pl, pr, pmid;
    //     uint32_t version;
    // FindRETRY:
    //     if (test_lock_set(
    //             version)){
    //         while(test_lock_set(version)){}
    //     } // Test whether the lock is set and record the version

        get_lock();
        if(page_count == 0){
            // if (test_lock_version_change(version)){
			// 		goto FindRETRY;
			// } // Test whether the version is changed or not
            release_lock();
            return false;
        }
        pl = 0;
        pr = page_count;
        while(pl < pr){
            pmid = pl + (pr - pl) / 2;
            if(buf_key[pmid] <= key){
                pl = pmid + 1;
            }else{
                pr = pmid;
            }
        }
        // *val = buf_payload[pl - 1];
        val = buf_payload[pl - 1];
        release_lock();
        // if (test_lock_version_change(version)){
        //         goto FindRETRY;
        // } // Test whether the version is changed or not
        return true;
    }

    bool predecessor_key_page(DTYPE& key, string_payload* val, DTYPE* pkey){
        uint32_t version;
        int pl, pr, pmid;
    PreRETRY:
        if (test_lock_set(
                version)){
            while(test_lock_set(version)){}
        } // Test whether the lock is set and record the version

        if(page_count == 0){
            if (test_lock_version_change(version)){
					goto PreRETRY;
			} // Test whether the version is changed or not
            return false;
        }
        pl = 0;
        pr = page_count;
        while(pl < pr){
            pmid = pl + (pr - pl) / 2;
            if(buf_key[pmid] <= key){
                pl = pmid + 1;
            }else{
                pr = pmid;
            }
        }
        if(pl - 1 > 0){
            *pkey = buf_key[pl - 2];
            val = buf_payload[pl - 2];
            if (test_lock_version_change(version)){
					goto PreRETRY;
			} // Test whether the version is changed or not
            return true;
        }
        if (test_lock_version_change(version)){
                goto PreRETRY;
        } // Test whether the version is changed or not
        return false;
    }

    bool update_key_page(DTYPE& key, string_payload* val){
        get_lock();
        if(page_count == 0){
            release_lock();
            return false;
        }
        int pl = 0, pr = page_count, pmid;
        while(pl < pr){
            pmid = pl + (pr - pl) / 2;
            if(buf_key[pmid] <= key){
                pl = pmid + 1;
            }else{
                pr = pmid;
            }
        }
        // *val = buf_payload[pl - 1];
        *buf_payload[pl - 1]  = *val;
        release_lock();
        return true;
    }

    // bool is_sort(){
    //     int flag = 0;
    //     for(uint64_t start = 1; start < page_count; start++){
    //         if(buf_key[start] < buf_key[start - 1])
    //             flag = 1;
    //     }
    //     if(flag == 1){
    //         return false;
    //     }
    //     return true;
    // }

};

class DRAM_index{
    public:
    DTYPE min_key;
    DTYPE* partial_keys;
    DTYPE* keys;
    string_payload* payloads;
    pgm_model_t* transfer_model[NR_DPUS];
    pgm::PGMIndex<DTYPE>* pgm_dram_index[NR_DPUS];
    size_t partiton_epsilon[NR_DPUS];
    uint64_t total_size;
    uint64_t partial_total_size;
    uint32_t dram_start_pos[NR_DPUS];
    buf_page* buffer_pages;
    uint64_t num_buf_pages; 
    btreeolc::BTree<DTYPE, string_payload>* overflow_trees;

    DRAM_index(){
        // nothing todo
        overflow_trees = new btreeolc::BTree<DTYPE, string_payload>[32]; // MERGE thread NUM
    }

    void get_payload_direct(int pos, string_payload* ret){
        *ret = payloads[pos];
    }
};

#endif