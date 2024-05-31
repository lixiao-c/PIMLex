#ifndef _CCONCURRENT_H_
#define _CCONCURRENT_H_

#include <utility>
#include <stdint.h>
#include <stdlib.h>
#include <cstdint>
#include <iostream>

#define CAS(_p, _u, _v)                                             \
  (__atomic_compare_exchange_n(_p, _u, _v, false, __ATOMIC_ACQUIRE, \
                               __ATOMIC_ACQUIRE))

const uint32_t lockSet = ((uint32_t)1 << 31);
const uint32_t lockMask = ((uint32_t)1 << 31) - 1;
const int counterMask = (1 << 19) - 1;

class ol_lock{
    public:
    uint32_t lock_;
    ol_lock(){
        lock_ = 0;
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

    inline bool try_get_lock() {
    uint32_t v = __atomic_load_n(&lock_, __ATOMIC_ACQUIRE);
    if (v & lockSet) {
        return false;
    }
    auto old_value = v & lockMask;
    auto new_value = v | lockSet;
    return CAS(&lock_, &old_value, new_value);
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
};

#endif