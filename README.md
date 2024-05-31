# PIMLex
PIMLex is a high-performance learned index optimized with Processing-in-Memory (PIM) devices. 

## Requirements
Current implementation of PIMLex is based on [UPMEM](https://www.upmem.com/), which is a commercial available PIM device. To build this codeset, you need to install [UPMEM SDK](https://sdk.upmem.com/).


## Dependencies

- intel-tbb 2020.3
- jemalloc

## Building

Before starting the build, you need to specify `NR_DPUS` as the number of DPU modules on the machine and `NR_TASKLETS` as the number of threads used on PIM modules. For exmaple: 

```
NR_DPUS=512 NR_TASKLETS=12 make all
```

## Running

To obtain the throughput:

```
./bin/lex_host \
--keys_file={dataset} \ 
--total_num_keys= {dataset size} \ 
--init_num_keys={bulk load number} \  
--query_num= {test number} \ 
--sample_distribution={zipf, uniform}
```

Search-only workload is executed by default. For additional features, add additional flags:

- Insert

```
--insert
```

- Range Query

```
--scan
```

- Mixed workload
```
--mix
```

- Predecessor
```
--predecessor
```

- To use uniform distribution

```
--sample_distribution=uniform
```
All the result will be output to the csv file specified in --output_path flag.


## Dataset

`books`, `osm` and `fb` are taken from [SOSD](https://github.com/learnedsystems/SOSD).

 `genome` and `planet` are taken from [GRE](https://github.com/gre4index/GRE).
 
 > Kipf, Andreas, et al. "SOSD: A Benchmark for Learned Indexes". NeurIPS Workshop on Machine Learning for Systems, 2019.

 > Chaichon Wongkham, et al. "Are Updatable Learned Indexes Ready?". PVLDB, 15(11): 3004 - 3017, 2022.

