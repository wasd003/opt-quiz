#!/bin/bash

# L1
# L1-dcache-loads
# L1-dcache-load-misses
# L1-dcache-stores

# L2
### Data Read
# l2_rqsts.all_demand_data_rd - Demand Data Read access L2 cache
# l2_rqsts.demand_data_rd_hit - Demand Data Read requests that hit L2 cache
# l2_rqsts.demand_data_rd_miss - Demand Data Read miss L2 cache
### All Access
# l2_request.all - All accesses to L2 cache
# l2_rqsts.all_demand_miss - Demand requests that miss L2 cache

# LLC
# LLC-loads
# LLC-load-misses
# LLC-stores
# LLC-prefetches

SAMPLE_EVENTS="L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,l2_rqsts.all_demand_data_rd,l2_rqsts.demand_data_rd_hit,l2_rqsts.demand_data_rd_miss,LLC-loads,LLC-load-misses"
CORE=12

# test l2 cache
export OMP_NUM_THREADS=1
perf stat -e ${SAMPLE_EVENTS} -C ${CORE}  -- taskset -c ${CORE} ./stream_c.exe


# test llc
perf stat -e ${SAMPLE_EVENTS}  -a  -- ./stream_c.exe