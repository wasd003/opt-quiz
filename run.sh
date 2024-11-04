#!/bin/bash

EXE_PATH=/home/jch/hpc-quiz/build/answer

likwid-pin -c M0:6 -- perf stat ${EXE_PATH}/branchless