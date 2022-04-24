#!/bin/bash

hw_type=$1

for batch in 1 4 8 16 24 32; do

    python 02_test_model_perf.py --batch_size=$batch --hw_type=$hw_type 2>&1 | tee log_test

    grep -inra "hw_type:" log_test >> infer_$hw_type.txt
done
