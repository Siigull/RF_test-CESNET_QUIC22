#!/bin/bash

epsilon=0.9
alpha=0.2
gamma=0.95
initial_batch=100
iters=10000
n_iterations=10

features="1,1,1,1,1,1,1,1,1,1,1,1"

for ((i=1; i<=n_iterations; i++))
do
    batch=$((initial_batch * i))

    echo "Running learn script"
    python3 learn.py -a "$alpha" -g "$gamma" -e "$epsilon" -f "$features" -b "$batch" -i "$iters"

    test_output_file="out_test[batches=${batch}].txt"
    test_input_file="out_state[batches=${batch}].txt"

    echo "Running test script"
    python3 test.py -fout "$test_output_file" -fin "$test_input_file"
done
