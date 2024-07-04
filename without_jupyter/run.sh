#!/bin/bash

epsilon=0.9
alpha=0.2
gamma=0.95
batch=400
initial_iters=400000
n_iterations=10
runs=3
nclass=0

features="1,1,1,1,1,1,1,1,1,1,1"

for ((i=1; i<=n_iterations; i++))
do
    iters=$((initial_iters * i))

    echo "Running learn script"
    python3 learn.py -a "$alpha" -g "$gamma" -e "$epsilon" -f "$features" -b "$batch" -i "$iters" -r "$runs" -n "$nclass"                     

    test_output_file="out_test[iters=${iters}].txt"
    test_input_file="out_state[iters=${iters}].txt"

    echo "Running test script"
    python3 test.py -fout "$test_output_file" -fin "$test_input_file"
done
