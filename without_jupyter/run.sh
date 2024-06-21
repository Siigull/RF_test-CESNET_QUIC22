#!/bin/bash

epsilon = 0.9
alpha = 0.2
gamma = 0.95

features = "1,1,1,1,1,1,1,1"

python3 learn.py -a "$alpha" -g "$gamma" -e "$epsilon" -f "$features"

python3 test.py