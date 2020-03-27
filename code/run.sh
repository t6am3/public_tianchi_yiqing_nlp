#!/bin/bash

python run.py \
    --data_dir ../data/Dataset/aug_input \
    --predict_file ../data/Dataset/test.example_20200228.csv \
    --use_reverse_in_train \
    --use_reverse_in_test \
    --do_train \
    --pseudo \
    --pseudo_data_dir ../data/Dataset/aug_input
    
    
    
