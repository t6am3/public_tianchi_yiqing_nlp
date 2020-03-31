#!/bin/bash

python run.py \
    --data_dir ../data/Dataset \
    --predict_file ../data/Dataset/dev.csv \
    --do_train \
    --do_eval \
    --adv_pgd \
    --pgd_k 3 \
#     --use_reverse_in_train \
#     --use_reverse_in_test \


    
    
    
