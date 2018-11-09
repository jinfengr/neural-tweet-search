#!/bin/bash
if [ -z "$1" ]; then
    TEST="trec-2011"
else
    TEST=$1
fi


for batch_size in 256 128 64
do
  for nb_filters in 256 128 64
  do
    for dropout in 0.1 0.2 0.3 0.4 0.5
    do
        CUDA_VISIBLE_DEVICES=0 python -u train.py -l -t $TEST -b ${batch_size} \
            -n ${nb_filters} -d ${dropout} --epochs 15 -v 2 &> \
            tune-logs/${TEST}_nbfilter${nb_filters}_d${dropout}_ttrue_b${batch_size}_mfalse.log ;
    done
  done
done