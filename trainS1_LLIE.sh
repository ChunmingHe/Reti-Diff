#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4  --use_env --master_port=4388 Reti-Diff/train.py -opt options/trainS1_LLIE.yml --launcher pytorch