#!/bin/bash
timestamp=`date +%Y%m%d%H%M%S`
rm Logs/*.log

python3 -m pdb  train.py 

#CUDA_VISIBLE_DEVICES=1 python validate.py --data ./T_data/iemocap --path './checkpoints/checkpoint_best.pt' --task emotion_prediction --valid-subset test --batch-size 4

