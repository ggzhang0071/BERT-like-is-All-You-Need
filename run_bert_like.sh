#!/bin/bash
timestamp=`date +%Y%m%d%H%M%S`
rm Logs/*.log

conda install -c conda-forge librosa==0.7.0

python  multimodal-speech-emotion-recognition/main.py  2>&1 >> Logs/$timestamp.log

#python SPEECH-BERT-TOKENIZATION/convert_aud_to_token.py 

CUDA_VISIBLE_DEVICES=0,1 python3 -m  pdb train.py --data ./T_data/iemocap --restore-file None --task emotion_prediction \
 --reset-optimizer --reset-dataloader --reset-meters --init-token 0 --separator-token 2 --arch robertEMO_large \
 --criterion emotion_prediction_cri --num-classes 8 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.1 \
 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 --clip-norm 0.0 --lr-scheduler polynomial_decay \
 --lr 1e-05 --total-num-update 2760 --warmup-updates 165 --max-epoch 1 --best-checkpoint-metric loss --encoder-attention-heads 2 \
 --batch-size 8192 --encoder-layers-cross 1 --no-epoch-checkpoints --update-freq 8 --find-unused-parameters --ddp-backend=no_c10d \
 --binary-target-iemocap --a-only --t-only --pooler-dropout 0.1 --log-interval 1 --data-raw /git/datasets/pre-processed_data/iemocap_data \
 2>&1 >> Logs/$timestamp.log
 

#python validate.py --data ./T_data/iemocap --path './checkpoints/checkpoint_best.pt' --task emotion_prediction --valid-subset test --batch-size 4
