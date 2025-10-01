#!/bin/bash
## Du-IN
# cd /home/ruicong/duin/train/duin
# 013
python run_vqvae.py --seed 42 --subjs 013

python run_mae.py --seed 42 --subjs 013 --vqkd_ckpt /home/ruicong/duin/summaries/2025-01-12/0/model/checkpoint-0.pth


python run_cls.py --seeds 42 0 1 2 3 4 --subjs 013 --subj_idxs 0
python run_cls.py --seeds 42 0 1 2 3 4 --subjs 013 --subj_idxs 0\
    --pt_ckpt /home/ruicong/duin/summaries/2025-01-12/0/model/checkpoint-0.pth
python run_cls.py --seeds 42 0 1 2 3 4 --subjs 013 --subj_idxs 0\
    --pt_ckpt /home/ruicong/duin/summaries/2025-01-12/1/model/checkpoint-0.pth