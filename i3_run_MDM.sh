#!/bin/bash
## Du-IN
cd ./train/duin

# 一共12个sub
# 001
python run_cls_self_distillation.py --seeds 42 0 1 2 3 4  --subjs 001 --subj_idxs 0 --isMDM  \
    --foldName i3_run_MDM

# 002
python run_cls_self_distillation.py --seeds 42 0 1 2 3 4  --subjs 002 --subj_idxs 0 --isMDM  \
    --foldName i3_run_MDM

# 003
python run_cls_self_distillation.py --seeds 42 0 1 2 3 4  --subjs 003 --subj_idxs 0 --isMDM  \
    --foldName i3_run_MDM

# 004
python run_cls_self_distillation.py --seeds 42 0 1 2 3 4  --subjs 004 --subj_idxs 0 --isMDM  \
    --foldName i3_run_MDM

# 005
python run_cls_self_distillation.py --seeds 42 0 1 2 3 4  --subjs 005 --subj_idxs 0 --isMDM  \
    --foldName i3_run_MDM

# 006
python run_cls_self_distillation.py --seeds 42 0 1 2 3 4  --subjs 006 --subj_idxs 0 --isMDM  \
    --foldName i3_run_MDM

# 007
python run_cls_self_distillation.py --seeds 42 0 1 2 3 4  --subjs 007 --subj_idxs 0 --isMDM  \
    --foldName i3_run_MDM

# 008
python run_cls_self_distillation.py --seeds 42 0 1 2 3 4  --subjs 008 --subj_idxs 0 --isMDM  \
    --foldName i3_run_MDM

# 009
python run_cls_self_distillation.py --seeds 42 0 1 2 3 4  --subjs 009 --subj_idxs 0 --isMDM  \
    --foldName i3_run_MDM

# 010
python run_cls_self_distillation.py --seeds 42 0 1 2 3 4  --subjs 010 --subj_idxs 0 --isMDM  \
    --foldName i3_run_MDM

# 011
python run_cls_self_distillation.py --seeds 42 0 1 2 3 4  --subjs 011 --subj_idxs 0 --isMDM  \
    --foldName i3_run_MDM

# 012
python run_cls_self_distillation.py --seeds 42 0 1 2 3 4  --subjs 012 --subj_idxs 0 --isMDM  \
    --foldName i3_run_MDM