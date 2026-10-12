#!/bin/bash
#python CA-HB2V-MoE/train_loso.py \
#  --manifest-root /home/user512-003/songcw/codex_vscode/scw_er/lcmr_tff_two_stage_project/manifests/loso \
#  --wavlm-name /home/user512-003/songcw/codex_vscode/scw_er/pretrained_model/wavlm-large \
#  --stage1-checkpoint /home/user512-003/songcw/codex_vscode/scw_er/lcmr_tff_two_stage_project/outputs/1024/stage1_no_aug/loso/loso_Ses02/cases/val_session_Ses05/best_stage1.pt \
#  --output-root /home/user512-003/songcw/codex_vscode/scw_er/lcmr_tff_two_stage_project/outputs/CA-HB2V-MoE/loso \
#  --fold loso_Ses02 \
#  --val-session Ses05 \
#  --variant ca_hb2v_moe \
#  --epochs 50 \
#  --batch-size 32 \
#  --num-workers 4 \
#  --device cuda


#python  CA-HB2V-MoE/train_ravdess.py \
#  --fold fold_01_test_01-04 \
#  --case val_05-08


python CA-HB2V-MoE/train_savee.py \
  --fold fold_02_test_JE \
  --case val_JK \
  --stage1-checkpoint \
  /home/user512-003/songcw/codex_vscode/scw_er/lcmr_tff_two_stage_project/outputs/1024/savee_stage1_no_aug/speaker_independent/fold_02_test_JE/cases/val_JK/stage1_topk_checkpoints/stage1_epoch0031.pt \
  --epochs 50 \
  --batch-size 16 \
  --num-workers 4

