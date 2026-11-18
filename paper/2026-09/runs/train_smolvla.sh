#!/usr/bin/env bash
# SmolVLA fine-tuning for one action representation (2026-09-paper).
# Fixed across representations: lerobot-main @ e595b7902 (env lerobot_main), smolvla_base, 180 episodes,
# batch 64, 12,893 steps (24 epochs of 34,382 frames), chunk 15, compile on, default optimizer/schedule, seed 1000.
# Usage: train_smolvla.sh <repr_name> <dataset_root> [extra lerobot-train args...]
set -e
REPR=$1; ROOT=$2; shift 2
OUT=/home/aivlab/kkb_capstone/2026-09-paper/runs/smolvla_$REPR
~/anaconda3/envs/lerobot_main/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/$(basename $ROOT) \
  --dataset.root=$ROOT \
  --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
  --policy.chunk_size=15 --policy.n_action_steps=15 \
  --policy.compile_model=true \
  --policy.push_to_hub=false --policy.device=cuda \
  --batch_size=64 --steps=12893 --save_freq=2000 --log_freq=100 --seed=1000 \
  --output_dir=$OUT --job_name=smolvla_$REPR \
  --wandb.enable=true --wandb.project=so101-action-repr-paper \
  "$@"
