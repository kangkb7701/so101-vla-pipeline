#!/usr/bin/env bash
# pi0.5 absolute-joint fine-tuning (~12 h), started after the SmolVLA queue finishes. Automatic recovery
# identical to run_queue.sh: resume from checkpoints/last on crash, kill + resume if the log is silent > 20 min.
# Recipe (2026-09-27 decision): lerobot/pi05_base, action expert only, torch.compile, bf16, gradient checkpointing,
# batch 16, 13,300 steps (~6.2 epochs of 34,382 frames; 3.19 s/step measured => ~12 h incl. compile/saves),
# chunk 15, default optimizer/schedule, seed 1000, dataset with exact quantile stats (pi0.5 uses quantile norm).
cd /home/aivlab/kkb_capstone/2026-09-paper/runs
TRAIN=~/anaconda3/envs/lerobot_main/bin/lerobot-train
DATA=/home/aivlab/kkb_capstone/data/lerobot
DS=so101_pick_place_fruit_180_q
NAME=pi05_abs_joint
OUT=$NAME
STEPS=13300
MAX_ATTEMPTS=6
QLOG=queue.log
log() { echo "$(date '+%F %T') $*" | tee -a $QLOG; }
done_run() { [ -d "$OUT/checkpoints/$(printf %06d $STEPS)/pretrained_model" ]; }

watch_pid() {
  local pid=$1 logf=$2
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    if [ -f "$logf" ] && [ $(( $(date +%s) - $(stat -c %Y "$logf") )) -gt 1200 ]; then
      log "HANG: $logf silent >20min, killing pid $pid"
      kill $pid; sleep 30; kill -9 $pid 2>/dev/null
    fi
  done
}

if [ -n "$1" ] && kill -0 "$1" 2>/dev/null; then
  log "pi05 waiting for SmolVLA queue pid $1"
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi

attempt=0
while ! done_run; do
  attempt=$((attempt+1))
  if [ $attempt -gt $MAX_ATTEMPTS ]; then log "GIVE UP $NAME after $MAX_ATTEMPTS attempts"; break; fi
  if [ -f "$OUT/checkpoints/last/pretrained_model/train_config.json" ]; then
    log "START $NAME attempt $attempt (resume from $(readlink -f $OUT/checkpoints/last | xargs basename))"
    $TRAIN --config_path=$OUT/checkpoints/last/pretrained_model/train_config.json --resume=true >> $NAME.log 2>&1 &
  else
    [ -d "$OUT" ] && { log "no checkpoint in $OUT, restarting from scratch"; rm -rf "$OUT"; }
    log "START $NAME attempt $attempt (fresh)"
    $TRAIN \
      --policy.path=lerobot/pi05_base \
      --dataset.repo_id=local/$DS --dataset.root=$DATA/$DS \
      --rename_map='{"observation.images.top":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}' \
      --policy.chunk_size=15 --policy.n_action_steps=15 \
      --policy.dtype=bfloat16 --policy.gradient_checkpointing=true \
      --policy.train_expert_only=true --policy.compile_model=true \
      --policy.push_to_hub=false --policy.device=cuda \
      --batch_size=16 --steps=$STEPS --save_freq=2500 --log_freq=100 --seed=1000 \
      --output_dir=$OUT --job_name=$NAME \
      --wandb.enable=true --wandb.project=so101-action-repr-paper \
      >> $NAME.log 2>&1 &
  fi
  pid=$!
  watch_pid $pid $NAME.log
  wait $pid; code=$?
  log "EXIT $NAME code=$code"
  done_run && log "DONE $NAME" || sleep 30
done
log "PI05 FINISHED: $(done_run && echo done || echo INCOMPLETE)"
