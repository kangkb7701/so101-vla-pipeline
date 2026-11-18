#!/usr/bin/env bash
# Sequential SmolVLA fine-tuning of the four action representations with automatic recovery.
# - waits for an already-running job (PID given as $1, optional)
# - a run is complete when checkpoints/012893 exists
# - on crash: resume from checkpoints/last (or restart from scratch if no checkpoint yet)
# - on hang (log silent > 20 min while training): kill and resume
# Fixed settings are identical to train_smolvla.sh (see 2026-09-paper/EXPERIMENT_PLAN.md).
cd /home/aivlab/kkb_capstone/2026-09-paper/runs
TRAIN=~/anaconda3/envs/lerobot_main/bin/lerobot-train
DATA=/home/aivlab/kkb_capstone/data/lerobot
URDF=/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf
STEPS=12893
MAX_ATTEMPTS=6
QLOG=queue.log
log() { echo "$(date '+%F %T') $*" | tee -a $QLOG; }

# name | base | dataset | extra args
RUNS=(
  "abs_joint|lerobot/smolvla_base|so101_pick_place_fruit_180|"
  "rel_joint|/home/aivlab/kkb_capstone/2026-09-paper/runs/bases/smolvla_base_rel_joint|so101_pick_place_fruit_180_rel_joint|--policy.use_relative_actions=true"
  "abs_ee|lerobot/smolvla_base|so101_pick_place_fruit_180_abs_ee|"
  "rel_ee|/home/aivlab/kkb_capstone/2026-09-paper/runs/bases/smolvla_base_rel_ee|so101_pick_place_fruit_180_rel_ee|--policy.use_relative_ee_actions=true --policy.ee_urdf_path=$URDF"
)

done_run() { [ -d "smolvla_$1/checkpoints/$(printf %06d $STEPS)/pretrained_model" ]; }

watch_pid() {  # wait for pid, killing it if its log goes silent for 20 minutes
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
  log "waiting for running job pid $1 (abs_joint)"
  watch_pid "$1" smolvla_abs_joint.log
fi

for spec in "${RUNS[@]}"; do
  IFS='|' read -r NAME BASE DS EXTRA <<< "$spec"
  OUT=smolvla_$NAME
  attempt=0
  while ! done_run $NAME; do
    attempt=$((attempt+1))
    if [ $attempt -gt $MAX_ATTEMPTS ]; then log "GIVE UP $NAME after $MAX_ATTEMPTS attempts"; break; fi
    if [ -f "$OUT/checkpoints/last/pretrained_model/train_config.json" ]; then
      log "START $NAME attempt $attempt (resume from $(readlink -f $OUT/checkpoints/last | xargs basename))"
      $TRAIN --config_path=$OUT/checkpoints/last/pretrained_model/train_config.json --resume=true \
        >> smolvla_$NAME.log 2>&1 &
    else
      [ -d "$OUT" ] && { log "no checkpoint in $OUT, restarting from scratch"; rm -rf "$OUT"; }
      log "START $NAME attempt $attempt (fresh)"
      # shellcheck disable=SC2086
      $TRAIN \
        --policy.path=$BASE \
        --dataset.repo_id=local/$DS --dataset.root=$DATA/$DS \
        --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
        --policy.chunk_size=15 --policy.n_action_steps=15 \
        --policy.compile_model=true \
        --policy.push_to_hub=false --policy.device=cuda \
        --batch_size=64 --steps=$STEPS --save_freq=2000 --log_freq=100 --seed=1000 \
        --output_dir=$OUT --job_name=smolvla_$NAME \
        --wandb.enable=true --wandb.project=so101-action-repr-paper \
        $EXTRA >> smolvla_$NAME.log 2>&1 &
    fi
    pid=$!
    watch_pid $pid smolvla_$NAME.log
    wait $pid; code=$?
    log "EXIT $NAME code=$code"
    done_run $NAME && log "DONE $NAME" || sleep 30
  done
done
log "QUEUE FINISHED: $(for s in "${RUNS[@]}"; do n=${s%%|*}; done_run $n && echo -n "$n=done " || echo -n "$n=INCOMPLETE "; done)"
