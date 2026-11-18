#!/usr/bin/env bash
# SmolVLA training speed on the pinned latest lerobot (~/lerobot-main @ e595b7902, env lerobot_main).
# Official example settings (batch 64), chunk 15. Measurement only.
cd /home/aivlab/kkb_capstone
OUT=2026-09-paper/smoke/main
mkdir -p $OUT
TRAIN=~/anaconda3/envs/lerobot_main/bin/lerobot-train
EPS=$(python -c "print(list(range(180)))" | tr -d ' ')
COMMON=(
  --policy.path=lerobot/smolvla_base
  --dataset.repo_id=local/so101_pick_place_fruit
  --dataset.root=/home/aivlab/kkb_capstone/data/lerobot/so101_pick_place_fruit
  --dataset.episodes="$EPS"
  --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'
  --policy.chunk_size=15 --policy.n_action_steps=15
  --policy.push_to_hub=false --policy.device=cuda --wandb.enable=false
  --batch_size=64 --log_freq=10 --save_checkpoint=false
)

run() {
  name=$1; shift
  log=$OUT/smolvla_$name.log
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > $OUT/smolvla_$name.mem &
  mon=$!
  $TRAIN "${COMMON[@]}" --output_dir=$OUT/smolvla_$name --job_name=smolvla_$name "$@" > $log 2>&1
  code=$?
  kill $mon
  rm -rf $OUT/smolvla_$name
  peak=$(sort -n $OUT/smolvla_$name.mem | tail -1)
  step=$(tr '\r' '\n' < $log | grep -o "updt_s:[0-9.]*" | tail -1)
  data=$(tr '\r' '\n' < $log | grep -o "data_s:[0-9.]*" | tail -1)
  echo "RESULT $name exit=$code peak_gpu_MiB=$peak last_$step last_$data oom=$(grep -c 'torch.OutOfMemoryError' $log)"
}

run bs64 --steps=60
run bs64_compile --steps=80 --policy.compile_model=true
echo ALL_DONE
