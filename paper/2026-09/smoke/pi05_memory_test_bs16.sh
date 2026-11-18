#!/usr/bin/env bash
# Short pi0.5 fine-tune runs (20 steps, batch 32) to measure peak GPU memory and step time per recipe.
# Measurement only: no recipe is chosen here.
# Uses pi05_base revision a538eb27 (same weights as main): main adds a processor step unknown to local lerobot 0.5.2.
cd /home/aivlab/kkb_capstone
OUT=2026-09-paper/smoke
EPS=$(python -c "print(list(range(180)))" | tr -d ' ')
COMMON=(
  --policy.path=/home/aivlab/kkb_capstone/2026-09-paper/smoke/pi05_base_rev_a538eb27
  --dataset.repo_id=local/so101_pick_place_fruit
  --dataset.root=/home/aivlab/kkb_capstone/data/lerobot/so101_pick_place_fruit
  --dataset.episodes="$EPS"
  --rename_map='{"observation.images.top":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}'
  --policy.chunk_size=15 --policy.n_action_steps=15
  --policy.dtype=bfloat16 --policy.gradient_checkpointing=true
  --policy.push_to_hub=false --policy.device=cuda --wandb.enable=false
  --batch_size=16 --steps=20 --log_freq=5 --save_checkpoint=false
)

run() {
  name=$1; shift
  log=$OUT/pi05_$name.log
  rm -rf $OUT/pi05_$name
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > $OUT/pi05_$name.mem &
  mon=$!
  ~/anaconda3/envs/lerobot/bin/lerobot-train "${COMMON[@]}" --output_dir=$OUT/pi05_$name --job_name=pi05_$name "$@" > $log 2>&1
  code=$?
  kill $mon
  peak=$(sort -n $OUT/pi05_$name.mem | tail -1)
  echo "RESULT $name exit=$code peak_gpu_MiB=$peak oom=$(grep -c 'OutOfMemory' $log)"
}

run expert_bs16 --policy.train_expert_only=true
echo ALL_DONE
