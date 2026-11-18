#!/usr/bin/env bash
# pi0.5 fine-tuning recipe measurements on the pinned latest lerobot (~/lerobot-main @ e595b7902, env lerobot_main).
# Measurement only (peak GPU memory, step time, trainable params): no recipe is chosen here.
cd /home/aivlab/kkb_capstone
OUT=2026-09-paper/smoke/main
mkdir -p $OUT
TRAIN=~/anaconda3/envs/lerobot_main/bin/lerobot-train
EPS=$(python -c "print(list(range(180)))" | tr -d ' ')
COMMON=(
  --policy.path=lerobot/pi05_base
  --dataset.repo_id=local/so101_pick_place_fruit
  --dataset.root=/home/aivlab/kkb_capstone/data/lerobot/so101_pick_place_fruit
  --dataset.episodes="$EPS"
  --rename_map='{"observation.images.top":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}'
  --policy.chunk_size=15 --policy.n_action_steps=15
  --policy.dtype=bfloat16 --policy.gradient_checkpointing=true
  --policy.push_to_hub=false --policy.device=cuda --wandb.enable=false
  --log_freq=5 --save_checkpoint=false
)
# openpi-style LoRA: adapters on the PaliGemma language model and the action expert (attention + MLP).
OPENPI_LORA='(.*\.(language_model|gemma_expert)\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|model\.(state_proj|action_in_proj|action_out_proj|time_mlp_in|time_mlp_out))'

run() {
  name=$1; shift
  log=$OUT/pi05_$name.log
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > $OUT/pi05_$name.mem &
  mon=$!
  $TRAIN "${COMMON[@]}" --output_dir=$OUT/pi05_$name --job_name=pi05_$name "$@" > $log 2>&1
  code=$?
  kill $mon
  rm -rf $OUT/pi05_$name
  peak=$(sort -n $OUT/pi05_$name.mem | tail -1)
  params=$(grep -o "num_learnable_params=[0-9]*" $log | head -1)
  step=$(tr '\r' '\n' < $log | grep -o "updt_s:[0-9.]*" | tail -1)
  echo "RESULT $name exit=$code peak_gpu_MiB=$peak $params last_$step oom=$(grep -c 'OutOfMemory' $log)"
}

run expert_bs16 --batch_size=16 --steps=30 --policy.train_expert_only=true
run lora_default_bs32 --batch_size=32 --steps=30 --peft.method_type=LORA
run lora_openpi_bs32 --batch_size=32 --steps=30 --peft.method_type=LORA --peft.r=16 --peft.target_modules="$OPENPI_LORA"
run lora_openpi_bs16 --batch_size=16 --steps=30 --peft.method_type=LORA --peft.r=16 --peft.target_modules="$OPENPI_LORA"
run expert_bs16_tok160 --batch_size=16 --steps=30 --policy.train_expert_only=true --policy.tokenizer_max_length=160
run expert_bs16_compile --batch_size=16 --steps=60 --policy.train_expert_only=true --policy.compile_model=true
echo ALL_DONE
