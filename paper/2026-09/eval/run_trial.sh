#!/usr/bin/env bash
# One evaluation trial on the real SO-101 (records 1 episode with lerobot-rollout, episodic strategy).
# Usage: ./run_trial.sh <repr> <trial_no> <position_no> <color>
#   repr   : abs_joint | rel_joint | abs_ee | rel_ee   (SmolVLA checkpoints, step 12893)
#   color  : blue | yellow | green
# Recording: eval/recordings/smolvla_<repr>/trial_<NN>_pos<PP>_<color>/  (LeRobot dataset: videos + actions)
# Fixed settings: 10 Hz, chunk 15 with all 15 actions executed (n_action_steps=15), max 60 s per trial,
# follower my_follower on /dev/ttyACM0, top=/dev/video0 -> camera1, side=/dev/video2 -> camera2.
# EE representations use IK inside the policy (eval/ee_ik_steps.py); IK log is written next to the recording.
set -e
REPR=$1; TRIAL=$2; POS=$3; COLOR=$4
case "$REPR" in abs_joint|rel_joint|abs_ee|rel_ee) ;; *) echo "repr must be abs_joint|rel_joint|abs_ee|rel_ee"; exit 1;; esac
case "$COLOR" in blue|yellow|green) ;; *) echo "color must be blue|yellow|green"; exit 1;; esac
[ -n "$TRIAL" ] && [ -n "$POS" ] || { echo "usage: $0 <repr> <trial_no> <position_no> <color>"; exit 1; }

HERE=/home/aivlab/kkb_capstone/2026-09-paper/eval
CKPT=/home/aivlab/kkb_capstone/2026-09-paper/runs/smolvla_$REPR/checkpoints/012893/pretrained_model
NAME=trial_$(printf %02d $TRIAL)_pos$(printf %02d $POS)_$COLOR
OUT=$HERE/recordings/smolvla_$REPR/$NAME
[ -e "$OUT" ] && { echo "already recorded: $OUT (delete it to re-run)"; exit 1; }
mkdir -p $(dirname $OUT)
TASK="pick the banana and place it in the $COLOR basket"

for port in 5000 5001; do
  if ss -ltn | grep -q ":$port "; then echo "camera page on port $port is running - stop it first (it holds the cameras)"; exit 1; fi
done

echo "=== $REPR | $NAME | task: $TASK"
echo "Keys: Right arrow = end episode (success or give up), Left arrow = discard and re-record, Esc = stop"
cd $HERE
EE_IK_LOG=$OUT.ik.csv EE_IK_JOINT_LIMITS=${EE_IK_JOINT_LIMITS:-urdf} \
~/anaconda3/envs/lerobot_main/bin/python eval_rollout.py \
  --strategy.type=episodic \
  --policy.path=$CKPT --policy.n_action_steps=15 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --robot.cameras="{camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
  --task="$TASK" --fps=10 \
  --dataset.repo_id=local/rollout_smolvla_${REPR}_$NAME --dataset.root=$OUT --dataset.single_task="$TASK" \
  --dataset.fps=10 --dataset.num_episodes=1 --dataset.episode_time_s=60 --dataset.reset_time_s=0 \
  --dataset.push_to_hub=false --dataset.no_stamp=true \
  --play_sounds=false 2>&1 | tee $OUT.log
echo "saved: $OUT"
