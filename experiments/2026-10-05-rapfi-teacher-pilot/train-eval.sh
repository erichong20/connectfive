# Fine-tune az-r7 on Rapfi teacher games, then gate vs az-r7 and run the Rapfi ladder.
set -e
cd "$(dirname "$0")/../.."
PY=./.venv/bin/python
R=runs/rapfi-teacher
echo "== train $(date)"
$PY scripts/train_supervised.py $R/pilot-v2-s300.npz --init-checkpoint runs/az-r7/model --steps 3000 --batch-size 256 --learning-rate 0.0005 --weight-decay 1e-4 --validation-fraction 0.1 --seed 0 --policy-target soft --value-target blend --value-weight 0.5 --value-weighting game --out $R/model-p1 > $R/train-p1.log 2>&1
grep -E "^(validation|saved)" $R/train-p1.log
echo "== gate vs az-r7 $(date)"
$PY scripts/promotion_gate.py $R/model-p1 runs/az-r7/model --games 200 --seed 17000 --time-limit 0.2 --parallel 5 --out $R/gate-p1 2>&1 | grep -v Warning | tail -1 || true
echo "== rapfi ladder $(date)"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
for N in 30 100 300 1000; do
  $PY scripts/play_external.py $R/model-p1 --engine runs/engines/rapfi/build/pbrain-rapfi --info MAX_NODE=$N \
    --info THREAD_NUM=1 --info TIMEOUT_TURN=1000 --info TIMEOUT_MATCH=10000000 \
    --openings experiments/2026-10-04-rapfi-anchor/openings.json --time-limit 1.0 \
    --json $R/ladder-p1-n$N.json > $R/ladder-p1-n$N.log 2>&1 &
done
wait
grep -h "vs runs" $R/ladder-p1-n*.log
echo "DONE $(date)"
