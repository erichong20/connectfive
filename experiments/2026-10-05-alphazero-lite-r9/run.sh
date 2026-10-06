# Round 9: native self-play from the Rapfi-DAgger candidate, fine-tune, gate vs az-r7 (+ confirmation).
set -e
cd "$(dirname "$0")/../.."
PY=./.venv/bin/python
R=runs/az-r9
echo "== self-play $(date)"
$PY scripts/selfplay_round.py runs/rapfi-teacher/model-d1 --games 4000 --seed 150000 --workers 9 --simulations 600 --full-search-fraction 0.25 --fast-simulations 100 --balance-threshold 0.3 --balance-attempts 64 --draw-ply-cap 150 --native-batch 8 --out $R/games.npz 2>&1 | grep -E "^(saved|\{)"
echo "== fine-tune $(date)"
$PY scripts/train_supervised.py $R/games.npz --extra-train runs/rapfi-teacher/label-r8.npz runs/rapfi-teacher/label-r7.npz runs/az-r8/games.npz runs/az-r7/games.npz runs/az-r6/games.npz --hold-out-extra --init-checkpoint runs/rapfi-teacher/model-d1 --steps 3000 --batch-size 256 --learning-rate 0.0003 --weight-decay 1e-4 --validation-fraction 0.1 --seed 0 --policy-target soft --value-target blend --value-weight 0.5 --value-weighting game --out $R/model > $R/train.log 2>&1
grep -E "^(validation|saved)" $R/train.log
echo "== gate vs az-r7 $(date)"
$PY scripts/promotion_gate.py $R/model runs/az-r7/model --games 200 --seed 19000 --time-limit 0.2 --parallel 5 --out $R/gate > $R/gate.log 2>&1 && GATE=pass || GATE=fail
grep -v Warning $R/gate.log | tail -1
if [ "$GATE" = pass ]; then
  echo "== confirmation vs az-r7 $(date)"
  $PY scripts/promotion_gate.py $R/model runs/az-r7/model --games 200 --seed 19100 --time-limit 0.2 --parallel 5 --out $R/confirm 2>&1 | grep -v Warning | tail -1 || true
fi
echo "DONE $(date)"
