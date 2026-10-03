set -e
PY=./.venv/bin/python
R=runs/az-r5
REPLAY="runs/az-r3/games.npz runs/az-r2/games.npz runs/az-r1/games-fixed.npz runs/teacher-v2/games.npz"
COMMON="--batch-size 256 --weight-decay 1e-4 --validation-fraction 0.1 --seed 0 --policy-target soft --value-target blend --value-weight 0.5 --value-weighting game"
gate() {  # $1 model, $2 tag, $3 first seed
  for k in 0 10 20 30 40; do
    $PY scripts/evaluate_guided.py $1 --opponent-checkpoint runs/az-r2/model --games 20 --seed $(($3 + k)) --time-limit 0.2 --json $R/eval/$2-$(($3 + k)).json > $R/eval/$2-$(($3 + k)).log 2>&1 &
  done
  wait
}
echo "== distil 64x4 from scratch $(date)"
$PY scripts/train_supervised.py runs/az-r4/games.npz --extra-train $REPLAY --channels 64 --blocks 4 --input-planes 4 --steps 6000 --learning-rate 0.002 $COMMON --out $R/distil | grep -E "^(validation|saved)"
echo "== gate distil vs az-r2 $(date)"
gate $R/distil distil 9000
$PY $R/gate.py "$R/eval/distil-*.json" 0.40 || { echo "STOP: distilled net too weak to generate self-play"; exit 1; }
echo "== self-play $(date)"
$PY scripts/selfplay_round.py $R/distil --games 600 --seed 110000 --workers 9 --simulations 600 --balance-threshold 0.3 --balance-attempts 64 --draw-ply-cap 150 --out $R/games.npz
echo "== fine-tune $(date)"
$PY scripts/train_supervised.py $R/games.npz --extra-train runs/az-r4/games.npz $REPLAY --init-checkpoint $R/distil --steps 2000 --learning-rate 0.0005 $COMMON --out $R/model | grep -E "^(validation|saved)"
echo "== gate model vs az-r2 $(date)"
gate $R/model model 9100
$PY $R/gate.py "$R/eval/model-*.json" 0.0
echo "DONE $(date)"
