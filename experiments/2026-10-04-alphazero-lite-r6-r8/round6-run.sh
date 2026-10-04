set -e
PY=./.venv/bin/python
R=runs/az-r6
echo "== self-play $(date)"
$PY scripts/selfplay_round.py runs/az-r5/model --games 2800 --seed 120000 --workers 9 --simulations 600 --full-search-fraction 0.25 --fast-simulations 100 --balance-threshold 0.3 --balance-attempts 64 --draw-ply-cap 150 --out $R/games.npz
echo "== fine-tune $(date)"
$PY scripts/train_supervised.py $R/games.npz --extra-train runs/az-r5/games.npz runs/az-r4/games.npz runs/az-r3/games.npz runs/az-r2/games.npz runs/az-r1/games-fixed.npz runs/teacher-v2/games.npz --hold-out-extra --init-checkpoint runs/az-r5/model --steps 2000 --batch-size 256 --learning-rate 0.0005 --weight-decay 1e-4 --validation-fraction 0.1 --seed 0 --policy-target soft --value-target blend --value-weight 0.5 --value-weighting game --out $R/model | grep -E "^(validation|saved)"
echo "== gate vs az-r2 $(date)"
$PY scripts/promotion_gate.py $R/model runs/az-r2/model --games 200 --seed 12000 --time-limit 0.2 --parallel 5 --out $R/gate || true
echo "DONE $(date)"
