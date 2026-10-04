# Overnight AlphaZero-lite rounds after round 6. Local CPU only.
# Stops: deadline (no round starts with < 2.5 h left; watchdog kills at DEADLINE),
# any command error, gate error/illegal moves (exit != 0/3), NaN in training.
set -u
cd "$(dirname "$0")/../.."
PY=./.venv/bin/python
DEADLINE=$1
LOG=runs/overnight/loop.log
log() { echo "[$(date '+%H:%M:%S')] $*" >> $LOG; }

R6PID=$(cat runs/az-r6/run.pid)
log "waiting for round 6 (pid $R6PID)"
while kill -0 $R6PID 2>/dev/null; do sleep 30; done
CHAMPION=runs/az-r2/model
if [ -f runs/az-r6/gate/gate.json ] && grep -q '"passed": true' runs/az-r6/gate/gate.json; then
  CHAMPION=runs/az-r6/model
fi
[ -f runs/az-r6/model.msgpack ] || { log "STOP: round 6 produced no model"; exit 1; }
log "round 6 done; champion $CHAMPION"

REPLAY="runs/az-r6/games.npz runs/az-r5/games.npz runs/az-r4/games.npz runs/az-r3/games.npz runs/az-r2/games.npz runs/az-r1/games-fixed.npz runs/teacher-v2/games.npz"
for K in 7 8 9 10; do
  LEFT=$((DEADLINE - $(date +%s)))
  if [ $LEFT -lt 9000 ]; then log "not starting round $K: only $((LEFT / 60)) min left"; break; fi
  R=runs/az-r$K; PREV=runs/az-r$((K - 1))/model
  mkdir -p $R; git rev-parse HEAD > $R/commit.txt; git status --short > $R/dirty.txt
  log "round $K: self-play from $PREV"
  $PY scripts/selfplay_round.py $PREV --games 2800 --seed $((120000 + (K - 6) * 10000)) --workers 9 --simulations 600 --full-search-fraction 0.25 --fast-simulations 100 --balance-threshold 0.3 --balance-attempts 64 --draw-ply-cap 150 --out $R/games.npz > $R/selfplay.log 2>&1 || { log "STOP: self-play failed"; exit 1; }
  log "round $K: $(grep -E '^\{' $R/selfplay.log | tail -1)"
  $PY scripts/train_supervised.py $R/games.npz --extra-train $REPLAY --hold-out-extra --init-checkpoint $PREV --steps 2000 --batch-size 256 --learning-rate 0.0005 --weight-decay 1e-4 --validation-fraction 0.1 --seed 0 --policy-target soft --value-target blend --value-weight 0.5 --value-weighting game --out $R/model > $R/train.log 2>&1 || { log "STOP: training failed"; exit 1; }
  if grep -E "^validation" $R/train.log | grep -qi nan; then log "STOP: NaN in validation metrics"; exit 1; fi
  log "round $K: $(grep -E '^validation' $R/train.log | cut -c1-200)"
  $PY scripts/promotion_gate.py $R/model $CHAMPION --games 200 --seed $((12000 + (K - 6) * 100)) --time-limit 0.2 --parallel 5 --out $R/gate > $R/gate.log 2>&1
  STATUS=$?
  log "round $K gate vs $CHAMPION: $(grep -v Warning $R/gate.log | tail -1)"
  if [ $STATUS -eq 0 ]; then CHAMPION=$R/model; log "PROMOTED: $CHAMPION"
  elif [ $STATUS -ne 3 ]; then log "STOP: gate error (exit $STATUS)"; exit 1; fi
  REPLAY="$R/games.npz $REPLAY"
done
log "DONE; champion $CHAMPION"
