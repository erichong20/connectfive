# Kill the overnight loop and its children at the deadline (epoch seconds).
LOOP=$1; DEADLINE=$2; cd "$(dirname "$0")/../.."
while kill -0 $LOOP 2>/dev/null; do
  if [ $(date +%s) -ge $DEADLINE ]; then
    echo "[$(date '+%H:%M:%S')] STOP: deadline reached" >> runs/overnight/loop.log
    pkill -f "scripts/(selfplay_round|train_supervised|promotion_gate|evaluate_guided).py runs/az-r([7-9]|10)"
    pkill -f "scripts/promotion_gate.py runs/az-r"; kill $LOOP; exit 0
  fi
  sleep 30
done
