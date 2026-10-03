# Run the round under a 5-hour wall-clock limit (macOS has no `timeout`).
cd "$(dirname "$0")/../.."
bash runs/az-r5/run.sh > runs/az-r5/run.log 2>&1 &
PID=$!
( sleep 18000; if kill -0 $PID 2>/dev/null; then echo "STOP: wall-clock limit" >> runs/az-r5/run.log; pkill -P $PID; kill $PID; pkill -f "runs/az-r5"; fi ) &
WATCHDOG=$!
wait $PID; STATUS=$?
kill $WATCHDOG 2>/dev/null
echo "exit $STATUS" >> runs/az-r5/run.log
