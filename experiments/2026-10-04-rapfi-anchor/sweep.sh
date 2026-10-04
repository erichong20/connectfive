# az-r7 (1 s/move, single-threaded XLA) vs Rapfi on 41 Rapfi-balanced openings, both colours.
cd "$(dirname "$0")/../.."
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
for N in 30 100 300 1000 0; do
  LIMIT=""; [ $N -gt 0 ] && LIMIT="--info MAX_NODE=$N"
  caffeinate -i -s ./.venv/bin/python scripts/play_external.py runs/az-r7/model --engine runs/engines/rapfi/build/pbrain-rapfi \
    $LIMIT --info THREAD_NUM=1 --info TIMEOUT_TURN=1000 --info TIMEOUT_MATCH=10000000 \
    --openings experiments/2026-10-04-rapfi-anchor/openings.json --time-limit 1.0 \
    --json runs/external/bal-r7-rapfi-n$N.json > runs/external/bal-r7-rapfi-n$N.log 2>&1 &
done
wait
grep -h "vs runs" runs/external/bal-r7-rapfi-n*.log
