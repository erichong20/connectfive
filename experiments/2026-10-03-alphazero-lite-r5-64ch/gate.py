"""Sum a gate's JSON results; exit 1 if the score is below the threshold."""
import glob, json, math, sys
pattern, threshold = sys.argv[1], float(sys.argv[2])
w = l = d = 0
for path in glob.glob(pattern):
    r = json.load(open(path)); w += r["wins"]; l += r["losses"]; d += r["draws"]
    if r["agent_illegal_moves"] or r["opponent_illegal_moves"]:
        sys.exit(f"illegal moves in {path}")
n = w + l + d; p = (w + d / 2) / n; z = 1.96
c = (p + z * z / 2 / n) / (1 + z * z / n)
h = z * math.sqrt(p * (1 - p) / n + z * z / 4 / n / n) / (1 + z * z / n)
print(f"{pattern}: {w}-{l}-{d}, {p:.1%} (95% CI {c - h:.1%}-{c + h:.1%})")
sys.exit(0 if p >= threshold else 1)
