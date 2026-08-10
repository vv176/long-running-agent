"""Real PLAN run. Usage: python scripts/try_plan.py <repo> "<goal>" """
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foreman import db, index, plan

repo, goal = Path(sys.argv[1]).resolve(), sys.argv[2]
db.init()
rid = db.create_run(goal, repo)
print(f"run {rid}\ngoal: {goal}\n")
index.build(rid, repo)
print()
t0 = time.time()
out = plan.run(rid, repo, goal)
st = db.fold(rid)
print(f"\n{time.time()-t0:.0f}s · ${st.cents/100:.4f} · {st.tokens_in+st.tokens_out:,} tokens")
