"""Run INDEX for real against a repo. Usage: python scripts/try_index.py <repo>"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foreman import db, index

repo = Path(sys.argv[1]).resolve()
db.init()
rid = db.create_run("index only", repo)
t0 = time.time()
out = index.build(rid, repo)
st = db.fold(rid)
print(f"\n{out}")
print(f"{time.time()-t0:.1f}s · ${st.cents/100:.4f} · {st.tokens_in+st.tokens_out:,} tokens")
