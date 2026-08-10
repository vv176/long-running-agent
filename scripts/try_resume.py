"""Start a run, SIGINT it mid-EXECUTE, then resume. Proves the ledger is enough."""
import re, signal, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PY_ = str(ROOT / ".venv/bin/python")

cmd = [PY_, "-u", "-m", "foreman.run",
       "migrate this service from Flask to FastAPI. Keep behaviour identical.",
       "--repo", "fixtures/miniledger"]
p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
run_id, done = None, 0
for line in p.stdout:
    sys.stdout.write(line)
    m = re.search(r"(run_[0-9a-f]{8})", line)
    if m: run_id = m.group(1)
    if line.strip().startswith("DONE "): done += 1
    if done == 3:                      # kill after 3 units land
        print("\n>>> sending SIGINT after 3 units\n")
        p.send_signal(signal.SIGINT)
        break
for line in p.stdout: sys.stdout.write(line)
p.wait()
print(f"\n>>> exit {p.returncode}\n>>> RESUMING {run_id}\n")
time.sleep(1)
subprocess.run([PY_, "-u", "-m", "foreman.run", "--resume", run_id], cwd=ROOT)
