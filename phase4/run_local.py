"""
Runs the remaining search steps one after another on this machine and keeps Windows from idle sleep while it runs.

    python phase4/run_local.py focal stage2 separate final
"""
import ctypes
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PACK = os.path.join(ROOT, "data", "colab_pack")
OUT = os.path.join(ROOT, "outputs", "phase4", "run")

if os.name == "nt":                      # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no idle sleep while this process lives
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

for step in sys.argv[1:]:
    print(f"=== {step}", flush=True)
    code = subprocess.call([sys.executable, os.path.join(HERE, "mlp_search.py"), step, "--pack", PACK, "--out", OUT])
    if code:
        print(f"=== {step} FAILED ({code})", flush=True)
        sys.exit(code)
print("=== ALL DONE", flush=True)
