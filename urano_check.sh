#!/usr/bin/env bash
# Smoke test for urano. Run from the repo root:
#   nohup bash urano_check.sh > urano_check.log 2>&1 &
# Then send back urano_check.log.
set -u
echo "== host"; hostname; date
echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,compute_cap --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
echo "== python / torch"
python -c "import sys, torch; print(sys.version); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)"
python -c "import custom_esm; print('custom_esm at', custom_esm.__file__)" || echo "custom_esm NOT importable: copy customs/custom_esm into site-packages"

echo "== ESM3 load"
python -u - <<'PY'
import torch, time
from custom_esm.models.esm3 import ESM3
t = time.time()
m = ESM3.from_pretrained("esm3-open")
torch.cuda.synchronize()
print(f"load {time.time()-t:.1f}s  dtype {next(m.parameters()).dtype}  device {next(m.parameters()).device}")
print(f"after load: allocated {torch.cuda.memory_allocated()/2**20:.0f} MB  reserved {torch.cuda.memory_reserved()/2**20:.0f} MB")
PY

echo "== 200-move one_pass chain (T=32), peak memory in run_info.txt"
cd softmax/monte_carlo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 1 > ../../urano_smi.csv &
SMI=$!
python -u run_nsites.py --T 32 --moves 200 --seed 0 --grad-mode one_pass \
    --log-step 50 --print-step 50 --out urano_smoke/one_pass
echo "exit code $?"
kill $SMI
cat urano_smoke/one_pass/run_info.txt
echo "max nvidia-smi memory.used (MiB):"; sort -t, -k2 -n ../../urano_smi.csv | tail -1
