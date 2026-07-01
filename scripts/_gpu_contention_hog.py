"""Stand-in for CosyVoice sharing the GPU: continuous fp16 matmuls that steal SM cycles
from MuseTalk's render, the same way vLLM bursts the card while streaming a reply. A COMPUTE
hog, not a memory hog. Use it to reproduce the shared-GPU long-turn A/V drift offline (run it
alongside scripts/_drive_frames.py) and to prove MUSETALK_TRT=1 keeps the render >=fps under
contention. See docs/PROBLEMS-AND-FIXES.md P16.

Run in the musetalk env; Ctrl-C / Stop-Process to stop:
  E:\\miniconda3\\envs\\musetalk\\python.exe -u scripts/_gpu_contention_hog.py [N]
N = matmul size (default 4096 = heavy, forces render < 12fps; 2048 = light, absorbed by headroom).
"""
import sys, time
import torch

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
ITERS = 120 if N >= 4096 else 40
d = torch.device("cuda")
a = torch.randn(N, N, device=d, dtype=torch.float16)
b = torch.randn(N, N, device=d, dtype=torch.float16)
print(f"gpu contention hog running (N={N}, iters/loop={ITERS})", flush=True)
while True:
    for _ in range(ITERS):
        a = (a @ b) * 0.0001 + 0.5
    torch.cuda.synchronize()
