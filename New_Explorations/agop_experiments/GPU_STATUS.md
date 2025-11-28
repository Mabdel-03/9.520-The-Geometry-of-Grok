# GPU Status and Availability on Cluster

**Date:** November 26, 2024

## Available Compatible GPUs (sm_70+)

Based on `sinfo` output:

**Available (not drained):**
- RTX 2080/2080Ti (nodes 071-076, 085-087) - sm_75 ✅
- Quadro RTX 6000 (nodes 078-084, 088-092) - sm_75 ✅  
- RTX A6000 (nodes 093-094, 097-098) - sm_86 ✅
- DGX V100 (dgx001-002) - sm_70 ✅ (idle!)

**Unavailable (drained/maintenance):**
- A100 nodes (100-116, apollo) - sm_80 ❌ Under maintenance
- Many other nodes drained

**Incompatible:**
- GTX 1080 Ti (nodes 055-070, 077) - sm_61 ❌

## Issue with A100 Request

Jobs requesting `gpu:a100:1` are pending because A100 nodes are unavailable:
```
ReqNodeNotAvail, UnavailableNodes:node[001-037,039-054,056,060,065,071,074,077,083,089-090,093,100-116]
```

## Solution

Use RTX 2080/Quadro/DGX GPUs which ARE available:
- Change to: `--gres=gpu:GEFORCERTX2080:1` or
- Change to: `--gres=gpu:QUADRORTX6000:1` or  
- Better: Just use CPU (slower but guaranteed to work)

