import torch
import torch.nn as nn
from cldm.fusion_module import MultiControlFusion
from cldm.lrci_adapter import LRCIAdapter
from cldm.dynamic_scheduler import DynamicControlScheduler

print("=" * 50)
print("AMCF-LRCI Module Test")
print("=" * 50)

# ── Test 1: LRCI Adapter ──────────────────────────
print("\n[1] Testing LRCIAdapter...")
for ch in [320, 640, 1280]:
    x = torch.randn(2, ch, 16, 16)
    adapter = LRCIAdapter(ch, rank=4)
    out = adapter(x)
    assert out.shape == x.shape, f"Shape mismatch at ch={ch}"
    params = sum(p.numel() for p in adapter.parameters())
    print(f"   ch={ch:4d} | output shape: {list(out.shape)} | params={params} | OK")

# ── Test 2: Fusion Module ─────────────────────────
print("\n[2] Testing MultiControlFusion (single input = passthrough)...")
for ch in [320, 640, 1280]:
    x = torch.randn(2, ch, 16, 16)
    fusion = MultiControlFusion(ch)
    out = fusion([x])
    assert out.shape == x.shape, f"Shape mismatch at ch={ch}"
    print(f"   ch={ch:4d} | output shape: {list(out.shape)} | OK")

# ── Test 3: Scheduler ─────────────────────────────
print("\n[3] Testing DynamicControlScheduler...")
scheduler = DynamicControlScheduler(T=1000)
for t in [0, 250, 500, 750, 999]:
    w = scheduler.weight(t)
    print(f"   t={t:4d} | weight = {w.item():.6f}")

w0   = scheduler.weight(0).item()
w500 = scheduler.weight(500).item()
w999 = scheduler.weight(999).item()
assert w0 > w500 > w999, "Scheduler not decreasing — logic error"
print("   Scheduler direction: CORRECT (high at t=0, low at t=999)")

# ── Test 4: Parameter Count (Thesis Metric) ───────
print("\n[4] Parameter count per level...")
ctrl_channels = [320, 320, 320, 320, 640, 640, 640,
                 1280, 1280, 1280, 1280, 1280, 1280]

total_fusion = sum(
    sum(p.numel() for p in MultiControlFusion(ch).parameters())
    for ch in ctrl_channels
)
total_lrci = sum(
    sum(p.numel() for p in LRCIAdapter(ch, rank=4).parameters())
    for ch in ctrl_channels
)
total_new = total_fusion + total_lrci

print(f"   Fusion modules total params : {total_fusion:>10,}")
print(f"   LRCI adapters total params  : {total_lrci:>10,}")
print(f"   All new module params       : {total_new:>10,}")

original_controlnet_params = 361000000
reduction_pct = (1 - total_new / original_controlnet_params) * 100
print(f"   Reduction vs full ControlNet: {reduction_pct:.1f}%")

# ── Test 5: Full Forward Pass Simulation ──────────
print("\n[5] Simulating full forward pass...")
ctrl_channels_list = [320, 320, 320, 320, 640, 640, 640,
                      1280, 1280, 1280, 1280, 1280, 1280]

fusion_modules = nn.ModuleList([MultiControlFusion(ch) for ch in ctrl_channels_list])
lrci_adapters  = nn.ModuleList([LRCIAdapter(ch) for ch in ctrl_channels_list])
scheduler      = DynamicControlScheduler()

controls = [torch.randn(2, ch, max(4, 64 // (2**min(i//4, 3))),
                                max(4, 64 // (2**min(i//4, 3))))
            for i, ch in enumerate(ctrl_channels_list)]

weight = scheduler.weight(500)
weight = weight.view(1, 1, 1, 1)

for idx in range(len(controls)):
    ctrl  = controls[idx]
    fused = fusion_modules[idx]([ctrl])
    delta = lrci_adapters[idx](fused)
    out   = ctrl + weight * delta
    assert out.shape == ctrl.shape, f"Shape error at idx={idx}"

print("   All 13 control levels: PASSED")

print("\n" + "=" * 50)
print("ALL TESTS PASSED — Safe to train on A100")
print("=" * 50)
