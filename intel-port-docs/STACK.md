# STACK -- the validated software and firmware stack for the Intel port on intel-host (owner note 76)

Part of the Intel port of radiance (milestone 3). Each line is what the milestone's measurements ran on, with the
date and the steps that validated it. **Any change to a line goes through a test like owner note 69's first**:
- the list probe;
- the tiny model's tests;
- the M-pair with identical text;
- one traced rotation.

| component | version | where | validated |
|---|---|---|---|
| DPC++ / Intel LLVM | 7.2.0 | build image localhost/ri-m1build:3 | the whole milestone (images :1-:76): E-tests, M-pairs, rotations |
| IGC (Intel Graphics Compiler) | 2.28.4 | runtime image (igc-core / igc-opencl .debs) | the same |
| NEO (compute-runtime) | 26.05.37020.3 (Level Zero 1.14.37020) | runtime image (libze-intel-gpu1) | the same; the 64 KB command buffer key (OverrideCmdListCmdBufferSizeInKb) measured on it (X9, H28, 2026-10-29) |
| Level Zero loader | 1.29.0 | runtime image (libze1) | the same |
| Linux kernel | 7.2.6-070206 (xe) | intel-host host | the same |
| GuC firmware | 70.58.0 | intel-host /lib/firmware/xe/bmg_guc_70.bin | 2026-10-26 to 2026-10-29 04:43: M43 / M44 (the M-pair), H25-H28 (rotations), and every earlier m3 step; reinstalled 2026-10-29 06:20 (linux-firmware commit 86725d73) |
| HuC firmware | 8.2.10 | intel-host | the same |

Host settings these numbers were taken with:
- **Until 2026-10-29 04:53:** CPU boost on (5950X, amd-pstate-epp, powersave / balance_performance), GPU min_freq
  1200 / max_freq 2800 MHz, ccs timeslice 1000 us.
- **From 05:11:** boost off (cpu-boost-off.service).
- **min_freq:** 2800 from 06:44 (the owner's clock-lock arm B).

## Not validated or known-bad

- **GuC 70.72.1** (2026-10-29 04:53-06:20).
  - Two-card Flash-Next hung at its first step (R2c, R2d). Run 86 then showed the same hang on GuC 70.58.0 with
    boost off (R6), so the hang is not attributable to this GuC.
  - The all-reduce test's ~1.3 ms an exchange is the same on 70.58.0 (X12), and the "89-101 us" baseline it was
    compared with was milestone 2's build of the test. So the speed is not attributable either.
  - What stays open is the rank-divergent peer case (X11). It also reproduces on 70.58.0 with boost off (X12-X15),
    so it is not shown to be this GuC's either.
  - Untested: 70.60, 70.65. A clean A/B with boost on comes when the owner is back (owner note 82).
- **NEO 26.35.39758.10 / IGC 2.41.5**: not yet tested (owner note 69).

## Open: the two-card first-step hang with boost off

Since boost went off (05:11), every two-card Flash-Next load hung at its first step: R2c, R2d (GuC 70.72.1) and R6
(70.58.0). Every two-card step before that (boost on) worked. The engine-side trace is in STATE.md (runs 87-).

Run 87 located it in the driver stack, not the engine. The mover's first copy on card1 spins in NEO's bind of the
copy's buffers: `DRM_IOCTL_XE_VM_BIND` returns EBUSY after 60-80 ms in the kernel, and NEO re-invokes it without
end. The buffers are host memory the shared two-card context imported from card0 and had not mapped on card1. It
is the same with copies off the copy engine (R8) and with 8 GiB less pinned memory (R12).

Run 88 named it. Card1 cannot bind a host buffer of more than 4-8 GiB that it imports from card0: rank 1's
16.6 GiB host pool gets EBUSY forever (R15). A model-free probe's 8 GiB crashed the kernel (xe_res_next WARNING
under xe_pt_stage_bind, then a NULL dereference; m3 fault 22). 4 GiB binds. On this stack (kernel 7.2.6, NEO
26.05.37020.3) keep every host allocation of a two-card context under 4 GiB. Drafted for Intel in briefs/.

Run 93 found the mechanism and a fix.
- SYCL kernels use indirect USM access, so NEO makes every allocation of the shared two-card context resident on
  the launching card. Card1 then binds both ranks' host pools and card0's device memory as imports, and some of
  those VM_BINDs return EBUSY without end; whether a boot shows it varies.
- The fix (8674044 + 4fb114b, behind switches): `RADIANCE_HOST_POOL_PIECE_MIB=2048`,
  `RADIANCE_L0_EXPLICIT_RESIDENCY=1`, NEO's `NEOReadDebugKeys=1 DisableIndirectAccess=1`.
- Every baseline number holds with it, with identical text (R23, M64, H60, B32). It waits for one boost-0 pass
  before becoming the default.
- `DisableIndirectAccess` is a NEO debug key measured on 26.05.37020.3: recheck it after any NEO update.

NEO's `MakeEachAllocationResident` is not usable on this stack: =1 dumps core at startup (R13); =2 faulted card1's
copy engine at startup (R14, m3 fault 21).
