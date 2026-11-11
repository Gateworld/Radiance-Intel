# DRAFT, not sent: xe oops binding a >4 GiB host buffer imported from another xe card (owner note 76)

Part of the Intel port of radiance (milestone 3, run 88; updated 2026-10-30 with a source reading of 7.2.6 and the
after-reset stalls). This is a draft bug report for Intel's drm/xe and compute-runtime trackers. It is for the owner
to send or edit; nothing here has been sent.

## Summary

In a Level Zero context that spans two Battlemage cards (Arc Pro B70, 0x8086:0xe223), host USM allocated on the
context is imported by the second card. Binding that import for a copy on the second card fails once the allocation
is somewhere between 4 GiB and 8 GiB or larger:
- **8 GiB:** the kernel warns in `xe_res_next` (`xe_res_cursor.h:167`) inside `xe_pt_stage_bind`, then
  dereferences NULL. A second oops follows in the process's `xe_file_close`, and the kernel asks for a reboot.
- **16.6 GiB:** `DRM_IOCTL_XE_VM_BIND` returns `EBUSY` every ~80 ms forever (each call spends ~78 ms in the
  kernel). NEO re-invokes it, so the application hangs in `zeCommandListAppendMemoryCopy`.

1.5 GiB and 4 GiB bind on both cards. The first card binds every size in under a millisecond.

## Stack

- Kernel 7.2.6-070206-generic (mainline build), xe, GuC 70.58.0, HuC 8.2.10.
- compute-runtime 26.05.37020.3 (Level Zero 1.14.37020), IGC 2.28.4, Level Zero loader 1.29.0.
- Intel oneAPI DPC++ (intel/llvm) 7.2.0.
- Two Arc Pro B70 (BMG-G31, 0x8086:0xe223): one ASRock board, one Intel board. They sit in an AMD X570 board
  (Gigabyte X570 AORUS PRO, BIOS F39; Ryzen 9 5950X, 128 GB), with the IOMMU in translated mode (DMA-FQ).

## Reproducer (SYCL, ~100 lines; xe_hostbind_probe in the radiance tree)

```
sycl::context ctx(sycl::device::get_devices(sycl::info::device_type::gpu));   // both cards
char* p = sycl::aligned_alloc_host<char>(4096, 8ull << 30, ctx);               // 8 GiB host USM
for (size_t b = 0; b < (8ull << 30); b += 4096) p[b] = 1;                       // populate
sycl::queue q0(ctx, devs[0], sycl::property::queue::in_order{});
sycl::queue q1(ctx, devs[1], sycl::property::queue::in_order{});
char* s0 = sycl::malloc_device<char>(4096, devs[0], ctx);
char* s1 = sycl::malloc_device<char>(4096, devs[1], ctx);
q0.memcpy(s0, p, 64).wait();   // fine, < 1 ms
q1.memcpy(s1, p, 64).wait();   // 8 GiB: kernel oops below; 16.6 GiB: VM_BIND EBUSY forever
```

## Kernel log (8 GiB case, 2026-10-29 13:22:41 UTC)

```
WARNING: drivers/gpu/drm/xe/xe_res_cursor.h:167 at xe_res_next+0x162/0x2f0 [xe], CPU#1: xe_hostbind_pro/296613
 xe_pt_walk_range+0x107/0x2f0 [xe]
 xe_pt_walk_range+0x28c/0x2f0 [xe]  (x3)
 xe_pt_stage_bind+0x39e/0xb00 [xe]
 xe_pt_update_ops_prepare+0xd3/0x110 [xe]
 xe_vm_bind_ioctl+0xc8e/0xd30 [xe]
 drm_ioctl+0x310/0x600
 xe_drm_ioctl+0x61/0xb0 [xe]
BUG: kernel NULL pointer dereference, address: 0000000000000018
Oops: Oops: 0000 [#1] SMP NOPTI
RIP: 0010:xe_res_next+0x166/0x2f0 [xe]
...
(13:23:02, the same process's other thread on exit)
BUG: unable to handle page fault for address: ffffd04716f1f8e0
Oops: Oops: 0000 [#2] SMP NOPTI
RIP: 0010:__ww_mutex_lock.constprop.0+0x99c/0xd20
 drm_exec_lock_obj+0x43/0x230 [drm_exec]
 xe_vm_lock_vma+0x56/0x60 [xe]
 xe_vma_destroy_unlocked+0x88/0xf0 [xe]
 xe_vm_close_and_put+0x4cb/0x8f0 [xe]
 xe_file_close+0x107/0x1e0 [xe]
Fixing recursive fault but reboot is needed!
```

The full logs are in the radiance-intel work tree (logs/HB1-oops1-klog.txt, logs/HB1-oops-klog.txt).

## The 16.6 GiB case (no oops, a hang)

- NEO's stack, from the 26.05.37020.3 dbgsym: `zeCommandListAppendMemoryCopy -> ...flushBcsTask /
  flushImmediateTask -> DrmCommandStreamReceiver::flush -> DrmMemoryOperationsHandlerBind::makeResidentWithinOsContext
  -> DrmAllocation::bindBOs -> BufferObject::bind -> IoctlHelperXe::xeVmBind -> Drm::ioctl`.
- strace: `ioctl(renderD129, DRM_IOCTL_XE_VM_BIND, ...) = -1 EBUSY <0.0785>`, 62 times in 5 s.
- On the importing card, `/proc/<pid>/fdinfo` shows the shared host BOs in the system placement, total above
  resident.
- The same with copies kept off the copy engine (`UR_L0_USE_COPY_ENGINE=0`).

## A second limit: the total imported

With the host memory split into allocations of at most 2 GiB (the single-buffer case avoided), the second card
still stops binding imports.
- In one process it bound twelve 2 GiB imports plus three smaller ones, ~26 GiB in all; the thirteenth 2 GiB
  `DRM_IOCTL_XE_VM_BIND` returned EBUSY without end (2026-10-29 17:3x UTC).
- Nine 2 GiB imports (18 GiB) bound fine in the reproducer. So there is a ceiling on the total a card can import,
  somewhere between 18 and ~26 GiB, separate from the single-buffer limit.
- In a later run, the second card's first step stuck the same way (EBUSY every ~7 ms) with the 2 GiB pieces. Its
  fdinfo then showed `drm-total-gtt` 53.4 GiB and `drm-total-system` 10.5 GiB of imports not yet bound.

## Why it surfaced only on 2026-10-29

What we can say:
- The same application, with the same 16 GiB-a-card host pool, ran two-card loads for days (2026-10-26 to the morning of 10-29) without this hang.
- From the first load after the 05:11 UTC reboot on 2026-10-29 (host CPU boost turned off), every two-card load
  hung at its first step. The GuC firmware was changed and changed back across the same reboots; the hang appeared
  on both versions.
- The pool's size and placement did not change. So what changed must be whether, or when, the second card first
  binds that import.
- A guess, not shown: with a slower host, the application's first copies on card1 come before something else
  (in NEO or the application) has bound the import on that card in a way that works. We have not found which path
  bound it before.
- What we can say for certain is that a direct bind of a >4 GiB import on the second card fails today (hang at
  16.6 GiB, oops at 8 GiB), in a 30-line reproducer with nothing else running.

## Also seen: the EBUSY loop on a small import when the importing card is fuller (2026-10-29 22:28, run 95)

With every import at most 2 GiB and explicit residency, the engine starts at a 2560 MiB VRAM headroom. At 2048 MiB
(about 0.5 GiB more VRAM in use a card) it never finishes starting, every time (3 of 3 loads).
- The stuck call: zeContextMakeMemoryResident of a 1.54 GiB pinned host allocation (owned by card0, imported by
  card1), on card1.
- strace: `DRM_IOCTL_XE_VM_BIND = -1 EBUSY` on card1, ~5 ms a call, 382 calls in 3 s, and NEO re-invokes forever.
- Card1's fdinfo at that moment: drm-total-system 17,966 MiB / drm-resident-system 1,580 MiB, drm-total-vram0
  27.0 GiB of a 32 GB card, drm-total-gtt 44 GiB.
- So EBUSY comes back without a large import and with ~4 GiB of VRAM still unused. It looks like the same no-exit
  path as above, reached through VRAM pressure on the importing card rather than import size.

## More of the same (2026-10-30, run 97): which imports bind on the second card varies

Every case is the same EBUSY loop on card1: host memory owned by card0 and imported by card1, at allocation, with
explicit residency.
- **The store with less pressure:** a 1.54 GiB import stalls even when bound before the weights load (M76).
- **The host pool's later pieces:** 2 GiB host-pool imports stall after ~5-6 pieces once ~0.9 GiB more VRAM is in
  use (H70).
- **The card's state does not decide it:** with MTP off, a fifth host-pool piece (1.6 GiB) stalls even on an
  otherwise empty card (250 MiB in use, M81). A 16 GiB host pool in nine pieces bound in the same form at 19:16
  (B32) but not at 03:33 (B37, after 10 h of uptime).
- **Small imports bind:** 2 GiB pieces up to 8 GiB, then 64 MiB buffers and a 225 MiB store (H73-H75, M80, M82).

No pattern of size, count or free VRAM fits all of these. It behaves like the first-step hang: it depends on the
boot's state, and each stall is the same `DRM_IOCTL_XE_VM_BIND = -1 EBUSY` retried forever.

## After a GT reset, the second card's import binds stop after a few GiB, even on an empty card (2026-10-30)

- At 10:08 UTC card1 had a GT reset (the teardown of a killed process: "Schedule disable failed to respond",
  reset from `guc_exec_queue_timedout_job`). Before it, on the same boot, card1 bound an 8 GiB host pool in 2 GiB
  imports several times.
- After it, every load stalled in the same EBUSY loop, after a cumulative ~2-9.5 GiB of imports per process:
  - 2 GiB pieces: 3 bound, then EBUSY;
  - 1 GiB pieces: 2 bound, then EBUSY;
  - with every import made before any VRAM allocation, card1 at 250 MiB of VRAM in use: three 2 GiB pieces bound,
    the fourth looped (~6.9 GiB).
- Only a reboot cleared it, as after two earlier resets (2026-10-29).

## Where the EBUSY comes from (our reading of the 7.2.6 source; not instrumented yet)

Confirmed by reading:
- The import's first bind validates the BO; for an import that runs the exporter's `xe_dma_buf_map()`. There,
  **any failure of `dma_map_sgtable(attach->dev, ...)` returns `-EBUSY`** (`drivers/gpu/drm/xe/xe_dma_buf.c`,
  the `error_free` label, ~line 150), whatever the DMA layer's error was (-EINVAL, -ENOMEM, ...). Userspace cannot
  tell a transient condition from a permanent failure, and NEO retries forever.
- The sg table is built by `drm_prime_pages_to_sg()` with `dma_max_mapping_size()` of the **exporting** device as
  the segment limit (`drivers/gpu/drm/drm_prime.c` ~850-867), which is effectively UINT_MAX here. The importing
  xe device sets `dma_set_max_seg_size()` to `xe_sg_segment_size()`, 0x7FFFF000 (`xe_device.c` ~706), and
  `xe_bo.h`'s comment for it says why: `iommu_dma_map_sg()`'s padding can overflow `sg->length` to 0. xe's own TT
  path respects that limit; the dma-buf export path does not, so a physically contiguous run of more than 2 GiB
  becomes one segment that the importer then maps through `iommu_dma_map_sg()`.
- In `iommu_dma_map_sg()` (`drivers/iommu/dma-iommu.c`), `prev->length += pad_len` (~1496) and `max_len - cur_len`
  (~1326) are unsigned-int arithmetic on such segments.

Inferred:
- The overflows give either `-EINVAL` from the short `iommu_map_sg()` (then EBUSY, the same on every retry: the
  hang) or a DMA list shorter than the buffer, whose walk runs off the end: `XE_WARN_ON(!sgl)` at
  `xe_res_cursor.h:167` and the NULL read in `xe_res_next`, which is our 8 GiB oops. The second oops (in
  `__ww_mutex_lock` on file close) looks like a consequence of the first thread dying with the resv lock held.
- `iommu_dma_unmap_sg()` stops at the first zero-length segment, so an overflowed mapping is only partly unmapped
  while its IOVA range is freed. Stale IOMMU entries like that would fail later maps (`-EADDRINUSE`, then EBUSY)
  until a reboot, which would fit the after-reset ceiling. We have not shown that this is the mechanism; IOMMU
  page-table allocation failing under `GFP_ATOMIC` is the other candidate.
- Runs of more than 2 GiB are most likely on a freshly booted 128 GB host, which may be why this appeared right
  after a reboot.
- For comparison, our other machine runs four AMD cards (R9700) with ROCm and `iommu=pt` (identity domains),
  and each card maps pinned host memory itself; that path never meets this code. We have not yet tried `iommu=pt`
  on the Intel machine.

## Related: kernels cannot read zexDriverImportExternalPointer memory (compute-runtime)

As a workaround we registered one contiguous mmap in 2 GiB pieces with zexDriverImportExternalPointer.
- Copies from it work on both cards.
- A SYCL kernel reading it reads zeros and page-faults on both cards (`Engine memory CAT error [18]: class=ccs`,
  2026-10-29 13:55:02).
- If kernel access to imported pointers is unsupported, the API documentation could say so.

## Asks

1. xe: build the exported sg table with the importing device's segment limit (`dma_get_max_seg_size(attach->dev)`,
   0x7FFFF000 for xe) instead of `drm_prime_pages_to_sg()`'s exporter-wide default, so no segment can overflow
   in `iommu_dma_map_sg()`. (Possibly a drm_prime fix for every driver: the helper uses the exporter's limit for a
   table that is mapped for the importer.)
2. xe: return the DMA layer's error from `xe_dma_buf_map()` instead of `-EBUSY`, so a permanent failure is not
   reported as "try again".
3. compute-runtime: a bounded retry (or an error) instead of re-invoking VM_BIND on EBUSY forever.
4. If our reading is wrong, the tracepoints or debug output you would want from us; we can run kretprobes on
   `xe_dma_buf_map`, `iommu_dma_map_sg` and `iommu_map_sg` on request.
