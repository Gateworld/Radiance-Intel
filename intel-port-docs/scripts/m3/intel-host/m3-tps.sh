#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 1 step 7; milestone 2 stage 1: tok/s after each tuned kernel).
# Inside ri-m1rt on card1, started by D/m1-step.sh with STEP_ENGINE=1. For each config: serve the
# production model, send /out/card/tps.body REPS times as a streamed request, then stop the server.
#   m2-tps.sh mtp,nomtp,plain,v1,v1nomtp [REPS=3]   (milestone 3: graph = RADIANCE_L0_TAPE=graph (the SYCL-graph tapes, the A/B), default = native command lists; milestone 2: v1 = RADIANCE_XE_MOE_V1=1, the old expert GEMV, the A/B)
#   mtp: /out/flashnext-card1.args; nomtp: --num-speculative-tokens 0; plain: RADIANCE_L0_TAPE=0
# Per config, /out/CFG/: serve.log (-v), and per run K:
#   tps-K.sse    one line per SSE line: "<epoch s.us> <cgroup usage_usec> <the line>"
#   stats-K.txt  /stats every STATS_PERIOD (0.5) s during the run: "T <epoch s.us> <usage_usec>" then
#                the body
# Only shell builtins in the per-chunk path (EPOCHREALTIME, read), so the sampling costs little.
H=/opt/ri/radiance_home; P=18421
CFGS=${1:-mtp}; CFGS=${CFGS//,/ }; REPS=${2:-3}   # "mtp,nomtp,plain": ssh flattens quoted spaces
CG=/sys/fs/cgroup/cpu.stat
usage() { local k v; while read -r k v; do [ "$k" = usage_usec ] && { echo "$v"; return; }; done < $CG; echo 0; }
get_stats() {   # one GET /stats (or the path in $1), body to stdout
  exec 4<>/dev/tcp/127.0.0.1/$P || return
  printf 'GET %s HTTP/1.1\r\nHost: l\r\nConnection: close\r\n\r\n' "${1:-/stats}" >&4
  local line; while IFS= read -r line <&4; do [ "$line" = $'\r' ] && break; done
  cat <&4; exec 4<&-
}
sampler() {     # /stats every STATS_PERIOD s (0.5: it costs the capped engine CPU) until the flag file goes
  while [ -e "$1" ]; do echo "T $EPOCHREALTIME $(usage)"; get_stats; echo; sleep "${STATS_PERIOD:-0.5}"; done
}
stream_req() {  # body-file out.sse
  local body line; body=$(cat "$1")
  exec 3<>/dev/tcp/127.0.0.1/$P || return 1
  printf 'POST /v1/completions HTTP/1.1\r\nHost: l\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s' \
    "${#body}" "$body" >&3
  while IFS= read -r line <&3; do echo "$EPOCHREALTIME $(usage) $line"; done > "$2"
  exec 3<&-
}
echo "cpu.stat: $(head -3 $CG | tr '\n' ' ')  cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
BODY=/tmp/tps-stream.body
sed 's/}$/,"stream":true}/' /out/card/tps.body > $BODY
for CFG in $CFGS; do
  O=/out/$CFG; mkdir -p $O
  ARGS=$(grep -v '^#' /out/flashnext-card1.args | tr '\n' ' ')
  # owner note 175 (run 169, 11-04): libavx serves the host rows ahead of libref in every step (its io_uring n-gram
  # gather; mstep.sh runs the container under note 45's seccomp profile; the image carries libavx.so from 193 on)
  ARGS=${ARGS/--kernels libxe:libref/--kernels libxe:libavx:libref}
  ENVV=()
  case $CFG in
    mtp) ;;
    nomtp) ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    plain) ENVV=(RADIANCE_L0_TAPE=0) ;;
    graph) ENVV=(RADIANCE_L0_TAPE=graph RADIANCE_L0_TAPE_STRETCHES=128) ;;
    graphnomtp) ENVV=(RADIANCE_L0_TAPE=graph RADIANCE_L0_TAPE_STRETCHES=128); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    v1) ENVV=(RADIANCE_XE_MOE_V1=1) ;;
    v1nomtp) ENVV=(RADIANCE_XE_MOE_V1=1); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    plainnomtp) ENVV=(RADIANCE_L0_TAPE=0); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # owner note 24: the compute runtime's debug keys (NEOReadDebugKeys=1 enables them in a release build,
    # per its FAQ); LogAllocationSummaryReport prints every allocation type's bytes by memory pool at exit;
    # nomtphost adds ForceSystemMemoryPlacement = bits (type-1) of commandBuffer 3, indirectObjectHeap 9,
    # internalHeap 11, linearStream 16, surfaceStateHeap 27 = 0x4008504: command lists and their heaps
    # in system memory; scratch (23) and kernel ISA (14) stay in VRAM.
    nomtplog)  ENVV=(NEOReadDebugKeys=1 PrintDebugSettings=1 LogAllocationSummaryReport=1)
               ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # milestone 3, run 42 (the pool accounting): the service setting (MTP on, the gate, 4 GiB headroom) with the
    # driver's allocation summary at exit, VRAM sampled beside it (STEP_VRAM=1): bytes per allocation type a tape
    mtplog)    ENVV=(RADIANCE_SPEC_MAX_SEQS=1 NEOReadDebugKeys=1 PrintDebugSettings=1 LogAllocationSummaryReport=1)
               ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
    # run 42: the service setting with the recorder using the runtime's kernel handles instead of its own kernel
    # objects (T8 compared this at the 2 GiB headroom, where both runs sat at the ceiling): A1 grew 4.8 GB on the
    # first prompt with 112 own kernel objects, and the object count tracks the VRAM steps on the rotation
    rtkernels4g) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_L0_NTAPE_OWN_KERNELS=0)
               ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
    # run 42: the service setting with the driver's command lists and their heaps in system memory (owner note 24's
    # key: ForceSystemMemoryPlacement bits commandBuffer 3, indirectObjectHeap 9, internalHeap 11, linearStream 16,
    # surfaceStateHeap 27 = 0x4008504; scratch and kernel ISA stay in VRAM): A1/A2 showed the first prompt's
    # recording takes 4.5 GB of VRAM that the driver never gives back -- is it the lists' heaps?
    mtphost4g) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 NEOReadDebugKeys=1 ForceSystemMemoryPlacement=0x4008504)
               ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
    # run 45: the service setting itself (MTP on, the gate, 4 GiB headroom), nothing else: the VRAM and speed reference
    mtp4g)     ENVV=(RADIANCE_SPEC_MAX_SEQS=1); ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
    # milestone 3 stage C (owner note 47, run 50): the service setting with the heat engine's minimum residency age
    # of 16 steps (RADIANCE_PLACE_MIN_AGE; m2-heat.sh's mtp4gage16 is the rotation's twin): the single warm prompt
    mtp4gage16) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_PLACE_MIN_AGE=16); ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
    # owner note 47 item 2 (run 52): cooperative verify (upstream d912c69, picked as 8b9a8bb) on and off at the service
    # setting: RADIANCE_MTP_COUPLED=1 makes the linear MTP drafter sample from its reranked top-8 and the sampler couple
    # the target's draw to it; the control is the same image with the flag unset
    mtp4gcoupled) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_MTP_COUPLED=1); ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}
                  ARGS=${ARGS/--kernels libxe:libref/--kernels libxe:libref:spechub} ;;
    # run 43 (A4): A3's bits plus the scratch surfaces (type 23) in system memory, the debug settings echoed so the
    # serve log shows the key took: if the first prompt's +4.5 GB of VRAM vanishes, the per-list scratch is the pool item
    mtphost4gscratch) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 NEOReadDebugKeys=1 PrintDebugSettings=1 ForceSystemMemoryPlacement=0x4808504)
               ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
    plainnomtplog) ENVV=(RADIANCE_L0_TAPE=0 NEOReadDebugKeys=1 PrintDebugSettings=1 LogAllocationSummaryReport=1)
               ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    rtkernels) ENVV=(RADIANCE_L0_NTAPE_OWN_KERNELS=0)
               ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    mtpskipdraft) ENVV=(RADIANCE_TAPE_SKIP=-1,1,2,3) ;;
    mtpskiptrunk) ENVV=(RADIANCE_TAPE_SKIP=0) ;;
    mtpskipall) ENVV=(RADIANCE_TAPE_SKIP=-1,0,1,2,3) ;;
    mtpskiphist) ENVV=(RADIANCE_TAPE_SKIP=-1) ;;
    mtpskiprounds) ENVV=(RADIANCE_TAPE_SKIP=1,2,3) ;;
    # run 18: the all-reduce's poll budget at 2e7 (~6 s) so a stalled collective reports itself in seconds
    mtpspin) ENVV=(XE_AR_SPIN=20000000) ;;
    # run 20: half the all-reduce's blocks a launch, so two launches overlapping on a card stay resident
    # (M10/M12: both compute engines 100 % busy with every rank's collectives spinning for the peer's)
    mtpblocks32) ENVV=(XE_AR_BLOCKS=32) ;;
    # run 27: the tape profiler (device timestamps around every replayed command; a per-kernel table in serve.log
    # every 8192 harvested lists)
    profnomtp) ENVV=(RADIANCE_L0_NTAPE_PROFILE=1); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # milestone 3, owner note 56: the hc mixing matrices expanded to FP16 at load (RADIANCE_XE_HC_F16=1)
    hcf16nomtp) ENVV=(RADIANCE_XE_HC_F16=1); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    hcf16mtp)   ENVV=(RADIANCE_XE_HC_F16=1) ;;
    # owner note 58's two-rank check (the tiny model's experts are too small for TP2): Flash-Next MTP off with the
    # first 3 tape appends failing as out of device memory -- retried, so the text must equal nomtp's
    fakeappendnomtp) ENVV=(RADIANCE_L0_FAKE_APPEND_FAIL=3); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # run 75 (owner note 59's "the M-pair at the 2 GiB floor"): the single prompt at the service's VRAM setting --
    # 4 GiB headroom, the 2 GiB recordings floor, no shedding, the gate (run.sh)
    svcnomtp) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    svcmtp)   ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
    # run 90 (owner note 68 item 1): the M-pair under note 68's two settings, untraced -- the early segment release
    # (RADIANCE_L0_TAPE_EARLY=1, the tree's default) and NEO's 64 KB command buffer per list
    svcnomtpe64) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=1
                NEOReadDebugKeys=1 OverrideCmdListCmdBufferSizeInKb=64)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # M54's MTP-on loss (48.93 against M44's 51.36), taken apart on the same image: the early release off (the old
    # code path), and the early release on without NEO's key
    svcmtpe0) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
    svcmtpe1) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
    # run 90 (owner note 65 item 4): the service with the key and the headroom at 3 GiB (1 GiB more for the experts)
    svcnomtpk64h3) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                NEOReadDebugKeys=1 OverrideCmdListCmdBufferSizeInKb=64)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 3072}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    svcmtpk64h3) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                NEOReadDebugKeys=1 OverrideCmdListCmdBufferSizeInKb=64)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 3072} ;;
    svcnomtpk64) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                NEOReadDebugKeys=1 OverrideCmdListCmdBufferSizeInKb=64)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    svcmtpk64) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                NEOReadDebugKeys=1 OverrideCmdListCmdBufferSizeInKb=64)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
    svcmtpe64) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=1
                NEOReadDebugKeys=1 OverrideCmdListCmdBufferSizeInKb=64)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
    # run 76 (owner note 61 item 0): svcmtp with the recordings' VRAM charged by window (RADIANCE_L0_VRAM_TRACE, image :71)
    vtracemtp) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_VRAM_TRACE=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
    # run 79, owner note 53 step 1: the service setting with the per-layer host step (RADIANCE_HOST_EXPERTS=1), against
    # svcnomtp / svcmtp (M43 / M44): the step's cost per token, and the text must be identical
    # (M45, run 79: serve refuses the libref-only op without --debug-accept-reference-kernels; it is step 1's no-op,
    # and no other op of this model resolves to libref, so the flag admits exactly these 48)
    hostexpnomtp) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_HOST_EXPERTS=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}
              ARGS="$ARGS --debug-accept-reference-kernels" ;;
    # run 79 (M45b): the same with the backend's host waits spinning instead of sleeping -- is the ~245 us a layer the
    # host's wake-up or the card's drained pipeline?
    hostexpspinnomtp) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_HOST_EXPERTS=1 RADIANCE_L0_WAIT=spin)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}
              ARGS="$ARGS --debug-accept-reference-kernels" ;;
    hostexpmtp) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_HOST_EXPERTS=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS="$ARGS --debug-accept-reference-kernels" ;;
    # run 80, owner note 65: the service setting with the early segment release, traced (against M43 / M44 and V2)
    earlynomtp) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=1 RADIANCE_L0_VRAM_TRACE=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    earlymtp) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=1 RADIANCE_L0_VRAM_TRACE=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
    # run 87 (owner note 82): svcnomtp with the shipped segment release (an image whose default is the early one)
    svcnomtpold) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # run 87 (R10: the hung bind is an xe VM_BIND failing EBUSY in the kernel's validate): svcnomtpold with the VRAM
    # trace, is card1 overcommitted at the hang; and H28's setting (the early release + NEO's 64 KB command buffer),
    # MTP off, the remedy if it is
    svcnomtpoldvt) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_L0_VRAM_TRACE=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # run 87 (R11: card0's GTT 61.19 GiB at the hang against ttm.pages_limit 60.73 GiB): svcnomtpold with production's
    # 12 GiB host pool a rank instead of 16 (rank 1's tier uses 8.92 GiB) -- does the hang go with ~8 GiB less pinned?
    svcnomtpoldhp12) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}
              ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 12288} ;;
    # run 87 (R12: the hang stays with the 12 GiB pool; card1's unmapped imports of the shared context's host BOs are
    # what its first copy binds): svcnomtpold with NEO binding every allocation on every engine at its creation
    svcnomtpoldmer1) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                NEOReadDebugKeys=1 MakeEachAllocationResident=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # R13: =1 dumps core at startup; =2 binds every allocation at each flush (the first is at startup)
    svcnomtpoldmer2) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                NEOReadDebugKeys=1 MakeEachAllocationResident=2)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # run 88 (owner note 84): svcnomtpold with every host allocation touched from each rank's mover and compute
    # streams before the first step (RADIANCE_L0_HOST_PREBIND, image :77)
    svcnomtpoldpb) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_L0_HOST_PREBIND=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # run 92 (owner note 90 item 3): R15's form with the host pool in pieces of 2 GiB (8674044, image :81)
    svcnomtpoldpbpc) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_L0_HOST_PREBIND=1 RADIANCE_HOST_POOL_PIECE_MIB=2048)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # run 93 (owner note 91 item 1): the fix alone, the prebind off -- card1 binds only what rank 1 uses
    svcnomtppc) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # run 93 (HB7: a kernel's indirect host access makes the card bind every host allocation; with peer access on,
    # card1 also maps card0's VRAM): R20 with the collectives through host memory and no peer access
    svcnomtppcnop2p) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}
              ARGS="$ARGS --p2p off" ;;
    # run 93 (HB7 b, R21): the pieces + explicit residency at allocation + NEO's indirect access off
    svcnomtppcres) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # run 93 (owner note 65 item 4's text check, M60 / M61 cut by run 90's hangs): the full candidate service setting --
    # the two-card fix, NEO's 64 KB command buffer, the headroom at 3 GiB
    svcnomtpall) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 3072}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    svcmtpall) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 3072} ;;
    # run 97 (owner note 95 item 3): step B's sizes (floor 1024, headroom 2048) with no checkpoint store, traced -- is
    # B's startup stall the store's bind on card1?
    svcmtpBck0) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=1024 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1)
              ARGS="$ARGS --checkpoint-slots 0" ;;
    # run 97: H71's setting on the single prompt -- the budget reads the real card, host pool 9 GiB, the store
    # pageable, headroom 2560, floor 256 (MTP on; svcnomtpcap the same with MTP off)
    svcmtpcap) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_CKPT_PAGEABLE=1 RADIANCE_VRAM_CAP_FROM_FREE=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 9216} ;;
    svcnomtpcap) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_CKPT_PAGEABLE=1 RADIANCE_VRAM_CAP_FROM_FREE=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 9216}
              ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # run 97: H73's setting (note 95 item 3's candidate): the real-card budget, host pool 9 GiB, a pinned store of 2
    # slots, headroom 2560, floor 256 -- MTP on / off
    svcmtpcapck2) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 9216}
              ARGS="$ARGS --checkpoint-slots 2" ;;
    svcnomtpcapck2) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 9216}
              ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}; ARGS="$ARGS --checkpoint-slots 2" ;;
    # the same with the host pool at 8 GiB: four 2 GiB pieces, no fifth import on card1 (M79 / M81 stalled at it, MTP off)
    svcmtpcapck2hp8) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
              ARGS="$ARGS --checkpoint-slots 2" ;;
    svcnomtpcapck2hp8) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
              ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}; ARGS="$ARGS --checkpoint-slots 2" ;;
    # run 99 (owner note 105 item 1): the candidate with the trunk W8A8 at 2-8 rows on the matrix unit, MTP on / off
    # ...fuse: also ar_hc_write's residual write inside the all-reduce (note 105 item 4); fuse alone: svc(no)mtpcapfuse
    svcmtpcaprows|svcnomtpcaprows|svcmtpcaprowsfuse|svcnomtpcaprowsfuse|svcmtpcapfuse|svcnomtpcapfuse)
              ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256
                RADIANCE_L0_TAPE_EARLY=0 RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1
                DisableIndirectAccess=1 OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1)
              case $CFG in *rows*) ENVV+=(RADIANCE_XE_GEMM_DPAS_ROWS=1) ;; esac
              case $CFG in *fuse) ENVV+=(RADIANCE_XE_AR_HC_FUSE=1) ;; esac
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
              case $CFG in svcnomtp*) ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;; esac
              ARGS="$ARGS --checkpoint-slots 2" ;;
    # run 100: the candidate (MTP on / off) with any of run 99-100's opt-in forms, named by tokens after the dash:
    # rows (note 105 item 1), fuse (item 4), hcsr (item 3: hc at 2-63 rows), moe8 (note 104: 8-lane teams + 4 rows a
    # team at K <= 256), h2 (note 104: f16x2 pairs); e.g. svcmtpx-rows-fuse, svcnomtpx-moe8, svcmtpx-base (none)
    svcmtpx-*|svcnomtpx-*) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256
                RADIANCE_L0_TAPE_EARLY=0 RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1
                DisableIndirectAccess=1 OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1)
              # run 102 (owner note 112 item 1): the m3 configs turn on the fail-fast exit and host-first (gates E33 /
              # M116 / M117 / M123); `noguard` drops both, for a comparison with the older runs
              case "$CFG-" in *-noguard-*) ;; *) ENVV+=(RADIANCE_STALL_EXIT_S=60 RADIANCE_HOST_FIRST=1) ;; esac
              # run 109 (owner note 126 item 5): the expert matrix form's staging (5de6d9b, bit-identical) is an m3
              # default; `nost` drops it
              case "$CFG-" in *-nost-*) ;; *) ENVV+=(RADIANCE_XE_MOE_MX_STAGE=1) ;; esac
              # run 104 (owner note 120): the all-reduce's LL form is the m3 configs' default -- the shipped form's
              # large-message race changes served answers from run to run (AG3 / AG6; LL repeats bit for bit, AG4 /
              # AG5). `ar0` keeps the shipped form, for comparisons with older runs. Run 105 (owner note 121): LL is the
              # engine's own default from image :127 on, so nothing is set here; `ar0` sets the opt-out explicitly
              # (on images before :127 the unset default is the old form: name `ar4` there)
              case "$CFG-" in *-ar0-*) ENVV+=(RADIANCE_XE_AR_PROTO=0) ;; esac
              case "$CFG-" in *-rows-*) ENVV+=(RADIANCE_XE_GEMM_DPAS_ROWS=1) ;; esac
              case "$CFG-" in *-fuse-*) ENVV+=(RADIANCE_XE_AR_HC_FUSE=1) ;; esac
              case "$CFG-" in *-hcsr-*) ENVV+=(RADIANCE_XE_HC_PF_DPAS=2) ;; esac
              case "$CFG-" in *-moe8-*) ENVV+=(RADIANCE_XE_MOE_LPR8=1 RADIANCE_XE_MOE_V2R_SMALLK=4) ;; esac
              case "$CFG-" in *-h2-*) ENVV+=(RADIANCE_XE_MOE_H2=1) ;; esac
              # run 206 (owner note 202 B.1): NEO's submission / in-list sync keys, one A/B each on test images
              # (NEOReadDebugKeys=1 is set above). neoprint = print the non-default settings NEO runs with; neods1 /
              # neods0 = direct submission forced on / off; neoprim = a list's command buffer dispatched as primary.
              # Keys that remove in-list synchronisation are not offered (supervisor review, run 206): a race cannot be
              # proved absent by matching texts, so none could become a default.
              case "$CFG-" in *-neoprint-*) ENVV+=(PrintDebugSettings=1) ;; esac
              case "$CFG-" in *-neods1-*) ENVV+=(EnableDirectSubmission=1) ;; esac
              case "$CFG-" in *-neods0-*) ENVV+=(EnableDirectSubmission=0) ;; esac
              case "$CFG-" in *-neoprim-*) ENVV+=(DispatchCmdlistCmdBufferPrimary=1) ;; esac
              # run 101 (owner notes 98 item 1 / 107): hf = the host pool bound before the VRAM pools; pg = the
              # checkpoint store in pageable memory (no card binds it); hp16 = the 16 GiB host pool (the service's)
              case "$CFG-" in *-gnt0-*) ENVV+=(RADIANCE_XE_GDN_NT=0) ;; esac   # run 101, note 108: no NT hint
              case "$CFG-" in *-hf-*) ENVV+=(RADIANCE_HOST_FIRST=1) ;; esac
              case "$CFG-" in *-pg-*) ENVV+=(RADIANCE_CKPT_PAGEABLE=1) ;; esac
              case "$CFG-" in *-hp16-*) HOST_POOL_MIB=${HOST_POOL_MIB:-16384} ;; esac
              # prof = the tape profiler (a per-kernel table in serve.log; P41's form): the P-run of a form's gates
              case "$CFG-" in *-prof-*) ENVV+=(RADIANCE_L0_NTAPE_PROFILE=1) ;; esac
              # edges (run 206, owner note 201 item 4; supervisor-reviewed): with prof, the idle inside plays by kernel pair
              case "$CFG-" in *-edges-*) ENVV+=(RADIANCE_L0_NTAPE_EDGES=1) ;; esac
              # ar4 = run 102's all-reduce LL form (note 109 item 1: a sequence number in every payload word)
              case "$CFG-" in *-ar4-*) ENVV+=(RADIANCE_XE_AR_PROTO=4) ;; esac
              # gnx = run 102's trunk GEMV with permute_group_by_xor (note 109 item 2, bit for bit)
              case "$CFG-" in *-gnx-*) ENVV+=(RADIANCE_XE_GEMV_NX=1) ;; esac
              # hc5 / hc8 = the hc read's V2 level 5 (run 98) / level 8 (run 103, owner note 119: a verify's down
              # product four rows a wave over column slices); with `prof`, note 119's timing pair
              case "$CFG-" in *-hc5-*) ENVV+=(RADIANCE_XE_HC_V2=5) ;; *-hc8-*) ENVV+=(RADIANCE_XE_HC_V2=8) ;; esac
              # rp = run 102's router GEMV + shared gate column in one launch (note 109 item 3, bit for bit)
              case "$CFG-" in *-rp-*) ENVV+=(RADIANCE_XE_ROUTER_PAIR=1) ;; esac
              # sx = owner note 112's fail-fast exit (a startup quiet 60 s inside a driver call exits 75)
              case "$CFG-" in *-sx-*) ENVV+=(RADIANCE_STALL_EXIT_S=60) ;; esac
              # cx / st = run 108: the expert matrix form's compact grid (823c79d) / its staging once a work-group (5de6d9b)
              case "$CFG-" in *-cx-*) ENVV+=(RADIANCE_XE_MOE_MX_COMPACT=1) ;; esac
              # m8m = run 110 (owner note 129 item 2): moe8 on several-token launches only (RADIANCE_XE_MOE8M=1)
              case "$CFG-" in *-m8m-*) ENVV+=(RADIANCE_XE_MOE8M=1) ;; esac
              case "$CFG-" in *-m8r8-*) ENVV+=(RADIANCE_XE_MOE8M=1 RADIANCE_XE_MOE8M_R=8) ;; esac
              # avx = run 111 (owner note 135 item 1): attention's split kernel with 8-byte loads + native butterflies
              case "$CFG-" in *-avx-*) ENVV+=(RADIANCE_XE_ATTN_VX=1) ;; esac
              # xp = run 112 (owner note 139 stage 1): the next-layer expert prefetch, gather mode (opt-in)
              case "$CFG-" in *-xp-*) ENVV+=(RADIANCE_XE_EXPERT_PREFETCH=1) ;; esac
              # gs2 / gs4 = runs 112-113 (owner note 138): the GDN decode update split over 2 / 4 workgroups a head
              case "$CFG-" in *-gs2-*) ENVV+=(RADIANCE_XE_GDN_SPLIT=2) ;; *-gs4-*) ENVV+=(RADIANCE_XE_GDN_SPLIT=4) ;; esac
              # lat7 = run 283 (briefs/2026-11-11-gdn-split-latency.md, KG17): the VS 2 split's folded norm N1, an exact opt-in
              # record (RADIANCE_XE_GDN_LAT=7 with -gs2-, else ignored by the launcher); L1 is folded into every form, L2 dropped
              case "$CFG-" in *-lat7-*) ENVV+=(RADIANCE_XE_GDN_LAT=7) ;; esac
              # rt = run 110 (owner note 129 item 3): the 4-row trunk GEMV's transpose-reduce (RADIANCE_XE_GEMV_RT=1, bit for bit)
              case "$CFG-" in *-rt-*) ENVV+=(RADIANCE_XE_GEMV_RT=1) ;; esac
              # gl = run 114 (owner note 141 item 1): the multi-row trunk GEMV a group a lane (RADIANCE_XE_GEMV_GL=1, bit for bit)
              case "$CFG-" in *-gl-*) ENVV+=(RADIANCE_XE_GEMV_GL=1) ;; esac
              # hrg = run 115 (owner note 142 item 2): hc_read level 8's columns and inject gains in one launch (bit for bit)
              case "$CFG-" in *-hrg-*) ENVV+=(RADIANCE_XE_HC_V2_RG=1) ;; esac
              # fc = run 117 (owner note 147): format C -- today's w4nl experts against int8 activations (opt-in)
              case "$CFG-" in *-fc-*) ENVV+=(RADIANCE_MOE_FORMAT_C=1) ;; esac
              # ad1 / ad2 / ad4 = run 160 (owner note 163 item 1a): the MTP draft depth (the verify's rows = depth + 1; 3 is
              # the service setting)
              case "$CFG-" in *-ad1-*) ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 1} ;;
                              *-ad2-*) ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 2} ;;
                              *-ad4-*) ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 4} ;; esac
              # tskip7 / tskip8 = run 160 (owner note 162 item 1): Strata's tail skip -- an expert off the device at rank >= 7 / 8
              # of a token's top-10 is dropped instead of read over the link (RADIANCE_XE_TAIL_SKIP)
              case "$CFG-" in *-tskip7-*) ENVV+=(RADIANCE_XE_TAIL_SKIP=7) ;; *-tskip8-*) ENVV+=(RADIANCE_XE_TAIL_SKIP=8) ;; esac
              # nopipe = run 162 (owner note 165 item 1): no step issued ahead (RADIANCE_STEP_AHEAD=0), the gate's prerequisite
              case "$CFG-" in *-nopipe-*) ENVV+=(RADIANCE_STEP_AHEAD=0) ;; esac
              # dgs / dg1 / dg2 / dg4 / dg8 = run 162 (owner note 165 item 1): the confidence-gated draft depth
              # (RADIANCE_DRAFT_GATE: stats only, or the last draft cut under a top-2 logit margin of 1 / 2 / 4 / 8)
              case "$CFG-" in *-dgs-*) ENVV+=(RADIANCE_DRAFT_GATE=stats RADIANCE_DRAFT_GATE_LOG=64) ;;
                              *-dg1-*) ENVV+=(RADIANCE_DRAFT_GATE=1 RADIANCE_DRAFT_GATE_LOG=64) ;;
                              *-dg2-*) ENVV+=(RADIANCE_DRAFT_GATE=2 RADIANCE_DRAFT_GATE_LOG=64) ;;
                              *-dg4-*) ENVV+=(RADIANCE_DRAFT_GATE=4 RADIANCE_DRAFT_GATE_LOG=64) ;;
                              *-dg8-*) ENVV+=(RADIANCE_DRAFT_GATE=8 RADIANCE_DRAFT_GATE_LOG=64) ;; esac
              # ar5 = run 161 (owner note 163 item 2): the all-reduce's LL form with 16-byte stores (RADIANCE_XE_AR_PROTO=5)
              case "$CFG-" in *-ar5-*) ENVV+=(RADIANCE_XE_AR_PROTO=5) ;; esac
              # nofc = run 135: format C off (the default since owner note 160)
              case "$CFG-" in *-nofc-*) ENVV+=(RADIANCE_MOE_FORMAT_C=0) ;; esac
              # arskew = run 122 (owner note 151 A): the all-reduce wait record, to /out/arskew (m3/ar_skew.py)
              case "$CFG-" in *-arskew-*) mkdir -p /out/arskew; ENVV+=(RADIANCE_XE_AR_SKEW=/out/arskew) ;; esac
              # ngs = run 123 (owner note 151 B): prompt lookup measured, a line every 64 verifies
              case "$CFG-" in *-ngs-*) ENVV+=(RADIANCE_NGRAM_STATS=64) ;; esac
              # lr2 = run 123 (owner note 151 C): the int8 lm_head's rows form, 2 weight rows a sub-group (bit-identical)
              case "$CFG-" in *-lr2-*) ENVV+=(RADIANCE_XE_LOGITS_R=2) ;; esac
              # wr2 = run 123 (owner note 151 C): the MTP draft head's w2a8 GEMV, 2 output rows a sub-group (bit-identical)
              case "$CFG-" in *-wr2-*) ENVV+=(RADIANCE_XE_W2A8_R=2) ;; esac
              # dstage = run 124 (owner note 151 C): a draft pass's host time by part (RADIANCE_DEBUG_STAGE)
              case "$CFG-" in *-dstage-*) ENVV+=(RADIANCE_DEBUG_STAGE=1) ;; esac
              # glue = run 126 (owner note 159): a stream's work behind its native play on the immediate list, no host
              # wait between a step's draft passes (RADIANCE_L0_GLUE=1)
              case "$CFG-" in *-glue-*) ENVV+=(RADIANCE_L0_GLUE=1) ;; esac
              # ring1k = run 127: the native ring at 1024 slots (RADIANCE_L0_RING; 256 the default)
              case "$CFG-" in *-ring1k-*) ENVV+=(RADIANCE_L0_RING=1024) ;; esac
              # hci8 = run 125 (owner note 158): hc's mixing matrices stored as int8 rows at load (V2 kernels only: hc8)
              # hci82 = the same plus stage 2: N in int8 and the down product's columns by DP4A (to 64 rows, level 8)
              case "$CFG-" in *-hci8-*) ENVV+=(RADIANCE_XE_HC_I8=1) ;; *-hci82-*) ENVV+=(RADIANCE_XE_HC_I8=2) ;; esac
              # am = run 109: attention's per-row matrix form (RADIANCE_XE_ATTN_MX=1; decode and verify rows too)
              case "$CFG-" in *-am-*) ENVV+=(RADIANCE_XE_ATTN_MX=1) ;; esac
              case "$CFG-" in *-st-*) ENVV+=(RADIANCE_XE_MOE_MX_STAGE=1) ;; esac
              # jm64 / jm32 / jm16 = run 108 (owner note 109 item 5): the GDN chunk scan on the matrix units
              case "$CFG-" in *-jm64-*) ENVV+=(RADIANCE_XE_GDN_CS_JM=64) ;; *-jm32-*) ENVV+=(RADIANCE_XE_GDN_CS_JM=32) ;;
                              *-jm16-*) ENVV+=(RADIANCE_XE_GDN_CS_JM=16) ;; esac
              # avx = run 169 (owner notes 166 / 174): libavx serves the host rows ahead of libref -- the n-gram gather
              # through its io_uring row cache (the step needs the io_uring seccomp profile: mstep.sh's *avx* rule)
              case "$CFG-" in *-avx-*) ARGS=${ARGS/--kernels libxe:libref/--kernels libxe:libavx:libref} ;; esac
              # fmha / ar6 = run 170 (owner notes 173 / 172 d): the prefill attention op (RADIANCE_ATTN_QSA_FMHA=1; prefill
              # chunks of 256+ rows only) and the all-reduce's bandwidth form (RADIANCE_XE_AR_PROTO=6; messages of 256 KiB
              # or more only) -- at decode both are off by construction; the M-pair under them is the gate's proof
              case "$CFG-" in *-fmha-*) ENVV+=(RADIANCE_ATTN_QSA_FMHA=1) ;; esac
              case "$CFG-" in *-ar6-*) ENVV+=(RADIANCE_XE_AR_PROTO=6) ;; esac
              # run 172 (owner note 165 item 3): lanes = three seats, for two decoders and a third user's 20K prompt;
              # bpc1k / bpc2k = the scheduler's busy prefill chunk (auto: >= 1024, the stager's smallest pass) pinned
              # at 1024 / 2048 tokens, so a prompt read beside decoders leaves them a step more often
              case "$CFG-" in *-lanes-*) ARGS=${ARGS/--max-num-seqs 2/--max-num-seqs 3} ;; esac
              # run 174: conc = m3-conc.sh's concurrency phases (1 / 3 / 6 decoders of the story at once) on today's
              # configs: six seats at a 16K context, as m3-conc.sh ran them (six KV sequences ~0.8 GiB instead of 3.2)
              case "$CFG-" in *-conc-*) ARGS=${ARGS/--max-num-seqs 2/--max-num-seqs 6}; ARGS=${ARGS/--max-model-len 65536/--max-model-len 16384} ;; esac
              # run 203 (owner notes 185 / 186 item 1): sg0 = MTP drafts kept on whatever the number of decoders
              # (RADIANCE_SPEC_MAX_SEQS=0 after the family's =1: env's last assignment wins); ns2 / ns1 = depth 2 / 1;
              # c1236 = the conc phases at 1, 2, 3 and 6 users
              case "$CFG-" in *-sg0-*) ENVV+=(RADIANCE_SPEC_MAX_SEQS=0) ;; esac
              case "$CFG-" in *-sg2-*) ENVV+=(RADIANCE_SPEC_MAX_SEQS=2) ;; esac   # drafts while <= 2 sequences decode
              # sg3 = the engine's own default from image 248 (owner note 199: drafting capped at 3 decoders, off at once
              # at 4+; with the default RADIANCE_SPEC_MULTI_DEPTH=1); c16 = the conc phases at 1-6 users
              case "$CFG-" in *-sg3-*) ENVV+=(RADIANCE_SPEC_MAX_SEQS=3) ;; esac
              case "$CFG-" in *-c16-*) CONC="1 2 3 4 5 6" ;; esac
              # hr4g = 4 GiB of VRAM headroom (C31: the 6-seat form ran out of VRAM at 5 users -- driver free 38 MiB, then
              # UR OUT_OF_RESOURCES -- after the policy's drafting at 2-3 users had recorded more pass shapes)
              case "$CFG-" in *-hr4g-*) ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;; esac
              # md1 = depth 1 while 2+ sequences decode, the configured depth alone (RADIANCE_SPEC_MULTI_DEPTH=1; with sg0)
              case "$CFG-" in *-md1-*) ENVV+=(RADIANCE_SPEC_MULTI_DEPTH=1) ;; esac
              case "$CFG-" in *-ns2-*) ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 2} ;; esac
              case "$CFG-" in *-ns1-*) ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 1} ;; esac
              case "$CFG-" in *-c1236-*) CONC="1 2 3 6" ;; esac
              # c123 = 1, 2 and 3 users only (C15: drafts kept on at 6 users faulted the card -- OUT_OF_RESOURCES)
              case "$CFG-" in *-c123-*) CONC="1 2 3" ;; esac
              # c456 = 1, 4, 5 and 6 users (run 204, owner note 197 item 5: where the batch variance starts)
              case "$CFG-" in *-c456-*) CONC="1 4 5 6" ;; esac
              # c44 = 1 user, then 4 users twice (run 205, item 5: is the 4+-sequence variance run-to-run?)
              case "$CFG-" in *-c44-*) CONC="1 4 4" ;; esac
              # notape = every pass issued live, none recorded (C33: a shape's first, live-recorded run gave other bits
              # than its replays -- the 4-user texts differ the first time only)
              case "$CFG-" in *-notape-*) ENVV+=(RADIANCE_L0_TAPE=0) ;; esac
              # nowarm = the startup warm-up pass off (C14: the conc mode's second request drafted worse than its first)
              case "$CFG-" in *-nowarm-*) ARGS="$ARGS --startup-warmup off" ;; esac
              # warmon = the startup warm-up on explicitly (run 204: the engine's default is off from image 247 until
              # note 197 item 1's gates pass)
              case "$CFG-" in *-warmon-*) ARGS="$ARGS --startup-warmup on" ;; esac
              case "$CFG-" in *-bpc1k-*) ARGS="$ARGS --busy-prefill-chunk 1024" ;; *-bpc2k-*) ARGS="$ARGS --busy-prefill-chunk 2048" ;;
                              *-bpcoff-*) ARGS="$ARGS --busy-prefill-chunk off" ;; esac   # off: the chunk takes the whole step, no decode rows beside it
              # tr = the per-step trace (RADIANCE_STEP_TRACE=1, the prefill configs' form) for a lanes step's chunk passes
              case "$CFG-" in *-tr-*) ENVV+=(RADIANCE_STEP_TRACE=1) ;; esac
              # dstage = Ctx::run_step's part times a step (RADIANCE_DEBUG_STAGE=1: the stager's begin, the residency
              # sync, the architecture's pass, the lane join, the stager's end, the route flush, the profile / flush)
              case "$CFG-" in *-dstage-*) ENVV+=(RADIANCE_DEBUG_STAGE=1) ;; esac
              # rb = event records as barriers (RADIANCE_L0_RECORD_BARRIER=1, run 173: the prefill item; here the decode check)
              case "$CFG-" in *-rb-*) ENVV+=(RADIANCE_L0_RECORD_BARRIER=1) ;; esac
              # run 174 (owner note 165 item 4): notape = the tapes off (every step issues its ops, so an op's host-side
              # instrument runs each step); rdump = libxe's router dump (RADIANCE_XE_ROUTE_DUMP: the logits, ids and
              # weights of every decode-sized router call into /out/CFG/route/, read by m3/route_union.py)
              case "$CFG-" in *-notape-*) ENVV+=(RADIANCE_L0_TAPE=0) ;; esac
              # run 176 (owner note 180 A): dhi8 = the draft rounds read the trunk's int8 lm_head, not its 2-bit copy
              case "$CFG-" in *-dhi8-*) ENVV+=(RADIANCE_MTP_DRAFT_HEAD_Q2=0) ;; esac
              # run 189: ga = kernel launches outside a recording on the backend's own list (RADIANCE_L0_GLUE_ALL=1, the
              # leak fix, owner notes 182 / 184); w2d4 = the draft head's 2-bit GEMV by DP4A (RADIANCE_XE_W2A8_DP4=1, note
              # 183 item 1); hcp = hc_read's down half as one persistent launch (RADIANCE_XE_HC_PERSIST=1, note 186 3b)
              case "$CFG-" in *-ga-*) ENVV+=(RADIANCE_L0_GLUE_ALL=1) ;; esac
              # ga2 / ga3 = the two-list form (the mover's stream on a copy engine's list / a second compute list)
              case "$CFG-" in *-ga2-*) ENVV+=(RADIANCE_L0_GLUE_ALL=2) ;; esac
              case "$CFG-" in *-ga3-*) ENVV+=(RADIANCE_L0_GLUE_ALL=3) ;; esac
              case "$CFG-" in *-w2d4-*) ENVV+=(RADIANCE_XE_W2A8_DP4=1) ;; esac
              case "$CFG-" in *-hcp-*) ENVV+=(RADIANCE_XE_HC_PERSIST=1) ;; esac
              # hcp2 = stage 1b (owner note 187: work by ticket, per-row flags, occupancy-sized)
              case "$CFG-" in *-hcp2-*) ENVV+=(RADIANCE_XE_HC_PERSIST=2) ;; esac
              # run 193: dp2 = the draft depth by content, prose at 2 (owner note 191 item 1, RADIANCE_DRAFT_DEPTH_PROSE);
              # hcab = upstream 1.14's hc_read merge on level 8 (owner note 190, RADIANCE_XE_HC_AB)
              case "$CFG-" in *-dp2-*) ENVV+=(RADIANCE_DRAFT_DEPTH_PROSE=2) ;; esac
              case "$CFG-" in *-hcab-*) ENVV+=(RADIANCE_XE_HC_AB=1) ;; esac
              # run 202 (owner note 196 item 1): coopq = the L0 cooperative limits logged (RADIANCE_L0_COOP_QUERY; nothing changes)
              case "$CFG-" in *-coopq-*) ENVV+=(RADIANCE_L0_COOP_QUERY=1) ;; esac
              # run 205 (owner note 200 item 1a): hcnr = the norm + columns as one shape-matched launch (RADIANCE_XE_HC_NR)
              case "$CFG-" in *-hcnr-*) ENVV+=(RADIANCE_XE_HC_NR=1) ;; esac
              # hcgu = the gains + up mix as one shape-matched launch (RADIANCE_XE_HC_GU, owner note 200 item 1b)
              case "$CFG-" in *-hcgu-*) ENVV+=(RADIANCE_XE_HC_GU=1) ;; esac
              # hcwf (run 206, the fusion map's pair 1; supervisor-reviewed): hc_write folded into the next hc_read
              case "$CFG-" in *-hcwf-*) ENVV+=(RADIANCE_HC_WRITE_FUSE=1) ;; esac
              # gq (run 206, the fusion map's pair 4; supervisor-reviewed): the attention gate as one gate_quant_fp8
              case "$CFG-" in *-gq-*) ENVV+=(RADIANCE_XE_GATE_QUANT=1) ;; esac
              # gh (run 207, the fusion map's pair 2; supervisor-reviewed): the routed gate as one gated_had_quant_i8g
              case "$CFG-" in *-gh-*) ENVV+=(RADIANCE_XE_GATED_HAD=1) ;; esac
              # fz (run 207, supervisor-reviewed): the five fusion-map defaults all opted out -- the old path
              case "$CFG-" in *-fz-*) ENVV+=(RADIANCE_XE_HC_GU=0 RADIANCE_XE_HC_D16=0 RADIANCE_HC_WRITE_FUSE=0
                                              RADIANCE_XE_GATE_QUANT=0 RADIANCE_XE_GATED_HAD=0) ;; esac
              # d16 = the up mix's d staged as bf16 (RADIANCE_XE_HC_D16, owner note 201 item 1)
              case "$CFG-" in *-d16-*) ENVV+=(RADIANCE_XE_HC_D16=1) ;; esac
              # sel = format C's run form decodes codes by a register select chain (RADIANCE_XE_I4I8_RUN_SEL, note 202 A.3)
              case "$CFG-" in *-sel-*) ENVV+=(RADIANCE_XE_I4I8_RUN_SEL=1) ;; esac
              # grf = the run form with 256 GRF a thread (RADIANCE_XE_I4I8_RUN_GRF, owner note 202 A.2)
              case "$CFG-" in *-grf-*) ENVV+=(RADIANCE_XE_I4I8_RUN_GRF=1) ;; esac
              # gwgN = the GEMVs' rows a work-group (RADIANCE_XE_GEMV_WG, run 207, owner note 203 item 2; default 8)
              case "$CFG-" in *-gwg4-*) ENVV+=(RADIANCE_XE_GEMV_WG=4) ;; *-gwg16-*) ENVV+=(RADIANCE_XE_GEMV_WG=16) ;; *-gwg32-*) ENVV+=(RADIANCE_XE_GEMV_WG=32) ;; esac
              # rgrf = the router's top-k scatter with 256 GRF a thread (RADIANCE_XE_ROUTER_GRF, run 207, note 203 item 2)
              case "$CFG-" in *-rgrf-*) ENVV+=(RADIANCE_XE_ROUTER_GRF=1) ;; esac
              # tkgrf / qgrf = the row top-k's two stages / qsa_score with 256 GRF a thread (RADIANCE_XE_TOPK_GRF /
              # RADIANCE_XE_QSA_GRF, run 208, owner note 204 item 1: at 128 GRF they spill 131-194 / 182 variables)
              case "$CFG-" in *-tkgrf-*) ENVV+=(RADIANCE_XE_TOPK_GRF=1) ;; esac
              case "$CFG-" in *-qgrf-*) ENVV+=(RADIANCE_XE_QSA_GRF=1) ;; esac
              # kc = the K-specialised instances of the GEMVs and format C's gate_up (RADIANCE_XE_GEMV_KC /
              # RADIANCE_XE_I4I8_KC, run 209, owner note 206 item 1)
              case "$CFG-" in *-kc-*) ENVV+=(RADIANCE_XE_GEMV_KC=1 RADIANCE_XE_I4I8_KC=1) ;; esac
              # wccN = the weights' / codes' LSC cache-control hint N (2 / 5 / 6) on gemv_i8a8 and format C's decode
              # instances (RADIANCE_XE_GEMV_WCC / RADIANCE_XE_I4I8_WCC, run 211, owner note 206 item 2)
              # nob4 = the run 212 default bundle opted out (TOPK / QSA / ROUTER GRF and the GEMV hint all =0)
              case "$CFG-" in *-nob4-*) ENVV+=(RADIANCE_XE_TOPK_GRF=0 RADIANCE_XE_QSA_GRF=0 RADIANCE_XE_ROUTER_GRF=0 RADIANCE_XE_GEMV_WCC=0) ;; esac
              # dr1 / dr4 = format C down's decode instance at 1 / 4 columns a team (RADIANCE_XE_I4I8_DOWN_R, run 212)
              case "$CFG-" in *-dr1-*) ENVV+=(RADIANCE_XE_I4I8_DOWN_R=1) ;; *-dr4-*) ENVV+=(RADIANCE_XE_I4I8_DOWN_R=4) ;; esac
              # gwcc5 = the GEMV's weights alone at hint 5 (format C's codes keep the default; run 211: down needs L3)
              case "$CFG-" in *-gwcc5-*) ENVV+=(RADIANCE_XE_GEMV_WCC=5) ;; esac
              # wr16 = the draft head's w2a8 GEMV in the wide form, 16 B of codes a lane (RADIANCE_XE_W2A8_R=16, run 212)
              case "$CFG-" in *-wr16-*) ENVV+=(RADIANCE_XE_W2A8_R=16) ;; esac
              # dc4 / dc8 = hc_read's level-8 columns at 4 / 8 a workgroup (RADIANCE_XE_HC_DCOLS, run 212)
              case "$CFG-" in *-dc4-*) ENVV+=(RADIANCE_XE_HC_DCOLS=4) ;; *-dc8-*) ENVV+=(RADIANCE_XE_HC_DCOLS=8) ;; esac
              # dgrf = MoE i4i8 decode DPAS under large register file (RADIANCE_XE_I4I8_DPAS_GRF, run 257/264; prefill configs use it)
              case "$CFG-" in *-dgrf-*) ENVV+=(RADIANCE_XE_I4I8_DPAS_GRF=1) ;; esac
              # rungrf = MoE i4i8 run form under large register file (RADIANCE_XE_I4I8_RUN_GRF)
              case "$CFG-" in *-rungrf-*) ENVV+=(RADIANCE_XE_I4I8_RUN_GRF=1) ;; esac
              # cuthost = the recorded pass cut after a host-arena producer (RADIANCE_TAPE_CUT_HOST_TOUCH, run 272, note 214 item 3)
              case "$CFG-" in *-cuthost-*) ENVV+=(RADIANCE_TAPE_CUT_HOST_TOUCH=1) ;; esac
              # nap50 / nap100 = the sleeping wait's nap cap in us (RADIANCE_L0_WAIT_SLEEP_MAX_US, run 272; default 200)
              case "$CFG-" in *-nap50-*) ENVV+=(RADIANCE_L0_WAIT_SLEEP_MAX_US=50) ;; *-nap100-*) ENVV+=(RADIANCE_L0_WAIT_SLEEP_MAX_US=100) ;; esac
              # gqp (run 273, note 214 item 3, the decode step's pair 1; coordinator-reviewed): the routed + shared gated quantisers as one launch
              case "$CFG-" in *-gqp-*) ENVV+=(RADIANCE_XE_GATED_QUANT_PAIR=1) ;; esac
              # qkp (run 273, the decode step's pair 3; coordinator-reviewed): libr4d's fused q / k norm + rope prologue row listed
              case "$CFG-" in *-qkp-*) ENVV+=(RADIANCE_XE_QK_PROLOGUE=1) ;; esac
              # rqp (run 273, the decode step's pair 2; coordinator-reviewed): the router's selection + sort and the block input's quantiser as one launch
              case "$CFG-" in *-rqp-*) ENVV+=(RADIANCE_XE_ROUTER_QUANT_PAIR=1) ;; esac
              # gvm (run 274, the decode step's pairs 4 / 5; coordinator-reviewed): several projections of one activation in one GEMV launch
              case "$CFG-" in *-gvm-*) ENVV+=(RADIANCE_XE_GEMV_MULTI=1) ;; esac
              # gvm2 (run 285, group A of owner note 232's audit): the fused router launch with the shared expert's gate_up in it
              case "$CFG-" in *-gvm2-*) ENVV+=(RADIANCE_XE_GEMV_MULTI2=1) ;; esac
              # gvm2a / gvm2b: one group at a time (a = the router + shared gate_up launch, b = the x3 + the indexer's qk)
              case "$CFG-" in *-gvm2a-*) ENVV+=(RADIANCE_XE_GEMV_MULTI2=a) ;; *-gvm2b-*) ENVV+=(RADIANCE_XE_GEMV_MULTI2=b) ;; esac
              # nopairs (run 279, owner note 215): the three decode fusions are the default from image 303 on; this opts them out (the old path)
              # run 294 (owner note 234): groups A / B are the default too; nopairs opts them out with the rest
              case "$CFG-" in *-nopairs-*) ENVV+=(RADIANCE_XE_GATED_QUANT_PAIR=0 RADIANCE_XE_QK_PROLOGUE=0 RADIANCE_XE_ROUTER_QUANT_PAIR=0 RADIANCE_XE_GEMV_MULTI=0 RADIANCE_XE_GEMV_MULTI2=0) ;; esac
              # gvm2off: only groups A / B off (the run-285 opt-out check)
              case "$CFG-" in *-gvm2off-*) ENVV+=(RADIANCE_XE_GEMV_MULTI2=0) ;; esac
              # pdev = with prof, the kernel times per device too (RADIANCE_L0_NTAPE_PERDEV, run 220, note 207 B7)
              case "$CFG-" in *-pdev-*) ENVV+=(RADIANCE_L0_NTAPE_PERDEV=1) ;; esac
              # lr12 = the int8 lm_head's split rows form, no spills (RADIANCE_XE_LOGITS_R=12, run 219, note 207 A2)
              case "$CFG-" in *-lr12-*) ENVV+=(RADIANCE_XE_LOGITS_R=12) ;; esac
              # dv32k / dv64k = the draft head over a draft vocabulary (RADIANCE_XE_W2A8_VOCAB, run 219, note 207 A3a):
              # 32K = the refined list (teaching + rotation + tool ids); 64K = the teaching-only list (the out-of-sample
              # test, supervisor); the image also carries draft-vocab-64k-v2.ids (the refined 64K) for a default
              case "$CFG-" in *-dv32k-*) ENVV+=(RADIANCE_XE_W2A8_VOCAB=/opt/ri/radiance_home/draft-vocab-32k.ids) ;;
                              *-dv64k-*) ENVV+=(RADIANCE_XE_W2A8_VOCAB=/opt/ri/radiance_home/draft-vocab-64k.ids) ;; esac
              case "$CFG-" in *-wcc2-*) ENVV+=(RADIANCE_XE_GEMV_WCC=2 RADIANCE_XE_I4I8_WCC=2) ;; *-wcc5-*) ENVV+=(RADIANCE_XE_GEMV_WCC=5 RADIANCE_XE_I4I8_WCC=5) ;; *-wcc6-*) ENVV+=(RADIANCE_XE_GEMV_WCC=6 RADIANCE_XE_I4I8_WCC=6) ;; esac
              # run 202 (owner note 195): map = the DPAS expert GEMM over its block map; rbd = the decode run form bounded
              # at the call's live count (the wrapped-leader fix, RADIANCE_XE_I4I8_RUN_BOUND)
              case "$CFG-" in *-map-*) ENVV+=(RADIANCE_XE_I4I8_DPAS_MAP=1) ;; esac
              case "$CFG-" in *-rbd-*) ENVV+=(RADIANCE_XE_I4I8_RUN_BOUND=1) ;; esac
              # hctx = upstream 64c652f's first-draft context (RADIANCE_MTP_HIST_CTX, owner note 190's list)
              case "$CFG-" in *-hctx-*) ENVV+=(RADIANCE_MTP_HIST_CTX=1) ;; esac
              # run 175 (owner note 167 item 1): ablN = the format C run kernel's ablations (RADIANCE_XE_I4I8_RUN_ABL=N,
              # TEST ONLY, wrong numbers): 1 no activation staging, 2 no level lookups, 3 no weight loads, 4 no scale loads
              case "$CFG-" in *-abl1-*) ENVV+=(RADIANCE_XE_I4I8_RUN_ABL=1) ;; *-abl2-*) ENVV+=(RADIANCE_XE_I4I8_RUN_ABL=2) ;;
                              *-abl3-*) ENVV+=(RADIANCE_XE_I4I8_RUN_ABL=3) ;; *-abl4-*) ENVV+=(RADIANCE_XE_I4I8_RUN_ABL=4) ;; esac
              case "$CFG-" in *-rdump-*) mkdir -p /out/$CFG/route; ENVV+=(RADIANCE_XE_ROUTE_DUMP=/out/$CFG/route) ;; esac
              # lean = the routing lean at verify steps (RADIANCE_XE_ROUTE_LEAN=2,0.005,4: the two lowest picks lean to a
              # stable expert of the step within 0.005 of softmax probability, at calls of <= 4 rows = the verify
              # step's shape, not a prompt's tail chunk; lossy, owner note 165 item 4); lean16 = the first form (<= 16
              # rows, M260: a 15-row prompt tail leaned too and the first token moved)
              case "$CFG-" in *-lean-*) ENVV+=(RADIANCE_XE_ROUTE_LEAN=2,0.005,4) ;; *-lean16-*) ENVV+=(RADIANCE_XE_ROUTE_LEAN=2,0.005) ;; esac
              # ce1 / ce0 = the SYCL Level Zero adapter's copy-engine choice for memcpy (the staging copies ride on
              # it or on the compute engine): both spellings of the knob, the newer unified-runtime one and the old
              case "$CFG-" in *-ce1-*) ENVV+=(UR_L0_USE_COPY_ENGINE=1 SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=1) ;;
                              *-ce0-*) ENVV+=(UR_L0_USE_COPY_ENGINE=0 SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0) ;; esac
              # ml32k = the context bound halved (65536 -> 32768): the third seat's KV reservation and the VRAM it takes
              # from the arena (L4: driver free 57 MiB with three seats; the staging ring lives in the arena)
              case "$CFG-" in *-ml32k-*) ARGS=${ARGS/--max-model-len 65536/--max-model-len 32768} ;; esac
              # maN = the heat engine's minimum residency age (stage C's RADIANCE_PLACE_MIN_AGE, default 0): a unit's tier
              # holds N steps, against the churn a prompt read beside decoders showed (L4: 60 % readmits, 30 GiB)
              case "$CFG-" in *-ma64-*) ENVV+=(RADIANCE_PLACE_MIN_AGE=64) ;; *-ma256-*) ENVV+=(RADIANCE_PLACE_MIN_AGE=256) ;;
                              *-ma1k-*) ENVV+=(RADIANCE_PLACE_MIN_AGE=1024) ;; esac
              # piN = the engine's own lanes: N decode-only steps after each prefill chunk while other sequences
              # decode (--prefill-interleave, default off)
              case "$CFG-" in *-pi4-*) ARGS="$ARGS --prefill-interleave 4" ;; *-pi8-*) ARGS="$ARGS --prefill-interleave 8" ;;
                              *-pi16-*) ARGS="$ARGS --prefill-interleave 16" ;; *-pi32-*) ARGS="$ARGS --prefill-interleave 32" ;; esac
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
              case $CFG in svcnomtpx-*) ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;; esac
              # ad4 = run 105 (owner note 122): note 101's adaptive draft window at a ceiling of 4 drafts
              case "$CFG-" in *-ad4-*) ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 4}
                                       ARGS="$ARGS --draft-window adaptive" ;; esac
              # ada3 = run 202 (owner note 193): the adaptive window at today's ceiling of 3 (with -dp2-: each request
              # starts at its depth by content, then adapts)
              case "$CFG-" in *-ada3-*) ARGS="$ARGS --draft-window adaptive" ;; esac
              # hr2048 / hr1536 / hr1280 = run 105 (owner note 123): the GPU headroom stepped down from 2560 MiB.
              # Since run 108 the svcmtpx / svcnomtpx default is 1280 (note 123's choice, H82); hr2560 the earlier
              case "$CFG-" in *-hr2560-*) ;;
                              *-hr2048-*) ARGS=${ARGS/--gpu-headroom-mib 2560/--gpu-headroom-mib 2048} ;;
                              *-hr1536-*) ARGS=${ARGS/--gpu-headroom-mib 2560/--gpu-headroom-mib 1536} ;;
                              *)          ARGS=${ARGS/--gpu-headroom-mib 2560/--gpu-headroom-mib 1280} ;; esac
              ARGS="$ARGS --checkpoint-slots 2" ;;
    # run 99 (owner note 103): the candidate (MTP on) with the early segment release (e1), and with its tape_end wait
    # made lazy (e1lazy); the trace line counts the release waits (the candidate itself: svcmtpcapck2hp8)
    svcmtpcape1|svcmtpcape1lazy) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256
                RADIANCE_L0_TAPE_EARLY=1 RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1
                DisableIndirectAccess=1 OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1)
              [ "$CFG" = svcmtpcape1lazy ] && ENVV+=(RADIANCE_L0_TAPE_EARLY_LAZY=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
              ARGS="$ARGS --checkpoint-slots 2" ;;
    # run 99 (owner note 101): the candidate (MTP on) with the adaptive draft window at a ceiling of 4 / 6 drafts; the
    # window chosen and the acceptance from /stats (stats-K.txt), the recordings' VRAM from the trace lines
    svcmtpcapad4|svcmtpcapad6) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256
                RADIANCE_L0_TAPE_EARLY=0 RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1
                DisableIndirectAccess=1 OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
              ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens ${CFG#svcmtpcapad}}
              ARGS="$ARGS --checkpoint-slots 2 --draft-window adaptive" ;;
    # run 98 (owner note 99): the hc levels on the candidate of note 95 (16 GiB host pools stall on this boot: M83, B37)
    svcmtpcaphc2) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1 RADIANCE_XE_HC_V2=2)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
              ARGS="$ARGS --checkpoint-slots 2" ;;
    svcnomtpcaphc2) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1 RADIANCE_XE_HC_V2=2)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
              ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}; ARGS="$ARGS --checkpoint-slots 2" ;;
    svcmtpcaphc5) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1 RADIANCE_XE_HC_V2=5)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
              ARGS="$ARGS --checkpoint-slots 2" ;;
    svcmtpcaphc7) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1 RADIANCE_XE_HC_V2=7)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
              ARGS="$ARGS --checkpoint-slots 2" ;;
    # run 99: level 5 MTP on with the tape profiler (the 4-row verify's hc kernels)
    profmtpcaphc5) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1 RADIANCE_XE_HC_V2=5 RADIANCE_L0_NTAPE_PROFILE=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
              ARGS="$ARGS --checkpoint-slots 2" ;;
    profmtpcaphc7) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1 RADIANCE_XE_HC_V2=7 RADIANCE_L0_NTAPE_PROFILE=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
              ARGS="$ARGS --checkpoint-slots 2" ;;
    svcnomtpcaphc5) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1 RADIANCE_XE_HC_V2=5)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
              ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}; ARGS="$ARGS --checkpoint-slots 2" ;;
    svcnomtpcaphc6) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1 RADIANCE_XE_HC_V2=6)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
              ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}; ARGS="$ARGS --checkpoint-slots 2" ;;
    profnomtpcaphc5) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1 RADIANCE_XE_HC_V2=5 RADIANCE_L0_NTAPE_PROFILE=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
              ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}; ARGS="$ARGS --checkpoint-slots 2" ;;
    # levels 6 / 7: level 5 with the down product's columns and gains timed apart (the gains at 8 / 4 loads in flight)
    profnomtpcaphc6) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1 RADIANCE_XE_HC_V2=6 RADIANCE_L0_NTAPE_PROFILE=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
              ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}; ARGS="$ARGS --checkpoint-slots 2" ;;
    profnomtpcaphc7) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1 RADIANCE_XE_HC_V2=7 RADIANCE_L0_NTAPE_PROFILE=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
              ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}; ARGS="$ARGS --checkpoint-slots 2" ;;
    # the same with the host pool allocated before the VRAM pools (RADIANCE_HOST_POOL_FIRST)
    svcmtpcapck2hf) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1 RADIANCE_HOST_POOL_FIRST=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 9216}
              ARGS="$ARGS --checkpoint-slots 2" ;;
    svcnomtpcapck2hf) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1 RADIANCE_HOST_POOL_FIRST=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 9216}
              ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}; ARGS="$ARGS --checkpoint-slots 2" ;;
    # the same sizes with the checkpoint store in pageable host memory (no card binds it)
    svcmtpBpg) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=1024 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_CKPT_PAGEABLE=1) ;;
    # the same sizes with the checkpoint store, the KV carve and the store before the weights load
    svcmtpBkvb) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=1024 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_KV_BEFORE_LOAD=1) ;;
    # run 96 (owner note 95 item 2): H61's setting (headroom 3072) with smaller per-list driver memory -- the command
    # buffer at 32 KB and NEO's default heaps at 16 KB (from 64) -- traced, with NEO's allocation log
    svcmtpheap16) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=32 ForceDefaultHeapSize=16 RADIANCE_L0_VRAM_TRACE=1
                LogAllocationType=1 LogAllocationStdout=1 LogAllocationMemoryPool=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 3072} ;;
    # its reference: the same with the service's 64 KB command buffer and the default heaps
    svcmtpheapref) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1
                LogAllocationType=1 LogAllocationStdout=1 LogAllocationMemoryPool=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 3072} ;;
    # run 95 (owner notes 92-93): the full candidate setting at note 92's step A (floor 1024, headroom 2560), traced
    svcnomtpfl1k) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=1024 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    svcmtpfl1k) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=1024 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560} ;;
    # run 94 (owner note 66): the fix + hc_read V2 (the norm once, the down product a wave a column, N in global)
    svcnomtpv2) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                RADIANCE_XE_HC_V2=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    svcmtpv2) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                RADIANCE_XE_HC_V2=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
    # the same two with the tape profiler (per-kernel device time), V1 and V2
    profnomtpfix) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                RADIANCE_L0_NTAPE_PROFILE=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    profnomtpv2) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                RADIANCE_L0_NTAPE_PROFILE=1 RADIANCE_XE_HC_V2=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # run 94, hc step 2: V2 plus hc_up_mix's staged W_up rows (RADIANCE_XE_HC_V2=2)
    svcnomtpv3) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                RADIANCE_XE_HC_V2=2)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    svcmtpv3) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                RADIANCE_XE_HC_V2=2)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
    profnomtpv3) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                RADIANCE_L0_NTAPE_PROFILE=1 RADIANCE_XE_HC_V2=2)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # level 4 (run 95): as 2 with the mix's codes by hc_quant_rows (image :89 also staged N in LDS for the down
    # product -- M72 / P32 -- since dropped as slower)
    svcnomtpl4) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                RADIANCE_XE_HC_V2=4)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    svcmtpl4) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                RADIANCE_XE_HC_V2=4)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
    profnomtpl4) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                RADIANCE_L0_NTAPE_PROFILE=1 RADIANCE_XE_HC_V2=4)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # level 5 (run 98, owner note 99): as 4 with the E4M3 codes decoded by e4m3x4_s8 and the gains with 8 loads in flight
    svcnomtpl5) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                RADIANCE_XE_HC_V2=5)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    svcmtpl5) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                RADIANCE_XE_HC_V2=5)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
    profnomtpl5) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                RADIANCE_L0_NTAPE_PROFILE=1 RADIANCE_XE_HC_V2=5)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # level 3: as 2 with hc_v2_down's inject gains in a launch of their own, to time the columns and the gains apart
    profnomtpv4) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                RADIANCE_L0_NTAPE_PROFILE=1 RADIANCE_XE_HC_V2=3)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    svcmtppcres) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
    earlynomtpcb64) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=1
                RADIANCE_L0_VRAM_TRACE=1 NEOReadDebugKeys=1 OverrideCmdListCmdBufferSizeInKb=64)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # run 87 (owner note 82, R8): svcnomtpold with every copy kept off the copy engine (bcs) -- R7's hang is the mover's
    # copy on the copy engine never completing
    svcnomtpnobcs) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                UR_L0_USE_COPY_ENGINE=0 SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}; ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # run 76: the same with tapes off -- does the first prompt's +1.8 GB before the first recording come without tapes?
    tapesoffmtp) ENVV=(RADIANCE_SPEC_MAX_SEQS=1 RADIANCE_L0_TAPE=0)
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
    profmtp) ENVV=(RADIANCE_L0_NTAPE_PROFILE=1) ;;
    # run 31, owner note 37: the n-gram table pinned in host memory instead of read from the DRAM-less drive
    # (M23: the 47.68 GiB table is one pinned allocation, past the adapter's max allocation size: relaxed limits)
    ngramram) ENVV=(UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1); ARGS=${ARGS/--ngram-placement disk/--ngram-placement ram} ;;
    ngramramnomtp) ENVV=(UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1); ARGS=${ARGS/--ngram-placement disk/--ngram-placement ram}
                   ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # run 27: the op oracle on the big model (libref on the host checks every issue of the named op at 2e-2;
    # the tiny model has no draft head, so mtp_enter can only be checked here)
    oraclemtp) ENVV=(RADIANCE_OP_ORACLE=${ORACLE_OP:-mtp_enter} RADIANCE_OP_ORACLE_MAX=${ORACLE_MAX:-100000} RADIANCE_OP_ORACLE_TOL=0.02) ;;
    # (the args already list --kernels libxe:libref, which the oracle needs)
    # run 31: the trunk W8A8 GEMM (gemm_nt_q, the DP4A gemv_i8a8), 4 checks a rank so the host load stays short
    oraclegemm) ENVV=(RADIANCE_OP_ORACLE=gemm_nt_q RADIANCE_OP_ORACLE_MAX=4 RADIANCE_OP_ORACLE_TOL=0.02) ;;
    # run 32: the expert GEMV (moe_gemm_q, moe_planes_v2m at several tokens), 4 checks a rank
    oraclemoe) ENVV=(RADIANCE_OP_ORACLE=moe_gemm_q RADIANCE_OP_ORACLE_MAX=4 RADIANCE_OP_ORACLE_TOL=0.02) ;;
    nomtphost) ENVV=(NEOReadDebugKeys=1 PrintDebugSettings=1 LogAllocationSummaryReport=1 ForceSystemMemoryPlacement=67142916)
               ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    # run 11, T4: the full mask aborted the driver at startup (linear_stream.h:84 -- a command stream's
    # buffer); the narrow mask places the command buffers alone (type 3 -> bit 2 = 4) in system memory.
    nomtphostcb) ENVV=(NEOReadDebugKeys=1 PrintDebugSettings=1 ForceSystemMemoryPlacement=4)
               ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
    *) echo "unknown config $CFG"; continue ;;
  esac
  # run 100: overrides for a boot whose card1 stops binding the host pool's fourth 2 GiB piece (M100 / M100b):
  # PIECE_MIB sets RADIANCE_HOST_POOL_PIECE_MIB (the last setting wins in env), HOST_POOL_MIB the pool's size
  [ -n "${PIECE_MIB:-}" ] && ENVV+=(RADIANCE_HOST_POOL_PIECE_MIB=$PIECE_MIB)
  [ -n "${HOST_POOL_MIB:-}" ] && ARGS=${ARGS/--host-pool-mib 8192/--host-pool-mib $HOST_POOL_MIB}
  # run 108: a libxe.so newer than the image, from the step folder; it replaces the container's copy only
  [ -f /out/libxe.so ] && cp /out/libxe.so $H/kernels/libxe.so && echo "libxe.so from /out"
  t0=$(date +%s)
  # run 163 (owner note 169 item 3): UT_PREFIX from the step command's environment wraps the engine (unitrace)
  env "${ENVV[@]}" timeout 3000 ${UT_PREFIX:-} /opt/ri/bin/radiance $ARGS --radiance-home $H -v > $O/serve.log 2>&1 &
  pid=$!; up=0
  for i in $(seq 2400); do
    kill -0 $pid 2>/dev/null || break
    if exec 3<>/dev/tcp/127.0.0.1/$P 2>/dev/null; then
      printf 'GET /health HTTP/1.1\r\nHost: l\r\nConnection: close\r\n\r\n' >&3
      head -1 <&3 | grep -q " 200" && up=1; exec 3<&-
      [ $up = 1 ] && break
    fi
    # owner note 175 (run 169): an engine that logs nothing for 180 s during startup sits on a card that has not
    # recovered from a fault (B99 / B101): stop it here, not at the step's limit. The count of lines other than the
    # engine's own "startup: nothing has been logged" watchdog lines (121, 241, 481 ... s) is the clock.
    n=$(grep -c -v 'startup: nothing has been logged' $O/serve.log 2>/dev/null)
    if [ "$n" != "${last_n:-}" ]; then last_n=$n; last_t=$(date +%s); fi
    if [ $(( $(date +%s) - ${last_t:-$t0} )) -ge 180 ]; then
      echo "$CFG startup: nothing logged for 180 s -- card not recovered after the last fault (owner note 175); engine stopped"
      kill $pid 2>/dev/null; break
    fi
    sleep 1
  done 2>/dev/null
  echo "$CFG up=$up after $(( $(date +%s) - t0 )) s"
  if [ $up = 1 ]; then
   case "$CFG-" in
   *-lanes-*)
    # run 172 (owner note 165 item 3, sFNX's lanes): what one user's 20K prompt does to two users decoding
    # meanwhile. A warm-up request first (the expert heat settles, the first tapes record); then phase A, the two
    # decoders alone (the baseline: the 256-token story, streamed, both at once); phase B, the same two decoders
    # with the 20K prompt (max_tokens 1, /out/card/prefill-20k.body) sent LANES_DELAY (3) s after they start.
    # Every SSE line carries the epoch, so lanes_gaps.py (amd-host) reads each decoder's worst inter-token gap and
    # its tok/s over the prompt's read; the prompt's own wall time is in p20k.time.
    LB=/out/card/prefill-20k.body
    # the decoders generate LANES_TOKENS (1200: ~17 s at 69 tok/s, longer than the 20K prompt's ~12 s read)
    LD=/tmp/lanes-dec.body
    sed 's/"max_tokens":256/"max_tokens":'"${LANES_TOKENS:-1200}"'/' $BODY > $LD
    timeout 900 bash -c "$(declare -f usage stream_req); CG=$CG; P=$P; stream_req $BODY /tmp/lanes-warm.sse"
    echo "$CFG lanes warm-up: $(grep -o '"predicted_per_second":[0-9.]*' /tmp/lanes-warm.sse | tail -1)"
    # dec1 = one decoder instead of two (run 173: is the shared chunk pass's slowdown per decoder or per state?)
    case "$CFG-" in *-dec1-*) NDEC=1 ;; *) NDEC=2 ;; esac
    for ph in A B; do
      O2=$O/lanes-$ph; mkdir -p $O2
      pids=""
      for k in $(seq $NDEC); do
        timeout 900 bash -c "$(declare -f usage stream_req); CG=$CG; P=$P; stream_req $LD $O2/dec-$k.sse" &
        pids="$pids $!"
      done
      if [ $ph = B ]; then
        sleep "${LANES_DELAY:-3}"
        w0=$EPOCHREALTIME
        timeout 900 bash -c "$(declare -f get_stats); P=$P; exec 5<>/dev/tcp/127.0.0.1/\$P; body=\$(cat $LB); printf 'POST /v1/completions HTTP/1.1\r\nHost: l\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s' \"\${#body}\" \"\$body\" >&5; cat <&5 > $O2/p20k.http; exec 5<&-"
        w1=$EPOCHREALTIME
        echo "$w0 $w1" > $O2/p20k.time
        echo "$CFG lanes $ph: the 20K prompt $(head -1 $O2/p20k.http | tr -d '\r') in $(echo "$w0 $w1" | awk '{printf "%.2f", $2-$1}') s"
      fi
      wait $pids
      for k in 1 2; do
        echo "$CFG lanes $ph decoder $k: $(grep -c '^[0-9.]* [0-9]* data: {' $O2/dec-$k.sse) chunks, $(grep -o '"predicted_per_second":[0-9.]*' $O2/dec-$k.sse | tail -1)"
      done
    done
    get_stats /metrics 2>/dev/null | grep -E '^radiance:draft' > $O/metrics-lanes.txt ;;
   *-conc-*)
    # run 174: the milestone's aggregate target (>= 120 tok/s at 6 users) on today's defaults -- m3-conc.sh's form:
    # a warm-up request, then for N in CONC (1 3 6) N streamed story requests at once; per request its tok/s and
    # chunks, and the aggregate both as the sum of the rates and as the tokens over the wall
    timeout 900 bash -c "$(declare -f usage stream_req); CG=$CG; P=$P; stream_req $BODY /tmp/conc-warm.sse"
    echo "$CFG conc warm-up: $(grep -o '"predicted_per_second":[0-9.]*' /tmp/conc-warm.sse | tail -1)"
    for N in ${CONC:-1 3 6}; do
      O2=$O/conc-$N; [ -d $O2 ] && O2=$O/conc-$N-again; mkdir -p $O2   # run 205: a repeated N keeps both phases
      u0=$(usage); w0=$EPOCHREALTIME
      pids=""
      for k in $(seq $N); do
        timeout 900 bash -c "$(declare -f usage stream_req); CG=$CG; P=$P; stream_req $BODY $O2/tps-$k.sse" &
        pids="$pids $!"
      done
      wait $pids
      u1=$(usage); w1=$EPOCHREALTIME
      wall=$(echo "$w0 $w1" | awk '{printf "%.2f", $2-$1}')
      per=""; tot=0; toks=0
      for k in $(seq $N); do
        r=$(grep -o '"predicted_per_second":[0-9.]*' $O2/tps-$k.sse | tail -1 | cut -d: -f2)
        n=$(grep -o '"predicted_n":[0-9]*' $O2/tps-$k.sse | tail -1 | cut -d: -f2)
        c=$(grep -c '^[0-9.]* [0-9]* data: {' $O2/tps-$k.sse)
        per="$per ${r:-?}($c)"
        tot=$(echo "$tot ${r:-0}" | awk '{printf "%.2f", $1+$2}')
        toks=$(( toks + ${n:-0} ))
      done
      echo "$CFG conc $N: wall ${wall}s, container CPU $(( (u1 - u0) / 1000 )) ms, aggregate $tot tok/s (sum of rates), $(echo "$toks $wall" | awk '{printf "%.1f", $1/$2}') tok/s ($toks tokens over the wall), per-request tok/s(chunks):$per"
    done
    get_stats /metrics 2>/dev/null | grep -E '^radiance:draft' > $O/metrics-conc.txt ;;
   *)
    for k in $(seq "$REPS"); do
      flag=/tmp/sampling-$CFG-$k; touch $flag
      sampler $flag > $O/stats-$k.txt 2>/dev/null &
      spid=$!
      u0=$(usage); w0=$EPOCHREALTIME
      timeout 900 bash -c "$(declare -f usage stream_req); CG=$CG; P=$P; stream_req $BODY $O/tps-$k.sse"
      u1=$(usage); w1=$EPOCHREALTIME
      rm -f $flag; wait $spid
      # run 99 (owner note 101): the draft metrics after each run (the adaptive window is only on /metrics)
      get_stats /metrics 2>/dev/null | grep -E '^radiance:draft' > $O/metrics-$k.txt
      echo "$CFG run $k: $(grep -c '^[0-9.]* [0-9]* data: {' $O/tps-$k.sse) chunks, $(grep -o '"predicted_per_second":[0-9.]*' $O/tps-$k.sse | tail -1)," \
           "container CPU $(( (u1 - u0) / 1000 )) ms over $(echo "$w0 $w1" | awk '{printf "%.1f", $2-$1}') s"
    done ;;
   esac
  fi
  kill $pid 2>/dev/null; wait $pid; echo "$CFG server exit $?"
  grep -E "^E |heat engine:|step decode" $O/serve.log | tail -8
done
