#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 2, stage 2 step 4): first-token time at ~7K and ~20K prompt
# tokens. Inside ri-m2rt on both cards, started by D/m2-step.sh with STEP_ENGINE=1: ONE model load (CFG as
# in m2-tps.sh), then /out/card/prefill-7k.body and prefill-20k.body (max_tokens 1), each REPS times, wall
# time a request to /out/prefill.txt ("<name> <k> <wall s> <status>"); the server's own "ttft=" and
# "prompt=N tok" lines come from /out/serve.log.
#   m2-prefill.sh CFG [REPS=2]
H=/opt/ri/radiance_home; P=18421
CFG=${1:-nomtp}; REPS=${2:-2}
post_raw() {
  local body; body=$(cat "$2")
  exec 5<>/dev/tcp/127.0.0.1/$P || return 1
  printf 'POST %s HTTP/1.1\r\nHost: l\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s' \
    "$1" "${#body}" "$body" >&5
  cat <&5 > "$3"; exec 5<&-
  head -1 "$3" | tr -d '\r'
}
ARGS=$(grep -v '^#' /out/flashnext-card1.args | tr '\n' ' ')
# owner note 175 (run 169, 11-04): libavx serves the host rows ahead of libref in every step -- its n-gram gather reads
# the table through an io_uring row cache with a read-ahead thread (B102: the 7K 10.49 -> 9.60 s, exact); the
# container runs under note 45's seccomp profile (bstep.sh / mstep.sh), and the image carries libavx.so (193 on)
ARGS=${ARGS/--kernels libxe:libref/--kernels libxe:libavx:libref}
ENVV=()
case $CFG in
  mtp) ;;
  nomtp) ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
  plain) ENVV=(RADIANCE_L0_TAPE=0) ;;
  plainnomtp) ENVV=(RADIANCE_L0_TAPE=0); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
  # milestone 3 run 60 (owner notes 49 and 50): MTP off with the matrix-unit expert form on (the service default
  # since note 49; the env switch on images up to :57) at the prefill chunk the args carry (4,096) and at 8,192 /
  # 16,384 tokens a chunk. B13 (run 60) showed the budget alone does not change the chunk: plan_chunk splits at
  # the linear-state checkpoint interval (--checkpoint-interval 4096) first, so the chunk needs --prefill-span N
  # (N intervals in one step; the checkpoint grid stays at 4,096) beside the budget; B13's budget-only form
  # (nomtpmx8k) is kept for the record (a larger arena and a staging threshold above the chunk: slower).
  nomtpmx)    ENVV=(RADIANCE_XE_MOE_MX=1); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
  # run 93 (owner note 91 item 2): nomtpmx with the two-card fix -- the host pool in 2 GiB pieces, explicit residency,
  # NEO's indirect access off (4fb114b)
  nomtpmxfix) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1
                    DisableIndirectAccess=1); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
  # run 96 (owner note 95 item 1): H61's full candidate setting (the fix, the 64 KB key, headroom 3072, MTP on as the
  # service) with the VRAM trace and its ledger, and NEO's allocation log; PREFILL_NAMES="16 1k 7k" REPS=1 gives
  # the first-use memory of each prompt size in one load
  mtpallvt) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1
                LogAllocationType=1 LogAllocationStdout=1 LogAllocationMemoryPool=1)
              PREFILL_NAMES=${PREFILL_NAMES:-16 1k 7k}
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 3072} ;;
  # run 97 (owner note 95 item 3): B32's form (MTP off) at step B's sizes (floor 1024, headroom 2048, the 64 KB key)
  # with the checkpoint store in pageable host memory -- the 7K and 20K prompts write snapshots (every 4,096 tokens)
  nomtpBpg) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1
                  DisableIndirectAccess=1 OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_TAPE_VRAM_FLOOR_MIB=1024
                  RADIANCE_CKPT_PAGEABLE=1); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
  # run 97 (owner note 95 item 3's candidate): B32's form (MTP off) at the real-card budget, host pool 8 GiB, 2 pinned
  # checkpoint slots, headroom 2560, floor 256, the 64 KB key
  nomtpcap) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1
                  DisableIndirectAccess=1 OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_TAPE_VRAM_FLOOR_MIB=256
                  RADIANCE_VRAM_CAP_FROM_FREE=1)
              ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
              ARGS="$ARGS --checkpoint-slots 2" ;;
  # run 99 (owner note 102): nomtpcap with the prefill hc on the matrix units; ...pfmlp the 7K prompt with logprobs
  # run 102: ...pfmar4 with the all-reduce's LL form (RADIANCE_XE_AR_PROTO=4: the prefill's large messages)
  # run 105 (owner note 123): ...pfmhr2048 / ...pfmhr1536 the GPU headroom stepped down from 2560
  # run 108 (owner note 109 item 5): ...pfmjm64 / 32 / 16 the chunk scan's matrix-unit form at that v block
  # run 108 (owner note 123's choice, H82 / B48): the family's headroom is 1280 MiB; ...pfmhr2560 the earlier
  nomtpcappfm|nomtpcappfmjm64tr|nomtpcappfmjm64trmbd|nomtpcappfmjm64trmbdprof|nomtpcappfmjm64trmbdtrace|nomtpcappfmjm64trv2prof|nomtpcappfmjm64trv2grfprof|nomtpcappfmjm64trv2|nomtpcappfmjm64trwarmon|nomtpcappfmjm64trhist|nomtpcappfmjm64trnohcpf|nomtpcappfmjm64trgrf|nomtpcappfmjm32tr|nomtpcappfmjm64tr2k|nomtpcappfmjm64trfz|nomtpcappfmjm64trnodia|nomtpcappfmjm64trnocb|nomtpcappfmjm64trnosplit|nomtpcappfmjm64trce0|nomtpcappfmjm64trmap|nomtpcappfmjm64trmapprof|nomtpcappfmjm64trtail|nomtpcappfmjm64trga|nomtpcappfmjm64trga2|nomtpcappfmjm64trga3|nomtpcappfmjm64trnowarm|nomtpcappfmjm64trpre|nomtpcappfmjm64trci8k|nomtpcappfmjm64trnofc|nomtpcappfmjm64trdpas|nomtpcappfmjm64trdpasprof|nomtpcappfmjm64trfmha|nomtpcappfmjm64trfmhaprof|nomtpcappfmjm64tropsprof|nomtpcappfmjm64trng|nomtpcappfmjm64trngavx|nomtpcappfmjm64tropsprofavx|nomtpcappfmjm64trar6|nomtpcappfmjm64trfmhaar6|nomtpcappfmjm64trfmhaar6ds|nomtpcappfmjm64trfmhaar6dsce1|nomtpcappfmjm64trfmhaar6dsce0|nomtpcappfmjm64trfmhaar6dsnoimm|nomtpcappfmjm64trfmhaar6ds2|nomtpcappfmjm64trfmhaar6dsrb|nomtpcappfmjm64trfmhaar6rb|nomtpcappfmjm64trfmhaar6nodec|nomtpcappfmjm64trfmhaar6nosc|nomtpcappfmjm64trfmhaar6nostage|nomtpcappfmjm64trfmhaar6nomma|nomtpcappfmjm64trfmhaar6w16|nomtpcappfmjm64trfmhaar6rt16|nomtpcappfmjm64trfmhaar6nodpas|nomtpcappfmdef|nomtpcappfmdefvx|nomtpcappfmdeftr|nomtpcappfmdeftrnotape|nomtpcappfmdeftrgraph|nomtpcappfmdefdefer|nomtpcappfmdeftrlp|nomtpcappfmdeftail|nomtpcappfmvec|nomtpcappfmvecprof|nomtpcappfmlpvec|nomtpcappfmtrace|nomtpcappfmtr|nomtpcappfmtrnotape|nomtpcappfmatprof|nomtpcappfmat|nomtpcappfmlpat|nomtpcappfmamprof|nomtpcappfmlpam|nomtpcappfmlp|nomtpcappfmlpcx|nomtpcappfmlpbd|nomtpcappfmlpcxst|nomtpcappfmjm64cxstprof|nomtpcappfmar4|nomtpcappfmhr2560|nomtpcappfmhr2048|nomtpcappfmhr1536|nomtpcappfmhr1280|nomtpcappfmjm64|nomtpcappfmjm32|nomtpcappfmjm16|nomtpcappfmjm64prof|nomtpcappfmjm64cx|nomtpcappfmjm64cxprof) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1
                  NEOReadDebugKeys=1 DisableIndirectAccess=1 OverrideCmdListCmdBufferSizeInKb=64
                  RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_VRAM_CAP_FROM_FREE=1 RADIANCE_XE_HC_PF_DPAS=1)
              # run 109 (owner note 126 item 5): the expert matrix form's staging is a default of this family;
              # run 110 (owner note 134 item 1): STAGE and VEC are libxe's defaults, so ...pfmdef sets neither
              case $CFG in nomtpcappfmdef*) ;; *) ENVV+=(RADIANCE_XE_MOE_MX_STAGE=1) ;; esac
              [ "$CFG" = nomtpcappfmlp ] && PREFILL_NAMES=${PREFILL_NAMES:-7klp}
              # run 108: the 7K logprob form with the expert matrix form's compact grid
              [ "$CFG" = nomtpcappfmlpcx ] && PREFILL_NAMES=${PREFILL_NAMES:-7klp} && ENVV+=(RADIANCE_XE_MOE_MX_COMPACT=1)
              # ... and with the per-row grid bounded at the regular experts' runs (RADIANCE_XE_MOE_MX_BOUND=1)
              [ "$CFG" = nomtpcappfmlpbd ] && PREFILL_NAMES=${PREFILL_NAMES:-7klp} && ENVV+=(RADIANCE_XE_MOE_MX_BOUND=1)
              # ... compact with the staging rework (RADIANCE_XE_MOE_MX_STAGE=1): must equal nomtpcappfmlpcx in every digit
              [ "$CFG" = nomtpcappfmlpcxst ] && PREFILL_NAMES=${PREFILL_NAMES:-7klp} && ENVV+=(RADIANCE_XE_MOE_MX_COMPACT=1 RADIANCE_XE_MOE_MX_STAGE=1)
              # run 109: the matrix-unit attention for a prefill pass (RADIANCE_XE_ATTN_JM=1), profiled / plain / logprobs
              [ "$CFG" = nomtpcappfmatprof ] && ENVV+=(RADIANCE_XE_ATTN_JM=1 RADIANCE_XE_LAUNCH_PROFILE=1 RADIANCE_L0_QUEUE_PROFILING=1)
              [ "$CFG" = nomtpcappfmat ] && ENVV+=(RADIANCE_XE_ATTN_JM=1)
              [ "$CFG" = nomtpcappfmlpat ] && PREFILL_NAMES=${PREFILL_NAMES:-7klp} && ENVV+=(RADIANCE_XE_ATTN_JM=1)
              # run 110 (owner note 129 item 1): the defaults, profiled, every prefill step's phases traced
              [ "$CFG" = nomtpcappfmtrace ] && ENVV+=(RADIANCE_STEP_TRACE=1 RADIANCE_XE_LAUNCH_PROFILE=1 RADIANCE_L0_QUEUE_PROFILING=1)
              # ... traced only, with and without the tapes (whether a tail step's cost is its recording + audit)
              [ "$CFG" = nomtpcappfmtr ] && ENVV+=(RADIANCE_STEP_TRACE=1)
              # run 110 (owner note 129 item 5): the expert matrix form's staging four bytes a load (bit for bit)
              [ "$CFG" = nomtpcappfmvec ] && ENVV+=(RADIANCE_XE_MOE_MX_VEC=1)
              # ... and with the attention split kernel's 8-byte loads + native butterflies (RADIANCE_XE_ATTN_VX=1, exact)
              [ "$CFG" = nomtpcappfmdefvx ] && ENVV+=(RADIANCE_XE_ATTN_VX=1)
              # owner note 134 item 2: the defaults traced, and with the prompt tail's experts staged
              # (RADIANCE_PREFILL_TAIL_STAGE=1, placement only), the 7K logprob form too
              [ "$CFG" = nomtpcappfmdeftr ] && ENVV+=(RADIANCE_STEP_TRACE=1)
              [ "$CFG" = nomtpcappfmdeftrnotape ] && ENVV+=(RADIANCE_STEP_TRACE=1 RADIANCE_L0_TAPE=0)
              # run 111 (owner note 135 item 2a): the tapes as graphs (no native lists) -- whether the tail's slow
              # collect comes with the backend's own immediate list
              [ "$CFG" = nomtpcappfmdeftrgraph ] && ENVV+=(RADIANCE_STEP_TRACE=1 RADIANCE_L0_TAPE=graph)
              # ... item 2a's fix: the decode step issued ahead behind a prompt's last chunk is not recorded on first
              # sight (RADIANCE_TAPE_DEFER_AFTER_PREFILL=1); 7k, 20k and the 7K logprob form, traced
              [ "$CFG" = nomtpcappfmdefdefer ] && PREFILL_NAMES=${PREFILL_NAMES:-7k 20k 7klp} &&
                  ENVV+=(RADIANCE_STEP_TRACE=1 RADIANCE_TAPE_DEFER_AFTER_PREFILL=1)
              [ "$CFG" = nomtpcappfmdeftrlp ] && PREFILL_NAMES=${PREFILL_NAMES:-7k 20k 7klp} && ENVV+=(RADIANCE_STEP_TRACE=1)
              [ "$CFG" = nomtpcappfmdeftail ] && PREFILL_NAMES=${PREFILL_NAMES:-7k 20k 7klp} &&
                  ENVV+=(RADIANCE_STEP_TRACE=1 RADIANCE_PREFILL_TAIL_STAGE=1)
              [ "$CFG" = nomtpcappfmlpvec ] && PREFILL_NAMES=${PREFILL_NAMES:-7klp} && ENVV+=(RADIANCE_XE_MOE_MX_VEC=1)
              [ "$CFG" = nomtpcappfmvecprof ] && ENVV+=(RADIANCE_XE_MOE_MX_VEC=1 RADIANCE_STEP_TRACE=1 RADIANCE_XE_LAUNCH_PROFILE=1
                                                         RADIANCE_L0_QUEUE_PROFILING=1)
              [ "$CFG" = nomtpcappfmtrnotape ] && ENVV+=(RADIANCE_STEP_TRACE=1 RADIANCE_L0_TAPE=0)
              [ "$CFG" = nomtpcappfmamprof ] && ENVV+=(RADIANCE_XE_ATTN_MX=1 RADIANCE_XE_LAUNCH_PROFILE=1 RADIANCE_L0_QUEUE_PROFILING=1)
              [ "$CFG" = nomtpcappfmlpam ] && PREFILL_NAMES=${PREFILL_NAMES:-7klp} && ENVV+=(RADIANCE_XE_ATTN_MX=1)
              [ "$CFG" = nomtpcappfmjm64cxstprof ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_XE_MOE_MX_COMPACT=1
                  RADIANCE_XE_MOE_MX_STAGE=1 RADIANCE_XE_LAUNCH_PROFILE=1 RADIANCE_L0_QUEUE_PROFILING=1)
              [ "$CFG" = nomtpcappfmar4 ] && ENVV+=(RADIANCE_XE_AR_PROTO=4)
              # run 163 (owner note 168 item 0): today's defaults (jm64) unprofiled, with the step trace -- a line a
              # chunk with its host phases and collect (the device time), to decompose the time outside the kernels
              [ "$CFG" = nomtpcappfmjm64tr ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1)
              # run 218 (owner YES ~06:50 11-07): jm32tr = jm64tr at v block 32 -- is the v64 JM kernel's 2.42.6 code
              # (3-4x the spill space) the prefill hang?
              [ "$CFG" = nomtpcappfmjm32tr ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=32 RADIANCE_STEP_TRACE=1)
              # run 258 (supervisor-approved, logs/run255-nohcpf-proposal.md): jm64tr with the prefill hc_read off the
              # matrix units -- is the hang r4d_hc_bf16's PF_DPAS kernels under IGC 2.42.6? (env applies left to right:
              # this =0 overrides the family's =1)
              [ "$CFG" = nomtpcappfmjm64trnohcpf ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_XE_HC_PF_DPAS=0)
              # run 258 (supervisor-approved, logs/run257-grf-proposal.md; owner note 209 item 4): jm64tr with the
              # prefill expert GEMM under the large register file
              [ "$CFG" = nomtpcappfmjm64trgrf ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_XE_I4I8_DPAS_GRF=1)
              # run 221 (the AR-overlap design's step 0): jm64tr at a 2048-row chunk budget -- is the prefill
              # chunk-split invariant (7klp logprobs), and what does a half-size chunk cost?
              [ "$CFG" = nomtpcappfmjm64tr2k ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1) && \
                ARGS=${ARGS/--max-num-batched-tokens 4096/--max-num-batched-tokens 2048}
              # run 208 (the supervisor's / owner's prefill separation of B156's hang): fz = the five fusion-map defaults
              # opted out; nodia / nocb = the family's NEO debug keys DisableIndirectAccess=1 / OverrideCmdListCmdBufferSizeInKb=64
              # dropped one at a time (removed from ENVV, so NEO's own default, not "=0")
              ri_drop() { local k=$1 e out=(); for e in "${ENVV[@]}"; do [ "${e%%=*}" = "$k" ] || out+=("$e"); done; ENVV=("${out[@]}"); }
              [ "$CFG" = nomtpcappfmjm64trfz ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_XE_HC_GU=0
                  RADIANCE_XE_HC_D16=0 RADIANCE_HC_WRITE_FUSE=0 RADIANCE_XE_GATE_QUANT=0 RADIANCE_XE_GATED_HAD=0)
              [ "$CFG" = nomtpcappfmjm64trnodia ] && ri_drop DisableIndirectAccess && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1)
              [ "$CFG" = nomtpcappfmjm64trnocb ] && ri_drop OverrideCmdListCmdBufferSizeInKb && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1)
              # run 208: nosplit = NEO's BCS split of large copies off (SplitBcsCopy=0; NEOReadDebugKeys=1 is already in the
              # family) -- the path of NEO 26.40's da1fc8be bcssplit race fix; ce0 = the SYCL adapter's copy engine off for
              # memcpy, so the staging copies run on the compute engine (the BCS path unused; staging may be slower)
              [ "$CFG" = nomtpcappfmjm64trnosplit ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 SplitBcsCopy=0)
              [ "$CFG" = nomtpcappfmjm64trce0 ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 UR_L0_USE_COPY_ENGINE=0 SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0)
              # run 188 (owner notes 182 / 184, the leak fix's gates): ga = the same with kernel launches outside a recording on
              # the backend's own immediate list (RADIANCE_L0_GLUE_ALL=1: no runtime event a launch, the adapter's pools stay put)
              [ "$CFG" = nomtpcappfmjm64trga ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_L0_GLUE_ALL=1)
              # run 189: the two-list form -- the mover's stream on a copy engine's list (2) or a second compute list (3)
              [ "$CFG" = nomtpcappfmjm64trga2 ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_L0_GLUE_ALL=2)
              [ "$CFG" = nomtpcappfmjm64trga3 ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_L0_GLUE_ALL=3)
              # run 175 (owner note 176 item 2): nowarm = the startup warm-up pass off (the truly cold first prompt,
              # kept beside the "first prompt after startup" figure the plain config now measures)
              [ "$CFG" = nomtpcappfmjm64trnowarm ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1) && ARGS="$ARGS --startup-warmup off"
              # run 262 (owner YES ~23:15 11-07, the warm-up gate; supervisor-approved): warmon = jm64tr with the startup
              # warm-up ON (the engine's default is off since image 247, note 197); the first 7K is then the warmed figure
              [ "$CFG" = nomtpcappfmjm64trwarmon ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1) && ARGS="$ARGS --startup-warmup on"
              # run 262 (owner note 210 item 1; supervisor-approved): hist = jm64tr with the expert launches' routing
              # histogram on stderr (RADIANCE_XE_I4I8_HIST=1, read-only), one B-step on image 284
              [ "$CFG" = nomtpcappfmjm64trhist ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_XE_I4I8_HIST=1)
              # run 263 (owner note 210 stage 3, the probe; supervisor-approved): v2prof = jm64tr with the expert GEMM's
              # V2 loop (RADIANCE_XE_I4I8_DPAS_V2=1: 128-K passes, register prefetch across the MMAs, SLM-only barriers,
              # dead sub-groups skipped) and the per-kernel launch profiler; the control is ...dpasprof. v2 = the same
              # without the profiler (the plain 7K / 20K the supervisor asked for before any default talk)
              [ "$CFG" = nomtpcappfmjm64trv2prof ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_XE_I4I8_DPAS_V2=1
                                                            RADIANCE_XE_LAUNCH_PROFILE=1 RADIANCE_L0_QUEUE_PROFILING=1)
              [ "$CFG" = nomtpcappfmjm64trv2 ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_XE_I4I8_DPAS_V2=1)
              # run 264 (supervisor-approved): v2grfprof = v2prof under the large register file (RADIANCE_XE_I4I8_DPAS_GRF=1
              # with _V2=1): V2 spills 7.2 KB a thread at 128 GRF; the AMD production kernel's own point is 141 of 256 VGPRs
              [ "$CFG" = nomtpcappfmjm64trv2grfprof ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_XE_I4I8_DPAS_V2=1
                                                               RADIANCE_XE_I4I8_DPAS_GRF=1 RADIANCE_XE_LAUNCH_PROFILE=1 RADIANCE_L0_QUEUE_PROFILING=1)
              # run 267 (owner lever #3, coordinator-approved): mbd = jm64tr with the protected experts' bf16 GEMM on the
              # matrix units (RADIANCE_XE_MOE_BF16_DPAS=1: opt-in, not bit-identical, note 149's bar); mbdprof adds the
              # launch profiler (the i4i8 matrix-unit form is the default -- dpas_on() is true unless =0 -- so this
              # compares with B187's dpasprof table as it is)
              [ "$CFG" = nomtpcappfmjm64trmbd ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_XE_MOE_BF16_DPAS=1)
              [ "$CFG" = nomtpcappfmjm64trmbdprof ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_XE_MOE_BF16_DPAS=1
                                                             RADIANCE_XE_LAUNCH_PROFILE=1 RADIANCE_L0_QUEUE_PROFILING=1)
              # run 268 (the coordinator's live-rows question, approved): mbdtrace = mbdprof + the launcher's per-call trace of
              # the live rows / live blocks it covered (RADIANCE_XE_MOE_BF16_DPAS_TRACE=1: a blocking copy a call, so this
              # config's first-token times are not comparable; the profiler's per-launch device times and the trace lines are)
              [ "$CFG" = nomtpcappfmjm64trmbdtrace ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_XE_MOE_BF16_DPAS=1
                                                              RADIANCE_XE_MOE_BF16_DPAS_TRACE=1 RADIANCE_XE_LAUNCH_PROFILE=1 RADIANCE_L0_QUEUE_PROFILING=1)
              # run 175 (owner note 176 item 3): pre = the expert codes decoded once a call into the op's scratch
              [ "$CFG" = nomtpcappfmjm64trpre ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_XE_I4I8_DPAS_PRE=1)
              # run 175 (owner note 177 item 2, measure first): ci8k = the checkpoint interval 8192, so the 7K prompt's
              # last chunk runs whole (1,508 rows, no 36-row tail step) -- the tail fold's gain without the mid-pass
              # checkpoint (and without the deep checkpoint it would keep)
              [ "$CFG" = nomtpcappfmjm64trci8k ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1) && ARGS=${ARGS/--checkpoint-interval 4096/--checkpoint-interval 8192}
              # run 164 (the supervisor's regression check): the same with format C off, so prefill runs the f16 matrix
              # expert form (moe_planes_mx) as before note 160 -- against B60
              [ "$CFG" = nomtpcappfmjm64trnofc ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_MOE_FORMAT_C=0)
              # run 164 (owner note 168 item 2): format C's expert GEMM on the matrix units at prefill (RADIANCE_XE_I4I8_DPAS=1)
              [ "$CFG" = nomtpcappfmjm64trdpas ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_XE_I4I8_DPAS=1)
              # run 202 (owner note 195): the DPAS expert GEMM over its block map (RADIANCE_XE_I4I8_DPAS_MAP=1), and profiled
              [ "$CFG" = nomtpcappfmjm64trmap ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_XE_I4I8_DPAS_MAP=1)
              # ...trtail: the row grid with the map's tail rule (rows past the call's live count zeroed; a test, SLMC=6)
              [ "$CFG" = nomtpcappfmjm64trtail ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_XE_I4I8_DPAS_SLMC=6)
              [ "$CFG" = nomtpcappfmjm64trmapprof ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_XE_I4I8_DPAS_MAP=1
                                                           RADIANCE_ATTN_QSA_FMHA=1 RADIANCE_XE_LAUNCH_PROFILE=1 RADIANCE_L0_QUEUE_PROFILING=1)
              [ "$CFG" = nomtpcappfmjm64trdpasprof ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_XE_I4I8_DPAS=1
                                                             RADIANCE_XE_LAUNCH_PROFILE=1 RADIANCE_L0_QUEUE_PROFILING=1)
              # run 167 (owner note 168 item 1): the sparse pass's attention on the matrix units (attn_qsa_fmha, opt-in)
              [ "$CFG" = nomtpcappfmjm64trfmha ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_ATTN_QSA_FMHA=1)
              [ "$CFG" = nomtpcappfmjm64trfmhaprof ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_ATTN_QSA_FMHA=1
                                                            RADIANCE_XE_LAUNCH_PROFILE=1 RADIANCE_L0_QUEUE_PROFILING=1)
              # run 169 (owner note 174 item 1 = note 168 item 4 + note 166): today's defaults under the engine's own
              # per-op timer (--profile-ops: host ops such as the n-gram gather on the wall clock, with their join;
              # device ops by event pairs, so the first-token time itself is not comparable); the table every chunk
              [ "$CFG" = nomtpcappfmjm64tropsprof ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_PROFILE_EVERY=1) &&
                                                   ARGS="$ARGS --profile-ops"
              # run 169 (note 166 item 1): today's defaults with the n-gram gather's own timing line a call (libavx,
              # RADIANCE_NGRAM_TIMING=1: cache hits / reads / waits and the phases' ms) beside the step trace
              [ "$CFG" = nomtpcappfmjm64trng ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_NGRAM_TIMING=1)
              # run 169 (owner notes 166 / 174): the same with libavx serving the host rows ahead of libref -- its n-gram
              # gather reads the table through an io_uring row cache with a read-ahead thread instead of libref's
              # page-faulting mmap walk (the step needs BSTEP_SECCOMP=1: the io_uring seccomp profile)
              [ "$CFG" = nomtpcappfmjm64trngavx ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_NGRAM_TIMING=1) &&
                                                  ARGS=${ARGS/--kernels libxe:libref/--kernels libxe:libavx:libref}
              # run 170 (owner note 172 d): the all-reduce's bandwidth form for prefill's large messages (proto 6: the
              # payload once, a read-back before the flag; the LL form 5 below 256 KiB), with and without the attention op
              [ "$CFG" = nomtpcappfmjm64trar6 ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_XE_AR_PROTO=6)
              [ "$CFG" = nomtpcappfmjm64trfmhaar6 ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_ATTN_QSA_FMHA=1
                                                           RADIANCE_XE_AR_PROTO=6)
              # run 173: ...ar6 with the step parts' timer (RADIANCE_DEBUG_STAGE=1: a prefill-sized pass's stager / residency /
              # pass / join / end parts, printed at once), the lone-prompt baseline for the lanes steps' shared chunk passes
              [ "$CFG" = nomtpcappfmjm64trfmhaar6ds ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_ATTN_QSA_FMHA=1
                                                             RADIANCE_XE_AR_PROTO=6 RADIANCE_DEBUG_STAGE=1)
              # run 173: RADIANCE_L0_RECORD_BARRIER=1 -- every event record a barrier instead of the queue's last event (B131:
              # the records blocked the host 3.5 s a chunk); dsrb with the timers, rb plain for the gate (the 7K logprobs)
              [ "$CFG" = nomtpcappfmjm64trfmhaar6dsrb ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_ATTN_QSA_FMHA=1
                                                               RADIANCE_XE_AR_PROTO=6 RADIANCE_DEBUG_STAGE=1 RADIANCE_L0_RECORD_BARRIER=1)
              [ "$CFG" = nomtpcappfmjm64trfmhaar6rb ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_ATTN_QSA_FMHA=1
                                                             RADIANCE_XE_AR_PROTO=6 RADIANCE_L0_RECORD_BARRIER=1)
              # run 173: RADIANCE_DEBUG_STAGE=2 -- the stager host-waits on each layer's copy list and logs its completion time
              # (the copy path's raw rate); it serialises the pipeline, a diagnostic only
              [ "$CFG" = nomtpcappfmjm64trfmhaar6ds2 ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_ATTN_QSA_FMHA=1
                                                              RADIANCE_XE_AR_PROTO=6 RADIANCE_DEBUG_STAGE=2)
              # run 173: the same with the SYCL Level Zero adapter's copy engine forced on / off for memcpy (the staging copies)
              [ "$CFG" = nomtpcappfmjm64trfmhaar6dsce1 ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_ATTN_QSA_FMHA=1
                                                                RADIANCE_XE_AR_PROTO=6 RADIANCE_DEBUG_STAGE=1
                                                                UR_L0_USE_COPY_ENGINE=1 SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=1)
              [ "$CFG" = nomtpcappfmjm64trfmhaar6dsce0 ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_ATTN_QSA_FMHA=1
                                                                RADIANCE_XE_AR_PROTO=6 RADIANCE_DEBUG_STAGE=1
                                                                UR_L0_USE_COPY_ENGINE=0 SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0)
              # run 173: the SYCL adapter's queues on batched regular command lists instead of immediate ones (one driver
              # flush a batch instead of one an append: the staging copies' host cost, R8's frames -- flushImmediateTask ->
              # DrmCommandStreamReceiver::flush -> makeResidentWithinOsContext -> bindBOs on every memcpy)
              [ "$CFG" = nomtpcappfmjm64trfmhaar6dsnoimm ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_ATTN_QSA_FMHA=1
                                                                  RADIANCE_XE_AR_PROTO=6 RADIANCE_DEBUG_STAGE=1
                                                                  UR_L0_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0)
              # run 170: ...ar6 with the expert DPAS kernel's 4-bit level decode SKIPPED (RADIANCE_XE_I4I8_DPAS_SLMC=2):
              # WRONG numbers by construction -- a timing-only step for the decode's share of the kernel
              [ "$CFG" = nomtpcappfmjm64trfmhaar6nodec ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_ATTN_QSA_FMHA=1
                                                                RADIANCE_XE_AR_PROTO=6 RADIANCE_XE_I4I8_DPAS_SLMC=2)
              # run 170: the same ablation for the kernel's other parts (wrong numbers by design, timing only):
              # =3 the scaling's loads skipped, =4 the B staging skipped, =5 the MMAs and tile loads skipped
              [ "$CFG" = nomtpcappfmjm64trfmhaar6nosc ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_ATTN_QSA_FMHA=1
                                                               RADIANCE_XE_AR_PROTO=6 RADIANCE_XE_I4I8_DPAS_SLMC=3)
              [ "$CFG" = nomtpcappfmjm64trfmhaar6nostage ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_ATTN_QSA_FMHA=1
                                                                  RADIANCE_XE_AR_PROTO=6 RADIANCE_XE_I4I8_DPAS_SLMC=4)
              [ "$CFG" = nomtpcappfmjm64trfmhaar6nomma ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_ATTN_QSA_FMHA=1
                                                                RADIANCE_XE_AR_PROTO=6 RADIANCE_XE_I4I8_DPAS_SLMC=5)
              # run 171: ...ar6 with the expert DPAS kernel's 128-row work-group of 16 sub-groups (RADIANCE_XE_I4I8_DPAS_WAVES=16)
              [ "$CFG" = nomtpcappfmjm64trfmhaar6w16 ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_ATTN_QSA_FMHA=1
                                                              RADIANCE_XE_AR_PROTO=6 RADIANCE_XE_I4I8_DPAS_WAVES=16)
              # run 171: ...ar6 with 16 rows a sub-group under the large register file (RADIANCE_XE_I4I8_DPAS_ROWS=16)
              [ "$CFG" = nomtpcappfmjm64trfmhaar6rt16 ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_ATTN_QSA_FMHA=1
                                                               RADIANCE_XE_AR_PROTO=6 RADIANCE_XE_I4I8_DPAS_ROWS=16)
              # run 171: ...ar6 with the expert run form at prefill (RADIANCE_XE_I4I8_DPAS=0): the reference bits for the
              # 64-row / 128-row DPAS blocks' disagreement
              [ "$CFG" = nomtpcappfmjm64trfmhaar6nodpas ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_ATTN_QSA_FMHA=1
                                                                 RADIANCE_XE_AR_PROTO=6 RADIANCE_XE_I4I8_DPAS=0)
              # ...tropsprofavx: the per-op timer (join / wait a call) with libavx serving, for what is left outside the kernels
              [ "$CFG" = nomtpcappfmjm64tropsprofavx ] && ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_STEP_TRACE=1 RADIANCE_PROFILE_EVERY=1) &&
                                                       ARGS="${ARGS/--kernels libxe:libref/--kernels libxe:libavx:libref} --profile-ops"
              # ...jm64cx / ...jm64cxprof: with the expert matrix form's compact grid (RADIANCE_XE_MOE_MX_COMPACT=1)
              # a libxe.so newer than the image, from the step folder (run 108: amd-host at its disk floor, no new image);
              # it replaces the container's copy only, never the image's
              [ -f /out/libxe.so ] && cp /out/libxe.so /opt/ri/radiance_home/kernels/libxe.so && echo "libxe.so from /out"
              case $CFG in nomtpcappfmjm64cx) ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_XE_MOE_MX_COMPACT=1) ;;
                           nomtpcappfmjm64cxprof) ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_XE_MOE_MX_COMPACT=1
                                                         RADIANCE_XE_LAUNCH_PROFILE=1 RADIANCE_L0_QUEUE_PROFILING=1) ;;
                           nomtpcappfmjm64prof) ENVV+=(RADIANCE_XE_GDN_CS_JM=64 RADIANCE_XE_LAUNCH_PROFILE=1 RADIANCE_L0_QUEUE_PROFILING=1) ;;
                           nomtpcappfmjm*) ENVV+=(RADIANCE_XE_GDN_CS_JM=${CFG#nomtpcappfmjm}) ;; esac
              ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
              case $CFG in nomtpcappfmhr*) ARGS=${ARGS/--gpu-headroom-mib 2560/--gpu-headroom-mib ${CFG#nomtpcappfmhr}} ;;
                           *)              ARGS=${ARGS/--gpu-headroom-mib 2560/--gpu-headroom-mib 1280} ;; esac
              ARGS="$ARGS --checkpoint-slots 2" ;;
  # the 7K prompt with logprobs (top 5) at the candidate and in B32's form: B35's 7K token differed from B32's
  nomtpcaplp) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1
                  DisableIndirectAccess=1 OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_TAPE_VRAM_FLOOR_MIB=256
                  RADIANCE_VRAM_CAP_FROM_FREE=1)
              PREFILL_NAMES=${PREFILL_NAMES:-7klp}
              ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}
              ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
              ARGS="$ARGS --checkpoint-slots 2" ;;
  nomtpmxfixlp) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1
                    DisableIndirectAccess=1); PREFILL_NAMES=${PREFILL_NAMES:-7klp}
                ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
  nomtpmx8k)  ENVV=(RADIANCE_XE_MOE_MX=1); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}
              ARGS=${ARGS/--max-num-batched-tokens 4096/--max-num-batched-tokens 8192} ;;
  nomtpmx8ks) ENVV=(RADIANCE_XE_MOE_MX=1); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}
              ARGS="${ARGS/--max-num-batched-tokens 4096/--max-num-batched-tokens 8192} --prefill-span 2" ;;
  nomtpmx16ks) ENVV=(RADIANCE_XE_MOE_MX=1); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}
              ARGS="${ARGS/--max-num-batched-tokens 4096/--max-num-batched-tokens 16384} --prefill-span 4" ;;
  # run 61: B14 showed the span is taken only for prompts of 8 x the interval (32,768 tokens; engine.cpp:1613-1620,
  # upstream's R9700 finding that a spanned step costs expert residency), so the 7K / 20K prompts kept 4,096-token
  # chunks. The interval form: --checkpoint-interval 8192 with the budget 8,192 makes every chunk 8,192 without a
  # span (the linear-state checkpoint grid coarsens with it); RADIANCE_PREFILL_SPAN_MIN (the commit after 6e3e3e5)
  # lowers the span's minimum instead, keeping the grid.
  nomtpmx8ki) ENVV=(RADIANCE_XE_MOE_MX=1); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}
              ARGS=${ARGS/--max-num-batched-tokens 4096 --checkpoint-interval 4096/--max-num-batched-tokens 8192 --checkpoint-interval 8192} ;;
  nomtpmx8km) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_PREFILL_SPAN_MIN=4096); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}
              ARGS="${ARGS/--max-num-batched-tokens 4096/--max-num-batched-tokens 8192} --prefill-span 2" ;;
  nomtpmx16km) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_PREFILL_SPAN_MIN=4096); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}
              ARGS="${ARGS/--max-num-batched-tokens 4096/--max-num-batched-tokens 16384} --prefill-span 4" ;;
  *) echo "unknown config $CFG"; exit 2 ;;
esac
echo "$CFG: env ${ENVV[*]:-(none)}; args $ARGS"
t0=$(date +%s)
# run 163 (owner note 169 item 3): UT_PREFIX, set by the environment of the step's command (e.g. unitrace's device
# timing and submission intervals: "unitrace --device-timing --kernel-submission -o /out/unitrace"), wraps the engine
env "${ENVV[@]}" timeout 3000 ${UT_PREFIX:-} /opt/ri/bin/radiance $ARGS --radiance-home $H -v > /out/serve.log 2>&1 &
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
  n=$(grep -c -v 'startup: nothing has been logged' /out/serve.log 2>/dev/null)
  if [ "$n" != "${last_n:-}" ]; then last_n=$n; last_t=$(date +%s); fi
  if [ $(( $(date +%s) - ${last_t:-$t0} )) -ge 180 ]; then
    echo "$CFG startup: nothing logged for 180 s -- card not recovered after the last fault (owner note 175); engine stopped"
    kill $pid 2>/dev/null; break
  fi
  sleep 1
done 2>/dev/null
echo "$CFG up=$up after $(( $(date +%s) - t0 )) s"
if [ $up = 1 ]; then
  mkdir -p /out/resp
  for name in ${PREFILL_NAMES:-7k 20k}; do   # milestone 3 run 48: PREFILL_NAMES="7klp" posts the logprobs body only
    for k in $(seq "$REPS"); do
      w0=$EPOCHREALTIME
      st=$(post_raw /v1/completions /out/card/prefill-$name.body /out/resp/prefill-$name-$k.http)
      w1=$EPOCHREALTIME
      line="prefill-$name $k $(echo "$w0 $w1" | awk '{printf "%.2f", $2-$1}') $st"
      echo "$line" | tee -a /out/prefill.txt
      # run 110 (owner note 129 item 1): /stats after each request (the experts' prefill staging counters)
      if exec 4<>/dev/tcp/127.0.0.1/$P; then
        printf 'GET /stats HTTP/1.1\r\nHost: l\r\nConnection: close\r\n\r\n' >&4
        cat <&4 > /out/stats-$name-$k.txt; exec 4<&-
      fi
    done
  done
  grep -E "req .* done" /out/serve.log | grep -oE "prompt=[0-9]+ .*ttft=[0-9.]+s" | tail -4
fi
# run 112: the stop's timing (B83's server died of the signal without draining)
echo "$CFG stop sent $(date -u +%H:%M:%S.%N)"
kill $pid 2>/dev/null; wait $pid; echo "$CFG server exit $? at $(date -u +%H:%M:%S.%N)"
grep -E "^E |step prefill|prefill" /out/serve.log | tail -4
