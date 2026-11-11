#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 115, owner note 146 item 1). Inside ri-m3rt on both cards,
# started by D/m3-step.sh with STEP_ENGINE=1 and STEP_CARD=both: GL's token-replay check. The engine's KLD mode
# teacher-forces /out/gl-replay.jsonl (the agreement requests with the control's answers, m3/tools/replay-corpus.py)
# through prefill twice, at m3-agree.sh's mtp4gm3 setting: first recording every position's log-probabilities with
# the GEMV (GL off), then scoring GL on against them -- mean / percentile KL and top-1 agreement, every position in
# /out/kld-gl.json.rows. RADIANCE_XE_GEMM_DPAS=0 in both, so every trunk row goes through the multi-row GEMV launches
# GL replaces (otherwise a prefill's 64-row blocks take the matrix units and only the tails would differ).
#   bash /out/kld-gl.sh
H=/opt/ri/radiance_home
[ -f /out/libxe.so ] && cp /out/libxe.so $H/kernels/libxe.so && echo "libxe.so from /out"
ARGS=$(grep -v '^#' /out/flashnext-card1.args | tr '\n' ' ')
ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}
ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}   # the KL mode scores the trunk's rows, drafts nothing
# run 187: mtpparity keeps the MTP drafter (its history pass drafts from each prompt's last row; MD5's first try
# refused at --num-speculative-tokens 0)
{ [ "${1:-}" = mtpparity ] || [ "${1:-}" = mtpparity3 ] || [ "${1:-}" = mtpparity3f ] || [ "${1:-}" = mtpparity3g ]; } && ARGS=${ARGS/--num-speculative-tokens 0/--num-speculative-tokens 3}
ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_XE_MOE_MX_STAGE=1 RADIANCE_XE_GEMM_DPAS=0)
# `dpas` (run 164): the matrix-unit expert form is the default from image 188, so the record pass must name the run
# form (the exact path) -- KDP1 scored the form against itself (the floor, 1.5e-8) before this line
[ "${1:-}" = dpas ] && ENVV+=(RADIANCE_XE_I4I8_DPAS=0)
# `hci82` (run 125): record and score in 64-token prefill chunks (the engine's least: GDN's chunk), which stage 2 serves
[ "${1:-}" = hci82 ] && ARGS=${ARGS/--max-num-batched-tokens 4096/--max-num-batched-tokens 64}
# `lean` / `lean64` (run 174, owner note 165 item 4): the routing lean at the verify step's shape -- record and score
# in 4-token prefill chunks (a verify step is 4 consecutive rows), the lean on in the score pass only; lean64 the
# same in 64-token chunks (the engine's chunk floor if it refuses 4), the lean's row cap raised to 64
[ "${1:-}" = lean ] && ARGS=${ARGS/--max-num-batched-tokens 4096/--max-num-batched-tokens 4}
# `fmhad` (run 174, owner note 176 item 1): the attention op is the default from image 213, so its broader replay
# (>= 5 long prompts) records with the op off and scores with it on
[ "${1:-}" = fmhad ] && ENVV+=(RADIANCE_ATTN_QSA_FMHA=0)
# `map` (run 202, owner note 195): the record pass on the row grid with the map's tail rule (RADIANCE_XE_I4I8_DPAS_SLMC=6,
# the wrapped-leader blocks zeroed), the score pass on the block map: exact against the corrected grid = the floor.
# `mapraw`: the record pass on today's row grid (with the wrapped-leader blocks), scored on the map: the fix's size.
[ "${1:-}" = map ] && ENVV+=(RADIANCE_XE_I4I8_DPAS_SLMC=6)
# `gsf16` (run 294, owner note 235): both passes print the GDN state's range (max |h| a layer, fp32 in the record pass,
# fp16 in the score pass) to their logs, every 2,000 probes and at exit
[ "${1:-}" = gsf16 ] && ENVV+=(RADIANCE_GDN_STATE_RANGE=1)
[ "${1:-}" = lean64 ] || [ "${1:-}" = lean64b ] || [ "${1:-}" = lean64c ] && ARGS=${ARGS/--max-num-batched-tokens 4096/--max-num-batched-tokens 64}
# `mtpdump` (run 179, owner note 181 phase 1): the record pass only, with the MTP head's pairs written to /out/mtpdump
# (RADIANCE_DUMP_HWIDE=1 keeps the head's input rows; RADIANCE_MTP_DUMP names the directory); no score pass
RTL=1200
if [ "${1:-}" = mtpdump ] || [ "${1:-}" = mtpdumpbig ] || [ "${1:-}" = mtpparity ] || [ "${1:-}" = mtpparity3 ] || [ "${1:-}" = mtpparity3f ] || [ "${1:-}" = mtpparity3g ]; then
  mkdir -p /out/mtpdump
  ENVV+=(RADIANCE_DUMP_HWIDE=1 RADIANCE_MTP_DUMP=/out/mtpdump)
  # mtpdumpbig (run 179): the real dump -- no reference written (RADIANCE_KLD_NO_REF=1: ~0.5 MB a position) and a
  # longer limit (~1M positions)
  [ "${1:-}" = mtpdumpbig ] && ENVV+=(RADIANCE_KLD_NO_REF=1) && RTL=6600
  # mtpparity (run 180, note 181's parity check): the dump plus the head's own round-1 top-32 at every document's
  # last prompt row (parity.bin), on the full-logits head (RADIANCE_MTP_DRAFT_HEAD_Q2=0); no reference written
  [ "${1:-}" = mtpparity ] && ENVV+=(RADIANCE_KLD_NO_REF=1 RADIANCE_MTP_PARITY=1 RADIANCE_MTP_DRAFT_HEAD_Q2=0) && RTL=3000
  # mtpparity3 (run 283, owner note 224 item 4, the v2 brief's parity2.bin): the same with the head's rounds 1-3 -- the
  # round-2 and round-3 draft passes fed the previous round's greedy draft, the coupling off -- into parity2.bin
  # (RADIANCE_MTP_PARITY_ROUNDS=3; the engine refuses it under a depth < 3 or with the coupling on)
  [ "${1:-}" = mtpparity3 ] && ENVV+=(RADIANCE_KLD_NO_REF=1 RADIANCE_MTP_PARITY=1 RADIANCE_MTP_PARITY_ROUNDS=3 RADIANCE_MTP_DRAFT_HEAD_Q2=0 RADIANCE_MTP_COUPLED=0) && RTL=3000
  # mtpparity3f (run 284, the coordinator's strict gate): mtpparity3 on the B155 prefill path -- the bf16 matrix-unit
  # expert form off (RADIANCE_XE_MOE_BF16_DPAS=0; on by default since note 212, it changes prompt rows > 256), so round
  # 1 must be byte-identical to the 11-04 parity.bin's rows
  [ "${1:-}" = mtpparity3f ] && ENVV+=(RADIANCE_KLD_NO_REF=1 RADIANCE_MTP_PARITY=1 RADIANCE_MTP_PARITY_ROUNDS=3 RADIANCE_MTP_DRAFT_HEAD_Q2=0 RADIANCE_MTP_COUPLED=0 RADIANCE_XE_MOE_BF16_DPAS=0) && RTL=3000
  # mtpparity3g (run 283, KPAR1's follow-up): mtpparity3f on the 11-04 numerics -- the format C row grid WITHOUT the
  # wrapped-leader fix (RADIANCE_XE_I4I8_DPAS_MAP=0, the map's opt-out; the map is the default since image 242, after
  # MD5c's image 227), so round 1 can be compared with the 11-04 parity.bin bug for bug
  [ "${1:-}" = mtpparity3g ] && ENVV+=(RADIANCE_KLD_NO_REF=1 RADIANCE_MTP_PARITY=1 RADIANCE_MTP_PARITY_ROUNDS=3 RADIANCE_MTP_DRAFT_HEAD_Q2=0 RADIANCE_MTP_COUPLED=0 RADIANCE_XE_MOE_BF16_DPAS=0 RADIANCE_XE_I4I8_DPAS_MAP=0) && RTL=3000
fi
# norefrec (run 187, owner note 182's memory question): the same record pass with nothing written -- no dump, no
# reference -- so a memory slope here is the engine's KL / prefill path, not the dump's
if [ "${1:-}" = norefrec ]; then ENVV+=(RADIANCE_KLD_NO_REF=1); RTL=3000; fi
# norefavx (run 187): the same on the service's kernels -- libavx's io_uring n-gram gather in place of libref's
# page-faulting one (the step needs the io_uring seccomp profile: kdstep.sh's 7th argument)
if [ "${1:-}" = norefavx ]; then
  ENVV+=(RADIANCE_KLD_NO_REF=1); RTL=3000
  ARGS=${ARGS/--kernels libxe:libref/--kernels libxe:libavx:libref}
fi
# norefctrace / norefevict (run 187): norefrec with the SYCL runtime's in-memory program cache traced (MD10's arenas held
# copies of one kernel's device image: a cache that adds a program per launch variant would do that), or bounded
if [ "${1:-}" = norefctrace ]; then ENVV+=(RADIANCE_KLD_NO_REF=1 SYCL_CACHE_TRACE=2); RTL=3000; fi
# norefprof (run 187; image ri-m3rtprof = rtneo:228 + jemalloc): norefrec under jemalloc's sampling heap profiler, a
# final profile at exit in /out/jeprof.* -- the call sites that still hold memory when the corpus ends
if [ "${1:-}" = norefprof ]; then
  ENVV+=(RADIANCE_KLD_NO_REF=1 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
         MALLOC_CONF=prof:true,prof_final:true,lg_prof_sample:19,prof_prefix:/out/jeprof)
  RTL=3000
fi
# norefinorder / norefnocache (run 187): MD12 put the growth in the Level Zero adapter's per-launch events
# (urEnqueueKernelLaunch -> EventCreate), live at exit; MD13 (no event asked) still grew. Two adapter knobs:
# the driver's in-order lists (counter-based events), or no event caching (a released event is destroyed)
if [ "${1:-}" = norefinorder ]; then ENVV+=(RADIANCE_KLD_NO_REF=1 UR_L0_USE_DRIVER_INORDER_LISTS=1); RTL=3000; fi
if [ "${1:-}" = norefnocache ]; then
  ENVV+=(RADIANCE_KLD_NO_REF=1 UR_L0_DISABLE_EVENTS_CACHING=1 SYCL_PI_LEVEL_ZERO_DISABLE_EVENTS_CACHING=1); RTL=3000
fi
# norefleaks (run 187): the adapter's own create / destroy table at exit (UR_L0_LEAKS_DEBUG=1) -- how many events and
# event pools each launch makes and how many outlive the run. (norefnocache, MD15: out of DEVICE memory at the first
# prefill step, rc 134, klog clean.)
if [ "${1:-}" = norefleaks ]; then ENVV+=(RADIANCE_KLD_NO_REF=1 UR_L0_LEAKS_DEBUG=1); RTL=3000; fi
# norefpool4k (run 187): MD16's table -- 17,310 event pools created, 0 destroyed (4.43M events, 256 a pool); 16x larger
# pools: if a pool's cost is mostly fixed, the growth falls ~16x
if [ "${1:-}" = norefpool4k ]; then
  ENVV+=(RADIANCE_KLD_NO_REF=1 UR_L0_LEAKS_DEBUG=1 UR_L0_MAX_NUMBER_OF_EVENTS_PER_EVENT_POOL=4096); RTL=3000
fi
if [ "${1:-}" = norefevict ]; then ENVV+=(RADIANCE_KLD_NO_REF=1 SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD=268435456); RTL=3000; fi
# gabase / gadump (run 187): the MTP dump (no reference) with the adapter's create / destroy table, without and with
# RADIANCE_L0_GLUE_ALL=1 -- the leak (pools) and exactness (top32.bin / hidden.bin byte-identical between the two)
if [ "${1:-}" = gabase ] || [ "${1:-}" = gadump ] || [ "${1:-}" = gadump2 ]; then
  mkdir -p /out/mtpdump
  ENVV+=(RADIANCE_DUMP_HWIDE=1 RADIANCE_MTP_DUMP=/out/mtpdump RADIANCE_KLD_NO_REF=1 UR_L0_LEAKS_DEBUG=1)
  [ "${1:-}" = gadump ] && ENVV+=(RADIANCE_L0_GLUE_ALL=1)
  [ "${1:-}" = gadump2 ] && ENVV+=(RADIANCE_L0_GLUE_ALL=2)   # run 189: the two-list form
  RTL=3000
fi
# norefsmaps (run 187): norefrec with the engine's /proc/PID/smaps copied every 2 minutes (smaps-N.txt): which mappings
# hold the anonymous growth (the malloc heap and its arenas, the driver's own mappings, or another)
if [ "${1:-}" = norefsmaps ]; then
  ENVV+=(RADIANCE_KLD_NO_REF=1)
  echo "=== record with smaps ($(date -u +%H:%M:%S))"
  env "${ENVV[@]}" timeout 3000 /opt/ri/bin/radiance $ARGS --radiance-home $H --kld-corpus /out/gl-replay.jsonl \
      --kld-record /out/kref > /out/kld-record.log 2>&1 &
  tp=$!
  sleep 5
  rp=""; read -r rp _ < /proc/$tp/task/$tp/children
  echo "timeout $tp, engine ${rp:-?}"
  i=0
  while kill -0 $tp 2>/dev/null; do
    sleep 120; i=$((i + 1))
    [ -n "$rp" ] && cp /proc/$rp/smaps /out/smaps-$i.txt 2>/dev/null
    # at 8 minutes: 8 MiB from each of the two fullest unnamed 64 MiB rw-p mappings (malloc arenas), what they hold
    if [ "$i" = 4 ] && [ -n "$rp" ]; then
      awk '/^[0-9a-f]+-[0-9a-f]+ /{split($1,r,"-"); s=r[1]; p=$2; path=(NF>=6)?$6:""; next}
           /^Size:/{sz=$2} /^Anonymous:/{if (sz==65536 && p=="rw-p" && path=="") print $2, s}' /out/smaps-4.txt |
        sort -rn | head -2 > /out/arena-pick.txt
      k=0
      while read -r an st; do
        k=$((k + 1))
        dd if=/proc/$rp/mem of=/out/arena-$k.bin bs=1M iflag=skip_bytes,count_bytes \
           skip=$(( 0x$st + 16777216 )) count=8388608 2>> /out/arena-dd.txt
      done < /out/arena-pick.txt
    fi
  done
  wait $tp; echo "record rc=$?"; tail -3 /out/kld-record.log; ls -la /out/smaps-*.txt
  exit 0
fi
echo "=== record, GL off ($(date -u +%H:%M:%S))"
env "${ENVV[@]}" timeout $RTL /opt/ri/bin/radiance $ARGS --radiance-home $H --kld-corpus /out/gl-replay.jsonl \
    --kld-record /out/kref > /out/kld-record.log 2>&1
echo "record rc=$?"; tail -3 /out/kld-record.log
if [ "${1:-}" = mtpdump ] || [ "${1:-}" = mtpdumpbig ] || [ "${1:-}" = mtpparity ] || [ "${1:-}" = mtpparity3 ] || [ "${1:-}" = mtpparity3f ] || [ "${1:-}" = mtpparity3g ] || [ "${1:-}" = norefrec ] || [ "${1:-}" = norefavx ] || [ "${1:-}" = norefctrace ] || [ "${1:-}" = norefevict ] || [ "${1:-}" = norefprof ] || [ "${1:-}" = norefinorder ] || [ "${1:-}" = norefnocache ] || [ "${1:-}" = norefleaks ] || [ "${1:-}" = norefpool4k ] || [ "${1:-}" = gabase ] || [ "${1:-}" = gadump ] || [ "${1:-}" = gadump2 ]; then
  grep -E "zeEventPoolCreate|zeEventCreate |glue-all" /out/kld-record.log | tail -4
  ls -la /out/mtpdump 2>/dev/null
  ls -la /out/jeprof.* 2>/dev/null
  grep -E "MTP head's (pairs|own|parity)|NO_REF|PARITY" /out/kld-record.log
  ls -la /out/mtpdump
  exit 0
fi
# `bash /out/kld-gl.sh ctl`: the control -- GL off scored against GL off (the KL mode's own run-to-run floor)
# `bash /out/kld-gl.sh amx` (run 116): another non-identical form, attention on the matrix units (RADIANCE_XE_ATTN_MX=1,
# note 130's, opt-in), scored the same way -- how far any last-bit change moves this model (the router's near-ties)
SV=(RADIANCE_XE_GEMV_GL=1); OUT=kld-gl
[ "${1:-}" = ctl ] && SV=(RADIANCE_XE_GEMV_GL=0) && OUT=kld-ctl
[ "${1:-}" = amx ] && SV=(RADIANCE_XE_GEMV_GL=0 RADIANCE_XE_ATTN_MX=1) && OUT=kld-amx
# `bash /out/kld-gl.sh fc` (run 117, owner note 147): format C -- today's w4nl experts against int8 activations
# (RADIANCE_MOE_FORMAT_C=1, opt-in), every prefill row through xe_moe_gemm_w4nl_i8, scored against off
[ "${1:-}" = fc ] && SV=(RADIANCE_XE_GEMV_GL=0 RADIANCE_MOE_FORMAT_C=1) && OUT=kld-fc
# `rows` / `hcsr` (run 124, owner note 153): the trunk's 4-row GEMV on the matrix units (RADIANCE_XE_GEMM_DPAS_ROWS=1) and
# hc on the matrix units (RADIANCE_XE_HC_PF_DPAS=2), each scored against off. (DPAS=0 keeps the prefill rows on the
# multi-row paths these forms replace.)
[ "${1:-}" = rows ] && SV=(RADIANCE_XE_GEMV_GL=0 RADIANCE_XE_GEMM_DPAS_ROWS=1) && OUT=kld-rows
[ "${1:-}" = hcsr ] && SV=(RADIANCE_XE_GEMV_GL=0 RADIANCE_XE_HC_PF_DPAS=2) && OUT=kld-hcsr
# `hci8` / `hc8` (run 125, owner note 158): hc's mixing matrices on int8 rows (RADIANCE_XE_HC_I8=1, read by the V2
# kernels only, so V2 level 8 on), and level 8 alone (E4M3, its own accumulation order), each scored against off
[ "${1:-}" = hci8 ] && SV=(RADIANCE_XE_GEMV_GL=0 RADIANCE_XE_HC_V2=8 RADIANCE_XE_HC_I8=1) && OUT=kld-hci8
[ "${1:-}" = hc8 ] && SV=(RADIANCE_XE_GEMV_GL=0 RADIANCE_XE_HC_V2=8) && OUT=kld-hc8
# run 164 (owner note 168 item 2): format C's expert GEMM on the matrix units at prefill chunks (the default from
# image 188; =0 the run form), replayed at the exact path's prompts
[ "${1:-}" = dpas ] && SV=(RADIANCE_XE_GEMV_GL=0 RADIANCE_XE_I4I8_DPAS=1) && OUT=kld-dpas
[ "${1:-}" = nodpas ] && SV=(RADIANCE_XE_GEMV_GL=0 RADIANCE_XE_I4I8_DPAS=0) && OUT=kld-nodpas
[ "${1:-}" = map ] && SV=(RADIANCE_XE_GEMV_GL=0 RADIANCE_XE_I4I8_DPAS_MAP=1) && OUT=kld-map
[ "${1:-}" = mapraw ] && SV=(RADIANCE_XE_GEMV_GL=0 RADIANCE_XE_I4I8_DPAS_MAP=1) && OUT=kld-mapraw
# run 167 (owner note 168 item 1): the sparse pass's attention on the matrix units (attn_qsa_fmha, opt-in by
# RADIANCE_ATTN_QSA_FMHA=1), replayed against the exact per-row path
[ "${1:-}" = fmha ] && SV=(RADIANCE_XE_GEMV_GL=0 RADIANCE_ATTN_QSA_FMHA=1) && OUT=kld-fmha
# `tskip7` / `tskip8` (run 160, owner note 162 item 1): the tail skip at rank 7 / 8 (RADIANCE_XE_TAIL_SKIP), scored
# against the exact path (format C is the default in both passes since note 160)
[ "${1:-}" = tskip7 ] && SV=(RADIANCE_XE_GEMV_GL=0 RADIANCE_XE_TAIL_SKIP=7) && OUT=kld-tskip7
[ "${1:-}" = tskip8 ] && SV=(RADIANCE_XE_GEMV_GL=0 RADIANCE_XE_TAIL_SKIP=8) && OUT=kld-tskip8
# `hci82`: stage 2 (RADIANCE_XE_HC_I8=2: N in int8 too, the down product by DP4A, to 64 rows), in 64-token chunks
[ "${1:-}" = hci82 ] && SV=(RADIANCE_XE_GEMV_GL=0 RADIANCE_XE_HC_V2=8 RADIANCE_XE_HC_I8=2) && OUT=kld-hci82
[ "${1:-}" = lean ] && SV=(RADIANCE_XE_GEMV_GL=0 RADIANCE_XE_ROUTE_LEAN=${LEAN:-2,0.005}) && OUT=kld-lean
[ "${1:-}" = fmhad ] && SV=(RADIANCE_XE_GEMV_GL=0 RADIANCE_ATTN_QSA_FMHA=1) && OUT=kld-fmhad
# `mbd` (run 268, owner lever #3): the protected experts' bf16 GEMM on the matrix units (RADIANCE_XE_MOE_BF16_DPAS=1,
# opt-in; xe_moe_bf16_dpas.cpp), scored against the WMMA form's exact path (the flag unset in the record pass)
[ "${1:-}" = mbd ] && SV=(RADIANCE_XE_GEMV_GL=0 RADIANCE_XE_MOE_BF16_DPAS=1) && OUT=kld-mbd
# `gsf16` (run 294, owner note 235): the GDN recurrent state cache in fp16 (RADIANCE_GDN_STATE_F16=1: storage fp16,
# arithmetic fp32, both the chunk scan and the decode update), scored against today's fp32 state (the record pass)
[ "${1:-}" = gsf16 ] && SV=(RADIANCE_XE_GEMV_GL=0 RADIANCE_GDN_STATE_F16=1) && OUT=kld-gsf16
[ "${1:-}" = lean64 ] && SV=(RADIANCE_XE_GEMV_GL=0 RADIANCE_XE_ROUTE_LEAN=2,0.005,64) && OUT=kld-lean64
# lean64b / lean64c: the margin 0.002 with the two lowest picks, and 0.005 with the lowest one (a variable does not
# cross m3-step.sh's ssh, so each setting is its own mode)
[ "${1:-}" = lean64b ] && SV=(RADIANCE_XE_GEMV_GL=0 RADIANCE_XE_ROUTE_LEAN=2,0.002,64) && OUT=kld-lean64b
[ "${1:-}" = lean64c ] && SV=(RADIANCE_XE_GEMV_GL=0 RADIANCE_XE_ROUTE_LEAN=1,0.005,64) && OUT=kld-lean64c
# `cold3` (run 176, owner note 178 item 3): the coldest-quarter mixed-bit container -- the record pass as above on the
# served container, the score pass on /d/flashnext-cold3.rad (3-bit levels for each layer's 128 least-routed experts)
SARGS=$ARGS
[ "${1:-}" = cold3 ] && SV=(RADIANCE_XE_GEMV_GL=0) && OUT=kld-cold3 && SARGS=${ARGS/qwen3.8-next-flash-fp8-iq4r-moe.rad/flashnext-cold3.rad}
echo "=== score, ${SV[*]} ($(date -u +%H:%M:%S)) model: $(echo $SARGS | grep -o -- '--model [^ ]*')"
env "${ENVV[@]}" "${SV[@]}" timeout 1200 /opt/ri/bin/radiance $SARGS --radiance-home $H \
    --kld-ref /out/kref --kld-out /out/$OUT.json > /out/kld-score.log 2>&1
echo "score rc=$?"; tail -12 /out/kld-score.log
