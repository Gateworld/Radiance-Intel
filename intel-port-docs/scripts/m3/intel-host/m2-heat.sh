#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 2, stage 2 step 3 / stage 1 step 5: the heat-engine replay).
# Inside ri-m2rt on both cards, started by D/m2-step.sh with STEP_ENGINE=1: ONE model load (CFG as in
# m2-tps.sh), then for MINUTES: the /out/card/NN-*.path + .body requests in rotation, back to back (greedy,
# <= 256 tokens each), /stats every 30 s to /out/stats-N.txt, one line a request to /out/requests.txt
# ("<epoch> <name> <wall s> <http status>"). At the end: /stats to /out/stats-final.txt and a summary from
# it (dispatches, promotions, demotions, readmits, misses, passes played/refused, link MB).
#   m2-heat.sh CFG [MINUTES=20]
H=/opt/ri/radiance_home; P=18421
CFG=${1:-nomtp}; MIN=${2:-20}
CG=/sys/fs/cgroup/cpu.stat
usage() { local k v; while read -r k v; do [ "$k" = usage_usec ] && { echo "$v"; return; }; done < $CG; echo 0; }
get() {
  exec 4<>/dev/tcp/127.0.0.1/$P || return
  printf 'GET %s HTTP/1.1\r\nHost: l\r\nConnection: close\r\n\r\n' "$1" >&4
  local line; while IFS= read -r line <&4; do [ "$line" = $'\r' ] && break; done
  cat <&4; exec 4<&-
}
post_raw() {   # post_raw PATH BODYFILE OUT -> prints the status line
  local body; body=$(cat "$2")
  exec 5<>/dev/tcp/127.0.0.1/$P || return 1
  printf 'POST %s HTTP/1.1\r\nHost: l\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s' \
    "$1" "${#body}" "$body" >&5
  cat <&5 > "$3"; exec 5<&-
  head -1 "$3" | tr -d '\r'
}
echo "cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
ARGS=$(grep -v '^#' /out/flashnext-card1.args | tr '\n' ' ')
ENVV=()
# run 283 (owner note 228 item 2): a "-gs2" suffix on any config = the GDN decode update's VS 2 split (+ L1, exact;
# RADIANCE_XE_GDN_SPLIT=2), the rest of the name parsed as before -- the rotation pair plain vs -gs2 on one image
GS2=0; case $CFG in *-gs2) GS2=1; CFG=${CFG%-gs2} ;; esac
# run 285 (group A of owner note 232's audit): a "-gvm2" suffix = the fused router launch with the shared expert's gate_up
# in it (RADIANCE_XE_GEMV_MULTI2=1, exact, opt-in), the rest of the name parsed as before
GVM2=0; case $CFG in *-gvm2) GVM2=1; CFG=${CFG%-gvm2} ;; esac
case $CFG in
  mtp) ;;
  nomtp) ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
  plain) ENVV=(RADIANCE_L0_TAPE=0) ;;
  plainnomtp) ENVV=(RADIANCE_L0_TAPE=0); ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0} ;;
  # milestone 3, run 37: MTP on with a 4 GiB headroom (H6 at 2 GiB: the recordings' VRAM floor fired at ~340 MiB
  # free on one user's rotation, so new pass shapes ran live: median 26.2 against H5's 37.5 before the floor)
  mtp4g) ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  # run 38: the same with the kept-tape bound at 8 a rank (RADIANCE_TAPE_MAX, 6ebd5d8; default kTapes = 24): H7 showed
  # more headroom only delays the floor, since the tapes grow into whatever is free
  mtp4gt8) ENVV=(RADIANCE_TAPE_MAX=8); ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  # run 40: MTP on with a 6 GiB headroom and no bound (C10: the bound churns; C11: at 6 GiB the floor stayed silent
  # through 6 users, 73.8 tok/s aggregate) -- does the rotation's set of shapes fit without the floor?
  mtp6g) ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 6144} ;;
  # run 41: the service setting with the composed-stretch cache bounded at 96 lists a tape (default 512; a 1,036-segment
  # pass is 65 fixed blocks plus the partials the mover's moving cuts make): the pool lever -- each list carries its own
  # command buffers and scratch, so the bound is the tape's VRAM
  mtp4gs96) ENVV=(RADIANCE_L0_TAPE_STRETCHES=96); ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  # run 42: the same rotation with no tapes at all (every pass live) at the service headroom: whether the VRAM still
  # climbs to the ceiling without recordings (H7 / H9: +5.3 GB a card in 2 minutes, then the floor)
  plain4g) ENVV=(RADIANCE_L0_TAPE=0); ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  # milestone 3 stage C (owner note 47, run 50): the service setting with the heat engine's minimum residency age
  # (RADIANCE_PLACE_MIN_AGE, steps; 0 = production's policy): a unit demoted on one step's noise is not fetched
  # straight back, a unit just promoted is not demoted on the next step. 16 steps is about one half-life of the
  # heat (decay 0.95) and ~0.9 s of MTP steps.
  mtp4gage16) ENVV=(RADIANCE_PLACE_MIN_AGE=16); ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  mtp4gage64) ENVV=(RADIANCE_PLACE_MIN_AGE=64); ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  # owner note 47 item 2 (run 52): cooperative verify (upstream d912c69, picked as 8b9a8bb) on the rotation; the control
  # is mtp4g on the same image
  mtp4gcoupled) ENVV=(RADIANCE_MTP_COUPLED=1); ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096}
                ARGS=${ARGS/--kernels libxe:libref/--kernels libxe:libref:spechub} ;;
  # milestone 3 run 60 (owner notes 49 and 50 item 2): the service setting with the matrix-unit expert form on
  # (RADIANCE_XE_MOE_MX=1, the default since note 49) -- the control -- and the heat engine's decay 0.98 / 0.99
  # (RADIANCE_PLACE_DECAY; shipped 0.95, half-life ~14 steps) and swap budget 32 / 16 a class a step
  # (RADIANCE_PLACE_MOVES; shipped 64) on the 5-minute rotation: misses a step, readmits, median. The two knobs
  # need the build after f49f641 (engine_bringup.cpp reads them); on earlier images they are ignored.
  mtp4gmx)    ENVV=(RADIANCE_XE_MOE_MX=1); ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  mtp4gd98)   ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_PLACE_DECAY=0.98); ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  mtp4gd99)   ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_PLACE_DECAY=0.99); ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  mtp4gm32)   ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_PLACE_MOVES=32); ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  mtp4gm16)   ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_PLACE_MOVES=16); ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  # run 63 (owner note 52 item 1): the tape pool under the VRAM floor -- mtp4gmx on an image with the new rule
  # (a tape played in the last RADIANCE_TAPE_KEEP_PASSES passes is never dropped) and mtp4gkeep0, the same image
  # with the earlier rule (any least recently used tape dropped per tape-less pass), the control
  mtp4gkeep0) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_TAPE_KEEP_PASSES=0); ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  # run 69 (owner note 56): the hc mixing matrices expanded to FP16 at load, against mtp4gmx on the same image
  mtp4ghcf16) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_XE_HC_F16=1); ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  # run 73 (owner note 58's fallback): M38 showed shedding frees no VRAM (the card stays under the floor and every
  # pass goes live), so: no shedding, and a 2 GiB recordings floor so the cards stop recording well clear of the
  # ceiling where m3 fault 14's append failed
  mtp4gfloor2g) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048)
                ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  # run 93 (owner note 91 item 2): the service rotation with the two-card fix (4fb114b)
  # the full candidate service setting: the fix, NEO's 64 KB key, the headroom 3072 (owner notes 65 item 4 / 68 / 90-91)
  mtp3gfloor2gall) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64)
                ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 3072} ;;
  mtp4gfloor2gfix) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1)
                ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  # run 97 (owner note 95 item 3): the budget reads the real card (RADIANCE_VRAM_CAP_FROM_FREE), the checkpoint store
  # pageable (d7b4da2), headroom 2560 sized from the measured table to leave ~0.5 GiB free at the rotation's steady
  # state, the floor at 256; the fix, the 64 KB key, traced
  capfree2560) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_CKPT_PAGEABLE=1
                RADIANCE_VRAM_CAP_FROM_FREE=1)
                ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560} ;;
  # the same with the host pool sized to what it holds (9 GiB of 16: H68's host tier held 8.82 GiB) -- rank 1's host
  # pool is an import on card1, and card1's import binds stall when its VRAM is fuller (H70)
  capfree2560hp9) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_CKPT_PAGEABLE=1
                RADIANCE_VRAM_CAP_FROM_FREE=1)
                ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 9216} ;;
  # the same with the pinned checkpoint store (no RADIANCE_CKPT_PAGEABLE): with the host pool at 9 GiB card1 imports
  # ~7 GiB less, so the store may bind again (H72)
  capfree2560hp9pin) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1)
                ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 9216} ;;
  # the same with a pinned store of 2 slots (~225 MiB): a smaller import than the 1.54 GiB that H72's could not bind
  capfree2560hp9ck2) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1)
                ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 9216}
                ARGS="$ARGS --checkpoint-slots 2" ;;
  # note 95 item 3's candidate: H73's setting with the host pool at 8 GiB (four pieces; M79 / M81 stalled at a fifth
  # with MTP off, M82 starts)
  capfree2560hp8ck2) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1)
                ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
                ARGS="$ARGS --checkpoint-slots 2" ;;
  # run 99 (owner note 103): the candidate rotation with the early segment release (e1), and its lazy tape_end (e1lazy)
  capfree2560hp8ck2e1|capfree2560hp8ck2e1lazy) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256
                RADIANCE_L0_TAPE_EARLY=1 RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1
                DisableIndirectAccess=1 OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1)
                [ "$CFG" = capfree2560hp8ck2e1lazy ] && ENVV+=(RADIANCE_L0_TAPE_EARLY_LAZY=1)
                ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 2560}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
                ARGS="$ARGS --checkpoint-slots 2" ;;
  # run 105 (owner note 123): the candidate rotation (e1) with M136's opt-ins (gnt0, hc level 8, the router pair) and the
  # m3 defaults (the fail-fast exit, host-first; the LL all-reduce is the image's default from :127), the GPU headroom
  # stepped down: bestr2560 (the earlier), bestr2048, bestr1536, bestr1280; plain `bestr` = 1280, the m3 headroom since
  # run 108 (owner note 123: the lowest step with >= ~0.5 GiB free at its worst, H82)
  # run 112 (owner note 135 item 2a): bestrdefer = bestr with RADIANCE_TAPE_DEFER_AFTER_PREFILL=1
  # run 112 (owner note 139 stage 0): bestrxg = bestr with the expert guess tap (RADIANCE_XP_GUESS, recorded passes off)
  # run 112 (owner note 140): calib = bestr, MTP off, with the Gram tap into /out/calib (500K tokens a half); the
  # corpus bodies (/out/card/NNNNN-<family>.body) are posted once each, the loop ending when the tap has written
  # run 112 (owner note 139 stage 1): bestrxp = bestr with the expert prefetch (RADIANCE_XE_EXPERT_PREFETCH=1)
  # run 114 (owner note 141 item 1): bestrgl = bestr with the multi-row trunk GEMV a group a lane (RADIANCE_XE_GEMV_GL=1)
  # run 115 (owner note 142 item 2): bestrhrg = bestr with hc_read level 8's columns and gains in one launch
  # run 117 (owner note 147): bestrfc = bestr with format C, today's w4nl experts against int8 activations
  # run 126 (owner note 159): bestrgu = bestr with the glue (RADIANCE_L0_GLUE=1: no host wait between draft passes)
  # run 160 (owner note 162 item 1): bestrts7 / bestrts8 = bestr with the tail skip at rank 7 / 8 (RADIANCE_XE_TAIL_SKIP)
  # run 161 (owner note 163 item 2): bestrar5 = bestr with the all-reduce's 16-byte LL form (RADIANCE_XE_AR_PROTO=5)
  # run 161 (owner note 164): bestrad2 = bestr at draft depth 2 (--num-speculative-tokens 2; the served depth is 3)
  # run 162 (owner note 165 item 1): bestrdg1 / bestrdg2 / bestrdg4 / bestrdg8 = bestr with the draft gate at that margin
  # run 162: bestrnp = bestr with no step issued ahead (RADIANCE_STEP_AHEAD=0); bestrnpdg2 / bestrnpdg4 = the same with the gate
  # run 174: bestrx = bestr as the service runs today (the trunk GEMV's permute form, libavx's n-gram gather) with the three
  # opt-in switches the owner is deciding (the attention op, the bandwidth all-reduce, the event records as barriers)
  bestr|bestrx|bestrxga|bestrxga2|bestrxw2d4|bestrxhcp|bestrxhcp2|bestrxdp2|bestrxhcab|bestrxhctx|bestrxad4dp2|bestrxad3dp2|bestrxxs|bestrxpin|bestrxprof|bestrxhcnr|bestrxhcgu|bestrxgud|bestrxgfw|bestrxgfw2|bestrxgrf3|bestrxgrf3w|bestrxgrf3w16|bestrxgrf3w16dv|bestrxgrf3w16dv64|bestrxgrf3w16p3|bestrxgrf3w16p4|bestrxpre|bestrx1024|bestrx768|bestrxdh8|bestrdefer|bestrxg|bestrxp|bestrgl|bestrhrg|bestrfc|bestrsk|bestrng|bestrlr|bestrlw|bestrgu|bestrts7|bestrts8|bestrar5|bestrad2|bestrdgs|bestrdg1|bestrdg2|bestrdg4|bestrdg8|bestrnp|bestrnpdg2|bestrnpdg4|calib|bestr2560|bestr2048|bestr1536|bestr1280) DEFER=0; XG=0; CAL=0; XP=0; GL=0; HRG=0; FC=0; SK=0; NG=0; LR=0; GU=0; TS=0; AR5=0; AD2=0; DG=0; NP=0; BX=0
                [ "$CFG" = bestrx ] && BX=1 && CFG=bestr
                # run 175 (owner note 176 item 3): bestrxpre = bestrx with the expert codes decoded once a call into
                # the op's scratch (RADIANCE_XE_I4I8_DPAS_PRE=1): the slab ~1.8 GiB a card smaller, so the misses a step
                [ "$CFG" = bestrxpre ] && BX=2 && CFG=bestr
                # run 176 (owner note 179 item 2): bestrx at a smaller headroom (the recorded-pass pool is ~0.24 GiB now)
                [ "$CFG" = bestrx1024 ] && BX=1 && CFG=bestr1024
                [ "$CFG" = bestrx768 ] && BX=1 && CFG=bestr768
                # run 176 (owner note 180 A): bestrxdh8 = bestrx with the draft rounds on the trunk's int8 lm_head
                [ "$CFG" = bestrxdh8 ] && BX=3 && CFG=bestr
                # run 189: bestrx with the leak fix (RADIANCE_L0_GLUE_ALL=1, notes 182 / 184), the draft head's GEMV by
                # DP4A (RADIANCE_XE_W2A8_DP4=1, note 183 item 1), hc_read's down half persistent (RADIANCE_XE_HC_PERSIST=1)
                [ "$CFG" = bestrxga ] && BX=4 && CFG=bestr
                [ "$CFG" = bestrxga2 ] && BX=7 && CFG=bestr   # run 189: the two-list form
                [ "$CFG" = bestrxhcp2 ] && BX=8 && CFG=bestr  # run 190: the megakernel's stage 1b (note 187)
                [ "$CFG" = bestrxdp2 ] && BX=9 && CFG=bestr   # run 193: the draft depth by content (note 191.1)
                [ "$CFG" = bestrxhcab ] && BX=10 && CFG=bestr # run 193: upstream 1.14's hc_read merge (note 190)
                [ "$CFG" = bestrxhctx ] && BX=11 && CFG=bestr # run 193: upstream 64c652f's first-draft context
                [ "$CFG" = bestrxad4dp2 ] && BX=12 && CFG=bestr # run 202: note 193, adaptive to 4 from depth by content
                [ "$CFG" = bestrxad3dp2 ] && BX=13 && CFG=bestr # run 202: note 193, adaptive to 3 from depth by content
                # run 203 (owner note 191 item 3): xs = the decode routing tally written to /out/xstats.txt (nothing else
                # changes: the control); pin = /out/xprofile.txt ranks the experts and the top half is pinned resident;
                # prof = the profile alone
                [ "$CFG" = bestrxxs ] && BX=14 && CFG=bestr
                [ "$CFG" = bestrxpin ] && BX=15 && CFG=bestr
                [ "$CFG" = bestrxprof ] && BX=16 && CFG=bestr
                [ "$CFG" = bestrxhcnr ] && BX=17 && CFG=bestr   # run 205 (owner note 200 item 1a): RADIANCE_XE_HC_NR=1
                [ "$CFG" = bestrxhcgu ] && BX=18 && CFG=bestr   # run 205 (owner note 200 item 1b): RADIANCE_XE_HC_GU=1
                [ "$CFG" = bestrxgud ] && BX=19 && CFG=bestr    # run 205: GU + D16 (owner notes 200 / 201 item 1)
                [ "$CFG" = bestrxgfw ] && BX=20 && CFG=bestr    # run 206: GU + D16 + the write-norm fold (the fusion group)
                [ "$CFG" = bestrxgfw2 ] && BX=21 && CFG=bestr   # run 207: the group + pairs 4 and 2
                [ "$CFG" = bestrxgrf3 ] && BX=22 && CFG=bestr   # run 211: top-k + qsa_score + router at 256 GRF (note 204 / 203 item 2)
                [ "$CFG" = bestrxgrf3w ] && BX=23 && CFG=bestr  # run 211: the GRF set + the GEMV's weights at LSC hint 5 (note 206 item 2)
                [ "$CFG" = bestrxgrf3w16 ] && BX=24 && CFG=bestr  # run 213: BX 23 + the draft head's wide w2a8 form (RADIANCE_XE_W2A8_R=16)
                [ "$CFG" = bestrxgrf3w16dv ] && BX=25 && CFG=bestr  # run 219: BX 24 + the 32K draft vocabulary (note 207 A3a)
                [ "$CFG" = bestrxgrf3w16dv64 ] && BX=26 && CFG=bestr  # run 219: BX 24 + the 64K teaching-only draft vocabulary
                [ "$CFG" = bestrxgrf3w16p3 ] && BX=27 && CFG=bestr  # run 273 (coordinator-reviewed): BX 24 + the three exact pairs (gqp / qkp / rqp)
                [ "$CFG" = bestrxgrf3w16p4 ] && BX=28 && CFG=bestr  # run 274 (coordinator-reviewed): BX 27 + the paired GEMV launches (gvm)
                [ "$CFG" = bestrxw2d4 ] && BX=5 && CFG=bestr
                [ "$CFG" = bestrxhcp ] && BX=6 && CFG=bestr
                [ "$CFG" = bestrnp ] && NP=1 && CFG=bestr
                [ "$CFG" = bestrnpdg2 ] && NP=1 && DG=2 && CFG=bestr
                [ "$CFG" = bestrnpdg4 ] && NP=1 && DG=4 && CFG=bestr
                [ "$CFG" = bestrdgs ] && DG=stats && CFG=bestr
                [ "$CFG" = bestrdg1 ] && DG=1 && CFG=bestr
                [ "$CFG" = bestrdg2 ] && DG=2 && CFG=bestr
                [ "$CFG" = bestrdg4 ] && DG=4 && CFG=bestr
                [ "$CFG" = bestrdg8 ] && DG=8 && CFG=bestr
                [ "$CFG" = bestrad2 ] && AD2=1 && CFG=bestr
                [ "$CFG" = bestrar5 ] && AR5=1 && CFG=bestr
                [ "$CFG" = bestrgu ] && GU=1 && CFG=bestr
                [ "$CFG" = bestrts7 ] && TS=7 && CFG=bestr
                [ "$CFG" = bestrts8 ] && TS=8 && CFG=bestr
                [ "$CFG" = bestrxp ] && XP=1 && CFG=bestr
                [ "$CFG" = bestrgl ] && GL=1 && CFG=bestr
                [ "$CFG" = bestrhrg ] && HRG=1 && CFG=bestr
                [ "$CFG" = bestrfc ] && FC=1 && CFG=bestr
                [ "$CFG" = bestrsk ] && SK=1 && CFG=bestr   # run 122 (owner note 151 A): the all-reduce wait record
                [ "$CFG" = bestrng ] && NG=1 && CFG=bestr   # run 123 (owner note 151 B): prompt lookup measured
                [ "$CFG" = bestrlr ] && LR=1 && CFG=bestr   # run 123 (owner note 151 C): the lm_head's rows form
                [ "$CFG" = bestrlw ] && LR=2 && CFG=bestr   # run 123 (owner note 151 C): lm_head + draft head rows forms
                [ "$CFG" = bestrdefer ] && DEFER=1 && CFG=bestr
                [ "$CFG" = bestrxg ] && XG=1 && CFG=bestr
                [ "$CFG" = calib ] && CAL=1 && CFG=bestr
                [ "$CFG" = bestr ] && CFG=bestr1280
                ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=256
                RADIANCE_L0_TAPE_EARLY=1 RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1
                DisableIndirectAccess=1 OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_VRAM_CAP_FROM_FREE=1
                RADIANCE_XE_GDN_NT=0 RADIANCE_XE_HC_V2=8 RADIANCE_XE_ROUTER_PAIR=1 RADIANCE_STALL_EXIT_S=60 RADIANCE_HOST_FIRST=1
                RADIANCE_XE_MOE_MX_STAGE=1)   # run 109 (owner note 126 item 5): the staging is an m3 default
                [ "$DEFER" = 1 ] && ENVV+=(RADIANCE_TAPE_DEFER_AFTER_PREFILL=1)
                [ "$XG" = 1 ] && ENVV+=(RADIANCE_XP_GUESS=300 RADIANCE_L0_TAPE=0)   # H87 heat-stopped at 2048
                [ "$XP" = 1 ] && ENVV+=(RADIANCE_XE_EXPERT_PREFETCH=1)
                [ "$GL" = 1 ] && ENVV+=(RADIANCE_XE_GEMV_GL=1)
                [ "$HRG" = 1 ] && ENVV+=(RADIANCE_XE_HC_V2_RG=1)
                [ "$FC" = 1 ] && ENVV+=(RADIANCE_MOE_FORMAT_C=1)
                [ "$SK" = 1 ] && mkdir -p /out/arskew && ENVV+=(RADIANCE_XE_AR_SKEW=/out/arskew)
                [ "$NG" = 1 ] && ENVV+=(RADIANCE_NGRAM_STATS=1)
                [ "$LR" = 1 ] && ENVV+=(RADIANCE_XE_LOGITS_R=2)
                [ "$LR" = 2 ] && ENVV+=(RADIANCE_XE_LOGITS_R=2 RADIANCE_XE_W2A8_R=2)
                [ "$GU" = 1 ] && ENVV+=(RADIANCE_L0_GLUE=1)
                [ "$TS" != 0 ] && ENVV+=(RADIANCE_XE_TAIL_SKIP=$TS)
                [ "$AR5" = 1 ] && ENVV+=(RADIANCE_XE_AR_PROTO=5)
                [ "$AD2" = 1 ] && ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 2}
                [ "$DG" != 0 ] && ENVV+=(RADIANCE_DRAFT_GATE=$DG)
                [ "$NP" = 1 ] && ENVV+=(RADIANCE_STEP_AHEAD=0)
                if [ "$BX" = 1 ] || [ "$BX" = 2 ] || [ "$BX" = 3 ] || [ "$BX" = 4 ] || [ "$BX" = 5 ] || [ "$BX" = 6 ] || [ "$BX" = 7 ] || [ "$BX" = 8 ] || [ "$BX" = 9 ] || [ "$BX" = 10 ] || [ "$BX" = 11 ] || [ "$BX" = 12 ] || [ "$BX" = 13 ] || [ "$BX" = 14 ] || [ "$BX" = 15 ] || [ "$BX" = 16 ] || [ "$BX" = 17 ] || [ "$BX" = 18 ] || [ "$BX" = 19 ] || [ "$BX" = 20 ] || [ "$BX" = 21 ] || [ "$BX" = 22 ] || [ "$BX" = 23 ] || [ "$BX" = 24 ] || [ "$BX" = 25 ] || [ "$BX" = 26 ] || [ "$BX" = 27 ] || [ "$BX" = 28 ]; then
                  ENVV+=(RADIANCE_XE_GEMV_NX=1 RADIANCE_ATTN_QSA_FMHA=1 RADIANCE_XE_AR_PROTO=6 RADIANCE_L0_RECORD_BARRIER=1)
                  [ "$BX" = 14 ] && ENVV+=(RADIANCE_EXPERT_STATS=/out/xstats.txt)
                  [ "$BX" = 15 ] && ENVV+=(RADIANCE_EXPERT_PROFILE=/out/xprofile.txt RADIANCE_EXPERT_PIN_TOP=0.5)
                  [ "$BX" = 16 ] && ENVV+=(RADIANCE_EXPERT_PROFILE=/out/xprofile.txt)
                  [ "$BX" = 17 ] && ENVV+=(RADIANCE_XE_HC_NR=1)
                  [ "$BX" = 18 ] && ENVV+=(RADIANCE_XE_HC_GU=1)
                  [ "$BX" = 19 ] && ENVV+=(RADIANCE_XE_HC_GU=1 RADIANCE_XE_HC_D16=1)
                  [ "$BX" = 20 ] && ENVV+=(RADIANCE_XE_HC_GU=1 RADIANCE_XE_HC_D16=1 RADIANCE_HC_WRITE_FUSE=1)
                  [ "$BX" = 21 ] && ENVV+=(RADIANCE_XE_HC_GU=1 RADIANCE_XE_HC_D16=1 RADIANCE_HC_WRITE_FUSE=1 RADIANCE_XE_GATE_QUANT=1 RADIANCE_XE_GATED_HAD=1)
                  [ "$BX" = 22 ] && ENVV+=(RADIANCE_XE_TOPK_GRF=1 RADIANCE_XE_QSA_GRF=1 RADIANCE_XE_ROUTER_GRF=1)
                  [ "$BX" = 23 ] && ENVV+=(RADIANCE_XE_TOPK_GRF=1 RADIANCE_XE_QSA_GRF=1 RADIANCE_XE_ROUTER_GRF=1 RADIANCE_XE_GEMV_WCC=5)
                  [ "$BX" = 24 ] && ENVV+=(RADIANCE_XE_TOPK_GRF=1 RADIANCE_XE_QSA_GRF=1 RADIANCE_XE_ROUTER_GRF=1 RADIANCE_XE_GEMV_WCC=5 RADIANCE_XE_W2A8_R=16)
                  [ "$BX" = 25 ] && ENVV+=(RADIANCE_XE_TOPK_GRF=1 RADIANCE_XE_QSA_GRF=1 RADIANCE_XE_ROUTER_GRF=1 RADIANCE_XE_GEMV_WCC=5 RADIANCE_XE_W2A8_R=16 RADIANCE_XE_W2A8_VOCAB=/opt/ri/radiance_home/draft-vocab-32k.ids)
                  [ "$BX" = 26 ] && ENVV+=(RADIANCE_XE_TOPK_GRF=1 RADIANCE_XE_QSA_GRF=1 RADIANCE_XE_ROUTER_GRF=1 RADIANCE_XE_GEMV_WCC=5 RADIANCE_XE_W2A8_R=16 RADIANCE_XE_W2A8_VOCAB=/opt/ri/radiance_home/draft-vocab-64k.ids)
                  [ "$BX" = 27 ] && ENVV+=(RADIANCE_XE_TOPK_GRF=1 RADIANCE_XE_QSA_GRF=1 RADIANCE_XE_ROUTER_GRF=1 RADIANCE_XE_GEMV_WCC=5 RADIANCE_XE_W2A8_R=16 RADIANCE_XE_GATED_QUANT_PAIR=1 RADIANCE_XE_QK_PROLOGUE=1 RADIANCE_XE_ROUTER_QUANT_PAIR=1)
                  [ "$BX" = 28 ] && ENVV+=(RADIANCE_XE_TOPK_GRF=1 RADIANCE_XE_QSA_GRF=1 RADIANCE_XE_ROUTER_GRF=1 RADIANCE_XE_GEMV_WCC=5 RADIANCE_XE_W2A8_R=16 RADIANCE_XE_GATED_QUANT_PAIR=1 RADIANCE_XE_QK_PROLOGUE=1 RADIANCE_XE_ROUTER_QUANT_PAIR=1 RADIANCE_XE_GEMV_MULTI=1)
                  [ "$BX" = 2 ] && ENVV+=(RADIANCE_XE_I4I8_DPAS_PRE=1)
                  [ "$BX" = 3 ] && ENVV+=(RADIANCE_MTP_DRAFT_HEAD_Q2=0)
                  [ "$BX" = 4 ] && ENVV+=(RADIANCE_L0_GLUE_ALL=1)
                  [ "$BX" = 5 ] && ENVV+=(RADIANCE_XE_W2A8_DP4=1)
                  [ "$BX" = 6 ] && ENVV+=(RADIANCE_XE_HC_PERSIST=1)
                  [ "$BX" = 7 ] && ENVV+=(RADIANCE_L0_GLUE_ALL=2)
                  [ "$BX" = 8 ] && ENVV+=(RADIANCE_XE_HC_PERSIST=2)
                  [ "$BX" = 9 ] && ENVV+=(RADIANCE_DRAFT_DEPTH_PROSE=2)
                  [ "$BX" = 10 ] && ENVV+=(RADIANCE_XE_HC_AB=1)
                  [ "$BX" = 11 ] && ENVV+=(RADIANCE_MTP_HIST_CTX=1)
                  if [ "$BX" = 12 ] || [ "$BX" = 13 ]; then ENVV+=(RADIANCE_DRAFT_DEPTH_PROSE=2); ARGS="$ARGS --draft-window adaptive"; fi
                  [ "$BX" = 12 ] && ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 4}
                  ARGS=${ARGS/--kernels libxe:libref/--kernels libxe:libavx:libref}   # the step runs under the io_uring profile
                fi
                if [ "$CAL" = 1 ]; then
                  mkdir -p /out/calib
                  ENVV+=(RADIANCE_CALIB_DIR=/out/calib RADIANCE_CALIB_TOKENS=500000)
                  ARGS=${ARGS/--num-speculative-tokens 3/--num-speculative-tokens 0}
                  ARGS=${ARGS/--max-num-seqs 2/--max-num-seqs 4}; ARGS=${ARGS/--max-model-len 65536/--max-model-len 32768}
                fi
                ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib ${CFG#bestr}}; ARGS=${ARGS/--host-pool-mib 16384/--host-pool-mib 8192}
                ARGS="$ARGS --checkpoint-slots 2" ;;
  # run 96 (owner note 95 item 1): H61's full candidate setting (floor 2048, headroom 3072) with the VRAM trace and
  # its ledger, and NEO's allocation log -- the steady state split by owner
  mtp3gallvt) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1
                LogAllocationType=1 LogAllocationStdout=1 LogAllocationMemoryPool=1)
                ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 3072} ;;
  # run 95 (owner notes 92-94): H61's full candidate setting with the VRAM trace, the floor and headroom stepped down
  # (step B 1024 / 2048, its fallback A 1024 / 2560, C 512 / 1536, D 256 / 1280; MiB)
  allfl1024hr2048|allfl1024hr2560|allfl512hr1536|allfl256hr1280)
                FL=${CFG#allfl}; FL=${FL%%hr*}; HR=${CFG##*hr}
                ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=$FL RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                OverrideCmdListCmdBufferSizeInKb=64 RADIANCE_L0_VRAM_TRACE=1)
                ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib $HR} ;;
  # run 94 (owner note 66): H60's rotation with hc_read V2 (6aa039d)
  mtp4gfloor2gv2) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                RADIANCE_HOST_POOL_PIECE_MIB=2048 RADIANCE_L0_EXPLICIT_RESIDENCY=1 NEOReadDebugKeys=1 DisableIndirectAccess=1
                RADIANCE_XE_HC_V2=1)
                ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  # run 76 (owner note 61 item 0): the service rotation with the recordings' VRAM traced against lists alive (image :72)
  mtp4gfloor2gvtrace) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_VRAM_TRACE=1)
                ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  # run 80 (owner note 65): H26's traced service rotation with the early segment release (against H26 / H25)
  mtp4gfloor2gearly) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_L0_TAPE_EARLY=1)
                ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  # run 80 (owner note 65 item 5, no build): H27b with the driver's command buffer per list at 64 KB (X9: 0.25 MiB a list)
  # run 90 (owner note 68): NEO's 64 KB command buffer alone, the early release explicitly off (it is the tree's
  # default from image :77 on), traced -- the recordings' VRAM against H26 (neither) and H28 (both)
  mtp4gfloor2gcb64) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_L0_TAPE_EARLY=0
                NEOReadDebugKeys=1 OverrideCmdListCmdBufferSizeInKb=64)
                ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  # run 90 (owner note 65 item 4): the service rotation with the key and 1 GiB of the headroom given to the experts
  # (4096 -> 3072 MiB; H56 ended with ~3.5 GiB free against the 2 GiB floor), untraced, for the median and misses
  mtp3gfloor2gcb64) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_TAPE_EARLY=0
                NEOReadDebugKeys=1 OverrideCmdListCmdBufferSizeInKb=64)
                ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 3072} ;;
  mtp4gfloor2gearlycb64) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048 RADIANCE_L0_VRAM_TRACE=1 RADIANCE_L0_TAPE_EARLY=1
                NEOReadDebugKeys=1 OverrideCmdListCmdBufferSizeInKb=64)
                ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  # run 74: owner note 56's FP16 hc form at the service setting (the 2 GiB floor), against mtp4gfloor2g on one image
  mtp4ghcf16floor2g) ENVV=(RADIANCE_XE_MOE_MX=1 RADIANCE_XE_HC_F16=1 RADIANCE_TAPE_SHED=0 RADIANCE_TAPE_VRAM_FLOOR_MIB=2048)
                ARGS=${ARGS/--gpu-headroom-mib 2048/--gpu-headroom-mib 4096} ;;
  *) echo "unknown config $CFG"; exit 2 ;;
esac
t0=$(date +%s)
# calib: the tap's host add makes a 4K step ~29 s (CL2), so a 500K-token half needs ~60 min
SRV_TO=3000; [ "${CAL:-0}" = 1 ] && SRV_TO=5100
[ "$GS2" = 1 ] && ENVV+=(RADIANCE_XE_GDN_SPLIT=2)   # run 283: the -gs2 suffix (above)
[ "$GVM2" = 1 ] && ENVV+=(RADIANCE_XE_GEMV_MULTI2=1)   # run 285: the -gvm2 suffix (above)
env "${ENVV[@]}" timeout $SRV_TO /opt/ri/bin/radiance $ARGS --radiance-home $H -v > /out/serve.log 2>&1 &
pid=$!; up=0
for i in $(seq 2400); do
  kill -0 $pid 2>/dev/null || break
  if exec 3<>/dev/tcp/127.0.0.1/$P 2>/dev/null; then
    printf 'GET /health HTTP/1.1\r\nHost: l\r\nConnection: close\r\n\r\n' >&3
    head -1 <&3 | grep -q " 200" && up=1; exec 3<&-
    [ $up = 1 ] && break
  fi
  sleep 1
done 2>/dev/null
echo "$CFG up=$up after $(( $(date +%s) - t0 )) s"
if [ $up = 1 ]; then
  mkdir -p /out/resp
  get /stats > /out/stats-0.txt
  get /metrics > /out/metrics-0.txt   # run 176 (owner note 179 item 3a): the per-position draft acceptance
  flag=/tmp/heat-sampling; touch $flag
  ( n=1; while [ -e $flag ]; do sleep 30; get /stats > /out/stats-$n.txt; n=$((n + 1)); done ) &
  spid=$!
  bodies=(/out/card/[0-9][0-9]-*.body)
  [ "${CAL:-0}" = 1 ] && bodies=(/out/card/[0-9][0-9][0-9][0-9][0-9]-*.body)
  # milestone 3 run 63 (owner note 52): PIECES pieces of PIECE_S seconds each (default one piece of MIN
  # minutes), the server idle for PAUSE_S between them, so a rotation can run past the tape pool's old
  # collapse (~request 22) inside the heat guard's limit; the requests continue the same rotation, and
  # /out/pieces.txt says which request each piece started at
  #   m2-heat.sh CFG MIN [PIECES [PIECE_S [PAUSE_S]]]   (the step container passes no environment through)
  PIECES=${3:-1}; PIECE_S=${4:-$(( MIN * 60 ))}; PAUSE_S=${5:-0}
  k=0; u0=$(usage)
  if [ "${CAL:-0}" = 1 ]; then
    # run 112 (owner note 140): the corpus once, by four posters, so the scheduler fills each 4K-token pass -- the
    # tap's per-expert host add is a fixed cost a pass (CL1's first try, one request at a time: ~23 s a request).
    # Done when the tap has written: its line is printed at layer 0 of the pass that crossed the target, and the
    # posters' requests in flight (that pass among them) are waited for before the stop, plus 30 s.
    rm -f /out/calib-stop; wp=()
    for w in 0 1 2 3; do
      ( i=$w
        while [ "$i" -lt "${#bodies[@]}" ] && [ ! -e /out/calib-stop ]; do
          b=${bodies[$i]}; name=$(basename "$b" .body); pth=$(cat "${b%.body}.path")
          w0=$EPOCHREALTIME; st=$(post_raw "$pth" "$b" /out/resp/$name.http); w1=$EPOCHREALTIME
          echo "$(date +%s) $name $(echo "$w0 $w1" | awk '{printf "%.2f", $2-$1}') $st" >> /out/requests.txt
          i=$((i + 4))
        done ) &
      wp+=($!)
    done
    end=$(( $(date +%s) + MIN * 60 ))
    while :; do
      grep -q "calibration wrote" /out/serve.log && { echo "calib: the tap wrote at $(date +%s)"; break; }
      alive=0; for p in "${wp[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
      [ "$alive" = 0 ] && { echo "calib: corpus exhausted before the tap wrote"; break; }
      [ "$(date +%s)" -ge "$end" ] && { echo "calib: out of time before the tap wrote"; break; }
      sleep 5
    done
    touch /out/calib-stop; wait "${wp[@]}"; sleep 30
    k=$(wc -l < /out/requests.txt)
    echo "calib: $k requests; $(ls /out/calib | wc -l) files, $(du -sh /out/calib | cut -f1)"
    PIECES=0
  fi
  for piece in $(seq "$PIECES"); do
    [ "$piece" -gt 1 ] && sleep "$PAUSE_S"
    echo "piece $piece starts at request $k, $(date +%s)" >> /out/pieces.txt
    end=$(( $(date +%s) + PIECE_S ))
    while [ "$(date +%s)" -lt "$end" ]; do
      b=${bodies[$(( k % ${#bodies[@]} ))]}; name=$(basename "$b" .body); pth=$(cat "${b%.body}.path")
      w0=$EPOCHREALTIME
      st=$(post_raw "$pth" "$b" /out/resp/$name-$k.http)
      w1=$EPOCHREALTIME
      echo "$(date +%s) $name $(echo "$w0 $w1" | awk '{printf "%.2f", $2-$1}') $st" >> /out/requests.txt
      k=$((k + 1))
    done
  done
  u1=$(usage)
  rm -f $flag; wait $spid
  get /stats > /out/stats-final.txt
  get /metrics > /out/metrics-final.txt
  echo "heat replay: $k requests in $MIN min, container CPU $(( (u1 - u0) / 1000000 )) s"
  python3 - /out/stats-0.txt /out/stats-final.txt <<'PY' 2>/dev/null || echo "(no python in the image: compare stats-0.txt and stats-final.txt on amd-host)"
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
ea, eb = a['experts'], b['experts']
for k in ('dispatches', 'promotions', 'demotions', 'readmits', 'routed', 'routed_res', 'refused', 'quarantined'):
    print('%-12s %d -> %d (+%d)' % (k, ea.get(k, 0), eb.get(k, 0), eb.get(k, 0) - ea.get(k, 0)))
pa, pb = a.get('passes', {}), b.get('passes', {})
for k in ('played', 'recorded', 'audited', 'issued_refused'):
    print('passes.%-10s %d -> %d' % (k, pa.get(k, 0), pb.get(k, 0)))
print('tiers', [(t['name'], t['units']) for t in eb['tiers'][:2]])
PY
fi
kill $pid 2>/dev/null; wait $pid; echo "$CFG server exit $?"
grep -E "^E |heat engine:|GPU|fault" /out/serve.log | tail -6
