#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, run 207, owner note 205): m3/build-igc.sh's in-container part --
# fetch the pinned revisions, build IGC, package it. Writes only under /o (= /srv/bulk/ri/igc-243).
# Pins: IGC = the head of releases/2.43.x on 11-06 (fda9face, 2026-11-05 "vISA: Revert merge scalar pass tuning");
# its "Update release dependencies" commit (f8484f61, 10-31) is empty, so the dependencies are 2.42.6's release set,
# the newest documented one (the same LLVM 22.1.8).
set -eux
JOBS=${JOBS:-12}
# opencl-clang applies its clang patches with `git am`, which wants a committer: a neutral, container-local one
export GIT_AUTHOR_NAME=ri-igc-build GIT_AUTHOR_EMAIL=ri-igc-build@localhost
export GIT_COMMITTER_NAME=ri-igc-build GIT_COMMITTER_EMAIL=ri-igc-build@localhost
# BUILD (build-igc.sh's 2nd argument): the build tree, /o/build by default. Run 1's failed `git am` left /o/build's
# opencl-clang clang patches marked done but never applied (no CL3.1 in the built clang); a fresh tree (/o/build2)
# re-runs the patch step. Nothing is deleted (supervisor, run 207); the stale tree stays for the owner.
BUILD=${BUILD:-/o/build}
mkdir -p /o/src "$BUILD" /o/pkg
cd /o/src
echo "# a PROBE, not Intel's 2.43 release set: IGC releases/2.43.x (IGC_REV=${IGC_REV:-fda9face}) with 2.42.6's dependency set" > /o/REVISIONS
echo "# (the branch's 10-31 'Update release dependencies' commit f8484f61 is empty; a later tag's dependency changes are NOT followed here)" >> /o/REVISIONS
pin() {   # url dir revision
    [ -d "$2/.git" ] || git clone -q "$1" "$2"
    # run 280 (owner note 223's fallback, IGC_REV=v2.43.9): a tag or commit the clone does not have yet is fetched first
    git -C "$2" checkout -q "$3" 2>/dev/null || { git -C "$2" fetch -q --tags origin; git -C "$2" checkout -q "$3"; }
    git -C "$2" log -1 --format="$2 %H %cI" | tee -a /o/REVISIONS
}
# IGC_REV (run 280, owner note 223's fallback): the IGC revision to build; the default is run 207's probe pin
IGC_REV=${IGC_REV:-fda9face0b56672ceaa31f46c5ab29065bef5aab}
[ -d llvm-project/.git ] || git clone -q --depth 1 -b llvmorg-22.1.8 https://github.com/llvm/llvm-project llvm-project
git -C llvm-project log -1 --format="llvm-project %H %cI" | tee -a /o/REVISIONS
pin https://github.com/intel/opencl-clang llvm-project/llvm/projects/opencl-clang b953ecee
# SPIRV-LLVM-Translator DEVIATES from 2.42.6's documented 0e672448 (09-16): IGC 2.43 needs
# TranslatorOpts::setErrorHandlingKind (ee30cde3, 09-29); re-pinned to llvm_release_220's head as of IGC's 10-31
# dependency marker, a2c61558 (10-29) -- "as of 10-31" is the rule for the set (supervisor-approved, run 207)
echo "# SPIRV-LLVM-Translator re-pinned to llvm_release_220 a2c61558 (10-29) from 2.42.6's 0e672448: IGC 2.43 needs setErrorHandlingKind" >> /o/REVISIONS
pin https://github.com/KhronosGroup/SPIRV-LLVM-Translator llvm-project/llvm/projects/llvm-spirv a2c6155836cfb24e4d6a3fe704fb60122422e5b4
# vc-intrinsics DEVIATES from 2.42.6's documented v0.25.0 (27f7c4f3, 02-04): that one uses APIs LLVM 22 deprecates and
# IGC builds with -Werror; master e2bb84f5 (10-01, the last before 2.42.6's release) holds the LLVM 22 fixes
# (supervisor-approved re-pin, run 207)
echo "# vc-intrinsics re-pinned to master e2bb84f5 (10-01) from 2.42.6's documented v0.25.0: LLVM 22 deprecations under -Werror" >> /o/REVISIONS
pin https://github.com/intel/vc-intrinsics vc-intrinsics e2bb84f509f613b0e36bd131c3a90800b4fe8f88
pin https://github.com/KhronosGroup/SPIRV-Tools SPIRV-Tools f8035151
pin https://github.com/KhronosGroup/SPIRV-Headers SPIRV-Headers 575b6512
pin https://github.com/intel/intel-graphics-compiler igc "$IGC_REV"
git -C igc describe --tags --always 2>/dev/null | sed 's/^/# igc describe: /' | tee -a /o/REVISIONS
# run 1's failed `git am` may have left an am in progress in the source checkout: abort it (restores the checkout)
git -C /o/src/llvm-project status --short | head -20 || true
if [ -d /o/src/llvm-project/.git/rebase-apply ]; then echo "am in progress in llvm-project: aborting"; git -C /o/src/llvm-project am --abort; fi
cd "$BUILD"
cmake -G Ninja /o/src/igc -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DLLVM_PARALLEL_LINK_JOBS=2 \
  2>&1 | tee "$BUILD/configure.log"
# all nine of opencl-clang's clang patches applied in this tree, or stop (supervisor, run 207)
# (a re-configure of a patched tree applies none: $BUILD/.ri-patches-ok marks a tree whose first configure showed 9)
n=$(grep -c "OPENCL-CLANG\] Applying .*patches/clang/000[1-9]-" "$BUILD/configure.log" || true)
echo "opencl-clang clang patches applied: $n of 9"
if [ "$n" = 9 ]; then date -u > "$BUILD/.ri-patches-ok"
elif [ "$n" = 0 ] && [ -f "$BUILD/.ri-patches-ok" ]; then echo "patched at an earlier configure ($(cat "$BUILD/.ri-patches-ok"))"
else echo "STOP: not all nine clang patches applied"; exit 3; fi
ninja -j "$JOBS"
cpack -G DEB -B /o/pkg
ls -la /o/pkg
