#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Part of the Intel port of radiance (milestone 3, the prefill-hang bisect; run 250 revision, for the supervisor's
# review before use). libxe links with relocatable device code: one ocloc compiles every unit at the link, so a
# single link cannot mix compilers, and Intel LLVM 7.2.0 has removed -fsycl-link=image (run 250: "option
# '-fsycl-link=image' is not supported and has been removed from the compiler"). So the halves become two shared
# objects, each an ordinary full -fsycl link inside its own build image -- the way libxe_fmha.so (2.42.6) already
# runs beside a 2.28.4 libxe.so (X2, B162, clean):
#   libxe_b.so  half B, linked and AOT-compiled by IGC 2.42.6 (localhost/ri-m1buildneo:2, the 2640 tree's objects);
#   libxe.so    half A, linked and AOT-compiled by IGC 2.28.4 (localhost/ri-m1build:3, build-m1l0's objects),
#               DT_NEEDED libxe_b.so, rpath /out (the step folder, where bstepx copies both files).
# The units' device code references no other unit (the only device externs are IGC builtins: xe_dp4a.h, xe_lsc.h),
# so each half device-links alone; host calls between the halves resolve through the dynamic linker.
# The object list is libxe's link line in build-m1l0/build.ninja (47 objects; fmha is its own module now, so the
# stale libxe.dir/fmha/*.o is not part of it).
#   bash m3/split-link.sh NAME "HALF_B_OBJECTS (paths under libxe/CMakeFiles/libxe.dir/)"
# Output: split/NAME/{libxe.so,libxe_b.so,kernels-A.txt,kernels-B.txt,dups.txt} and logs/split-NAME.txt.
set -u
NAME=$1; HALFB=$2
W=$HOME/code/radiance-intel
T=/srv/bulk/ri/build-m1l0neo2640
O=$W/split/$NAME
L=$W/logs/split-$NAME.txt
mkdir -p "$O"
P=libxe/CMakeFiles/libxe.dir/
ALL=$(grep '^build radiance_home/kernels/libxe.so:' "$W/build-m1l0/build.ninja" | tr ' ' '\n' | grep "^$P.*\.o$" \
      | sed "s#^$P##" | sort)
A=""; B=""
for f in $ALL; do
  case " $HALFB " in *" $f "*) B="$B $f" ;; *) A="$A $f" ;; esac
done
for f in $HALFB; do case " $(echo $ALL) " in *" $f "*) ;; *) echo "not in libxe's link line: $f" | tee "$L"; exit 2 ;; esac; done
echo "half A (2.28.4): $(echo $A | wc -w) objects; half B (2.42.6): $(echo $B | wc -w) objects" | tee "$L"
echo "B:$B" >> "$L"
FL="-fPIC -O2 -g -DNDEBUG -Xlinker --build-id=sha1 -fsycl -fsycl-targets=intel_gpu_bmg_g31,spir64 -foffload-fp32-prec-div -foffload-fp32-prec-sqrt"
rm -f "$O/libxe.so" "$O/libxe_b.so"
# half B: libxe_b.so, linked + AOT by 2.42.6
BB=$(for f in $B; do printf ' %s/%s%s' "$T" "$P" "$f"; done)
timeout 3h podman run --rm --name ri-m3-splitB --cpus 4 --memory 10g -v "$W:/w" -v /srv/bulk/ri:/srv/bulk/ri \
  localhost/ri-m1buildneo:2 nice -n 19 bash -c "cd $T && /opt/sycl/bin/clang++ $FL -shared -Wl,-soname,libxe_b.so \
    -o /w/split/$NAME/libxe_b.so $BB -Wl,-rpath,$T/lib: -ldl lib/librad_l0k.so" >> "$L" 2>&1
echo "libxe_b rc=$?" | tee -a "$L"
[ -f "$O/libxe_b.so" ] || exit 1
# half A: libxe.so, linked + AOT by 2.28.4, needing libxe_b.so from /out
AA=$(for f in $A; do printf ' /w/build-m1l0/%s%s' "$P" "$f"; done)
timeout 3h podman run --rm --name ri-m3-splitA --cpus 4 --memory 10g -v "$W:/w" localhost/ri-m1build:3 \
  nice -n 19 bash -c "cd /w/build-m1l0 && /opt/sycl/bin/clang++ $FL -shared -o /w/split/$NAME/libxe.so $AA \
    /w/split/$NAME/libxe_b.so -Wl,-rpath,/out:/w/build-m1l0/lib: -ldl lib/librad_l0k.so" >> "$L" 2>&1
echo "libxe rc=$?" | tee -a "$L"
[ -f "$O/libxe.so" ] || exit 1
# proofs on amd-host: the dependency and rpath; each half's kernel names (the integration headers' _ZTS strings in the
# host objects, minus RTTI symbol names) and the names in both halves (a kernel in both images -> keep its TUs
# together, or say which compiler the runtime picked)
readelf -d "$O/libxe.so" | grep -E 'NEEDED|RPATH|RUNPATH' | tee -a "$L"
readelf -d "$O/libxe_b.so" | grep -E 'SONAME' | tee -a "$L"
kn() { local d=$1; shift; for f in "$@"; do strings -a "$d/$P$f" | grep '^_ZTS'; done | sort -u > /tmp/kn.$$;
       for f in "$@"; do nm "$d/$P$f" 2>/dev/null | awk '{print $NF}' | grep '^_ZTS'; done | sort -u > /tmp/rt.$$;
       comm -23 /tmp/kn.$$ /tmp/rt.$$; rm -f /tmp/kn.$$ /tmp/rt.$$; }
kn "$W/build-m1l0" $A > "$O/kernels-A.txt"
kn "$T" $B > "$O/kernels-B.txt"
comm -12 "$O/kernels-A.txt" "$O/kernels-B.txt" > "$O/dups.txt"
echo "kernel names: A $(wc -l < "$O/kernels-A.txt"), B $(wc -l < "$O/kernels-B.txt"), in both $(wc -l < "$O/dups.txt")" | tee -a "$L"
ls -la "$O" | tee -a "$L"
