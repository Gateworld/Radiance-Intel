#!/bin/bash
# ri-sample.sh STEP CTR FLAG -- sample intel-host once a second, read-only, into cpu-STEP.csv until FLAG
# is removed. Only builtins and small /proc and /sys reads per sample (plus `sleep 1`); the
# container's cgroup is looked up once with podman inspect. Its own CPU time goes to
# cpu-STEP.times at exit (the `times` builtin).
#   t, host_busy_pct (all threads, /proc/stat deltas: (total - idle - iowait) / total),
#   ctr_cpu_pct (the container's cgroup usage_usec delta, % of one CPU; empty without one),
#   tctl_c (k10temp Tctl).
D=/mnt/fast-storage/models/test/radiance-intel
step=$1; ctr=$2; flag=$3
out=$D/cpu-$step.csv
tin=""
for h in /sys/class/hwmon/hwmon*; do
    read -r n < "$h/name" 2>/dev/null
    [ "$n" = k10temp ] || continue
    for l in "$h"/temp*_label; do
        read -r lab < "$l"
        [ "$lab" = Tctl ] && tin=${l%_label}_input
    done
done
CGBASE=/sys/fs/cgroup
cgf=""; tries=0
stat_now() {
    local cpu u n s i w q sq st rest
    read -r cpu u n s i w q sq st rest < /proc/stat
    tot=$((u + n + s + i + w + q + sq + st)); idl=$((i + w))
}
usec_now() { local v=${EPOCHREALTIME/./}; now_us=$((10#$v)); }
ctr_now() {
    cu=""
    [ -n "$cgf" ] || return
    local k v
    while read -r k v; do [ "$k" = usage_usec ] && { cu=$v; return; }; done < "$cgf"
}
echo "t,host_busy_pct,ctr_cpu_pct,tctl_c" > "$out"
stat_now; pt=$tot; pi=$idl; usec_now; pw=$now_us; pc=""
while [ -e "$flag" ]; do
    sleep 1
    if [ -z "$cgf" ] && [ "$ctr" != - ] && [ $tries -lt 30 ]; then
        tries=$((tries + 1))
        cp=$(podman inspect --format '{{.State.CgroupPath}}' "$ctr" 2>/dev/null)
        if [ -n "$cp" ]; then
            for c in "$CGBASE$cp/cpu.stat" "$CGBASE/user.slice/user-1000.slice/user@1000.service/$cp/cpu.stat"; do
                [ -r "$c" ] && { cgf=$c; break; }
            done
        fi
    fi
    stat_now; usec_now; ctr_now
    dt=$((tot - pt)); di=$((idl - pi)); dw=$((now_us - pw))
    busy=""; [ $dt -gt 0 ] && busy=$(( (dt - di) * 1000 / dt ))
    cpct=""
    if [ -n "$cu" ] && [ -n "$pc" ] && [ $dw -gt 0 ]; then cpct=$(( (cu - pc) * 1000 / dw )); fi
    tc=""; [ -n "$tin" ] && read -r tc < "$tin"
    printf '%s,%s,%s,%s\n' "${EPOCHREALTIME%.*}" \
        "${busy:+$((busy / 10)).$((busy % 10))}" "${cpct:+$((cpct / 10)).$((cpct % 10))}" \
        "${tc:+$((tc / 1000)).$((tc % 1000 / 100))}" >> "$out"
    pt=$tot; pi=$idl; pw=$now_us; pc=$cu
done
times > "$D/cpu-$step.times"
