#!/usr/bin/env bash
# Quick reference verifier from PROJECT_GUIDE.md §15.
ok=true
fail() { echo "  FAIL: $1"; ok=false; }
pass() { echo "  ok:   $1"; }

echo "isolated core(s)"
iso=$(cat /sys/devices/system/cpu/isolated)
[ -n "$iso" ] && pass "isolated=$iso" || fail "no isolated core"

echo "frequency policy"
# Read governor directly from sysfs — cpupower output format varies.
gov_file=/sys/devices/system/cpu/cpu${iso:-0}/cpufreq/scaling_governor
[ -f "$gov_file" ] || gov_file=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
if [ -f "$gov_file" ]; then
    gov=$(cat "$gov_file")
    [ "$gov" = "performance" ] && pass "governor=$gov" || fail "governor=$gov (need performance)"
else
    fail "no cpufreq sysfs"
fi

echo "turbo off"
if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    [ "$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)" = "1" ] && pass "intel turbo off" || fail "intel turbo on"
fi
if [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
    [ "$(cat /sys/devices/system/cpu/cpufreq/boost)" = "0" ] && pass "amd boost off" || fail "amd boost on"
fi

echo "perf access"
p=$(cat /proc/sys/kernel/perf_event_paranoid)
[ "$p" -le 1 ] && pass "perf_paranoid=$p" || fail "perf_paranoid=$p (need ≤1)"

echo "swap"
[ -z "$(swapon --show)" ] && pass "swap off" || fail "swap is on"

echo "AVX2"
lscpu | grep -q avx2 && pass "avx2" || fail "no avx2 — stage 5 will not work"
lscpu | grep -q avx_vnni && echo "  + VNNI available" || echo "  - no VNNI (use non-VNNI fallback)"

$ok && echo -e "\nrig OK" || (echo -e "\nrig FAILED"; exit 1)
