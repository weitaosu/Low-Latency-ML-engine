#!/usr/bin/env bash
# Source before every benchmark. Sets policy and prints rig state.
set -e

ISOLATED_CORE=${ISOLATED_CORE:-3}

echo "=== rig state ==="
echo "isolated cores:    $(cat /sys/devices/system/cpu/isolated)"
echo "governor:          $(cpupower frequency-info 2>/dev/null | grep 'current policy' | head -1)"
[ -f /sys/devices/system/cpu/intel_pstate/no_turbo ] && \
    echo "no_turbo:          $(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"
[ -f /sys/devices/system/cpu/cpufreq/boost ] && \
    echo "boost (AMD):       $(cat /sys/devices/system/cpu/cpufreq/boost)"
echo "perf_paranoid:     $(cat /proc/sys/kernel/perf_event_paranoid)"
echo "swap:              $(swapon --show | wc -l) entries"
echo "bound to core:     $ISOLATED_CORE"
echo "================="

if [ -n "$1" ]; then
    exec taskset -c $ISOLATED_CORE "$@"
fi
