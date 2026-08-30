#!/usr/bin/env python3
"""run_diagnostic.py — orchestrate a GPU stress test and emit a Pass/Fail report.

This mirrors the day-to-day of a data-center diagnostics engineer: launch a
workload, watch telemetry for throttling/instability, decide health. It runs the
CUDA load generator as a subprocess while sampling NVML in the parent, then
applies pass/fail criteria and prints a report.

PASS/FAIL LOGIC (transparent, tunable):
  FAIL if the GPU spends > fail_frac of the loaded window in HW thermal or HW
       power-brake slowdown (hardware-protection throttling -- a real fault
       signal), OR if the SM clock collapses below min_clock_frac of its loaded
       peak (instability). Otherwise PASS.
  SW power-cap throttling on a 35W part is EXPECTED and does NOT fail the test --
  distinguishing "designed power management" from "fault" is the whole skill.

    python3 run_diagnostic.py --seconds 60
"""
import argparse, subprocess, threading, time, sys, statistics

try:
    import pynvml
except ImportError:
    sys.exit("Need pynvml:  pip install nvidia-ml-py")

THROTTLE_BITS = {
    "sw_power_cap":            0x04,
    "hw_slowdown":             0x08,
    "sw_thermal_slowdown":     0x20,
    "hw_thermal_slowdown":     0x40,
    "hw_power_brake_slowdown": 0x80,
}

class Sampler(threading.Thread):
    """Samples NVML in the background while the load runs."""
    def __init__(self, handle, hz=5.0):
        super().__init__(daemon=True)
        self.h, self.period, self.stop = handle, 1.0/hz, False
        self.rows = []
    def run(self):
        while not self.stop:
            temp  = pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(self.h)/1000.0
            smclk = pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM)
            mask  = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.h)
            self.rows.append((time.time(), temp, power, smclk, mask))
            time.sleep(self.period)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--load-bin", default="./gpu_load")
    ap.add_argument("--fail-frac", type=float, default=0.10,
                    help="max fraction of window in HW-protection throttle before FAIL")
    ap.add_argument("--min-clock-frac", type=float, default=0.60,
                    help="SM clock must stay above this fraction of loaded peak")
    args = ap.parse_args()

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(h)
    if isinstance(name, bytes): name = name.decode()
    print(f"=== GPU diagnostic: {name} ===")

    idle_temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
    print(f"idle temp: {idle_temp} C.  launching load for {args.seconds:.0f}s...")

    sampler = Sampler(h); sampler.start()
    try:
        proc = subprocess.run([args.load_bin, str(args.seconds)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print("load generator failed:\n", proc.stderr); sys.exit(2)
    finally:
        sampler.stop = True; sampler.join()

    rows = sampler.rows
    if not rows: sys.exit("no telemetry captured")

    # analyse only the loaded window (skip first 10% ramp)
    n = len(rows); loaded = rows[n//10:]
    temps  = [r[1] for r in loaded]
    powers = [r[2] for r in loaded]
    clocks = [r[3] for r in loaded]
    peak_clk = max(clocks)
    hw_fault_bits = THROTTLE_BITS["hw_thermal_slowdown"] | THROTTLE_BITS["hw_power_brake_slowdown"] | THROTTLE_BITS["hw_slowdown"]
    hw_fault_frac = sum(1 for r in loaded if r[4] & hw_fault_bits) / len(loaded)
    sw_power_frac = sum(1 for r in loaded if r[4] & THROTTLE_BITS["sw_power_cap"]) / len(loaded)
    min_clk_frac  = min(clocks) / peak_clk if peak_clk else 1.0

    print("\n--- results (loaded window) ---")
    print(f"  temp   : {min(temps)}-{max(temps)} C  (peak rise {max(temps)-idle_temp} K)")
    print(f"  power  : {min(powers):.1f}-{max(powers):.1f} W")
    print(f"  SM clk : {min(clocks)}-{peak_clk} MHz")
    print(f"  time in SW power-cap throttle : {100*sw_power_frac:.0f}%  (expected on 35W part)")
    print(f"  time in HW-protection throttle: {100*hw_fault_frac:.0f}%  (fault signal)")

    fail_reasons = []
    if hw_fault_frac > args.fail_frac:
        fail_reasons.append(f"HW-protection throttle {100*hw_fault_frac:.0f}% > {100*args.fail_frac:.0f}%")
    if min_clk_frac < args.min_clock_frac:
        fail_reasons.append(f"SM clock collapsed to {100*min_clk_frac:.0f}% of peak")

    verdict = "PASS" if not fail_reasons else "FAIL"
    print(f"\n  VERDICT: {verdict}")
    for r in fail_reasons: print(f"    - {r}")
    pynvml.nvmlShutdown()
    sys.exit(0 if verdict == "PASS" else 1)

if __name__ == "__main__":
    main()
