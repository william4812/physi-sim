#!/usr/bin/env python3
"""nvml_logger.py — sample GPU telemetry over time via NVML.

Logs temperature, power, clocks, utilization, and (critically) the THROTTLE
REASON bitmask to CSV. The throttle reasons are what a diagnostic engineer
actually cares about: they tell you *why* the clock dropped -- thermal cap,
power cap, or reliability voltage limit -- which raw temperature alone cannot.

Runs on any NVIDIA GPU with the driver installed. Tested target: GTX 1650
(35W cap) -- a strict power ceiling that makes throttling easy to observe.

    pip install nvidia-ml-py          # provides the 'pynvml' module
    python3 nvml_logger.py --seconds 60 --hz 5 --out telemetry.csv
"""
import argparse, csv, time, sys

try:
    import pynvml
except ImportError:
    sys.exit("Need pynvml:  pip install nvidia-ml-py")

# Throttle-reason bits (from nvml.h). Decoding these is the diagnostic payload.
THROTTLE_BITS = {
    "gpu_idle":                 0x0000000000000001,
    "app_clocks_setting":       0x0000000000000002,
    "sw_power_cap":             0x0000000000000004,   # driver power cap
    "hw_slowdown":              0x0000000000000008,   # HW protection tripped
    "sync_boost":               0x0000000000000010,
    "sw_thermal_slowdown":      0x0000000000000020,   # driver thermal cap
    "hw_thermal_slowdown":      0x0000000000000040,   # HW thermal protection
    "hw_power_brake_slowdown":  0x0000000000000080,   # external power brake
    "display_clock_setting":    0x0000000000000100,
}

def decode_throttle(mask):
    """Return the '|'-joined names of active throttle reasons, or 'none'."""
    active = [name for name, bit in THROTTLE_BITS.items() if mask & bit]
    # gpu_idle is not a real throttle -- drop it when the GPU is doing work
    active = [a for a in active if a != "gpu_idle"] or (["idle"] if mask & 1 else [])
    return "|".join(active) if active else "none"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--hz", type=float, default=5.0, help="samples per second")
    ap.add_argument("--out", default="telemetry.csv")
    ap.add_argument("--index", type=int, default=0, help="GPU index")
    args = ap.parse_args()

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(args.index)
    name = pynvml.nvmlDeviceGetName(h)
    if isinstance(name, bytes): name = name.decode()
    cap = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0   # mW -> W
    print(f"logging {name}  (power cap {cap:.0f} W)  ->  {args.out}")

    period = 1.0 / args.hz
    t0 = time.time()
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s","temp_C","power_W","sm_clock_MHz","mem_clock_MHz",
                    "gpu_util_pct","mem_util_pct","throttle_reasons"])
        while time.time() - t0 < args.seconds:
            t = time.time() - t0
            temp  = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            smclk = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
            mclk  = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
            util  = pynvml.nvmlDeviceGetUtilizationRates(h)
            mask  = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h)
            w.writerow([f"{t:.3f}", temp, f"{power:.2f}", smclk, mclk,
                        util.gpu, util.memory, decode_throttle(mask)])
            f.flush()
            time.sleep(max(0.0, period - ((time.time()-t0) - t)))
    pynvml.nvmlShutdown()
    print("done.")

if __name__ == "__main__":
    main()
