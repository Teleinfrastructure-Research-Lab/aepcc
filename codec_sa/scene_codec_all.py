import argparse
import csv
import math
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
import subprocess
import numpy as np


ARCHS_DEFAULT = ["foldingnet", "gae", "tgae"]
DS_DEFAULT = [128, 256, 512]
QS_DEFAULT = [32, 16, 8, 4, 2]

PATTERNS = {
    "rate": re.compile(r"Scene rate\s*:\s*([-+]?\d+(?:\.\d+)?)"),
    "psnr": re.compile(r"Scene PSNR\s*:\s*([-+]?\d+(?:\.\d+)?)"),
    "spars": re.compile(r"Sparsification\s*:\s*([-+]?\d+(?:\.\d+)?)"),
    "enc": re.compile(r"Encode time\s*:\s*([-+]?\d+(?:\.\d+)?)"),
    "dec": re.compile(r"Decode time\s*:\s*([-+]?\d+(?:\.\d+)?)"),
}

def parse_first(log_text: str, key: str):
    m = PATTERNS[key].search(log_text)
    return float(m.group(1)) if m else None

def fmt6(x):
    return f"{x:.6f}"

def safe_stdev(values):
    if len(values) >= 2:
        return float(np.std(values, ddof=1))
    return 0.0


def run_one(
    python_bin: str,
    scene_codec_py: Path,
    input_file: Path,
    output_file: Path,
    arch: str,
    d: int,
    q: int,
    timeout_s: int,
    log_file: Path,
):
    """
    Run a single trial. Returns (success, log_text).
    """
    cmd = [
        python_bin, "-u", str(scene_codec_py),
        "-a", arch, "-d", str(d), "-q", str(q), "-m", "both",
        str(input_file), str(output_file),
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    env["TORCH_SHOW_CPP_STACKTRACES"] = "1"

    log_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"    CMD: {' '.join(cmd)}")
    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"CMD: {' '.join(cmd)}\n")
        lf.flush()
        try:
            proc = subprocess.run(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout_s,
                check=False,
            )
            success = (proc.returncode == 0)
        except subprocess.TimeoutExpired:
            lf.write(f"\n[runner] TIMEOUT after {timeout_s}s\n")
            success = False

    log_text = log_file.read_text(encoding="utf-8", errors="replace")
    if not success:
        tail = "\n".join(log_text.splitlines()[-40:])
        print("    FAILED (see log). Tail:\n" + tail)

    return success, log_text

def aggregate_combo(
    python_bin: str,
    scene_codec_py: Path,
    input_file: Path,
    out_dir: Path,
    logs_dir: Path,
    arch: str,
    d: int,
    q: int,
    runs: int,
    timeout_s: int,
):
    out_ply = out_dir / f"{input_file.stem}_{arch}_{d}_{q}.ply"
    print(f"Running arch={arch} d={d} q={q} -> {out_ply.name} ...")

    enc_times = []
    dec_times = []
    fixed_rate = None
    fixed_psnr = None
    fixed_spars = None
    success_count = 0

    for i in range(1, runs + 1):
        log_file = logs_dir / f"{input_file.stem}_{arch}_{d}_{q}_run{i}.log"
        print(f"  [{arch} d={d} q={q}] Run {i}/{runs} ...")

        ok, log_text = run_one(
            python_bin, scene_codec_py,
            input_file, out_ply, arch, d, q,
            timeout_s, log_file
        )
        if not ok:
            continue

        rate = parse_first(log_text, "rate")
        psnr = parse_first(log_text, "psnr")
        spars = parse_first(log_text, "spars")
        enc = parse_first(log_text, "enc")
        dec = parse_first(log_text, "dec")

        if (rate is None) or (psnr is None) or (enc is None) or (dec is None):
            print("    Warning: missing required metrics (rate/psnr/enc/dec); skipping this run.")
            continue

        if fixed_rate is None: fixed_rate = rate
        if fixed_psnr is None: fixed_psnr = psnr
        if fixed_spars is None and spars is not None: fixed_spars = spars

        enc_times.append(enc)
        dec_times.append(dec)
        success_count += 1
        print(f"    ok: rate={fmt6(rate)}, psnr={fmt6(psnr)}, spars={fmt6(spars) if spars is not None else 'NA'}, enc={fmt6(enc)}s, dec={fmt6(dec)}s")

    if success_count == 0:
        print(f"  ERROR: no successful runs for arch={arch}, d={d}, q={q} — skipping CSV row.")
        return None

    enc_mean = float(np.mean(enc_times))
    enc_std  = safe_stdev(enc_times)
    dec_mean = float(np.mean(dec_times))
    dec_std  = safe_stdev(dec_times)
    if fixed_spars is None:
        fixed_spars = float("nan")

    return {
        "arch": arch,
        "d": d,
        "q": q,
        "n_runs": success_count,
        "rate": fixed_rate,
        "psnr": fixed_psnr,
        "spars": fixed_spars,
        "enc_time_mean_s": enc_mean,
        "enc_time_std_s": enc_std,
        "dec_time_mean_s": dec_mean,
        "dec_time_std_s": dec_std,
    }

def main():
    parser = argparse.ArgumentParser(description="Sweep scene_codec.py and aggregate metrics into ./codec_sa.csv")
    parser.add_argument("input_file", type=Path, help="Input OBJ/PLY")
    parser.add_argument("out_dir", type=Path, help="Output directory for recon PLYs")
    parser.add_argument("-a", "--architecture", choices=ARCHS_DEFAULT, help="Only run this architecture")
    parser.add_argument("-r", "--runs", type=int, default=10, help="Runs per (arch,d,q) [default: 10]")
    parser.add_argument("-t", "--timeout", type=int, default=900, help="Per-run timeout seconds [default: 900]")
    parser.add_argument("--python", dest="python_bin", default=sys.executable, help="Python interpreter to run scene_codec.py [default: current]")
    parser.add_argument("--scene-codec", dest="scene_codec_path", type=Path, default=None, help="Path to scene_codec.py (default: alongside this script)")
    args = parser.parse_args()

    input_file = args.input_file
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    if args.scene_codec_path is None:
        script_dir = Path(__file__).resolve().parent
        scene_codec_py = script_dir / "scene_codec.py"
    else:
        scene_codec_py = args.scene_codec_path

    if not scene_codec_py.exists():
        print(f"ERROR: scene_codec.py not found at {scene_codec_py}", file=sys.stderr)
        sys.exit(1)

    archs = [args.architecture] if args.architecture else ARCHS_DEFAULT
    ds = DS_DEFAULT
    qs = QS_DEFAULT

    csv_path = Path("codec_sa.csv")
    print(f"Python  : {args.python_bin}")
    print(f"Script  : {scene_codec_py}")
    print(f"Input   : {input_file}")
    print(f"Outdir  : {out_dir}")
    print(f"Runs    : {args.runs}")
    print(f"Timeout : {args.timeout}s per run\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "arch","d","q","n_runs",
            "rate","psnr","spars",
            "enc_time_mean_s","enc_time_std_s",
            "dec_time_mean_s","dec_time_std_s",
        ])

        for arch in archs:
            for d in ds:
                for q in qs:
                    row = aggregate_combo(
                        python_bin=args.python_bin,
                        scene_codec_py=scene_codec_py,
                        input_file=input_file,
                        out_dir=out_dir,
                        logs_dir=logs_dir,
                        arch=arch, d=d, q=q,
                        runs=args.runs,
                        timeout_s=args.timeout,
                    )
                    if row is None:
                        continue
                    writer.writerow([
                        row["arch"], row["d"], row["q"], row["n_runs"],
                        fmt6(row["rate"]), fmt6(row["psnr"]), fmt6(row["spars"]),
                        fmt6(row["enc_time_mean_s"]), fmt6(row["enc_time_std_s"]),
                        fmt6(row["dec_time_mean_s"]), fmt6(row["dec_time_std_s"]),
                    ])
                    fcsv.flush()

    print(f"\nSaved results to ./{csv_path.name}")
    print("Per-run logs are in ./logs/")

if __name__ == "__main__":
    main()
