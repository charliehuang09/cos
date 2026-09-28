#!/usr/bin/env python3
"""Repeat calibration on random, time-separated recorded ChArUco views.

Run on dev-orin after calib_helper exports the recording's detections. Random
candidates conflicting with an accepted timestamp are rejected; if a sample
cannot reach the requested count, the entire draw is retried. This measures
variation under that sampling procedure, not a uniform draw of all valid sets.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess


def usable(detection):
    points = detection["object_points"]
    if len(detection["charuco_corners"]) <= 3 or not detection["image_points"] or not points:
        return False
    x0, y0, _ = points[0]
    x1, y1, _ = points[1]
    # Board coordinates are in meters; this tolerance is far below one square
    # while accounting for float serialization of collinear diagonal points.
    return any(abs((x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)) > 1e-8
               for x, y, _ in points[2:])


def sample_statistics(values):
    values = list(values)
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return {"mean": mean, "stddev": math.sqrt(variance)}


def spaced_sample(detections, count, gap, rng):
    timestamps = [Decimal(Path(d["filename"]).stem) for d in detections]
    if not all(t.is_finite() for t in timestamps):
        raise ValueError("Frame timestamps must be finite seconds")
    earliest = []
    for timestamp in sorted(timestamps):
        if not earliest or timestamp - earliest[-1] >= gap:
            earliest.append(timestamp)
    if len(earliest) < count:
        raise ValueError(f"Only {len(earliest)} frames can be spaced {gap}s apart; "
                         f"{count} requested")
    rejected = 0
    for _ in range(10000):
        indices = list(range(len(detections)))
        rng.shuffle(indices)
        selected = []
        for index in indices:
            if any(abs(timestamps[index] - timestamps[other]) < gap
                   for other in selected):
                rejected += 1
                continue
            selected.append(index)
            if len(selected) == count:
                result = [detections[index] for index in selected]
                ordered = sorted(timestamps[index] for index in selected)
                minimum_gap = min(b - a for a, b in zip(ordered, ordered[1:]))
                assert minimum_gap >= gap
                assert len({d["filename"] for d in result}) == count
                return result, rejected, str(minimum_gap)
    raise RuntimeError("Unable to draw a full spaced sample after 10000 retries")


def calibrate(binary, sample, output, seed, count):
    command = [str(binary), f"--detections_path={sample}",
               f"--max_detections={count}", f"--sampling_seed={seed}",
               f"--intrinsics_output_path={output}"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    log = result.stdout + result.stderr
    output.with_suffix(".log").write_text(log)
    if result.returncode:
        raise RuntimeError(f"Calibration failed ({result.returncode}): {log}")
    if f"Calibrating with {count} captured frames" not in log:
        raise RuntimeError("Calibration did not use the requested frame count")
    error = re.search(r"Reprojection error: ([^\s]+)", log)
    values = json.loads(output.read_text())
    values["reprojection_error_px"] = float(error.group(1))
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("Nonfinite calibration result")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path,
                        default=Path("/root/tools/intrinsics_calibrate_disk"))
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--min-gap-seconds", type=Decimal, default=Decimal("0.5"))
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--jobs", type=int, default=1,
                        help="number of independent calibration processes")
    args = parser.parse_args()
    if args.runs < 2 or args.frames < 2 or args.min_gap_seconds < 0 or args.jobs < 1:
        parser.error("Require at least two runs/frames and a nonnegative gap")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("Seed must fit uint32")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    saved = json.loads(args.detections.read_text())
    detections = [d for d in saved["detections"] if usable(d)]
    trials = []
    sample_hashes = set()
    def run_trial(run):
        seed = (args.seed + run) % 2**32
        sample, rejected, gap = spaced_sample(
            detections, args.frames, args.min_gap_seconds, random.Random(seed))
        sample_path = args.output_dir / f"sample_{run:03d}.json"
        sample_path.write_text(json.dumps({"image_size": saved["image_size"],
                                          "detections": sample}))
        filenames = sorted(d["filename"] for d in sample)
        sample_hash = hashlib.sha256(json.dumps(filenames).encode()).hexdigest()
        output = args.output_dir / f"intrinsics_{run:03d}.json"
        values = calibrate(args.binary, sample_path, output, seed, args.frames)
        if run == 0:
            repeated = args.output_dir / "intrinsics_repeat.json"
            repeat_values = calibrate(args.binary, sample_path, repeated,
                                      seed, args.frames)
            determinism = {
                "seed": seed, "sample_sha256": sample_hash,
                "identical_intrinsics_bytes": output.read_bytes() == repeated.read_bytes(),
                "identical_values": values == repeat_values,
                "absolute_differences": {
                    key: abs(value - repeat_values[key])
                    for key, value in values.items()},
            }
            (args.output_dir / "determinism.json").write_text(
                json.dumps(determinism, indent=2))
            print("Seeded repeatability: " + json.dumps(determinism), flush=True)
        return {"run": run, "seed": seed, "filenames": filenames,
                "minimum_gap_seconds": gap, "rejected_candidates": rejected,
                "values": values}

    def save_trial(trial):
        sample_hash = hashlib.sha256(json.dumps(trial["filenames"]).encode()).hexdigest()
        if sample_hash in sample_hashes:
            raise RuntimeError("Duplicate sample across trials")
        sample_hashes.add(sample_hash)
        trials.append(trial)
        (args.output_dir / "trials.json").write_text(json.dumps(trials, indent=2))
        print(f"Run {trial['run'] + 1}/{args.runs}: "
              f"gap >= {trial['minimum_gap_seconds']}s, "
              f"RMS {trial['values']['reprojection_error_px']:.6g}px", flush=True)

    # Finish both identical seeded runs before varying samples in parallel.
    save_trial(run_trial(0))
    determinism = json.loads((args.output_dir / "determinism.json").read_text())
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        for trial in executor.map(run_trial, range(1, args.runs)):
            save_trial(trial)
    summary = {
        "runs": args.runs, "frames_per_run": args.frames,
        "min_gap_seconds": str(args.min_gap_seconds), "base_seed": args.seed,
        "usable_detections": len(detections), "stddev_definition": "sample (N-1)",
        "jobs": args.jobs,
        "determinism": determinism,
        "statistics": {
            key: sample_statistics(trial["values"][key] for trial in trials)
            for key in trials[0]["values"]},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
