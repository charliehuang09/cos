#!/usr/bin/env python3
"""Run unattended original/calibrated Orin replays and summarize disagreements."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import traceback


REPO = Path(__file__).resolve().parents[1]
CAMERAS = ("front", "left", "right")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def metric(values):
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "sample_stddev": statistics.stdev(values) if len(values) > 1 else None,
        "rms": math.sqrt(statistics.mean(x * x for x in values)) if values else None,
    }


def analyze(path, max_span_ms):
    groups = defaultdict(lambda: {"translation_m": [], "yaw_deg": []})
    excluded = 0
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            values = {key: float(row[key]) for key in
                      ("translation_m", "yaw_deg", "time_a_s", "time_b_s")}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Non-finite disagreement in {path}")
            if values["translation_m"] < 0 or not 0 <= values["yaw_deg"] <= 180:
                raise ValueError(f"Invalid disagreement in {path}")
            if abs(values["time_a_s"] - values["time_b_s"]) * 1000 >= max_span_ms:
                excluded += 1
                continue
            pair = "-".join(sorted((row["camera_a"], row["camera_b"])))
            for group in ("all", pair):
                for key in ("translation_m", "yaw_deg"):
                    groups[group][key].append(values[key])
    if len(groups["all"]["translation_m"]) < 2:
        raise ValueError(f"Fewer than two simultaneous selected pairs in {path}")
    return groups, {
        "excluded_pairs_at_or_above_max_span": excluded,
        "groups": {group: {key: metric(values) for key, values in metrics.items()}
                   for group, metrics in groups.items()},
    }


def format_metric(value):
    if value["mean"] is None:
        return "n/a"
    stddev = value["sample_stddev"]
    return f'{value["mean"]:.6f} ± {stddev:.6f}' if stddev is not None else f'{value["mean"]:.6f}'


def summarize(output, records, expected_runs, max_span_ms):
    collected = defaultdict(lambda: defaultdict(list))
    run_means = defaultdict(lambda: defaultdict(list))
    summaries = []
    for record in records:
        values, result = analyze(output / record["csv"], max_span_ms)
        summaries.append({**record, **result})
        for group, metrics in values.items():
            for key, samples in metrics.items():
                collected[(record["configuration"], group)][key].extend(samples)
                run_means[(record["configuration"], group)][key].append(statistics.mean(samples))
    result = {"expected_runs_per_configuration": expected_runs,
              "max_pair_timestamp_span_ms_strict": max_span_ms,
              "definition": "Euclidean 3D translation distance and absolute wrapped yaw difference between cameras selected by the live solver; equal weight per accepted camera pair.",
              "stddev_definition": "Sample standard deviation (n-1); pooled pairs and across-run means are reported separately.",
              "runs": summaries, "configurations": {}}
    for configuration in ("original", "calibrated"):
        result["configurations"][configuration] = {
            group: {key: {"pooled_pairs": metric(samples),
                          "across_run_means": metric(run_means[(configuration, group)][key])}
                    for key, samples in metrics.items()}
            for (config, group), metrics in collected.items() if config == configuration}
    paired = {}
    for key in ("translation_m", "yaw_deg"):
        deltas = []
        for run in range(1, expected_runs + 1):
            pair = {r["configuration"]: r for r in summaries if r["run"] == run}
            if len(pair) == 2:
                deltas.append(pair["original"]["groups"]["all"][key]["mean"] -
                              pair["calibrated"]["groups"]["all"][key]["mean"])
        paired[key] = metric(deltas)
    result["paired_mean_reduction_original_minus_calibrated"] = paired
    write_json(output / "summary.json", result)
    lines = ["# Chezychamps localization comparison", "",
             f"Completed {len(records)} of {expected_runs * 2} full replays.", "",
             "Translation is 3D distance in meters. Rotation is absolute wrapped yaw difference in degrees. Only camera pairs actually selected by localization with replay timestamps strictly less than "
             f"{max_span_ms:g} ms apart are included. All standard deviations use n−1.", "",
             "## Variation across runs", "",
             "Each row gives the mean of the per-run means ± their sample standard deviation.", "",
             "| Configuration | Runs | Translation (m) | Yaw (°) |",
             "| --- | ---: | ---: | ---: |"]
    for configuration in ("original", "calibrated"):
        all_metrics = result["configurations"][configuration].get("all")
        if all_metrics:
            translation = all_metrics["translation_m"]["across_run_means"]
            yaw = all_metrics["yaw_deg"]["across_run_means"]
            lines.append(f"| {configuration} | {translation['count']} | {format_metric(translation)} | {format_metric(yaw)} |")
    lines += ["", "Positive original-minus-calibrated reductions mean less disagreement:", "",
              "| Metric | Paired runs | Mean reduction ± stddev |",
              "| --- | ---: | ---: |"]
    for key, value in paired.items():
        lines.append(f"| {key} | {value['count']} | {format_metric(value)} |")
    lines += ["", "## Pooled camera pairs", "",
              "This standard deviation describes the spread of disagreements within the replays, including variation over the match.", "",
              "| Configuration | Camera pair | Pairs | Translation mean ± stddev (m) | Yaw mean ± stddev (°) |",
              "| --- | --- | ---: | ---: | ---: |"]
    for configuration, groups in result["configurations"].items():
        for group, values in sorted(groups.items()):
            translation = values["translation_m"]["pooled_pairs"]
            yaw = values["yaw_deg"]["pooled_pairs"]
            lines.append(f"| {configuration} | {group} | {translation['count']} | {format_metric(translation)} | {format_metric(yaw)} |")
    lines += ["", "## Individual replays", "",
              "| Run | Configuration | Pairs | Translation mean ± stddev (m) | Yaw mean ± stddev (°) |",
              "| ---: | --- | ---: | ---: | ---: |"]
    for record in summaries:
        values = record["groups"]["all"]
        lines.append(f"| {record['run']} | {record['configuration']} | {values['translation_m']['count']} | {format_metric(values['translation_m'])} | {format_metric(values['yaw_deg'])} |")
    lines += ["", "The cameras are compared with each other; there is no ground-truth robot trajectory. Async replay scheduling can change which frames are processed. Configuration hashes, binary hash, raw CSVs, console logs, and WPILOGs accompany this report.", ""]
    (output / "summary.md").write_text("\n".join(lines))


def run_batch(args, output):
    state = {"state": "running", "pid": os.getpid(), "runs_per_configuration": args.runs,
             "completed_replays": [], "started_at": datetime.now(timezone.utc).isoformat()}
    write_json(output / "status.json", state)
    (output / "summary.md").write_text(
        f"# Chezychamps localization comparison\n\nAutomation is running. "
        f"Scheduled: {args.runs} original and {args.runs} calibrated replays.\n\n"
        "This report will be updated automatically after each replay. "
        "See status.json and automation.log for progress or failures.\n")
    remote = args.device_root.rstrip("/") + "/comparison-batches/" + output.name

    def ssh(arguments):
        subprocess.run(["ssh", "-o", "BatchMode=yes", args.device, shlex.join(arguments)], check=True)

    try:
        manifest = {"device": args.device, "device_output_directory": remote,
                    "replay_directory": args.replay_directory, "configurations": {},
                    "runs_per_configuration": args.runs,
                    "max_pair_timestamp_span_ms_strict": args.max_span_ms,
                    "order": "Original first in odd numbered runs, calibrated first in even numbered runs.",
                    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        for name, source in (("original", args.original_configs), ("calibrated", args.calibrated_configs)):
            snapshot = output / (name + "-configs")
            snapshot.mkdir()
            manifest["configurations"][name] = {}
            for camera in CAMERAS:
                filename = camera + "_camera.json"
                shutil.copy2(source / filename, snapshot / filename)
                manifest["configurations"][name][filename] = hashlib.sha256((snapshot / filename).read_bytes()).hexdigest()
        write_json(output / "manifest.json", manifest)
        ssh(["mkdir", "-p", remote + "/tests"])
        ssh(["cp", args.device_root + "/tests/unambiguous_solver_node_test", remote + "/tests/"])
        ssh(["cp", "-a", args.device_root + "/lib", remote + "/lib"])
        manifest["binary_sha256"] = subprocess.check_output(
            ["ssh", "-o", "BatchMode=yes", args.device, shlex.join(["sha256sum", remote + "/tests/unambiguous_solver_node_test"])], text=True).split()[0]
        write_json(output / "manifest.json", manifest)
        for name in ("original", "calibrated"):
            subprocess.run(["scp", "-r", str(output / (name + "-configs")), args.device + ":" + remote + "/"], check=True)
        for run in range(1, args.runs + 1):
            order = ("original", "calibrated") if run % 2 else ("calibrated", "original")
            for configuration in order:
                basename = f"run-{run:02d}-{configuration}"
                state["current_replay"] = basename
                write_json(output / "status.json", state)
                command = ["timeout", str(args.timeout_seconds), remote + "/tests/unambiguous_solver_node_test",
                           "--log_path=" + args.replay_directory,
                           "--camera_config_directory=" + remote + "/" + configuration + "-configs",
                           "--disagreement_csv_path=" + remote + "/" + basename + ".csv",
                           "--wpilog_path=" + remote + "/" + basename + ".wpilog"]
                print(f"Starting {basename}", flush=True)
                with (output / (basename + ".log")).open("w") as log:
                    execution = subprocess.run(["ssh", "-o", "BatchMode=yes", args.device, shlex.join(command)],
                                               stdout=log, stderr=subprocess.STDOUT,
                                               timeout=args.timeout_seconds + 60)
                if execution.returncode != 0:
                    raise RuntimeError(f"{basename} failed with exit code {execution.returncode}")
                for extension in ("csv", "wpilog"):
                    subprocess.run(["scp", args.device + ":" + remote + "/" + basename + "." + extension,
                                    str(output / (basename + "." + extension))], check=True)
                with (output / (basename + ".wpilog")).open("rb") as wpilog:
                    if wpilog.read(6) != b"WPILOG":
                        raise ValueError(f"Invalid WPILOG for {basename}")
                record = {"run": run, "configuration": configuration, "exit_code": 0,
                          "csv": basename + ".csv", "wpilog": basename + ".wpilog"}
                state["completed_replays"].append(record)
                summarize(output, state["completed_replays"], args.runs, args.max_span_ms)
                write_json(output / "status.json", state)
                print(f"Finished {basename}; report updated", flush=True)
        state.update(state="completed", finished_at=datetime.now(timezone.utc).isoformat())
        state.pop("current_replay", None)
        write_json(output / "status.json", state)
        print(f"Completed. Report: {output / 'summary.md'}", flush=True)
    except Exception as error:
        state.update(state="failed", error=str(error), finished_at=datetime.now(timezone.utc).isoformat())
        write_json(output / "status.json", state)
        traceback.print_exc()
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="root@dev-orin")
    parser.add_argument("--device-root", default="/root/extrinsics-calibration-20261001")
    parser.add_argument("--replay-directory", default="/cos-logs/second_bot/extrinsics-chezychamps")
    parser.add_argument("--original-configs", type=Path, default=REPO / "calibration-results/chezychamps/original")
    parser.add_argument("--calibrated-configs", type=Path, default=REPO / "calibration-results/chezychamps/calibrated")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-span-ms", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=int, default=360)
    parser.add_argument("--detach", action="store_true", help="Launch without keeping this terminal or agent active")
    args = parser.parse_args()
    if args.runs < 1 or not math.isfinite(args.max_span_ms) or args.max_span_ms <= 0 or args.timeout_seconds < 1:
        parser.error("Runs, finite timestamp span, and timeout must be positive")
    for source in (args.original_configs, args.calibrated_configs):
        for camera in CAMERAS:
            if not (source / (camera + "_camera.json")).is_file():
                parser.error(f"Missing {camera} config in {source}")
    output = (args.output_directory or REPO / "calibration-results/chezychamps/batches" /
              datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")).resolve()
    if args.detach:
        output.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, str(Path(__file__).resolve()),
                   *[argument for argument in sys.argv[1:] if argument != "--detach"]]
        if args.output_directory is None:
            command += ["--output-directory", str(output)]
        write_json(output / "status.json", {"state": "starting", "runs_per_configuration": args.runs})
        with (output / "automation.log").open("w") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        (output / "automation.pid").write_text(str(process.pid) + "\n")
        print(f"Started automation PID {process.pid}\nResults: {output}\nStatus: {output / 'status.json'}")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        parser.error("Output directory already contains a batch; choose a new directory")
    return run_batch(args, output)


if __name__ == "__main__":
    sys.exit(main())
