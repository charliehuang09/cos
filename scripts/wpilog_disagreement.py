#!/usr/bin/env python3
"""Camera disagreement for COS WPILOGs, using the estimates retained in fusion.

Usage: python3 scripts/wpilog_disagreement.py replay.wpilog
Requires only Python's standard library. Supports front/left/back/right cameras.
Pairs must share a logged context and have image timestamps strictly <10 ms
apart. Reports Euclidean 3D distance and absolute wrapped yaw, equally weighted
per pair, with sample standard deviation (n-1). No residual trimming is applied.
"""

import argparse
from collections import defaultdict
from itertools import combinations, product
import math
from pathlib import Path
import struct


def contexts(path):
    """COS writes each context under a mutex in ascending entry-ID order."""
    data = path.read_bytes()
    if data[:8] != b"WPILOG\x00\x01" or len(data) < 12:
        raise ValueError("Expected a version 1.0 WPILOG")
    offset = 12 + struct.unpack_from("<I", data, 8)[0]
    if offset > len(data):
        raise ValueError("Truncated WPILOG header")
    names, context, previous = {}, {}, 0
    while offset < len(data):
        header = data[offset]
        sizes = ((header & 3) + 1, ((header >> 2) & 3) + 1,
                 ((header >> 4) & 7) + 1)
        end = offset + 1 + sum(sizes)
        if header & 128 or end > len(data):
            raise ValueError("Invalid or truncated WPILOG record header")
        entry = int.from_bytes(data[offset + 1:offset + 1 + sizes[0]], "little")
        length = int.from_bytes(data[offset + 1 + sizes[0]:end - sizes[2]], "little")
        payload = data[end:end + length]
        offset = end + length
        if len(payload) != length:
            raise ValueError("Truncated WPILOG record")
        if entry == 0:
            if payload and payload[0] == 0:  # Start entry: ID, name, type, metadata.
                entry, size = struct.unpack_from("<II", payload, 1)
                names[entry] = payload[9:9 + size].decode()
            elif payload and payload[0] == 1:
                names.pop(struct.unpack_from("<I", payload, 1)[0], None)
            continue
        name = names[entry]
        if name.startswith(".schema/"):
            continue
        if entry <= previous:
            yield context
            context = {}
        context[name], previous = payload, entry
    if context:
        yield context


def doubles(data):
    return struct.unpack("<" + "d" * (len(data) // 8), data)


def tags(data):
    return struct.unpack("<" + "q" * (len(data) // 8), data)


def average(pool):
    if len(pool) == 1:
        return pool[0][2]
    total = sum(1 / item[3] for item in pool)
    xyz, quaternion = [0.] * 3, [0.] * 4
    for _, _, pose, variance, _ in pool:
        weight = 1 / variance / total
        sign = -1 if sum(a * b for a, b in zip(quaternion, pose[3:])) < 0 else 1
        xyz = [a + weight * b for a, b in zip(xyz, pose[:3])]
        quaternion = [a + sign * weight * b for a, b in zip(quaternion, pose[3:])]
    norm = math.sqrt(sum(q * q for q in quaternion))
    return (*xyz, *(q / norm for q in quaternion))


def retained(context):
    """Recover retained cameras/branches by matching tags AND the fused pose.

    Enumerating omissions also handles the square-solve rejection patch. This
    does not assume pos1 won or impose a new 40 cm cutoff during analysis.
    """
    target, ids = doubles(context["pose/pose"]), tags(context["pose/tag_ids"])
    choices = []
    # Iterate publications in their logged order, which is also fusion order.
    for key in context:
        if not key.endswith(":multitag_solver/estimate/pos1/pose"):
            continue
        camera = key.split("/")[0].removeprefix("second_bot_")
        if camera not in ("front", "left", "back", "right"):
            continue
        base = key.rsplit("/pos1/pose", 1)[0]
        time = doubles(context[f"second_bot_{camera}/jpeg_buffer/timestamp"])[0]
        options = [None]
        for branch in ("pos1", "pos2"):
            if branch == "pos2" and context[base + "/pos2_present"] == b"\x00":
                continue
            prefix = base + "/" + branch
            options.append((camera, tags(context[prefix + "/tag_ids"]),
                            doubles(context[prefix + "/pose"]),
                            doubles(context[prefix + "/variance"])[0], time))
        choices.append(options)
    matches = []
    for selection in product(*choices):
        pool = [item for item in selection if item is not None]
        if not pool or tuple(tag for item in pool for tag in item[1]) != ids:
            continue
        pose = average(pool)
        quaternion_error = min(math.dist(pose[3:], target[3:]),
                               math.dist(pose[3:], [-q for q in target[3:]]))
        if math.dist(pose[:3], target[:3]) < 1e-8 and quaternion_error < 1e-8:
            matches.append(pool)
    if len(matches) != 1:
        raise ValueError(f"Cannot uniquely recover fused candidates ({len(matches)} matches)")
    return matches[0]


def yaw(pose):
    w, x, y, z = pose[3:]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def stats(values):
    mean = math.fsum(values) / len(values)
    sd = (math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1))
          if len(values) > 1 else float("nan"))
    return mean, sd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wpilog", type=Path)
    args = parser.parse_args()
    samples, excluded, fused = defaultdict(list), 0, 0
    for context in contexts(args.wpilog):
        if "pose/pose" not in context:
            continue
        fused += 1
        for a, b in combinations(retained(context), 2):
            if abs(a[4] - b[4]) * 1000 >= 10:
                excluded += 1
                continue
            distance = 100 * math.dist(a[2][:3], b[2][:3])
            angle = abs(math.degrees(math.remainder(yaw(a[2]) - yaw(b[2]), 2 * math.pi)))
            for group in ("all", "-".join(sorted((a[0], b[0])))):
                samples[group].append((distance, angle))
    if not samples:
        raise ValueError("No retained camera pairs with timestamps <10 ms apart")
    print(f"{fused} fused contexts; {excluded} pairs excluded by timestamp gap")
    print("Pair             N     Translation cm (mean / stddev)    Yaw deg (mean / stddev)")
    for group, values in sorted(samples.items()):
        distance, angle = map(stats, zip(*values))
        print(f"{group:<14} {len(values):5d}     {distance[0]:7.3f} / {distance[1]:7.3f}"
              f"                 {angle[0]:7.3f} / {angle[1]:7.3f}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, struct.error, ZeroDivisionError) as error:
        raise SystemExit(f"Error: {error}")
