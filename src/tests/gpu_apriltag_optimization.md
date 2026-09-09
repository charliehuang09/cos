# GPU AprilTag optimization on dev-orin

The baseline is the uncommitted working tree at the start of this optimization,
including its existing decimation support. The baseline benchmark executable was
saved as `/root/apriltag-optimization/baseline` before implementation changes.
The detector still defaults to full resolution; this optimization does not change
decimation, target IDs, Hamming thresholds, or retry policy.

## Recording results

Measured on dev-orin on 2026-09-09. Full-resolution and decimated before/after
exports match byte-for-byte across all 2,154 selected frames (4,308 frame/mode
comparisons). All normal benchmarks exited successfully.

| Dataset | Frames | Decimation | Before ms/frame | After ms/frame | Speedup | Time reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev-orin/log2 | 1,724 | 1 | 7.78734 | 5.03821 | 1.55x | 35.3% |
| pmatch2_compressed/right, every fifth filename | 330 | 1 | 23.9976 | 11.1177 | 2.16x | 53.7% |
| log22/right, first 100 frames | 100 | 1 | 12.6742 | 8.04730 | 1.57x | 36.5% |
| dev-orin/log2 | 1,724 | 2 | 6.08223 | 2.96085 | 2.05x | 51.3% |
| pmatch2_compressed/right, every fifth filename | 330 | 2 | 10.5342 | 4.59461 | 2.29x | 56.4% |
| log22/right, first 100 frames | 100 | 2 | 6.34331 | 2.99669 | 2.12x | 52.8% |

The latter two datasets are under `/cos-logs/second_bot/`. Comparisons are
against the same decimation setting in the saved baseline, not against another
resolution. The existing decimation accuracy tradeoff is unchanged. Full-resolution
tag totals are 1,540 / 533 / 300; decimated totals are 1,562 / 472 / 293.
Results describe warmed recording replays, not camera-to-result latency.

Logs and TSV exports are retained on dev-orin in `/root/apriltag-optimization/`.

## Four concurrent detectors

Measured on 2026-09-09 with `gpu_apriltag_concurrency_benchmark`, using the
first 100 log22/right frames, 10 passes per detector, and decimation 2. Input is
1280x800, segmentation is 640x400, and decoding/refinement uses the input image.
Images are preloaded and every detector is warmed before a synchronized start.
Four host threads each own a detector in the same process, sharing the production
geometry thread pool and CUDA context. Timings exclude capture and JPEG decoding.

| Instances | Frames per detector | Combined FPS | Per-camera FPS | Mean time per detection |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1,000 | 343.77 | 343.77 | 2.91 ms |
| 4, run 1 | 1,000 | 521.12 | 130.28 | 7.66–7.68 ms |
| 4, run 2 | 1,000 | 520.25 | 130.06 | 7.66–7.69 ms |

Each worker's final pass matched the sequential reference's IDs and corners on
all 100 frames. Per-camera FPS is combined FPS divided by the number of detectors;
individual completion rates are also printed. These results replay the same
recording through four detectors; actual four-camera throughput also depends on
capture, decoding, and scene content.

```sh
/root/tests/gpu_apriltag_concurrency_benchmark --input_folder=/cos-logs/second_bot/log22/right --instances=4 --decimate=2 --max_images=100 --iterations=10
```

Both normal four-instance runs exited successfully. A subsequent two-frame
Valgrind smoke check (`--error-exitcode=99 --undef-value-errors=no --leak-check=no
--num-callers=30`, four instances, one iteration) aborted with exit 134 before
completion: `The futex facility returned an unexpected error code.` The stack
passes through `__futex_lock_pi64`, `pthread_mutex_lock`, `libcuda.so.1.1`, and
`cuLaunchKernel`. Valgrind also reported the previously observed loader invalid
read (23 errors from one context). This check provides no leak verdict; it does
not invalidate the separately completed normal timing runs. The log is retained
at `/root/apriltag-optimization/concurrency-valgrind.log`.

## Stage measurements

On the first 100 sampled frames of `second_bot/pmatch2_compressed/right`
(`--sample_step=5 --max_images=100 --profile`, full resolution):

| Stage | Before ms/frame | After ms/frame |
| --- | ---: | ---: |
| Input, preprocessing, labeling, extraction | 3.228 | 3.254 |
| Boundary sorting and deduplication | 15.440 | 4.284 |
| Quad geometry and bit locations | 1.990 | 2.020 |
| Tag decoding | 1.242 | 0.458 |
| Corner refinement and output | 0.096 | 0.098 |
| Complete detection | 22.104 | 10.228 |

The sample's exported detections are byte-for-byte identical. Temporary detailed
instrumentation localized most of the original sorting cost to CPU coordinate
deduplication. A segmented radix-sort experiment was slower and was discarded;
the retained packed-key path uses a global radix sort.

## Implementation

- Replace double-precision `atan2` sort keys with a monotonic angular ratio.
  For coordinate spans up to 8191 on each axis, pack the segment index and a
  32-bit angular rank into one key and use a global radix sort. The rank has
  finer resolution than the minimum separation of distinct integer-coordinate
  rays in that range. Larger spans retain double keys and segmented sorting.
- Replace per-segment coordinate hash tables with a reusable stamp array for
  bounded image coordinates, preserving the first occurrence of each point.
  Cap that workspace at 64 MiB and use a scalar-key hash fallback for sparse
  coordinate ranges. Reuse the original segment storage when compacting points.
- Assign a warp to each candidate tag. Cooperatively accumulate border samples,
  assemble codewords with ballots, and compare target codes. Reduce Hamming
  distance together with original scan order to preserve ID/rotation tie breaks.
- Add optional `GPUApriltagDetector::Profile` and benchmark `--profile` reporting.
  Timings are wall times; extraction includes input staging, preprocessing, and
  connected components. Decode includes decimation retries. Profiling adds no
  device-wide synchronization and is off by default.

## Reproduction

Build and deploy using the repository target:

```sh
cmake --build build --target dev-orin -j 4
```

The deployment target places tests in `/root/tests`. Run on dev-orin:

```sh
/root/tests/gpu_apriltag_pipeline_test
/root/tests/gpu_segment_extractor_test
/root/tests/gpu_apriltag_benchmark --input_folder=/cos-logs/dev-orin/log2 --iterations=3
/root/tests/gpu_apriltag_benchmark --input_folder=/cos-logs/second_bot/pmatch2_compressed/right --sample_step=5 --iterations=3
/root/tests/gpu_apriltag_benchmark --input_folder=/cos-logs/second_bot/log22/right --max_images=100 --iterations=3
```

Repeat each benchmark with `--decimate=2`. Add `--detections_output=FILE.tsv`
to compare each frame's IDs and corners with the saved baseline executable.
Add `--profile` for stage timings. Images are loaded and warmed before three
timed repetitions; times exclude JPEG decoding, construction, and TSV output.
Run benchmarks sequentially to avoid device contention. No power-mode, clock,
service, or system configuration changes are needed.

Tests cover GPU/CPU preprocessing agreement, mapped and padded host inputs,
nonadjacent duplicate points, packed and fallback angular ordering, changing
coordinate ranges, batched rotations and bit errors, target filtering/capacity,
parallel CPU geometry, and boundary extraction.

## Memory check

All 10 pipeline/geometry/decoder tests and all 4 extraction tests passed on the
device. After normal runs passed, both the optimized and saved baseline binaries
were checked with:

```sh
valgrind --error-exitcode=99 --undef-value-errors=no --leak-check=full --show-leak-kinds=definite,possible --num-callers=30 /root/tests/gpu_apriltag_benchmark --input_folder=/cos-logs/second_bot/pmatch2_compressed/right --max_images=1 --iterations=1 --decimate=2
```

Both completed frame processing, then Valgrind returned 99 with identical totals:
652 bytes definitely lost in 17 blocks, 48 bytes indirectly lost in 6 blocks,
14,606 bytes possibly lost in 94 blocks, and 104 errors from 82 contexts.
The definite-loss allocation stacks originate in `libnvcucompat.so.39.2.0`
during CUDA initialization. The invalid read is in the dynamic loader's `strcmp`
while CUDA loads libraries; possible losses include CUDA, Protobuf, and Abseil
flag-registry allocations. No project-owned allocation leak stack was identified.
No new leak totals were introduced, but this is not a clean Valgrind result or a
GPU-memory sanitizer pass. Logs are `valgrind.log` and `valgrind-baseline.log`
in the same remote results directory. Valgrind timings are excluded above.

## Pre-commit review validation

The review added validation before constructor division, widened generic box
averaging and decoder indexing arithmetic, rejected nonpositive extraction
thresholds, cleared Hamming outputs for empty decodes, and made degenerate or
nonfinite refinement return a whole fallback quad. The concurrency timer now
starts and stops in barrier completion callbacks to include all worker activity.

After rebuilding and deploying, all 13 pipeline/geometry/decoder tests and all
5 extraction tests passed on dev-orin. Added coverage includes invalid arguments,
box averaging at factors 2/3/4 with padded and mapped inputs, empty decode output,
and degenerate refinement. The first 100 log22 frames at decimation 2 still
matched the archived optimized detection TSV byte-for-byte (293 detections).
This rerun measured 3.297 ms/frame for one detector. Four concurrent detectors
measured 487.28 combined FPS, or 121.82 FPS per camera, with every final-pass
output matching the sequential reference. These are additional observed timings,
not replacements for the earlier runs; device clocks and power modes were not
controlled. The Valgrind limitations above remain unresolved.
