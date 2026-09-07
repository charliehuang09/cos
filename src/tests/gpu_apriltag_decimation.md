# AprilTag image decimation experiment

Measured on dev-orin on 2026-09-07 using the existing cross-compiled build.
The benchmark adds `--decimate=2`: OpenCV INTER_AREA reduction from 1280x800
 to 640x400 before running the complete GPU detector. The default remains 1.
Exported corners are mapped back to original pixel-center coordinates.

Each image is loaded once, both detectors are warmed on it, then each runs
three timed repetitions in alternating order. Times include resize and detection,
but exclude JPEG decoding, detector construction, warmup and TSV export.
Tag counts are counted once per image, restricted to the existing target IDs 1–32.

| Dataset | Frames | Full tags | Decimated tags | Baseline IDs retained | Full ms/frame | Decimated ms/frame | Speedup | Time reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev-orin/log2 (all frames) | 1724 | 1538 | 1557 | 1532/1538 (99.6%) | 7.25969 | 4.08444 | 1.7774x | 43.7% |
| second_bot/pmatch2_compressed/right (every fifth filename) | 330 | 524 | 451 | 451/524 (86.1%) | 23.6932 | 8.02533 | 2.9523x | 66.1% |

Per-image ID multisets match on 1693/1724 and 291/330 frames, respectively.
Counts are similar on log2 (+1.2%), but the second sample loses 13.9% of
baseline detections. This experiment does not establish that 2x decimation
preserves detection quality generally. Baseline ID agreement is not ground-truth
accuracy or a geometric corner-accuracy test. Timings describe these warmed
recording replays on the current device configuration.

Build and deployment succeeded with:

```sh
cmake --build build --target dev-orin -j 4
```

The current deployment target places test executables in `/root/tests`.
Both measured commands completed normally on the Orin:

```sh
/root/tests/gpu_apriltag_benchmark --input_folder=/cos-logs/dev-orin/log2 --decimate=2 --iterations=3
/root/tests/gpu_apriltag_benchmark --input_folder=/cos-logs/second_bot/pmatch2_compressed/right --decimate=2 --sample_step=5 --iterations=3
```

An initial dev-orin/log1 sample contained no baseline detections and was excluded
from the detection-quality comparison. Uncompressed second_bot folders contained
zero-byte JPEG files; the successful comparison uses pmatch2_compressed.

A separate one-image Valgrind run completed frame processing but reported 105
errors from 83 contexts, 652 bytes definitely lost and 16,086 bytes possibly lost
(48 bytes indirectly lost). The first definite leak is an allocation in
libnvcucompat during cuInit, reached through cudaMalloc in GpuSegmentSorter's
constructor and GPUApriltagDetector's constructor. An invalid read also appears
in the dynamic loader during CUDA initialization. This is not a clean memory
check. The command used --error-exitcode=99; its actual exit status was not
captured because the shell subsequently printed the log. Valgrind timings were
excluded from the performance results.

The `gpu_apriltag` example also accepts `--decimate` (default 1):

```sh
/root/examples/gpu_apriltag --image_path=/root/apriltag.png --decimate=2
```

Input dimensions must be divisible by `4 * decimate`. The annotated image stays
at the original resolution, with detection corners mapped back accordingly;
intermediate debug images use the reduced resolution. Its reported timing
includes resizing and corner-coordinate conversion.
