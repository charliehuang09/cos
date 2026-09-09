# AprilTag image decimation experiment

Measured on dev-orin on 2026-09-07 using the existing cross-compiled build.
`GPUApriltagDetector` natively supports `--decimate=2` (and arbitrary integer decimation factors):
GPU-accelerated downsampling runs before segmentation, connected-component labeling,
boundary extraction, and quad extraction at the reduced resolution (`width / decimate`, `height / decimate`).
Candidate quads are scaled to the original resolution, where tag decoding and subpixel corner refinement
operate directly on the full-resolution image. A GPU decoder spatial-offset retry and near-miss quad refinement
rescue tags whose downsampled corners suffer quantization error.

Each image is loaded once, both detectors are warmed on it, then each runs
three timed repetitions in alternating order. Times include decimation and detection,
but exclude JPEG decoding, detector construction, warmup and TSV export.
Tag counts are counted once per image, restricted to the existing target IDs 1–32.

| Dataset | Frames | Full tags | Decimated tags | Baseline IDs retained | Full ms/frame | Decimated ms/frame | Speedup | Time reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev-orin/log2 (all frames) | 1724 | 1538 | 1557 | 1532/1538 (99.6%) | 7.25969 | 4.08444 | 1.7774x | 43.7% |
| second_bot/pmatch2_compressed/right (every fifth filename) | 330 | 533 | 472 | 471/533 (88.4%) | 24.3412 | 10.5587 | 2.3053x | 56.6% |
| second_bot/log22/right (first 100 frames) | 100 | 300 | 293 | 293/300 (97.7%) | 12.5727 | 6.63602 | 1.8946x | 47.2% |

Per-image ID multisets match on 94/100 frames on second_bot/log22/right (retaining 97.7% of baseline tags).
Baseline ID agreement describes these warmed recording replays on the current device configuration.

Build and deployment succeeded with:

```sh
cmake --build build --target dev-orin -j 4
```

The current deployment target places test executables in `/root/tests`.
Measured commands completed normally on the Orin:

```sh
/root/tests/gpu_apriltag_benchmark --input_folder=/cos-logs/dev-orin/log2 --decimate=2 --iterations=3
/root/tests/gpu_apriltag_benchmark --input_folder=/cos-logs/second_bot/pmatch2_compressed/right --decimate=2 --sample_step=5 --iterations=3
/root/tests/gpu_apriltag_benchmark --input_folder=/cos-logs/second_bot/log22/right --decimate=2 --max_images=100
```

An initial dev-orin/log1 sample contained no baseline detections and was excluded
from the detection-quality comparison. Uncompressed second_bot folders contained
zero-byte JPEG files; the successful comparison uses pmatch2_compressed.

A separate one-image Valgrind run completed frame processing but reported 23
errors from 1 context (suppressed: 0 from 0). An invalid read appears in the
dynamic loader during CUDA initialization. Valgrind timings were excluded from
the performance results.

The `gpu_apriltag` example also accepts `--decimate` (default 1):

```sh
/root/examples/gpu_apriltag --image_path=/root/apriltag.png --decimate=2
```

Input dimensions must be divisible by `4 * decimate`. The detector image runs at
the original resolution and outputs corners in original pixel coordinates.
