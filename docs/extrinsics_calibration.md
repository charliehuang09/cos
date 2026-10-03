# Reusable extrinsics calibration

Build and deploy with `./scripts/build.sh` and `./scripts/deploy.sh`. CMake fetches
Ceres 2.2.0 with a pinned checksum and builds it statically with its bundled
minimal logger. No Ceres installation on the Orin is required.
The build script defaults to two jobs; set `CMAKE_BUILD_PARALLEL_LEVEL` to
override that limit.
For simultaneous worktrees, deploy to a separate device directory:

```sh
COS_DEV_ORIN_DEPLOY_ROOT=/root/extrinsics-calibration ./scripts/deploy.sh
```

The default destination is `/root`. A separate destination receives the binaries,
libraries, and camera constants without updating the shared systemd service.

Clean macOS metadata sidecars from extracted logs before replay:

```sh
python3 scripts/remove_appledouble.py /path/to/extracted/log
```

This recursively deletes files named `._*`. Replay contains no special handling
for these metadata files.

On dev-orin:

```sh
/root/tests/extrinsics_calibration_test
/root/tools/extrinsics_calibrate \
  --replay_directory=/cos-logs/second_bot/chezychamps \
  --camera_config_directory=/root/constants/second_bot \
  --output_directory=/root/extrinsics-calibration-candidate
```

The output directory must be empty. The executable writes candidate camera
JSONs, `report.json`, and `observations.csv`; the input JSONs stay intact.
Intrinsics and the existing AprilTag field layout stay fixed. Candidates are
not installed as active robot configurations.

The default identities are `second_bot_front,second_bot_left,second_bot_right`,
with `front_camera.json,left_camera.json,right_camera.json`. Replay subdirectories
can use those identities or the existing `front,left,right` names. For another
configuration, supply comma-separated `--camera_names` and `--config_files` in
matching order. `--anchor_camera` defaults to `second_bot_front` and must name
one of the supplied cameras. `--reject_far_tags=false` disables the existing
localization sanity checks for diagnostic runs.

`calibration/extrinsics_replay.h` provides enumeration, injectable detection,
observation collection, matching, and deterministic splitting. Enumeration
accepts `.jpg` and `.jpeg` case insensitively; filenames must be nonnegative
decimal capture times in seconds with at most nine fractional digits. Invalid
JPEG filenames or decode failures abort the run instead of skipping a frame.
The existing NVIDIA hardware decoder decodes each JPEG, then the existing VPI
AprilTag detector runs synchronously on its luminance. PnP and unambiguous selection complete before advancing to
the next frame. Selection uses the previous valid pose across the whole replay,
including frames that will not later match another camera. Each observation
retains the selected candidate's camera and original robot poses.

Matching happens after replay completes. It sorts valid observations by original
capture time, seeds from the earliest available frame, and takes the earliest
frame from each different camera with a total span strictly below 10 ms. A camera
appears at most once per group. Frames are consumed only by successful groups.
`--reuse_frames=true` enables overlapping groups. Every fifth chronological group
is held out; overlapping groups sharing any frame are assigned together, preventing
training/validation frame leakage. With fewer than five independent groups there
is no held-out portion; its report contains zero pairs. All cameras must have a
training observation path to the anchor, or calibration fails.

`calibration/extrinsics_solver.h` provides the joint Ceres solve and pair metrics.
The internal extrinsics are `T_C<-R`, so a fixed selected field camera pose `Y`
predicts robot pose `Y E`. Every camera pair contributes translation disagreement
at a 0.10 m scale and the shortest SO(3) rotation logarithm at a 10 degree scale.
Residuals are divided by the square root of the training pair count. Every
optimized camera receives a rotation correction residual centered on its input
rotation at a 5 degree scale. There is no translation prior or robust loss. The
anchor's complete transform is fixed exactly. JSON output inverts `E` back to the
existing camera position in robot coordinates, with Euler rotations in degrees.

The report gives processed JPEGs, selected observations, groups and pairs,
training and held-out pair errors before and after optimization, and camera
translation changes in robot coordinates and rotation angle changes. RMS
translation and rotation errors are unweighted physical quantities; the normalized
half mean squared pair error follows the objective's scales and excludes the
rotation penalty. `observations.csv` records capture nanoseconds and both poses
in WPILib coordinates. Rotation priors can leave a nonzero pair error even for
noise-free observations when the input rotations differ from the true rotations.

## Chezychamps validation, 2026-10-01

`./scripts/build.sh` and `./scripts/deploy.sh` succeeded using the two-job default.
Deployment and testing used `/root/extrinsics-calibration-20261001` on `root@dev-orin`
to avoid another workspace's deployment to `/root`. The committed left camera
configuration needed its Unicode minus replaced with an ASCII minus at `k2`;
its numeric intrinsics value was preserved.

All 37 calibration, hardware decoder, single-tag rotation, logging, geometry,
and solver sanity unit tests passed. The lossless hardware replay completed
with exit code 0 and Ceres convergence in four iterations:

| Replay statistic | Count |
| --- | ---: |
| JPEGs processed | 52,933 |
| Valid selected PnP observations | 16,391 |
| Matched groups | 5,411 |
| Matched pairs | 9,679 |
| Training groups / pairs | 4,329 / 7,737 |
| Held-out groups / pairs | 1,082 / 1,942 |

| Pair metric | Initial | Calibrated |
| --- | ---: | ---: |
| Training translation RMS | 0.142814 m | 0.135823 m |
| Training rotation RMS | 6.341265° | 5.605175° |
| Held-out translation RMS | 0.126064 m | 0.118185 m |
| Held-out rotation RMS | 6.252480° | 5.505518° |
| Held-out normalized half mean squared error | 0.990080 | 0.849940 |

The held-out scaled pair error decreased by 14.15%. Left camera position changed
by 0.02810 m and rotation by 0.70821°; right camera position changed by 0.04068 m
and rotation by 0.71027°. The front candidate JSON is exactly equal to its input
JSON, and all three cameras' intrinsics are unchanged.

Candidate configurations and the observation CSV remain on the device under
`/root/extrinsics-calibration-20261001/candidate`. A local copy, the report,
input manifest, and build/deployment/unit-test logs are in
`build/extrinsics-validation`. The active input configurations were not replaced
by the candidates.

Durable copies of the original and calibrated camera JSONs and the calibration
report are now in `calibration-results/chezychamps/`, outside `build/`.
`scripts/localization_comparison.py --detach` runs an unattended five-run
original-versus-calibrated localization comparison and writes translation and
yaw disagreement statistics, raw CSVs, and WPILOGs there. See
[`calibration-results/chezychamps/README.md`](../calibration-results/chezychamps/README.md)
for the metric definitions and automation outputs.

The existing `unambiguous_solver_node_test` also completed the full paced log
replay with exit code 0 and no crashes, using the isolated input configurations.
Its device log is retained as `build/extrinsics-validation/unambiguous-replay.log`.
