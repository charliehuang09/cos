To calibrate intrinsics from recorded frames, first export ChArUco detections
for every image in the directory, then randomly sample usable detections.
On the device (use `build/tools/` for local build paths):

```sh
./tools/calib_helper --camera_folder=/path/to/frames --detections_output_path=detections.json
./tools/intrinsics_calibrate_disk --detections_path=detections.json --max_detections=100 --intrinsics_output_path=intrinsics.json
```

The helper processes JPEG, PNG, BMP, and TIFF files one frame at a time and
writes all results, including frames without enough corners, directly to JSON.
Frames must have matching dimensions. The JSON records image dimensions and
an array of detections containing filenames, ChArUco corners and IDs, image
points, and object points. The helper does not run calibration.
`intrinsics_calibrate_disk` samples without replacement from usable detections;
`--max_detections=0` uses all usable results. The saved JSON can be reused
without decoding frames or running detection again. For live calibration, run
`intrinsics_calibrate --config_path=/path/to/config.json`,
press Enter to capture each frame, and type `q` then Enter to calibrate and quit.

```sh
python3 calibration_sampling_test.py --detections=detections.json --output-dir=sampling-results --runs=100 --frames=50 --min-gap-seconds=0.5 --seed=20260927 --jobs=4
```

The script is in `src/tests/calibration_sampling_test.py`. It rejects random
frame candidates closer than the minimum time gap, repeats the first seeded
calibration twice, and saves every sample and result. `summary.json` reports
means and sample standard deviations (N-1) for all intrinsics and reprojection
error. Timestamp filename stems must be in seconds.
