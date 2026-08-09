#!/usr/bin/env bash
# Export a grayscale Ultralytics detector with NMS embedded in ONNX, then build
# a TensorRT engine. Run this script on the Jetson Orin.
#
# Usage: export_yolo_engine.sh MODEL.pt [OUTPUT.engine]

set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 MODEL.pt [OUTPUT.engine]" >&2
  exit 2
fi

MODEL_PATH="$1"
MODEL_DIR="$(cd -- "$(dirname -- "${MODEL_PATH}")" 2>/dev/null && pwd)" || {
  echo "Model directory not found: $(dirname -- "${MODEL_PATH}")" >&2
  exit 1
}
MODEL_PATH="${MODEL_DIR}/$(basename -- "${MODEL_PATH}")"
MODEL_NAME="$(basename -- "${MODEL_PATH}")"
MODEL_STEM="${MODEL_NAME%.*}"
ENGINE_PATH="${2:-${MODEL_DIR}/${MODEL_STEM}_nms_gray.engine}"
ONNX_PATH="${ONNX_PATH:-${ENGINE_PATH%.*}.onnx}"
IMAGE_SIZE="${IMAGE_SIZE:-640}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.25}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.45}"
MAX_DETECTIONS="${MAX_DETECTIONS:-300}"
ONNX_OPSET="${ONNX_OPSET:-17}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model not found: ${MODEL_PATH}" >&2
  exit 1
fi

mkdir -p -- "$(dirname -- "${ONNX_PATH}")" "$(dirname -- "${ENGINE_PATH}")"

if command -v trtexec >/dev/null 2>&1; then
  TRTEXEC="$(command -v trtexec)"
elif [[ -x /usr/src/tensorrt/bin/trtexec ]]; then
  TRTEXEC=/usr/src/tensorrt/bin/trtexec
else
  echo "trtexec was not found in PATH or /usr/src/tensorrt/bin." >&2
  exit 1
fi

echo "Exporting ${MODEL_PATH} to ${ONNX_PATH}"
python3 - "${MODEL_PATH}" "${ONNX_PATH}" "${IMAGE_SIZE}" \
  "${CONF_THRESHOLD}" "${IOU_THRESHOLD}" "${MAX_DETECTIONS}" \
  "${ONNX_OPSET}" <<'PY'
import shutil
import sys
from pathlib import Path

try:
    import onnx
    import torch
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        f"Missing Python dependency: {exc.name}. Install ultralytics and onnx "
        "in the Orin-side Python environment."
    ) from exc

model_path = Path(sys.argv[1]).resolve()
onnx_path = Path(sys.argv[2]).resolve()
image_size = int(sys.argv[3])
conf = float(sys.argv[4])
iou = float(sys.argv[5])
max_det = int(sys.argv[6])
opset = int(sys.argv[7])

if image_size <= 0 or max_det <= 0:
    raise SystemExit("IMAGE_SIZE and MAX_DETECTIONS must be positive integers.")
if not 0.0 <= conf <= 1.0 or not 0.0 <= iou <= 1.0:
    raise SystemExit("CONF_THRESHOLD and IOU_THRESHOLD must be between 0 and 1.")

model = YOLO(str(model_path))
first_conv = next(
    (module for module in model.model.modules() if isinstance(module, torch.nn.Conv2d)),
    None,
)
if first_conv is None:
    raise SystemExit("The checkpoint contains no Conv2d layer.")
if first_conv.in_channels != 1:
    raise SystemExit(
        "Refusing to export a non-grayscale checkpoint: first convolution has "
        f"{first_conv.in_channels} input channels, expected 1."
    )

exported_path = Path(
    model.export(
        format="onnx",
        imgsz=image_size,
        batch=1,
        dynamic=False,
        nms=True,
        conf=conf,
        iou=iou,
        max_det=max_det,
        opset=opset,
        simplify=False,
        device="cpu",
    )
).resolve()

if exported_path != onnx_path:
    shutil.move(str(exported_path), str(onnx_path))

graph = onnx.load(str(onnx_path), load_external_data=False)
input_dims = graph.graph.input[0].type.tensor_type.shape.dim
input_channels = input_dims[1].dim_value if len(input_dims) >= 2 else 0
if input_channels != 1:
    raise SystemExit(
        f"Exported ONNX input is not grayscale (channel dimension={input_channels})."
    )
if not any(node.op_type == "NonMaxSuppression" for node in graph.graph.node):
    raise SystemExit("Exported ONNX graph does not contain NonMaxSuppression.")

print(f"Verified grayscale input and embedded NMS in {onnx_path}")
PY

echo "Building FP16 TensorRT engine ${ENGINE_PATH}"
"${TRTEXEC}" \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${ENGINE_PATH}" \
  --fp16 \
  --buildOnly

if [[ ! -s "${ENGINE_PATH}" ]]; then
  echo "trtexec completed without producing a non-empty engine: ${ENGINE_PATH}" >&2
  exit 1
fi

echo "Created ${ENGINE_PATH}"
