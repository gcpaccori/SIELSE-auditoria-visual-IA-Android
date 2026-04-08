from pathlib import Path
from onnxruntime.quantization import QuantType, quantize_dynamic
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "source_models"
OUT = ROOT / "app" / "src" / "main" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    (SRC / "display_detection.pt", OUT / "display_detection.onnx", OUT / "display_detection_int8.onnx"),
    (SRC / "digit_recognition.pt", OUT / "digit_recognition.onnx", OUT / "digit_recognition_int8.onnx"),
]

for pt_path, onnx_path, int8_path in MODELS:
    if not pt_path.exists():
        raise FileNotFoundError(f"Falta: {pt_path}")

    model = YOLO(str(pt_path))
    exported = model.export(format="onnx", imgsz=320, simplify=False, opset=12, device="cpu")
    exported_path = Path(exported)
    if exported_path != onnx_path:
        onnx_path.write_bytes(exported_path.read_bytes())

    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
    print(f"OK: {pt_path.name} -> {int8_path.name}")
