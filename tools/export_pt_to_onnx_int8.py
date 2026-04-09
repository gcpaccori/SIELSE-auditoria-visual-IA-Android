import argparse
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

def parse_imgsz(value: str) -> tuple[int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) == 1:
        size = int(parts[0])
        if size <= 0:
            raise ValueError("imgsz debe ser mayor que 0")
        return size, size
    if len(parts) == 2:
        height, width = int(parts[0]), int(parts[1])
        if height <= 0 or width <= 0:
            raise ValueError("alto y ancho deben ser mayores que 0")
        return height, width
    raise ValueError("Usa --imgsz N o --imgsz H,W")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imgsz",
        default="320,320",
        help="Tamaño de exportación ONNX. Formato: N o H,W (ej: 256,640)",
    )
    args = parser.parse_args()
    img_h, img_w = parse_imgsz(args.imgsz)
    imgsz_arg = [img_h, img_w]

    for pt_path, onnx_path, int8_path in MODELS:
        if not pt_path.exists():
            raise FileNotFoundError(f"Falta: {pt_path}")

        model = YOLO(str(pt_path))
        exported = model.export(format="onnx", imgsz=imgsz_arg, simplify=False, opset=12, device="cpu")
        exported_path = Path(exported)
        if exported_path != onnx_path:
            onnx_path.write_bytes(exported_path.read_bytes())

        quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
        print(f"OK: {pt_path.name} -> {int8_path.name} ({img_h}x{img_w})")

if __name__ == "__main__":
    main()
