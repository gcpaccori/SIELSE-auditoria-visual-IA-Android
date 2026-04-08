Este proyecto ya incluye los modelos convertidos desde tus .pt originales.

Archivos incluidos:
- display_detection_int8.onnx
- digit_recognition_int8.onnx
- labels_display.txt
- labels_digits.txt

Nota técnica:
- Los .pt se exportaron a ONNX y luego se cuantizaron a INT8 dinámico.
- En estos modelos la salida real es tipo Ultralytics ONNX: [1, canales, cajas]
- display: [1, 5, 2100]
- digits: [1, 15, 2100]
- El parser Android usa 4 bbox + clases, sin objectness separado.
