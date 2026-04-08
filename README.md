# OfflineMeterReader (Android, Kotlin, CameraX, ONNX Runtime)

Proyecto Android offline para lectura de medidores, completado sobre tu proyecto previo y alineado con la lógica real de `electro-api`.

## Qué se hizo ahora

- se tomaron tus modelos reales `display_detection.pt` y `digit_recognition.pt`
- se exportaron a **ONNX**
- se generaron versiones **INT8 dinámicas** para bajar tamaño
- se reemplazó la capa TFLite por **ONNX Runtime Android**
- se ajustó el parser a la salida real de tus modelos
- se mantuvo tu flujo de negocio y heurísticas de `app.py`

## Modelos ya incluidos

En `app/src/main/assets` ya vienen:

- `display_detection_int8.onnx`
- `digit_recognition_int8.onnx`
- `labels_display.txt`
- `labels_digits.txt`

## Formato real detectado en tus exports

### Display
- salida: `[1, 5, 2100]`
- interpretación: `cx, cy, w, h, clase_display`

### Dígitos
- salida: `[1, 15, 2100]`
- interpretación: `cx, cy, w, h, 11 clases`
- clases: `0..9, dot`

En otras palabras, para estos ONNX exportados el parser se maneja como:

`4 bbox + clases`

y no como `4 bbox + objectness + clases`.

## Lógica heredada de tu `app.py`

- display: `conf=0.25`, `iou=0.50`
- dígitos: `conf=0.30`, `iou=0.20`
- vertical: rota izquierda y derecha, procesa ambas y elige por score `count * 10 + avg_conf`
- cuadrado: escalado proporcional
- horizontal: resize a `400x150`
- dígitos superpuestos: gana el de mayor confianza si están a menos de `20 px` en X
- múltiples puntos: se conserva solo el último a la derecha
- punto decimal reconocido como `dot`, `point`, `.`, o `10`

## Dependencias clave

- CameraX
- ONNX Runtime Android
- Kotlin Coroutines

## Estado del proyecto y Próximos Pasos

La app está estructuralmente completa para ejecutar inferencia local.

### Problemas conocidos / Optimización en curso:
- **Detección en ángulos**: Al inclinar el dispositivo (orientación horizontal/perpendicular al display), el cuadro de detección tiende a deformarse o estirarse, dificultando la detección. Se está trabajando en mejorar la normalización del aspect ratio antes de la inferencia y en la lógica de rotación de los frames de la cámara.

## Limitación honesta

No pude generar TFLite dentro de este entorno porque la exportación de Ultralytics a TFLite exige TensorFlow y aquí no estaba disponible. Por eso la app quedó terminada con ONNX Runtime Android, que sí soporta ejecución local de modelos ONNX y encaja bien con tus `.pt` exportados.

## Cómo abrir

1. abre la carpeta en Android Studio
2. sincroniza Gradle
3. ejecuta en dispositivo físico Android
4. concede permiso de cámara

## Archivos principales

- `MainActivity.kt`: cámara, estados, geometría y pipeline
- `OnnxYoloDetector.kt`: inferencia ONNX + parser YOLO exportado
- `ReadingLogic.kt`: colisión y puntos
- `OverlayView.kt`: overlay
- `BitmapUtils.kt`: rotaciones, resize, letterbox
- `tools/export_pt_to_onnx_int8.py`: script de conversión reproducible

## Nota sobre Gradle Wrapper

Se incluyen `gradlew`, `gradlew.bat` y `gradle-wrapper.properties`, pero el `gradle-wrapper.jar` no fue generado aquí. Si Android Studio lo reclama, basta regenerarlo localmente o crear un proyecto vacío y copiar encima el módulo `app`.
