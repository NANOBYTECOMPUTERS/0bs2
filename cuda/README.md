# 0BS CUDA Build

This folder contains the separate CUDA/TensorRT build surface for 0BS.
The root `0BS_box_2.vcxproj` remains the DML build and should stay free of
`USE_CUDA`, `CudaCompile`, TensorRT sources, and CUDA-only project imports.

Build from this folder with:

```powershell
.\build-cuda.ps1
```

If local PowerShell policy blocks scripts, use a process-local bypass:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build-cuda.ps1
```

The script builds `0BS_cuda.vcxproj` with the `CUDA|x64` configuration and
writes output to `..\x64\CUDA\`.

TensorRT is expected to be provided separately. Set `TensorRTDir` to the
TensorRT install folder if it is not available under `..\modules\tensorrt` or
an auto-detected `..\modules\TensorRT-*` folder with `include\NvInfer.h` and
matching `lib\nvinfer_*.lib` / `lib\nvonnxparser_*.lib` import libraries.
The repository default is `..\modules\TensorRT-11.1.0.106`.

OpenCV is resolved from `..\modules\opencv-5.0.0\build\cuda\install` by
default and must be a CUDA-enabled OpenCV install (`include\opencv2\cvconfig.h`
with `HAVE_CUDA`). Override it with `OpenCVDir` or `-OpenCVDir` if needed.
The CUDA targets link `opencv_world500.lib` by default and copy the OpenCV
runtime DLLs from the selected install.
The CUDA project defaults to CUDA Toolkit 13.3 and imports the Visual Studio
CUDA build customization files directly from the CUDA-enabled project only.
Pass `-CudaVersion` and/or `-CudaToolkitDir` when testing another installed
toolkit.
Post-build copies use the existing prebuilt OpenCV DLL and TensorRT DLLs; the
CUDA build does not rebuild OpenCV.

Export an ONNX model to a TensorRT engine with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\export-engine.ps1 -Onnx ..\models\yolo26.onnx
```

The same detector path is `models\yolo26.onnx` when referenced from the
repository root.

By default the script writes a sibling `.engine` file next to the ONNX model.
Preview the generated `trtexec.exe` command without writing an engine:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\export-engine.ps1 -Onnx ..\models\yolo26.onnx -DryRun
```

For dynamic input models, pass explicit TensorRT shapes, for example:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\export-engine.ps1 -Onnx ..\models\model.onnx -MinShapes "images:1x3x160x160" -OptShapes "images:1x3x320x320" -MaxShapes "images:1x3x640x640"
```

NanoSim is only a vague reference harness for this CUDA work. Acceptance and
performance comparisons should use the main dynamic test environment, with
NanoSim treated as a quick smoke/reference check at most.
