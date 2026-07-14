param(
    [string]$Configuration = "CUDA",
    [string]$Platform = "x64",
    [string]$Project = "$PSScriptRoot\yolo_annotation_worker.vcxproj",
    [string]$CudaVersion = "13.3",
    [string]$CudaToolkitDir = $env:CudaToolkitDir,
    [string]$TensorRTDir = $env:TensorRTDir,
    [string]$OpenCVDir = $env:OpenCVDir
)

$ErrorActionPreference = "Stop"

& "$PSScriptRoot\build-cuda.ps1" `
    -Configuration $Configuration `
    -Platform $Platform `
    -Project $Project `
    -CudaVersion $CudaVersion `
    -CudaToolkitDir $CudaToolkitDir `
    -TensorRTDir $TensorRTDir `
    -OpenCVDir $OpenCVDir

exit $LASTEXITCODE
