param(
    [string]$Configuration = "CUDA",
    [string]$Platform = "x64",
    [string]$Project = "$PSScriptRoot\0BS_cuda.vcxproj",
    [string]$CudaVersion = "13.2",
    [string]$TensorRTDir = $env:TensorRTDir
)

$ErrorActionPreference = "Stop"

$msbuild = "C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe"
if (!(Test-Path $msbuild)) {
    throw "MSBuild was not found at $msbuild"
}

$cudaProps = "C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Microsoft\VC\v180\BuildCustomizations\CUDA $CudaVersion.props"
if (!(Test-Path $cudaProps)) {
    throw "CUDA $CudaVersion Visual Studio build customizations were not found at $cudaProps"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($TensorRTDir)) {
    $defaultTensorRTDir = Join-Path $repoRoot "modules\tensorrt"
    if (Test-Path (Join-Path $defaultTensorRTDir "include\NvInfer.h")) {
        $TensorRTDir = $defaultTensorRTDir
    }
    else {
        $TensorRTDir = Get-ChildItem -Path (Join-Path $repoRoot "modules") -Directory -Filter "TensorRT-*" -ErrorAction SilentlyContinue |
            Where-Object {
                (Test-Path (Join-Path $_.FullName "include\NvInfer.h")) -and
                (Test-Path (Join-Path $_.FullName "lib\nvinfer_10.lib"))
            } |
            Sort-Object Name -Descending |
            Select-Object -First 1 -ExpandProperty FullName
    }
}

$tensorRTInclude = Join-Path $TensorRTDir "include"
if (!(Test-Path $tensorRTInclude)) {
    throw "TensorRT headers were not found at $tensorRTInclude. Set TensorRTDir to the TensorRT install folder."
}

$properties = @(
    "/p:Configuration=$Configuration",
    "/p:Platform=$Platform",
    "/p:TensorRTDir=$TensorRTDir"
)

& $msbuild $Project @properties /m /v:minimal
exit $LASTEXITCODE
