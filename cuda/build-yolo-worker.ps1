param(
    [string]$Configuration = "CUDA",
    [string]$Platform = "x64",
    [string]$Project = "$PSScriptRoot\yolo_annotation_worker.vcxproj"
)

$ErrorActionPreference = "Stop"

$msbuild = "C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe"
if (!(Test-Path $msbuild)) {
    throw "MSBuild was not found at $msbuild"
}

$properties = @(
    "/p:Configuration=$Configuration",
    "/p:Platform=$Platform"
)

& $msbuild $Project @properties /m /v:minimal
exit $LASTEXITCODE
