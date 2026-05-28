param(
    [string]$Configuration = "CUDA",
    [string]$Platform = "x64",
    [string]$Project = "$PSScriptRoot\0BS_cuda.vcxproj",
    [string]$CudaVersion = "13.2",
    [string]$TensorRTDir = $env:TensorRTDir
)

$ErrorActionPreference = "Stop"

function Get-VsWherePath {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere -PathType Leaf) {
        return $vswhere
    }
    return $null
}

function Get-VisualStudioInstallRoots {
    $roots = @()
    $vswhere = Get-VsWherePath
    if ($vswhere) {
        $roots += & $vswhere -products * -property installationPath
    }

    $visualStudioRoot = Join-Path $env:ProgramFiles "Microsoft Visual Studio"
    if (Test-Path -LiteralPath $visualStudioRoot) {
        $roots += Get-ChildItem -Path $visualStudioRoot -Directory -ErrorAction SilentlyContinue |
            ForEach-Object {
                Get-ChildItem -Path $_.FullName -Directory -ErrorAction SilentlyContinue
            } |
            ForEach-Object FullName
    }

    $roots |
        Where-Object { ![string]::IsNullOrWhiteSpace($_) -and (Test-Path -LiteralPath $_) } |
        Select-Object -Unique
}

function Resolve-MSBuild {
    $cmd = Get-Command MSBuild.exe -ErrorAction SilentlyContinue
    if ($cmd -and (Test-Path -LiteralPath $cmd.Source -PathType Leaf)) {
        return $cmd.Source
    }

    $vswhere = Get-VsWherePath
    if ($vswhere) {
        $found = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -find "MSBuild\Current\Bin\MSBuild.exe" |
            Select-Object -First 1
        if ($found -and (Test-Path -LiteralPath $found -PathType Leaf)) {
            return $found
        }
    }

    foreach ($installRoot in Get-VisualStudioInstallRoots) {
        $candidate = Join-Path $installRoot "MSBuild\Current\Bin\MSBuild.exe"
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }

    throw "MSBuild.exe was not found. Open a Developer PowerShell or install Visual Studio Build Tools."
}

function Find-CudaPropsUnder {
    param([string]$Root)

    if ([string]::IsNullOrWhiteSpace($Root) -or !(Test-Path -LiteralPath $Root)) {
        return $null
    }

    $match = Get-ChildItem -Path $Root -Recurse -Filter "CUDA $CudaVersion.props" -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($match) {
        return $match.FullName
    }
    return $null
}

function Resolve-CudaProps {
    if (![string]::IsNullOrWhiteSpace($env:VCTargetsPath)) {
        $candidate = Join-Path $env:VCTargetsPath "BuildCustomizations\CUDA $CudaVersion.props"
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }

    foreach ($installRoot in Get-VisualStudioInstallRoots) {
        $candidate = Find-CudaPropsUnder -Root (Join-Path $installRoot "MSBuild\Microsoft\VC")
        if ($candidate) {
            return $candidate
        }
    }

    throw "CUDA $CudaVersion Visual Studio build customizations were not found. Install CUDA build customizations or run from a Developer PowerShell with VCTargetsPath set."
}

$msbuild = Resolve-MSBuild
$cudaProps = Resolve-CudaProps

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

if ([string]::IsNullOrWhiteSpace($TensorRTDir)) {
    throw "TensorRT headers were not found. Set TensorRTDir to the TensorRT install folder."
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
