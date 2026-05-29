param(
    [Parameter(Mandatory = $true)]
    [string]$Onnx,

    [string]$Engine = "",
    [string]$TensorRTDir = $env:TensorRTDir,

    [bool]$UseFp16 = $true,
    [bool]$UseFp8 = $false,

    [string]$MinShapes = "",
    [string]$OptShapes = "",
    [string]$MaxShapes = "",

    [string[]]$AdditionalArgs = @(),
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# No external dependencies: this script only shells out to TensorRT trtexec.exe.
$repoRoot = Split-Path -Parent $PSScriptRoot

function replace_extension {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Extension
    )

    return [System.IO.Path]::ChangeExtension($Path, $Extension)
}

function Resolve-ExistingFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    if (!(Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description was not found: $Path"
    }

    return (Resolve-Path -LiteralPath $Path).Path
}

function Resolve-TensorRTDir {
    param([string]$ConfiguredDir)

    if (![string]::IsNullOrWhiteSpace($ConfiguredDir)) {
        if (Test-Path -LiteralPath (Join-Path $ConfiguredDir "bin\trtexec.exe") -PathType Leaf) {
            return (Resolve-Path -LiteralPath $ConfiguredDir).Path
        }
        throw "trtexec.exe was not found under TensorRTDir: $ConfiguredDir"
    }

    $defaultTensorRTDir = Join-Path $repoRoot "modules\tensorrt"
    if (Test-Path -LiteralPath (Join-Path $defaultTensorRTDir "bin\trtexec.exe") -PathType Leaf) {
        return (Resolve-Path -LiteralPath $defaultTensorRTDir).Path
    }

    $autoDetected = Get-ChildItem -Path (Join-Path $repoRoot "modules") -Directory -Filter "TensorRT-*" -ErrorAction SilentlyContinue |
        Where-Object {
            (Test-Path -LiteralPath (Join-Path $_.FullName "bin\trtexec.exe") -PathType Leaf) -and
            (Test-Path -LiteralPath (Join-Path $_.FullName "include\NvInfer.h") -PathType Leaf)
        } |
        Sort-Object Name -Descending |
        Select-Object -First 1 -ExpandProperty FullName

    if ($autoDetected) {
        return $autoDetected
    }

    $pathCommand = Get-Command "trtexec.exe" -ErrorAction SilentlyContinue
    if ($pathCommand) {
        return Split-Path -Parent (Split-Path -Parent $pathCommand.Source)
    }

    throw "TensorRT trtexec.exe was not found. Set TensorRTDir to the TensorRT install folder."
}

$onnxPath = Resolve-ExistingFile -Path $Onnx -Description "ONNX model"
if ([System.IO.Path]::GetExtension($onnxPath).ToLowerInvariant() -ne ".onnx") {
    throw "Input model must be an .onnx file: $onnxPath"
}

if ([string]::IsNullOrWhiteSpace($Engine)) {
    $Engine = replace_extension -Path $onnxPath -Extension ".engine"
}

$enginePath = $Engine
if (![System.IO.Path]::IsPathRooted($enginePath)) {
    $enginePath = Join-Path (Get-Location).Path $enginePath
}
$enginePath = [System.IO.Path]::GetFullPath($enginePath)

if ((Test-Path -LiteralPath $enginePath -PathType Leaf) -and !$Force) {
    Write-Host "Engine already exists: $enginePath"
    Write-Host "Use -Force to overwrite it."
    exit 0
}

$engineDir = Split-Path -Parent $enginePath
if (![string]::IsNullOrWhiteSpace($engineDir) -and !(Test-Path -LiteralPath $engineDir -PathType Container)) {
    New-Item -ItemType Directory -Path $engineDir | Out-Null
}

$resolvedTensorRTDir = Resolve-TensorRTDir -ConfiguredDir $TensorRTDir
$trtexec = Join-Path $resolvedTensorRTDir "bin\trtexec.exe"
if (!(Test-Path -LiteralPath $trtexec -PathType Leaf)) {
    $pathCommand = Get-Command "trtexec.exe" -ErrorAction SilentlyContinue
    if (!$pathCommand) {
        throw "trtexec.exe was not found in TensorRT bin or PATH."
    }
    $trtexec = $pathCommand.Source
}

$trtBin = Split-Path -Parent $trtexec
$env:PATH = "$trtBin;$env:PATH"

$trtArgs = @(
    "--onnx=$onnxPath",
    "--saveEngine=$enginePath"
)

if ($UseFp16) {
    $trtArgs += "--fp16"
}
if ($UseFp8) {
    $trtArgs += "--fp8"
}
if (![string]::IsNullOrWhiteSpace($MinShapes)) {
    $trtArgs += "--minShapes=$MinShapes"
}
if (![string]::IsNullOrWhiteSpace($OptShapes)) {
    $trtArgs += "--optShapes=$OptShapes"
}
if (![string]::IsNullOrWhiteSpace($MaxShapes)) {
    $trtArgs += "--maxShapes=$MaxShapes"
}
if ($AdditionalArgs.Count -gt 0) {
    $trtArgs += $AdditionalArgs
}

Write-Host "TensorRT: $resolvedTensorRTDir"
Write-Host "ONNX:     $onnxPath"
Write-Host "Engine:   $enginePath"

if ($DryRun) {
    Write-Host "Dry run command:"
    Write-Host "`"$trtexec`" $($trtArgs -join ' ')"
    exit 0
}

& $trtexec @trtArgs
exit $LASTEXITCODE
