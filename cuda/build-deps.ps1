function Find-FirstExistingPath {
    param([Parameter(Mandatory)][string[]]$Paths)

    foreach ($path in $Paths) {
        if (![string]::IsNullOrWhiteSpace($path) -and (Test-Path -LiteralPath $path)) {
            return (Resolve-Path -LiteralPath $path).Path
        }
    }
    return $null
}

function Get-VersionFromModuleName {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Prefix
    )

    $text = $Name -replace "^$([regex]::Escape($Prefix))", ""
    try {
        return [version]$text
    }
    catch {
        return [version]"0.0"
    }
}

function Find-OpenCVWorldLibrary {
    param([Parameter(Mandatory)][string]$OpenCVDir)

    $libDirs = @(
        (Join-Path $OpenCVDir "x64\vc18\lib"),
        (Join-Path $OpenCVDir "lib")
    ) | Where-Object { Test-Path -LiteralPath $_ -PathType Container }

    if ($libDirs.Count -eq 0) {
        return $null
    }

    return Get-ChildItem -Path $libDirs -Filter "opencv_world*.lib" -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        Select-Object -First 1
}

function Test-OpenCVInstall {
    param(
        [Parameter(Mandatory)][string]$OpenCVDir,
        [switch]$RequireCuda
    )

    if (!(Test-Path -LiteralPath (Join-Path $OpenCVDir "include\opencv2\core\version.hpp") -PathType Leaf)) {
        return $false
    }
    if (!(Find-OpenCVWorldLibrary -OpenCVDir $OpenCVDir)) {
        return $false
    }
    if ($RequireCuda) {
        $cvConfig = Join-Path $OpenCVDir "include\opencv2\cvconfig.h"
        if (!(Test-Path -LiteralPath $cvConfig -PathType Leaf)) {
            return $false
        }
        if (!(Select-String -Path $cvConfig -Pattern "#define HAVE_CUDA" -Quiet)) {
            return $false
        }
    }
    return $true
}

function Resolve-OpenCVDependency {
    param(
        [Parameter(Mandatory)][string]$RepoRoot,
        [string]$OpenCVDir,
        [switch]$RequireCuda
    )

    if (![string]::IsNullOrWhiteSpace($OpenCVDir)) {
        $resolved = [System.IO.Path]::GetFullPath($OpenCVDir)
        if (Test-OpenCVInstall -OpenCVDir $resolved -RequireCuda:$RequireCuda) {
            $worldLib = Find-OpenCVWorldLibrary -OpenCVDir $resolved
            return [pscustomobject]@{
                Dir = $resolved
                WorldLib = $worldLib.Name
                WorldDll = [System.IO.Path]::ChangeExtension($worldLib.Name, ".dll")
            }
        }
        throw "OpenCV install was not valid or did not match CUDA requirements: $OpenCVDir"
    }

    $modulesDir = Join-Path $RepoRoot "modules"
    $candidates = @(
        (Join-Path $RepoRoot "modules\opencv-5.0.0\build\cuda\install"),
        (Join-Path $RepoRoot "modules\opencv\build\install")
    )

    if (Test-Path -LiteralPath $modulesDir -PathType Container) {
        $candidates += Get-ChildItem -Path $modulesDir -Directory -Filter "opencv-*" -ErrorAction SilentlyContinue |
            Sort-Object @{ Expression = { Get-VersionFromModuleName -Name $_.Name -Prefix "opencv-" }; Descending = $true } |
            ForEach-Object {
                @(
                    (Join-Path $_.FullName "build\cuda\install"),
                    (Join-Path $_.FullName "build\install")
                )
            }
    }

    foreach ($candidate in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        if ((Test-Path -LiteralPath $candidate -PathType Container) -and
            (Test-OpenCVInstall -OpenCVDir $candidate -RequireCuda:$RequireCuda)) {
            $resolved = (Resolve-Path -LiteralPath $candidate).Path
            $worldLib = Find-OpenCVWorldLibrary -OpenCVDir $resolved
            return [pscustomobject]@{
                Dir = $resolved
                WorldLib = $worldLib.Name
                WorldDll = [System.IO.Path]::ChangeExtension($worldLib.Name, ".dll")
            }
        }
    }

    if ($RequireCuda) {
        throw "OpenCV CUDA install was not found. Set OpenCVDir to an OpenCV install folder with include\opencv2\cvconfig.h declaring HAVE_CUDA."
    }
    throw "OpenCV install was not found. Set OpenCVDir to the OpenCV install folder."
}

function Find-TensorRTLibrarySet {
    param(
        [Parameter(Mandatory)][string]$TensorRTDir,
        [switch]$RequirePlugin
    )

    $libDirs = @(
        (Join-Path $TensorRTDir "lib"),
        (Join-Path $TensorRTDir "lib\x64")
    ) | Where-Object { Test-Path -LiteralPath $_ -PathType Container }

    $coreLibs = Get-ChildItem -Path $libDirs -Filter "nvinfer_*.lib" -ErrorAction SilentlyContinue |
        ForEach-Object {
            if ($_.Name -match "^nvinfer_(\d+)\.lib$") {
                [pscustomobject]@{
                    File = $_
                    Suffix = [int]$Matches[1]
                }
            }
        } |
        Sort-Object Suffix -Descending

    foreach ($core in $coreLibs) {
        $suffix = $core.Suffix
        $onnx = Find-FirstExistingPath -Paths @(
            (Join-Path $TensorRTDir "lib\nvonnxparser_$suffix.lib"),
            (Join-Path $TensorRTDir "lib\x64\nvonnxparser_$suffix.lib")
        )
        $plugin = Find-FirstExistingPath -Paths @(
            (Join-Path $TensorRTDir "lib\nvinfer_plugin_$suffix.lib"),
            (Join-Path $TensorRTDir "lib\x64\nvinfer_plugin_$suffix.lib")
        )

        if ($onnx -and (!$RequirePlugin -or $plugin)) {
            $coreLib = "nvinfer_$suffix.lib"
            $onnxParserLib = "nvonnxparser_$suffix.lib"
            $pluginLib = if ($plugin) { "nvinfer_plugin_$suffix.lib" } else { "" }
            $libraries = @($coreLib, $onnxParserLib)
            if ($plugin) {
                $libraries += $pluginLib
            }
            return [pscustomobject]@{
                Suffix = $suffix
                Libraries = ($libraries -join ";")
                CoreLib = $coreLib
                OnnxParserLib = $onnxParserLib
                PluginLib = $pluginLib
            }
        }
    }

    return $null
}

function Test-TensorRTInstall {
    param(
        [Parameter(Mandatory)][string]$TensorRTDir,
        [switch]$RequirePlugin
    )

    if (!(Test-Path -LiteralPath (Join-Path $TensorRTDir "include\NvInfer.h") -PathType Leaf)) {
        return $false
    }
    return [bool](Find-TensorRTLibrarySet -TensorRTDir $TensorRTDir -RequirePlugin:$RequirePlugin)
}

function Resolve-TensorRTDependency {
    param(
        [Parameter(Mandatory)][string]$RepoRoot,
        [string]$TensorRTDir,
        [switch]$RequirePlugin
    )

    if (![string]::IsNullOrWhiteSpace($TensorRTDir)) {
        $resolved = [System.IO.Path]::GetFullPath($TensorRTDir)
        if (Test-TensorRTInstall -TensorRTDir $resolved -RequirePlugin:$RequirePlugin) {
            $libSet = Find-TensorRTLibrarySet -TensorRTDir $resolved -RequirePlugin:$RequirePlugin
            return [pscustomobject]@{
                Dir = $resolved
                Libraries = $libSet.Libraries
                Major = $libSet.Suffix
                CoreLib = $libSet.CoreLib
                OnnxParserLib = $libSet.OnnxParserLib
                PluginLib = $libSet.PluginLib
            }
        }
        throw "TensorRT install was not valid or did not include required libraries: $TensorRTDir"
    }

    $modulesDir = Join-Path $RepoRoot "modules"
    $candidates = @((Join-Path $RepoRoot "modules\tensorrt"))
    if (Test-Path -LiteralPath $modulesDir -PathType Container) {
        $candidates += Get-ChildItem -Path $modulesDir -Directory -Filter "TensorRT-*" -ErrorAction SilentlyContinue |
            Sort-Object @{ Expression = { Get-VersionFromModuleName -Name $_.Name -Prefix "TensorRT-" }; Descending = $true } |
            ForEach-Object FullName
    }

    foreach ($candidate in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        if ((Test-Path -LiteralPath $candidate -PathType Container) -and
            (Test-TensorRTInstall -TensorRTDir $candidate -RequirePlugin:$RequirePlugin)) {
            $resolved = (Resolve-Path -LiteralPath $candidate).Path
            $libSet = Find-TensorRTLibrarySet -TensorRTDir $resolved -RequirePlugin:$RequirePlugin
            return [pscustomobject]@{
                Dir = $resolved
                Libraries = $libSet.Libraries
                Major = $libSet.Suffix
                CoreLib = $libSet.CoreLib
                OnnxParserLib = $libSet.OnnxParserLib
                PluginLib = $libSet.PluginLib
            }
        }
    }

    throw "TensorRT headers/libraries were not found. Set TensorRTDir to the TensorRT install folder."
}
