#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <exception>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <optional>
#include <deque>
#include <random>
#include <array>
#include <cwchar>
#include <memory>

#include <opencv2/core/utils/logger.hpp>

#include "capture.h"
#include "mouse.h"
#include "0BS_box_2.h"
#include "keyboard_listener.h"
#include "overlay.h"
#include "Game_overlay.h"
#include "other_tools.h"
#include "virtual_camera.h"
#include "mem/gpu_resource_manager.h"
#include "mem/cpu_affinity_manager.h"
#include "runtime/thread_loops.h"
#include "runtime/RuntimeSupervisor.h"
#include "diagnostics/Logger.h"

#ifdef USE_CUDA
#include "tensorrt/nvinf.h"
#endif

std::condition_variable frameCV;
std::atomic<bool> shouldExit(false);
std::atomic<bool> aiming(false);
std::atomic<bool> detectionPaused(false);
std::recursive_mutex configMutex;

#ifdef USE_CUDA
TrtDetector trt_detector;
#endif

DirectMLDetector* dml_detector = nullptr;
MouseThread* globalMouseThread = nullptr;
Config config;
diagnostics::Logger appLogger;


GhubMouse* gHub = nullptr;
RzctlMouse* razerControl = nullptr;
Arduino* arduinoSerial = nullptr;
KmboxNetConnection* kmboxNetSerial = nullptr;
KmboxAConnection* kmboxASerial = nullptr;
MakcuConnection* makcuSerial = nullptr;
std::unique_ptr<IMouseInput> activeMouseInputOwner;

std::atomic<bool> detection_resolution_changed(false);
std::atomic<unsigned long long> detection_resolution_generation(0);
std::atomic<bool> capture_method_changed(false);
std::atomic<bool> capture_cursor_changed(false);
std::atomic<bool> capture_borders_changed(false);
std::atomic<bool> capture_fps_changed(false);
std::atomic<bool> capture_window_changed(false);
std::atomic<bool> detector_model_changed(false);
std::atomic<bool> show_window_changed(false);
std::atomic<bool> input_method_changed(false);

std::atomic<bool> zooming(false);
std::atomic<bool> shooting(false);


std::string g_iconLastError;

static int FatalExit(const std::string& message)
{
    std::cerr << message << std::endl;
    std::cout << "Press Enter to exit...";
    std::cin.get();
    return -1;
}

static void HandleThreadCrash(const char* name, const std::exception* ex)
{
    std::cerr << "[Thread] " << name << " crashed: "
              << (ex ? ex->what() : "unknown exception") << std::endl;
    appLogger.log(std::string("[Thread] ") + name + " crashed: " + (ex ? ex->what() : "unknown exception"));
    shouldExit = true;
    gameOverlayShouldExit.store(true);
    if (dml_detector)
    {
        dml_detector->shouldExit = true;
        dml_detector->inferenceCV.notify_all();
    }
    detectionBuffer.cv.notify_all();
}

void createInputDevices()
{
    if (globalMouseThread)
        globalMouseThread->setMouseInput(nullptr);

    activeMouseInputOwner.reset();
    gHub = nullptr;
    razerControl = nullptr;
    arduinoSerial = nullptr;
    kmboxNetSerial = nullptr;
    kmboxASerial = nullptr;
    makcuSerial = nullptr;

    activeMouseInputOwner = CreateMouseInputDevice(config);

    arduinoSerial = activeMouseInputOwner->arduino();
    gHub = activeMouseInputOwner->ghub();
    razerControl = activeMouseInputOwner->razer();
    kmboxNetSerial = activeMouseInputOwner->kmboxNet();
    kmboxASerial = activeMouseInputOwner->kmboxA();
    makcuSerial = activeMouseInputOwner->makcu();

    std::string message = std::string("[Mouse] Using ") + activeMouseInputOwner->name() + " input.";
    if (!activeMouseInputOwner->isOpen())
        message += " Device not connected; input disabled until the selected method is available.";
    std::cout << message << std::endl;
    appLogger.log(message);
}

void assignInputDevices()
{
    if (globalMouseThread)
    {
        globalMouseThread->setMouseInput(activeMouseInputOwner.get());
    }
}

bool SaveRuntimeConfig(const std::string& filename)
{
    std::lock_guard<std::recursive_mutex> lock(configMutex);
    return config.saveConfig(filename);
}

static void RefreshRuntimeAfterConfigLoadUnlocked(const Config& previousConfig)
{
    if (previousConfig.detection_resolution != config.detection_resolution)
    {
        detection_resolution_changed.store(true);
        detection_resolution_generation.fetch_add(1, std::memory_order_relaxed);
        detector_model_changed.store(true);
    }

    if (previousConfig.capture_fps != config.capture_fps)
        capture_fps_changed.store(true);

    if (previousConfig.capture_method != config.capture_method)
        capture_method_changed.store(true);
    if (previousConfig.capture_cursor != config.capture_cursor)
        capture_cursor_changed.store(true);
    if (previousConfig.capture_borders != config.capture_borders)
        capture_borders_changed.store(true);
    if (previousConfig.capture_target != config.capture_target ||
        previousConfig.capture_window_title != config.capture_window_title ||
        previousConfig.monitor_idx != config.monitor_idx ||
        previousConfig.virtual_camera_name != config.virtual_camera_name ||
        previousConfig.virtual_camera_width != config.virtual_camera_width ||
        previousConfig.virtual_camera_heigth != config.virtual_camera_heigth ||
        previousConfig.udp_ip != config.udp_ip ||
        previousConfig.udp_port != config.udp_port)
    {
        capture_window_changed.store(true);
    }

    if (previousConfig.ai_model != config.ai_model ||
        previousConfig.backend != config.backend ||
        previousConfig.dml_device_id != config.dml_device_id ||
        previousConfig.confidence_threshold != config.confidence_threshold ||
        previousConfig.nms_threshold != config.nms_threshold ||
        previousConfig.max_detections != config.max_detections ||
        previousConfig.fixed_input_size != config.fixed_input_size)
    {
        detector_model_changed.store(true);
    }

    if (previousConfig.input_method != config.input_method ||
        previousConfig.arduino_port != config.arduino_port ||
        previousConfig.arduino_baudrate != config.arduino_baudrate ||
        previousConfig.arduino_16_bit_mouse != config.arduino_16_bit_mouse ||
        previousConfig.arduino_enable_keys != config.arduino_enable_keys ||
        previousConfig.teensy_hid_serial != config.teensy_hid_serial ||
        previousConfig.teensy_hid_vid_filter != config.teensy_hid_vid_filter ||
        previousConfig.teensy_hid_pid_filter != config.teensy_hid_pid_filter ||
        previousConfig.teensy_hid_usage_page != config.teensy_hid_usage_page ||
        previousConfig.teensy_hid_usage_id != config.teensy_hid_usage_id ||
        previousConfig.teensy_hid_open_index != config.teensy_hid_open_index ||
        previousConfig.teensy_hid_packet_timeout_ms != config.teensy_hid_packet_timeout_ms ||
        previousConfig.teensy_hid_reconnect_interval_ms != config.teensy_hid_reconnect_interval_ms ||
        previousConfig.kmbox_net_ip != config.kmbox_net_ip ||
        previousConfig.kmbox_net_port != config.kmbox_net_port ||
        previousConfig.kmbox_net_uuid != config.kmbox_net_uuid ||
        previousConfig.kmbox_a_pidvid != config.kmbox_a_pidvid ||
        previousConfig.makcu_port != config.makcu_port ||
        previousConfig.makcu_baudrate != config.makcu_baudrate)
    {
        input_method_changed.store(true);
    }

    // DIRECT (stub) is covered by the generic input_method change detection above.

    if (previousConfig.show_window != config.show_window)
        show_window_changed.store(true);

    appLogger.configure(config.debug_log_file_enabled, config.debug_log_file_path);

    if (globalMouseThread)
    {
        globalMouseThread->updateConfig(
            config.detection_resolution,
            config.fovX,
            config.fovY,
            config.auto_shoot,
            config.bScope_multiplier);
    }
}

void RefreshRuntimeAfterConfigLoad(const Config& previousConfig)
{
    std::lock_guard<std::recursive_mutex> lock(configMutex);
    RefreshRuntimeAfterConfigLoadUnlocked(previousConfig);
}

bool LoadRuntimeConfigMerge(const std::string& filename)
{
    std::lock_guard<std::recursive_mutex> lock(configMutex);
    Config previousConfig = config;
    if (!config.loadConfigMerged(filename))
        return false;

    RefreshRuntimeAfterConfigLoadUnlocked(previousConfig);
    return true;
}

int main()
{
    SetConsoleOutputCP(CP_UTF8);
    SetRandomConsoleTitle();
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);

    {
        wchar_t exePath[MAX_PATH]{};
        if (GetModuleFileNameW(nullptr, exePath, MAX_PATH) > 0)
        {
            std::filesystem::path exeDir = std::filesystem::path(exePath).parent_path();
            std::error_code ec;
            std::filesystem::current_path(exeDir, ec);
            if (ec && config.verbose)
            {
                std::cout << "[Config] Failed to set working dir: " << exeDir.u8string()
                          << " (" << ec.message() << ")" << std::endl;
            }
        }
    }

    if (!config.loadConfig())
    {
        std::cerr << "[Config] Error with loading config!" << std::endl;
        std::cin.get();
        return -1;
    }
    appLogger.configure(config.debug_log_file_enabled, config.debug_log_file_path);
    appLogger.log("[MAIN] 0BS starting.");

    CPUAffinityManager cpuManager;

    if (config.cpuCoreReserveCount > 0)
    {
        if (!cpuManager.reserveCPUCores(config.cpuCoreReserveCount))
            return FatalExit("[MAIN] Failed to reserve CPU cores.");
    }

    if (config.systemMemoryReserveMB > 0)
    {
        if (!cpuManager.reserveSystemMemory(config.systemMemoryReserveMB))
            return FatalExit("[MAIN] Failed to reserve system memory.");
    }

    try
    {
#ifdef USE_CUDA
        int cuda_runtime_version = 0;
        cudaError_t runtime_status = cudaRuntimeGetVersion(&cuda_runtime_version);

        if (runtime_status != cudaSuccess)
        {
            std::cerr << "[MAIN] CUDA runtime check failed: " << cudaGetErrorString(runtime_status) << std::endl;
            std::cin.get();
            return -1;
        }

        if (config.verbose)
            std::cout << "[CUDA] Version: " << cuda_runtime_version << std::endl;

        const int required_cuda_version = 13010;
        if (cuda_runtime_version < required_cuda_version)
        {
            int required_major = required_cuda_version / 1000;
            int required_minor = (required_cuda_version % 1000) / 10;
            int runtime_major = cuda_runtime_version / 1000;
            int runtime_minor = (cuda_runtime_version % 1000) / 10;
            std::cerr << "[MAIN] CUDA " << required_major << "." << required_minor
                << " required. Detected " << runtime_major << "." << runtime_minor << "." << std::endl;
            const wchar_t* title = L"CUDA Update Required";
            std::wstring message =
                L"An outdated CUDA version was detected. "
                L"Please update your graphics drivers to the latest version "
                L"and install CUDA 13.1.\n\n"
                L"The program will now attempt to continue.";
            MessageBoxW(nullptr, message.c_str(), title, MB_OK | MB_ICONWARNING);
        }

        GPUResourceManager gpuManager;
        if (config.gpuMemoryReserveMB > 0)
        {
            if (!gpuManager.reserveGPUMemory(config.gpuMemoryReserveMB))
                return FatalExit("[MAIN] Failed to reserve GPU memory.");
        }
        
        if (config.enableGpuExclusiveMode)
        {
            if (!gpuManager.setGPUExclusiveMode())
                return FatalExit("[MAIN] Failed to set GPU exclusive mode.");
        }

        int cuda_devices = 0;
        if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0)
        {
            std::cerr << "[MAIN] CUDA required but no devices found." << std::endl;
            std::cin.get();
            return -1;
        }
#endif
        if (!CreateDirectory(L"screenshots", NULL) && GetLastError() != ERROR_ALREADY_EXISTS)
        {
            std::cout << "[MAIN] Error with screenshot folder" << std::endl;
            std::cin.get();
            return -1;
        }

        if (!CreateDirectory(L"models", NULL) && GetLastError() != ERROR_ALREADY_EXISTS)
        {
            std::cout << "[MAIN] Error with models folder" << std::endl;
            std::cin.get();
            return -1;
        }
        if (config.capture_method == "virtual_camera")
        {
            auto cams = VirtualCameraCapture::GetAvailableVirtualCameras(true);
            if (!cams.empty())
            {
                if (config.virtual_camera_name != "None" &&
                    std::find(cams.begin(), cams.end(), config.virtual_camera_name) == cams.end())
                {
                    config.virtual_camera_name = "None";
                    config.saveConfig("config.ini");
                    std::cout << "[MAIN] virtual_camera_name reset to None (auto-select)." << std::endl;
                }
                std::cout << "[MAIN] Virtual cameras loaded: " << cams.size() << std::endl;
            }
            else
            {
                std::cerr << "[MAIN] No virtual cameras found" << std::endl;
            }
        }

        std::string modelPath = "models/" + config.ai_model;

        if (!std::filesystem::exists(modelPath))
        {
            std::cerr << "[MAIN] Specified model does not exist: " << modelPath << std::endl;

            std::vector<std::string> modelFiles = getModelFiles();

            if (!modelFiles.empty())
            {
                config.ai_model = modelFiles[0];
                config.saveConfig();
                std::cout << "[MAIN] Loaded first available model: " << config.ai_model << std::endl;
            }
            else
            {
                std::cerr << "[MAIN] No models found in 'models' directory." << std::endl;
                std::cin.get();
                return -1;
            }
        }

        createInputDevices();

        MouseThread mouseThread(
            config.detection_resolution,
            config.fovX,
            config.fovY,
            config.auto_shoot,
            config.bScope_multiplier,
            activeMouseInputOwner.get()
        );

        globalMouseThread = &mouseThread;
        assignInputDevices();

        std::vector<std::string> availableModels = getAvailableModels();

        if (!config.ai_model.empty())
        {
            std::string modelPath = "models/" + config.ai_model;
            if (!std::filesystem::exists(modelPath))
            {
                std::cerr << "[MAIN] Specified model does not exist: " << modelPath << std::endl;

                if (!availableModels.empty())
                {
                    config.ai_model = availableModels[0];
                    config.saveConfig("config.ini");
                    std::cout << "[MAIN] Loaded first available model: " << config.ai_model << std::endl;
                }
                else
                {
                    std::cerr << "[MAIN] No models found in 'models' directory." << std::endl;
                    std::cin.get();
                    return -1;
                }
            }
        }
        else
        {
            if (!availableModels.empty())
            {
                config.ai_model = availableModels[0];
                config.saveConfig();
                std::cout << "[MAIN] No AI model specified in config. Loaded first available model: " << config.ai_model << std::endl;
            }
            else
            {
                std::cerr << "[MAIN] No AI models found in 'models' directory." << std::endl;
                std::cin.get();
                return -1;
            }
        }

        RuntimeSupervisor runtime(HandleThreadCrash);

        if (config.backend == "DML")
        {
            dml_detector = new DirectMLDetector("models/" + config.ai_model);
            std::cout << "[MAIN] DML detector initialized." << std::endl;
            appLogger.log("[MAIN] DML detector initialized.");
            runtime.start("DmlDetector", [] {
                dml_detector->dmlInferenceThread();
                });
        }
#ifdef USE_CUDA
        else
        {
            trt_detector.initialize("models/" + config.ai_model);
        }
#endif

        detection_resolution_changed.store(true);
        detection_resolution_generation.fetch_add(1, std::memory_order_relaxed);

        runtime.start("KeyboardListener", [] {
            keyboardListener();
            });
        runtime.start("CaptureThread", [] {
            captureThread(config.detection_resolution, config.detection_resolution);
            });

#ifdef USE_CUDA
        runtime.start("TrtDetector", [] {
            trt_detector.inferenceThread();
            });
#endif
        runtime.start("MouseThread", [&mouseThread] {
            mouseThreadFunction(mouseThread);
            });
        runtime.start("OverlayThread", [] {
            OverlayThread();
            });

        gameOverlayShouldExit.store(false);
        runtime.start("GameOverlay", [] {
            gameOverlayRenderLoop();
            });

        welcome_message();

        while (!shouldExit.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(50));

        if (dml_detector)
        {
            dml_detector->shouldExit = true;
            dml_detector->inferenceCV.notify_all();
        }
        gameOverlayShouldExit.store(true);
        detectionBuffer.cv.notify_all();
        runtime.joinAll();

        if (dml_detector)
        {
            delete dml_detector;
            dml_detector = nullptr;
        }

        if (gameOverlayPtr)
        {
            gameOverlayPtr->Stop();
            delete gameOverlayPtr;
            gameOverlayPtr = nullptr;
        }
        globalMouseThread = nullptr;
        activeMouseInputOwner.reset();
        arduinoSerial = nullptr;
        gHub = nullptr;
        razerControl = nullptr;
        kmboxNetSerial = nullptr;
        kmboxASerial = nullptr;
        makcuSerial = nullptr;
        appLogger.log("[MAIN] 0BS stopped.");

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[MAIN] An error has occurred in the main stream: " << e.what() << std::endl;
        appLogger.log(std::string("[MAIN] Exception: ") + e.what());
        std::cout << "Press Enter to exit...";
        std::cin.get();
        return -1;
    }
}
