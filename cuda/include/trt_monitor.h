#pragma once

#include <NvInfer.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <string>

struct TrtProgressPhase
{
    int current = 0;
    int max = 0;
};

inline std::mutex gProgressMutex;
inline std::map<std::string, TrtProgressPhase> gProgressPhases;
inline std::atomic<bool> gIsTrtExporting{ false };
inline std::atomic<bool> gTrtExportCancelRequested{ false };
inline std::atomic<long long> gTrtExportLastUpdateMs{ 0 };

inline long long TrtNowMs()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

inline void TrtExportResetState()
{
    std::lock_guard<std::mutex> lock(gProgressMutex);
    gProgressPhases.clear();
    gTrtExportCancelRequested = false;
    gTrtExportLastUpdateMs = TrtNowMs();
}

class ImGuiProgressMonitor : public nvinfer1::IProgressMonitor
{
public:
    void phaseStart(char const* phaseName, char const*, int32_t nbSteps) noexcept override
    {
        std::lock_guard<std::mutex> lock(gProgressMutex);
        const std::string name = phaseName ? phaseName : "TensorRT";
        gProgressPhases[name] = TrtProgressPhase{ 0, std::max(1, nbSteps) };
        gTrtExportLastUpdateMs = TrtNowMs();
    }

    bool stepComplete(char const* phaseName, int32_t step) noexcept override
    {
        std::lock_guard<std::mutex> lock(gProgressMutex);
        const std::string name = phaseName ? phaseName : "TensorRT";
        auto& phase = gProgressPhases[name];
        if (phase.max <= 0)
            phase.max = std::max(1, step + 1);
        phase.current = std::clamp(step + 1, 0, phase.max);
        gTrtExportLastUpdateMs = TrtNowMs();
        return !gTrtExportCancelRequested.load();
    }

    void phaseFinish(char const* phaseName) noexcept override
    {
        std::lock_guard<std::mutex> lock(gProgressMutex);
        const std::string name = phaseName ? phaseName : "TensorRT";
        auto it = gProgressPhases.find(name);
        if (it != gProgressPhases.end())
            it->second.current = it->second.max;
        gTrtExportLastUpdateMs = TrtNowMs();
    }
};
