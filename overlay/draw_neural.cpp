#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <string>
#include <vector>

#include "imgui/imgui.h"

#include "0BS_box_2.h"
#include "overlay/config_dirty.h"
#include "overlay/ui_sections.h"

namespace
{
char neuralModelPathBuf[260] = {};
char temporalModelPathBuf[260] = {};
char neuralTargetingModelPathBuf[260] = {};
char neuralTelemetryLogPathBuf[260] = {};
bool neuralUiInitialized = false;

bool hasNeuralModelExtension(const std::filesystem::path& path)
{
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".onnx" || ext == ".engine";
}

std::vector<std::string> getAvailableNeuralTrackerModels()
{
    std::vector<std::string> models;
    const std::filesystem::path roots[] = {
        std::filesystem::path("training") / "models",
        std::filesystem::path("models")
    };

    for (const auto& root : roots)
    {
        std::error_code ec;
        std::filesystem::directory_iterator it(root, ec);
        if (ec)
            continue;

        for (const auto& entry : it)
        {
            if (!entry.is_regular_file() || !hasNeuralModelExtension(entry.path()))
                continue;

            models.push_back((root / entry.path().filename()).generic_string());
        }
    }

    std::sort(models.begin(), models.end());
    models.erase(std::unique(models.begin(), models.end()), models.end());
    return models;
}

void syncNeuralBuffers()
{
    if (neuralUiInitialized)
        return;

    strncpy_s(neuralModelPathBuf, config.neural_tracker_model_path.c_str(), _TRUNCATE);
    strncpy_s(temporalModelPathBuf, config.temporal_prediction_model_path.c_str(), _TRUNCATE);
    strncpy_s(neuralTargetingModelPathBuf, config.neural_targeting_model_path.c_str(), _TRUNCATE);
    strncpy_s(neuralTelemetryLogPathBuf, config.neural_control_telemetry_log_path.c_str(), _TRUNCATE);
    neuralUiInitialized = true;
}

void refreshMouseThreadFromNeural()
{
    if (!globalMouseThread)
        return;

    globalMouseThread->updateConfig(
        config.detection_resolution,
        config.fovX,
        config.fovY,
        config.auto_shoot,
        config.bScope_multiplier);
}

void applyPerfectAimPreset(const char* preset)
{
    config.neural_control_preset = preset ? preset : "Balanced";
    config.temporal_prediction_feed_forward_enabled = true;
    config.temporal_prediction_adaptive_influence_enabled = true;
    config.pid_smart_blending_enabled = true;

    if (config.neural_control_preset == "Aggressive")
    {
        config.temporal_prediction_influence = 0.50f;
        config.temporal_prediction_max_lead_px = 60.0f;
        config.neural_targeting_influence = 0.55f;
        config.neural_targeting_max_refinement_px = 45.0f;
        config.pid_smart_blending_aggression = 0.85f;
        config.pid_smart_blending_near_damping = 0.45f;
        config.pid_smart_blending_jerk_limit_px = 1.15f;
        config.pid_smart_blending_confidence_floor = 0.40f;
    }
    else if (config.neural_control_preset == "Smooth")
    {
        config.temporal_prediction_influence = 0.22f;
        config.temporal_prediction_max_lead_px = 38.0f;
        config.neural_targeting_influence = 0.28f;
        config.neural_targeting_max_refinement_px = 28.0f;
        config.pid_smart_blending_aggression = 0.48f;
        config.pid_smart_blending_near_damping = 0.92f;
        config.pid_smart_blending_jerk_limit_px = 0.32f;
        config.pid_smart_blending_confidence_floor = 0.50f;
    }
    else if (config.neural_control_preset == "Sniper")
    {
        config.temporal_prediction_influence = 0.18f;
        config.temporal_prediction_max_lead_px = 28.0f;
        config.neural_targeting_influence = 0.20f;
        config.neural_targeting_max_refinement_px = 22.0f;
        config.pid_smart_blending_aggression = 0.38f;
        config.pid_smart_blending_near_damping = 0.95f;
        config.pid_smart_blending_jerk_limit_px = 0.22f;
        config.pid_smart_blending_confidence_floor = 0.60f;
    }
    else
    {
        config.neural_control_preset = "Balanced";
        config.temporal_prediction_influence = 0.32f;
        config.temporal_prediction_max_lead_px = 45.0f;
        config.neural_targeting_influence = 0.40f;
        config.neural_targeting_max_refinement_px = 35.0f;
        config.pid_smart_blending_aggression = 0.65f;
        config.pid_smart_blending_near_damping = 0.75f;
        config.pid_smart_blending_jerk_limit_px = 0.65f;
        config.pid_smart_blending_confidence_floor = 0.45f;
    }

    config.temporal_prediction_adaptive_ema_alpha = 0.62f;
    config.pid_smart_blending_deadzone_px = 0.0f;
    OverlayConfig_MarkDirty();
    refreshMouseThreadFromNeural();
}
}

void draw_neural()
{
    syncNeuralBuffers();

    if (OverlayUI::BeginSection("Perfect Aim v1.0 Presets", "neural_section_perfect_aim_presets"))
    {
        const char* presets[] = { "Balanced", "Aggressive", "Smooth", "Sniper" };
        int presetIndex = 0;
        for (int i = 0; i < IM_ARRAYSIZE(presets); ++i)
        {
            if (config.neural_control_preset == presets[i])
            {
                presetIndex = i;
                break;
            }
        }

        if (ImGui::Combo("Control preset", &presetIndex, presets, IM_ARRAYSIZE(presets)))
        {
            applyPerfectAimPreset(presets[presetIndex]);
        }

        if (ImGui::Checkbox("Neural telemetry overlay", &config.neural_control_telemetry_overlay_enabled))
        {
            OverlayConfig_MarkDirty();
        }

        if (ImGui::Checkbox("Neural telemetry logging", &config.neural_control_telemetry_logging_enabled))
        {
            OverlayConfig_MarkDirty();
        }

        ImGui::SetNextItemWidth(OverlayUI::AdaptiveItemWidth(0.78f));
        if (ImGui::InputText("Telemetry log path", neuralTelemetryLogPathBuf, IM_ARRAYSIZE(neuralTelemetryLogPathBuf)))
        {
            config.neural_control_telemetry_log_path = neuralTelemetryLogPathBuf;
            OverlayConfig_MarkDirty();
        }

        if (ImGui::SliderInt("Telemetry log interval (ms)", &config.neural_control_telemetry_log_interval_ms, 50, 5000))
        {
            config.neural_control_telemetry_log_interval_ms =
                std::clamp(config.neural_control_telemetry_log_interval_ms, 50, 5000);
            OverlayConfig_MarkDirty();
        }

        if (globalMouseThread)
        {
            NeuralControlTelemetry telemetry;
            if (globalMouseThread->getNeuralControlTelemetry(telemetry))
            {
                ImGui::Text("Current influence: %.1f%%", telemetry.adaptiveInfluence * 100.0);
                ImGui::Text("Prediction confidence: %.2f", telemetry.confidence);
                ImGui::Text("Predicted lead: %.2f, %.2f", telemetry.predictionLeadX, telemetry.predictionLeadY);
                ImGui::Text("Smart jitter: %.2f", telemetry.jitterScore);
            }
        }

        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Neural Tracker", "neural_section_tracker"))
    {
        if (ImGui::Checkbox("Enable neural association", &config.neural_tracker_enabled))
        {
            OverlayConfig_MarkDirty();
        }

        const char* runtimeItems[] = {
            "CPU (ONNX Runtime)",
#ifdef USE_CUDA
            "CUDA (TensorRT)"
#endif
        };
        int runtimeIndex = config.neural_tracker_runtime == "CUDA" ? 1 : 0;
#ifndef USE_CUDA
        runtimeIndex = 0;
#endif
        if (ImGui::Combo("Association runtime", &runtimeIndex, runtimeItems, IM_ARRAYSIZE(runtimeItems)))
        {
            if (runtimeIndex == 1)
                config.neural_tracker_runtime = "CUDA";
            else
                config.neural_tracker_runtime = "CPU";
            OverlayConfig_MarkDirty();
        }

        std::vector<std::string> availableModels = getAvailableNeuralTrackerModels();
        if (!availableModels.empty())
        {
            auto it = std::find(availableModels.begin(), availableModels.end(), config.neural_tracker_model_path);
            if (it == availableModels.end())
            {
                availableModels.push_back(config.neural_tracker_model_path);
                it = std::prev(availableModels.end());
            }

            int currentModelIndex = static_cast<int>(std::distance(availableModels.begin(), it));
            std::vector<const char*> modelItems;
            modelItems.reserve(availableModels.size());
            for (const auto& model : availableModels)
                modelItems.push_back(model.c_str());

            ImGui::SetNextItemWidth(OverlayUI::AdaptiveItemWidth(0.78f));
            if (ImGui::Combo("Association model", &currentModelIndex, modelItems.data(), static_cast<int>(modelItems.size())))
            {
                config.neural_tracker_model_path = availableModels[currentModelIndex];
                strncpy_s(neuralModelPathBuf, config.neural_tracker_model_path.c_str(), _TRUNCATE);
                OverlayConfig_MarkDirty();
            }
        }

        ImGui::SetNextItemWidth(OverlayUI::AdaptiveItemWidth(0.78f));
        if (ImGui::InputText("Association path", neuralModelPathBuf, IM_ARRAYSIZE(neuralModelPathBuf)))
        {
            config.neural_tracker_model_path = neuralModelPathBuf;
            OverlayConfig_MarkDirty();
        }

        if (ImGui::SliderFloat("Association blend", &config.neural_tracker_blend, 0.0f, 1.0f, "%.2f"))
        {
            config.neural_tracker_blend = std::clamp(config.neural_tracker_blend, 0.0f, 1.0f);
            OverlayConfig_MarkDirty();
        }

        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Learned Temporal Predictor", "neural_section_temporal_predictor"))
    {
        if (ImGui::Checkbox("Enable temporal prediction", &config.temporal_prediction_enabled))
        {
            OverlayConfig_MarkDirty();
        }

        ImGui::SetNextItemWidth(OverlayUI::AdaptiveItemWidth(0.78f));
        if (ImGui::InputText("Temporal model path", temporalModelPathBuf, IM_ARRAYSIZE(temporalModelPathBuf)))
        {
            config.temporal_prediction_model_path = temporalModelPathBuf;
            OverlayConfig_MarkDirty();
        }

        if (ImGui::SliderInt("History length", &config.temporal_prediction_history_length, 2, 64))
        {
            config.temporal_prediction_history_length = std::clamp(config.temporal_prediction_history_length, 2, 64);
            OverlayConfig_MarkDirty();
        }

        if (ImGui::SliderInt("Prediction horizon", &config.temporal_prediction_horizon, 1, 64))
        {
            config.temporal_prediction_horizon = std::clamp(config.temporal_prediction_horizon, 1, 64);
            OverlayConfig_MarkDirty();
        }

        if (ImGui::SliderInt("Predict every N frames", &config.temporal_prediction_interval_frames, 1, 16))
        {
            config.temporal_prediction_interval_frames = std::clamp(config.temporal_prediction_interval_frames, 1, 16);
            OverlayConfig_MarkDirty();
        }

        if (ImGui::Checkbox("Prediction feed-forward", &config.temporal_prediction_feed_forward_enabled))
        {
            OverlayConfig_MarkDirty();
        }

        if (ImGui::SliderFloat("Base prediction influence##Prediction influence", &config.temporal_prediction_influence, 0.0f, 1.0f, "%.2f"))
        {
            config.temporal_prediction_influence = std::clamp(config.temporal_prediction_influence, 0.0f, 1.0f);
            OverlayConfig_MarkDirty();
        }

        if (ImGui::Checkbox("Adaptive prediction influence", &config.temporal_prediction_adaptive_influence_enabled))
        {
            OverlayConfig_MarkDirty();
        }

        if (ImGui::SliderFloat("Adaptive influence EMA", &config.temporal_prediction_adaptive_ema_alpha, 0.05f, 1.0f, "%.2f"))
        {
            config.temporal_prediction_adaptive_ema_alpha =
                std::clamp(config.temporal_prediction_adaptive_ema_alpha, 0.05f, 1.0f);
            OverlayConfig_MarkDirty();
        }

        if (ImGui::SliderFloat("Max prediction lead (px)", &config.temporal_prediction_max_lead_px, 20.0f, 80.0f, "%.1f"))
        {
            config.temporal_prediction_max_lead_px = std::clamp(config.temporal_prediction_max_lead_px, 20.0f, 80.0f);
            OverlayConfig_MarkDirty();
        }

        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Neural Targeting Head", "neural_section_targeting_head"))
    {
        if (ImGui::Checkbox("Enable neural targeting", &config.neural_targeting_enabled))
        {
            OverlayConfig_MarkDirty();
        }

        ImGui::SetNextItemWidth(OverlayUI::AdaptiveItemWidth(0.78f));
        if (ImGui::InputText("Targeting model path", neuralTargetingModelPathBuf, IM_ARRAYSIZE(neuralTargetingModelPathBuf)))
        {
            config.neural_targeting_model_path = neuralTargetingModelPathBuf;
            OverlayConfig_MarkDirty();
        }

        if (ImGui::SliderFloat("Targeting influence", &config.neural_targeting_influence, 0.0f, 1.0f, "%.2f"))
        {
            config.neural_targeting_influence = std::clamp(config.neural_targeting_influence, 0.0f, 1.0f);
            OverlayConfig_MarkDirty();
        }

        if (ImGui::SliderFloat("Max targeting refinement (px)", &config.neural_targeting_max_refinement_px, 1.0f, 80.0f, "%.1f"))
        {
            config.neural_targeting_max_refinement_px = std::clamp(config.neural_targeting_max_refinement_px, 1.0f, 80.0f);
            OverlayConfig_MarkDirty();
        }

        if (ImGui::SliderInt("Targeting feedback iterations", &config.neural_targeting_max_iterations, 1, 2))
        {
            config.neural_targeting_max_iterations = std::clamp(config.neural_targeting_max_iterations, 1, 2);
            OverlayConfig_MarkDirty();
        }

        OverlayUI::EndSection();
    }
}
