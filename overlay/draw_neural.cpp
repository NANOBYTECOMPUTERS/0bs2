#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <algorithm>
#include <cstring>

#include "imgui/imgui.h"

#include "0BS_box_2.h"
#include "overlay/config_dirty.h"
#include "overlay/ui_sections.h"

namespace
{
char neuralModelPathBuf[260] = {};
bool neuralUiInitialized = false;

void syncNeuralBuffers()
{
    if (neuralUiInitialized)
        return;

    strncpy_s(neuralModelPathBuf, config.neural_tracker_model_path.c_str(), _TRUNCATE);
    neuralUiInitialized = true;
}
}

void draw_neural()
{
    syncNeuralBuffers();

    if (OverlayUI::BeginSection("Neural Tracker", "neural_section_tracker"))
    {
        if (ImGui::Checkbox("Enable neural association", &config.neural_tracker_enabled))
        {
            OverlayConfig_MarkDirty();
        }

        ImGui::SetNextItemWidth(OverlayUI::AdaptiveItemWidth(0.78f));
        if (ImGui::InputText("Association model", neuralModelPathBuf, IM_ARRAYSIZE(neuralModelPathBuf)))
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
}
