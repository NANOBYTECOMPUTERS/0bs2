#include "overlay/config_dirty.h"

#include "0BS_box_2.h"
#include "imgui/imgui.h"

namespace
{
    bool cfgDirty = false;
    double cfgDirtyAt = 0.0;
    constexpr double kSaveDelaySec = 0.35;
}

void OverlayConfig_MarkDirty()
{
    cfgDirty = true;
    cfgDirtyAt = ImGui::GetTime();
}

void OverlayConfig_ClearDirty()
{
    cfgDirty = false;
    cfgDirtyAt = 0.0;
}

bool OverlayConfig_SaveNow(const char* filename)
{
    const bool saved = SaveRuntimeConfig(filename ? filename : "config.ini");
    if (saved)
        OverlayConfig_ClearDirty();
    return saved;
}

void OverlayConfig_TrySave(const char* filename)
{
    if (!cfgDirty)
        return;

    const double now = ImGui::GetTime();
    if ((now - cfgDirtyAt) < kSaveDelaySec)
        return;

    if (ImGui::IsAnyItemActive())
        return;

    OverlayConfig_SaveNow(filename);
}
