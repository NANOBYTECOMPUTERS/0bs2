#pragma once

void OverlayConfig_MarkDirty();
void OverlayConfig_ClearDirty();
bool OverlayConfig_SaveNow(const char* filename = "config.ini");
void OverlayConfig_TrySave(const char* filename = "config.ini");
