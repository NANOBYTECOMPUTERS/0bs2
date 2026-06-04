#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <algorithm>

#include "d3d11.h"
#include "imgui/imgui.h"

#include "overlay.h"
#include "overlay/config_dirty.h"
#include "draw_settings.h"
#include "overlay/ui_sections.h"
#include "0BS_box_2.h"
#include "other_tools.h"
#include "memory_images.h"

ID3D11ShaderResourceView* bodyTexture = nullptr;
ImVec2 bodyImageSize;

bool prev_disable_headshot = config.disable_headshot;
float prev_body_y_offset = config.body_y_offset;
float prev_head_y_offset = config.head_y_offset;
bool prev_auto_aim = config.auto_aim;
bool prev_easynorecoil = config.easynorecoil;
float prev_easynorecoilstrength = config.easynorecoilstrength;

void draw_target()
{
    if (OverlayUI::BeginSection("Targeting", "target_section_targeting"))
    {
        ImGui::Checkbox("Disable Headshot", &config.disable_headshot);
        ImGui::Checkbox("Auto Aim", &config.auto_aim);
        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Offsets", "target_section_offsets"))
    {
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Arrow keys: Adjust body offset");
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Shift+Arrow keys: Adjust head offset");

        ImGui::SliderFloat("Approximate Body Y Offset", &config.body_y_offset, Config::kBodyYOffsetMin, Config::kBodyYOffsetMax, "%.2f");
        ImGui::SliderFloat("Approximate Head Y Offset", &config.head_y_offset, Config::kHeadYOffsetMin, Config::kHeadYOffsetMax, "%.2f");
        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Preview", "target_section_preview"))
    {
        if (bodyTexture)
        {
            ImGui::Image((void*)bodyTexture, bodyImageSize);

            ImVec2 image_pos = ImGui::GetItemRectMin();
            ImVec2 image_size = ImGui::GetItemRectSize();

            ImDrawList* draw_list = ImGui::GetWindowDrawList();

            const float bodyOffset = std::clamp(config.body_y_offset, Config::kBodyYOffsetMin, Config::kBodyYOffsetMax);
            const float headOffset = std::clamp(config.head_y_offset, Config::kHeadYOffsetMin, Config::kHeadYOffsetMax);
            float body_line_y = image_pos.y + bodyOffset * image_size.y;
            ImVec2 body_line_start = ImVec2(image_pos.x, body_line_y);
            ImVec2 body_line_end = ImVec2(image_pos.x + image_size.x, body_line_y);
            draw_list->AddLine(body_line_start, body_line_end, IM_COL32(255, 0, 0, 255), 2.0f);

            float head_line_y = image_pos.y + headOffset * image_size.y;

            ImVec2 head_line_start = ImVec2(image_pos.x, head_line_y);
            ImVec2 head_line_end = ImVec2(image_pos.x + image_size.x, head_line_y);
            draw_list->AddLine(head_line_start, head_line_end, IM_COL32(0, 255, 0, 255), 2.0f);

            draw_list->AddText(ImVec2(body_line_end.x + 5, body_line_y - 7), IM_COL32(255, 0, 0, 255), "Body");
            draw_list->AddText(ImVec2(head_line_end.x + 5, head_line_y - 7), IM_COL32(0, 255, 0, 255), "Head");
        }
        else
        {
            ImGui::Text("Image not found!");
        }
        ImGui::Text("Note: There is a different value for each game, as the sizes of the player models may vary.");
        OverlayUI::EndSection();
    }

    if (prev_disable_headshot != config.disable_headshot ||
        prev_body_y_offset != config.body_y_offset ||
        prev_head_y_offset != config.head_y_offset ||
        prev_auto_aim != config.auto_aim ||
        prev_easynorecoil != config.easynorecoil ||
        prev_easynorecoilstrength != config.easynorecoilstrength)
    {
        prev_disable_headshot = config.disable_headshot;
        prev_body_y_offset = config.body_y_offset;
        prev_head_y_offset = config.head_y_offset;
        prev_auto_aim = config.auto_aim;
        prev_easynorecoil = config.easynorecoil;
        prev_easynorecoilstrength = config.easynorecoilstrength;
        OverlayConfig_MarkDirty();
    }
}

void load_body_texture()
{
    int image_width = 0;
    int image_height = 0;

    std::string body_image = std::string(bodyImageBase64_1) + std::string(bodyImageBase64_2) + std::string(bodyImageBase64_3);

    bool ret = LoadTextureFromMemory(body_image, g_pd3dDevice, &bodyTexture, &image_width, &image_height);
    if (!ret)
    {
        std::cerr << "[Overlay] Can't load image!" << std::endl;
    }
    else
    {
        bodyImageSize = ImVec2((float)image_width, (float)image_height);
    }
}

void release_body_texture()
{
    if (bodyTexture)
    {
        bodyTexture->Release();
        bodyTexture = nullptr;
    }
}
