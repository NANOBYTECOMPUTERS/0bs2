#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cstring>
#include <string>
#include <vector>

#include "imgui/imgui.h"

#include "0BS_box_2.h"
#include "overlay/config_dirty.h"
#include "overlay/ui_sections.h"
#include "include/other_tools.h"
#include "kmbox_net/picture.h"
#include "rzctl.h"

std::string ghub_version = get_ghub_version();

namespace
{
int prev_fovX = config.fovX;
int prev_fovY = config.fovY;
int prev_pid_actuator_hz = config.pid_actuator_hz;
float prev_pid_kp = config.pid_kp;
float prev_pid_ki = config.pid_ki;
float prev_pid_kd = config.pid_kd;
float prev_pid_deadzone_px = config.pid_deadzone_px;
float prev_pid_max_pixel_step = config.pid_max_pixel_step;
float prev_pid_output_scale = config.pid_output_scale;
float prev_pid_min_output_scale = config.pid_min_output_scale;
float prev_pid_max_output_scale = config.pid_max_output_scale;
float prev_pid_size_reference_px = config.pid_size_reference_px;
float prev_pid_size_min_scale = config.pid_size_min_scale;
float prev_pid_size_max_scale = config.pid_size_max_scale;
float prev_pid_precision_radius_scale = config.pid_precision_radius_scale;
float prev_pid_slowdown_radius_scale = config.pid_slowdown_radius_scale;
float prev_pid_overshoot_brake = config.pid_overshoot_brake;
float prev_pid_divergence_boost = config.pid_divergence_boost;
float prev_pid_scale_response = config.pid_scale_response;
float prev_pid_max_integral = config.pid_max_integral;
float prev_pid_max_derivative_term = config.pid_max_derivative_term;
float prev_pid_derivative_filter_tau_ms = config.pid_derivative_filter_tau_ms;
float prev_pid_target_loss_timeout_ms = config.pid_target_loss_timeout_ms;
bool prev_pid_feed_forward_enabled = config.pid_feed_forward_enabled;
float prev_pid_feed_forward_gain = config.pid_feed_forward_gain;
float prev_pid_feed_forward_lookahead_ms = config.pid_feed_forward_lookahead_ms;
float prev_pid_feed_forward_max_step = config.pid_feed_forward_max_step;
float prev_pid_feed_forward_min_speed = config.pid_feed_forward_min_speed;
float prev_pid_feed_forward_confidence_floor = config.pid_feed_forward_confidence_floor;
bool prev_pid_governor_enabled = config.pid_governor_enabled;
float prev_pid_governor_blend = config.pid_governor_blend;
float prev_pid_governor_max_speed_multiple = config.pid_governor_max_speed_multiple;
bool prev_auto_shoot = config.auto_shoot;
float prev_bScope_multiplier = config.bScope_multiplier;

void refreshMouseThread()
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

void drawGameProfileEditor()
{
    std::vector<std::string> profileNames;
    for (const auto& kv : config.game_profiles)
        profileNames.push_back(kv.first);

    static int selectedIndex = 0;
    for (size_t i = 0; i < profileNames.size(); ++i)
    {
        if (profileNames[i] == config.active_game)
        {
            selectedIndex = static_cast<int>(i);
            break;
        }
    }

    std::vector<const char*> profileItems;
    profileItems.reserve(profileNames.size());
    for (const auto& name : profileNames)
        profileItems.push_back(name.c_str());

    if (!profileItems.empty() &&
        ImGui::Combo("Active Game Profile", &selectedIndex, profileItems.data(), static_cast<int>(profileItems.size())))
    {
        config.active_game = profileNames[selectedIndex];
        OverlayConfig_MarkDirty();
        refreshMouseThread();
    }

    const auto& gp = config.currentProfile();
    ImGui::Text("Current profile: %s", gp.name.c_str());
    ImGui::Text("Sens: %.4f", gp.sens);
    ImGui::Text("Yaw: %.4f", gp.yaw);
    ImGui::Text("Pitch: %.4f", gp.pitch);
    ImGui::Text("FOV scaled: %s", gp.fovScaled ? "true" : "false");

    if (gp.name == "UNIFIED")
        return;

    Config::GameProfile& modifiable = config.game_profiles[gp.name];
    bool changed = false;

    float sens = static_cast<float>(modifiable.sens);
    float yaw = static_cast<float>(modifiable.yaw);
    float pitch = static_cast<float>(modifiable.pitch);
    float baseFov = static_cast<float>(modifiable.baseFOV);

    changed |= ImGui::SliderFloat("Sensitivity", &sens, 0.001f, 10.0f, "%.4f");
    changed |= ImGui::SliderFloat("Yaw", &yaw, 0.001f, 0.1f, "%.4f");
    changed |= ImGui::SliderFloat("Pitch", &pitch, 0.001f, 0.1f, "%.4f");
    changed |= ImGui::Checkbox("FOV Scaled", &modifiable.fovScaled);
    if (modifiable.fovScaled)
        changed |= ImGui::SliderFloat("Base FOV", &baseFov, 10.0f, 180.0f, "%.1f");

    if (!changed)
        return;

    modifiable.sens = static_cast<double>(sens);
    modifiable.yaw = static_cast<double>(yaw);
    modifiable.pitch = static_cast<double>(pitch);
    modifiable.baseFOV = static_cast<double>(baseFov);
    OverlayConfig_MarkDirty();
    refreshMouseThread();
}

void drawProfileManager()
{
    static char newProfileName[64] = "";
    ImGui::InputText("New profile name", newProfileName, sizeof(newProfileName));
    ImGui::SameLine();
    if (ImGui::Button("Add Profile"))
    {
        std::string name = std::string(newProfileName);
        if (!name.empty() && config.game_profiles.count(name) == 0)
        {
            Config::GameProfile gp;
            gp.name = name;
            gp.sens = 1.0;
            gp.yaw = 0.022;
            gp.pitch = 0.022;
            gp.fovScaled = false;
            gp.baseFOV = 90.0;
            config.game_profiles[name] = gp;
            config.active_game = name;
            newProfileName[0] = '\0';
            OverlayConfig_MarkDirty();
            refreshMouseThread();
        }
    }

    const auto& gp = config.currentProfile();
    if (gp.name == "UNIFIED")
        return;

    ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(200, 50, 50, 255));
    if (ImGui::Button("Delete Current Profile"))
    {
        config.game_profiles.erase(gp.name);
        config.active_game = config.game_profiles.empty() ? "UNIFIED" : config.game_profiles.begin()->first;
        OverlayConfig_MarkDirty();
        refreshMouseThread();
    }
    ImGui::PopStyleColor();
}

void drawInputMethod()
{
    std::vector<std::string> inputMethods = { "WIN32", "GHUB", "RAZER", "ARDUINO", "KMBOX_NET", "KMBOX_A", "MAKCU" };
    std::vector<const char*> methodItems;
    methodItems.reserve(inputMethods.size());
    for (const auto& item : inputMethods)
        methodItems.push_back(item.c_str());

    int inputMethodIndex = 0;
    for (size_t i = 0; i < inputMethods.size(); ++i)
    {
        if (inputMethods[i] == config.input_method)
        {
            inputMethodIndex = static_cast<int>(i);
            break;
        }
    }

    if (ImGui::Combo("Mouse Input Method", &inputMethodIndex, methodItems.data(), static_cast<int>(methodItems.size())))
    {
        std::string next = inputMethods[inputMethodIndex];
        if (next != config.input_method)
        {
            config.input_method = next;
            OverlayConfig_MarkDirty();
            input_method_changed.store(true);
        }
    }

    if (config.input_method == "ARDUINO")
    {
        if (arduinoSerial && arduinoSerial->isOpen())
            ImGui::TextColored(ImVec4(0, 255, 0, 255), "Arduino connected");
        else
            ImGui::TextColored(ImVec4(255, 0, 0, 255), "Arduino not connected");

        std::vector<std::string> ports;
        for (int i = 1; i <= 30; ++i)
            ports.push_back("COM" + std::to_string(i));

        std::vector<const char*> portItems;
        portItems.reserve(ports.size());
        for (const auto& port : ports)
            portItems.push_back(port.c_str());

        int portIndex = 0;
        for (size_t i = 0; i < ports.size(); ++i)
        {
            if (ports[i] == config.arduino_port)
            {
                portIndex = static_cast<int>(i);
                break;
            }
        }

        if (ImGui::Combo("Arduino Port", &portIndex, portItems.data(), static_cast<int>(portItems.size())))
        {
            config.arduino_port = ports[portIndex];
            OverlayConfig_MarkDirty();
            input_method_changed.store(true);
        }

        std::vector<int> baudRates = { 9600, 19200, 38400, 57600, 115200 };
        std::vector<std::string> baudStrings;
        for (int rate : baudRates)
            baudStrings.push_back(std::to_string(rate));

        std::vector<const char*> baudItems;
        baudItems.reserve(baudStrings.size());
        for (const auto& rate : baudStrings)
            baudItems.push_back(rate.c_str());

        int baudIndex = 0;
        for (size_t i = 0; i < baudRates.size(); ++i)
        {
            if (baudRates[i] == config.arduino_baudrate)
            {
                baudIndex = static_cast<int>(i);
                break;
            }
        }

        if (ImGui::Combo("Arduino Baudrate", &baudIndex, baudItems.data(), static_cast<int>(baudItems.size())))
        {
            config.arduino_baudrate = baudRates[baudIndex];
            OverlayConfig_MarkDirty();
            input_method_changed.store(true);
        }

        if (ImGui::Checkbox("Arduino 16-bit Mouse", &config.arduino_16_bit_mouse) ||
            ImGui::Checkbox("Arduino Enable Keys", &config.arduino_enable_keys))
        {
            OverlayConfig_MarkDirty();
            input_method_changed.store(true);
        }
    }
    else if (config.input_method == "GHUB")
    {
        if (ghub_version == "13.1.4")
            ImGui::Text("The correct version of GHub is installed: %s", ghub_version.c_str());
        else
        {
            ImGui::Text("The wrong version of GHub is installed or the path to GHub is not set by default.");
            ImGui::Text("Install GHub 13.1.4 using the default path.");
        }
    }
    else if (config.input_method == "RAZER")
    {
        if (razerControl && razerControl->isOpen())
            ImGui::TextColored(ImVec4(0, 255, 0, 255), "Razer rzctl connected");
        else
            ImGui::TextColored(ImVec4(255, 0, 0, 255), "Razer rzctl not connected");

        ImGui::Text("Requires rzctl.dll next to 0BS.exe.");
    }
    else if (config.input_method == "WIN32")
    {
        ImGui::TextColored(ImVec4(255, 255, 255, 255), "Standard Windows mouse input.");
    }
    else if (config.input_method == "KMBOX_NET")
    {
        static char ip[32] = "";
        static char port[8] = "";
        static char uuid[16] = "";
        static std::string lastIp;
        static std::string lastPort;
        static std::string lastUuid;

        if (lastIp != config.kmbox_net_ip || lastPort != config.kmbox_net_port || lastUuid != config.kmbox_net_uuid)
        {
            strncpy(ip, config.kmbox_net_ip.c_str(), sizeof(ip));
            strncpy(port, config.kmbox_net_port.c_str(), sizeof(port));
            strncpy(uuid, config.kmbox_net_uuid.c_str(), sizeof(uuid));
            ip[sizeof(ip) - 1] = '\0';
            port[sizeof(port) - 1] = '\0';
            uuid[sizeof(uuid) - 1] = '\0';
            lastIp = config.kmbox_net_ip;
            lastPort = config.kmbox_net_port;
            lastUuid = config.kmbox_net_uuid;
        }

        ImGui::InputText("IP", ip, sizeof(ip));
        ImGui::InputText("Port", port, sizeof(port));
        ImGui::InputText("UUID", uuid, sizeof(uuid));
        if (ImGui::Button("Save & Reconnect"))
        {
            config.kmbox_net_ip = ip;
            config.kmbox_net_port = port;
            config.kmbox_net_uuid = uuid;
            lastIp = config.kmbox_net_ip;
            lastPort = config.kmbox_net_port;
            lastUuid = config.kmbox_net_uuid;
            OverlayConfig_MarkDirty();
            input_method_changed.store(true);
        }

        if (kmboxNetSerial && kmboxNetSerial->isOpen())
            ImGui::TextColored(ImVec4(0, 255, 0, 255), "kmboxNet connected");
        else
            ImGui::TextColored(ImVec4(255, 0, 0, 255), "kmboxNet not connected");

        if (ImGui::Button("Reboot box") && kmboxNetSerial)
            kmboxNetSerial->reboot();
        if (ImGui::Button("Change Kmbox image") && kmboxNetSerial)
        {
            kmboxNetSerial->lcdColor(0);
            kmboxNetSerial->lcdPicture(gImage_128x160);
        }
    }
    else if (config.input_method == "KMBOX_A")
    {
        static char pidvid[32] = "";
        static std::string lastPidvid;
        if (lastPidvid != config.kmbox_a_pidvid)
        {
            strncpy(pidvid, config.kmbox_a_pidvid.c_str(), sizeof(pidvid));
            pidvid[sizeof(pidvid) - 1] = '\0';
            lastPidvid = config.kmbox_a_pidvid;
        }

        ImGui::InputText("PIDVID", pidvid, sizeof(pidvid));
        if (ImGui::Button("Save & Reconnect##kmbox_a"))
        {
            config.kmbox_a_pidvid = pidvid;
            lastPidvid = config.kmbox_a_pidvid;
            OverlayConfig_MarkDirty();
            input_method_changed.store(true);
        }

        if (kmboxASerial && kmboxASerial->isOpen())
            ImGui::TextColored(ImVec4(0, 255, 0, 255), "kmboxA connected");
        else
            ImGui::TextColored(ImVec4(255, 0, 0, 255), "kmboxA not connected");
    }
    else if (config.input_method == "MAKCU")
    {
        std::vector<std::string> ports;
        ports.push_back("AUTO");
        for (int i = 1; i <= 30; ++i)
            ports.push_back("COM" + std::to_string(i));

        std::vector<const char*> portItems;
        portItems.reserve(ports.size());
        for (const auto& port : ports)
            portItems.push_back(port.c_str());

        int portIndex = 0;
        for (size_t i = 0; i < ports.size(); ++i)
        {
            if (ports[i] == config.makcu_port)
            {
                portIndex = static_cast<int>(i);
                break;
            }
        }

        if (ImGui::Combo("Makcu Port", &portIndex, portItems.data(), static_cast<int>(portItems.size())))
        {
            config.makcu_port = ports[portIndex];
            OverlayConfig_MarkDirty();
            input_method_changed.store(true);
        }

        std::vector<int> baudRates = { 115200, 230400, 460800, 921600, 1000000, 2000000, 4000000 };
        std::vector<std::string> baudStrings;
        for (int rate : baudRates)
            baudStrings.push_back(std::to_string(rate));

        std::vector<const char*> baudItems;
        baudItems.reserve(baudStrings.size());
        for (const auto& rate : baudStrings)
            baudItems.push_back(rate.c_str());

        int baudIndex = 0;
        for (size_t i = 0; i < baudRates.size(); ++i)
        {
            if (baudRates[i] == config.makcu_baudrate)
            {
                baudIndex = static_cast<int>(i);
                break;
            }
        }

        if (ImGui::Combo("Makcu Baudrate", &baudIndex, baudItems.data(), static_cast<int>(baudItems.size())))
        {
            config.makcu_baudrate = baudRates[baudIndex];
            OverlayConfig_MarkDirty();
            input_method_changed.store(true);
        }

        if (makcuSerial && makcuSerial->isOpen())
            ImGui::TextColored(ImVec4(0, 255, 0, 255), "Makcu connected");
        else
            ImGui::TextColored(ImVec4(255, 0, 0, 255), "Makcu not connected");
    }
}

}

void draw_mouse()
{
    if (OverlayUI::BeginSection("FOV", "mouse_section_fov"))
    {
        ImGui::SliderInt("FOV X", &config.fovX, 10, 120);
        ImGui::SliderInt("FOV Y", &config.fovY, 10, 120);
        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Pure PID Movement", "mouse_section_pid"))
    {
        ImGui::SliderInt("Actuator Hz", &config.pid_actuator_hz, 30, 2000);
        ImGui::SliderFloat("Kp", &config.pid_kp, 0.0f, 1.5f, "%.4f");
        ImGui::SliderFloat("Ki", &config.pid_ki, 0.0f, 0.5f, "%.4f");
        ImGui::SliderFloat("Kd", &config.pid_kd, 0.0f, 0.25f, "%.4f");
        ImGui::SliderFloat("Deadzone (px)", &config.pid_deadzone_px, 0.0f, 10.0f, "%.3f");
        ImGui::SliderFloat("Max step (px/tick)", &config.pid_max_pixel_step, 0.01f, 20.0f, "%.3f");
        ImGui::SliderFloat("Output scale", &config.pid_output_scale, 0.01f, 3.0f, "%.3f");
        ImGui::SliderFloat("Min output scale", &config.pid_min_output_scale, 0.0f, 3.0f, "%.3f");
        ImGui::SliderFloat("Max output scale", &config.pid_max_output_scale, 0.01f, 3.0f, "%.3f");
        ImGui::SliderFloat("Size reference (px)", &config.pid_size_reference_px, 1.0f, 240.0f, "%.1f");
        ImGui::SliderFloat("Small target scale", &config.pid_size_min_scale, 0.01f, 1.0f, "%.3f");
        ImGui::SliderFloat("Large target scale", &config.pid_size_max_scale, 0.05f, 2.0f, "%.3f");
        ImGui::SliderFloat("Precision radius / size", &config.pid_precision_radius_scale, 0.0f, 0.10f, "%.4f");
        ImGui::SliderFloat("Slowdown radius / size", &config.pid_slowdown_radius_scale, 0.01f, 1.0f, "%.3f");
        ImGui::SliderFloat("Overshoot brake", &config.pid_overshoot_brake, 0.01f, 1.0f, "%.3f");
        ImGui::SliderFloat("Divergence boost", &config.pid_divergence_boost, 0.0f, 2.0f, "%.3f");
        ImGui::SliderFloat("Scale response", &config.pid_scale_response, 0.1f, 40.0f, "%.1f");
        ImGui::SliderFloat("Max integral", &config.pid_max_integral, 0.0f, 10000.0f, "%.1f");
        ImGui::SliderFloat("Max derivative term", &config.pid_max_derivative_term, 0.0f, 5.0f, "%.3f");
        ImGui::SliderFloat("Derivative filter (ms)", &config.pid_derivative_filter_tau_ms, 0.0f, 250.0f, "%.1f");
        ImGui::SliderFloat("Target timeout (ms)", &config.pid_target_loss_timeout_ms, 10.0f, 1000.0f, "%.1f");
        ImGui::Checkbox("Feed-forward prediction", &config.pid_feed_forward_enabled);
        ImGui::SliderFloat("Feed-forward gain", &config.pid_feed_forward_gain, 0.0f, 2.0f, "%.3f");
        ImGui::SliderFloat("Feed-forward lookahead (ms)", &config.pid_feed_forward_lookahead_ms, 0.0f, 120.0f, "%.1f");
        ImGui::SliderFloat("Feed-forward max step (px/tick)", &config.pid_feed_forward_max_step, 0.0f, 5.0f, "%.3f");
        ImGui::SliderFloat("Feed-forward min speed (px/s)", &config.pid_feed_forward_min_speed, 0.0f, 3000.0f, "%.1f");
        ImGui::SliderFloat("Feed-forward confidence floor", &config.pid_feed_forward_confidence_floor, 0.0f, 1.0f, "%.3f");

        ImGui::Separator();
        ImGui::TextUnformatted("PID Governor");
        ImGui::Checkbox("Enable PID governor", &config.pid_governor_enabled);
        if (!config.pid_governor_enabled)
            ImGui::BeginDisabled();

        ImGui::SliderFloat("Governor blend", &config.pid_governor_blend, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Governor max speed multiple", &config.pid_governor_max_speed_multiple, 1.0f, 5.0f, "%.2f");

        if (!config.pid_governor_enabled)
            ImGui::EndDisabled();

        if (ImGui::Button("Reset PID defaults"))
        {
            config.pid_actuator_hz = 1000;
            config.pid_kp = 0.0085f;
            config.pid_ki = 0.0003f;
            config.pid_kd = 0.0001f;
            config.pid_deadzone_px = 0.0f;
            config.pid_max_pixel_step = 0.80f;
            config.pid_output_scale = 0.10f;
            config.pid_min_output_scale = 0.02f;
            config.pid_max_output_scale = 0.35f;
            config.pid_size_reference_px = 48.0f;
            config.pid_size_min_scale = 0.20f;
            config.pid_size_max_scale = 1.00f;
            config.pid_precision_radius_scale = 0.012f;
            config.pid_slowdown_radius_scale = 0.30f;
            config.pid_overshoot_brake = 0.35f;
            config.pid_divergence_boost = 0.35f;
            config.pid_scale_response = 8.0f;
            config.pid_max_integral = 120.0f;
            config.pid_max_derivative_term = 0.02f;
            config.pid_derivative_filter_tau_ms = 18.0f;
            config.pid_target_loss_timeout_ms = 90.0f;
            config.pid_feed_forward_enabled = true;
            config.pid_feed_forward_gain = 0.35f;
            config.pid_feed_forward_lookahead_ms = 24.0f;
            config.pid_feed_forward_max_step = 0.35f;
            config.pid_feed_forward_min_speed = 20.0f;
            config.pid_feed_forward_confidence_floor = 0.55f;
            OverlayConfig_MarkDirty();
            refreshMouseThread();
        }

        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Game Profile", "mouse_section_game_profile"))
    {
        drawGameProfileEditor();
        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Manage Profiles", "mouse_section_manage_profiles"))
    {
        drawProfileManager();
        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Auto Shoot", "mouse_section_auto_shoot"))
    {
        ImGui::Checkbox("Auto Shoot", &config.auto_shoot);
        if (!config.auto_shoot)
            ImGui::BeginDisabled();

        ImGui::SliderFloat("bScope Multiplier", &config.bScope_multiplier, 0.5f, 2.0f, "%.1f");

        if (!config.auto_shoot)
            ImGui::EndDisabled();

        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Input Method", "mouse_section_input_method"))
    {
        drawInputMethod();
        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Warning", "mouse_section_warning"))
    {
        ImGui::TextColored(ImVec4(255, 255, 255, 100), "Do not test shooting and aiming with the overlay open.");
        OverlayUI::EndSection();
    }

    if (prev_fovX != config.fovX ||
        prev_fovY != config.fovY ||
        prev_pid_actuator_hz != config.pid_actuator_hz ||
        prev_pid_kp != config.pid_kp ||
        prev_pid_ki != config.pid_ki ||
        prev_pid_kd != config.pid_kd ||
        prev_pid_deadzone_px != config.pid_deadzone_px ||
        prev_pid_max_pixel_step != config.pid_max_pixel_step ||
        prev_pid_output_scale != config.pid_output_scale ||
        prev_pid_min_output_scale != config.pid_min_output_scale ||
        prev_pid_max_output_scale != config.pid_max_output_scale ||
        prev_pid_size_reference_px != config.pid_size_reference_px ||
        prev_pid_size_min_scale != config.pid_size_min_scale ||
        prev_pid_size_max_scale != config.pid_size_max_scale ||
        prev_pid_precision_radius_scale != config.pid_precision_radius_scale ||
        prev_pid_slowdown_radius_scale != config.pid_slowdown_radius_scale ||
        prev_pid_overshoot_brake != config.pid_overshoot_brake ||
        prev_pid_divergence_boost != config.pid_divergence_boost ||
        prev_pid_scale_response != config.pid_scale_response ||
        prev_pid_max_integral != config.pid_max_integral ||
        prev_pid_max_derivative_term != config.pid_max_derivative_term ||
        prev_pid_derivative_filter_tau_ms != config.pid_derivative_filter_tau_ms ||
        prev_pid_target_loss_timeout_ms != config.pid_target_loss_timeout_ms ||
        prev_pid_feed_forward_enabled != config.pid_feed_forward_enabled ||
        prev_pid_feed_forward_gain != config.pid_feed_forward_gain ||
        prev_pid_feed_forward_lookahead_ms != config.pid_feed_forward_lookahead_ms ||
        prev_pid_feed_forward_max_step != config.pid_feed_forward_max_step ||
        prev_pid_feed_forward_min_speed != config.pid_feed_forward_min_speed ||
        prev_pid_feed_forward_confidence_floor != config.pid_feed_forward_confidence_floor ||
        prev_pid_governor_enabled != config.pid_governor_enabled ||
        prev_pid_governor_blend != config.pid_governor_blend ||
        prev_pid_governor_max_speed_multiple != config.pid_governor_max_speed_multiple ||
        prev_auto_shoot != config.auto_shoot ||
        prev_bScope_multiplier != config.bScope_multiplier)
    {
        prev_fovX = config.fovX;
        prev_fovY = config.fovY;
        prev_pid_actuator_hz = config.pid_actuator_hz;
        prev_pid_kp = config.pid_kp;
        prev_pid_ki = config.pid_ki;
        prev_pid_kd = config.pid_kd;
        prev_pid_deadzone_px = config.pid_deadzone_px;
        prev_pid_max_pixel_step = config.pid_max_pixel_step;
        prev_pid_output_scale = config.pid_output_scale;
        prev_pid_min_output_scale = config.pid_min_output_scale;
        prev_pid_max_output_scale = config.pid_max_output_scale;
        prev_pid_size_reference_px = config.pid_size_reference_px;
        prev_pid_size_min_scale = config.pid_size_min_scale;
        prev_pid_size_max_scale = config.pid_size_max_scale;
        prev_pid_precision_radius_scale = config.pid_precision_radius_scale;
        prev_pid_slowdown_radius_scale = config.pid_slowdown_radius_scale;
        prev_pid_overshoot_brake = config.pid_overshoot_brake;
        prev_pid_divergence_boost = config.pid_divergence_boost;
        prev_pid_scale_response = config.pid_scale_response;
        prev_pid_max_integral = config.pid_max_integral;
        prev_pid_max_derivative_term = config.pid_max_derivative_term;
        prev_pid_derivative_filter_tau_ms = config.pid_derivative_filter_tau_ms;
        prev_pid_target_loss_timeout_ms = config.pid_target_loss_timeout_ms;
        prev_pid_feed_forward_enabled = config.pid_feed_forward_enabled;
        prev_pid_feed_forward_gain = config.pid_feed_forward_gain;
        prev_pid_feed_forward_lookahead_ms = config.pid_feed_forward_lookahead_ms;
        prev_pid_feed_forward_max_step = config.pid_feed_forward_max_step;
        prev_pid_feed_forward_min_speed = config.pid_feed_forward_min_speed;
        prev_pid_feed_forward_confidence_floor = config.pid_feed_forward_confidence_floor;
        prev_pid_governor_enabled = config.pid_governor_enabled;
        prev_pid_governor_blend = config.pid_governor_blend;
        prev_pid_governor_max_speed_multiple = config.pid_governor_max_speed_multiple;
        prev_auto_shoot = config.auto_shoot;
        prev_bScope_multiplier = config.bScope_multiplier;

        refreshMouseThread();
        OverlayConfig_MarkDirty();
    }
}
