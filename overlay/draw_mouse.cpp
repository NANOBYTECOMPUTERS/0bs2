#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cstring>
#include <cstdio>
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
bool prev_runtime_latency_sweep_enabled = config.runtime_latency_sweep_enabled;
std::string prev_estimator_mode = config.estimator_mode;
bool prev_kalman_enabled = config.kalman_enabled;
float prev_kalman_process_noise_position = config.kalman_process_noise_position;
float prev_kalman_process_noise_velocity = config.kalman_process_noise_velocity;
float prev_kalman_measurement_noise = config.kalman_measurement_noise;
float prev_kalman_velocity_damping = config.kalman_velocity_damping;
float prev_kalman_max_velocity = config.kalman_max_velocity;
int prev_kalman_warmup_frames = config.kalman_warmup_frames;
bool prev_kalman_velocity_seed_enabled = config.kalman_velocity_seed_enabled;
int prev_kalman_acquisition_frames = config.kalman_acquisition_frames;
bool prev_kalman_compensate_detection_delay = config.kalman_compensate_detection_delay;
float prev_kalman_additional_prediction_ms = config.kalman_additional_prediction_ms;
float prev_kalman_reset_timeout_sec = config.kalman_reset_timeout_sec;
bool prev_ego_motion_compensation_enabled = config.ego_motion_compensation_enabled;
float prev_ego_motion_compensation_strength = config.ego_motion_compensation_strength;
float prev_ego_motion_compensation_max_shift_px = config.ego_motion_compensation_max_shift_px;
int prev_ego_motion_compensation_max_age_ms = config.ego_motion_compensation_max_age_ms;
float prev_target_deadzone_px = config.target_deadzone_px;
bool prev_target_stream_enabled = config.target_stream_enabled;
float prev_target_stream_interval_ms = config.target_stream_interval_ms;
float prev_target_stream_sharpness = config.target_stream_sharpness;
float prev_target_max_pixel_speed = config.target_max_pixel_speed;
int prev_target_state_max_age_ms = config.target_state_max_age_ms;
float prev_target_min_stream_confidence = config.target_min_stream_confidence;
float prev_target_max_pixel_step = config.target_max_pixel_step;
float prev_target_output_scale = config.target_output_scale;
bool prev_target_calibrated_pixel_counts_enabled = config.target_calibrated_pixel_counts_enabled;
float prev_target_counts_per_pixel_x = config.target_counts_per_pixel_x;
float prev_target_counts_per_pixel_y = config.target_counts_per_pixel_y;
float prev_target_prediction_blend = config.target_prediction_blend;
float prev_target_prediction_max_lead_px = config.target_prediction_max_lead_px;
bool prev_auto_shoot = config.auto_shoot;
float prev_bScope_multiplier = config.bScope_multiplier;

void refreshMouseThread()
{
    if (!globalMouseThread)
        return;

    globalMouseThread->updateConfig(
        config.detection_resolution,
        config.auto_shoot,
        config.bScope_multiplier);
}

void drawInputMethod()
{
    std::vector<std::string> inputMethods = { "WIN32", "GHUB", "RAZER", "DIRECT", "ARDUINO", "TEENSY41", "TEENSY41_HID", "KMBOX_NET", "KMBOX_A", "MAKCU" };
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

    if (config.input_method == "ARDUINO" || config.input_method == "TEENSY41")
    {
        if (arduinoSerial && arduinoSerial->isOpen())
            ImGui::TextColored(ImVec4(0, 255, 0, 255), config.input_method == "TEENSY41" ? "Teensy 4.1 connected" : "Arduino connected");
        else
            ImGui::TextColored(ImVec4(255, 0, 0, 255), config.input_method == "TEENSY41" ? "Teensy 4.1 not connected" : "Arduino not connected");

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

        if (ImGui::Combo(config.input_method == "TEENSY41" ? "Teensy Port" : "Arduino Port", &portIndex, portItems.data(), static_cast<int>(portItems.size())))
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

        if (ImGui::Combo(config.input_method == "TEENSY41" ? "Teensy Baudrate" : "Arduino Baudrate", &baudIndex, baudItems.data(), static_cast<int>(baudItems.size())))
        {
            config.arduino_baudrate = baudRates[baudIndex];
            OverlayConfig_MarkDirty();
            input_method_changed.store(true);
        }

        if (config.input_method == "TEENSY41")
        {
            ImGui::TextWrapped("Uses Teensy command protocol: move dx dy 0 0. Select the Teensy COM port.");
        }
        else if (ImGui::Checkbox("Arduino 16-bit Mouse", &config.arduino_16_bit_mouse) ||
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
    else if (config.input_method == "TEENSY41_HID")
    {
        if (activeMouseInputOwner && activeMouseInputOwner->isOpen())
            ImGui::TextColored(ImVec4(0, 255, 0, 255), "Generic HID Mouse connected");
        else
            ImGui::TextColored(ImVec4(255, 0, 0, 255), "Generic HID Mouse not connected");

        static char manufacturer[64] = "";
        static char product[64] = "";
        static char serial[64] = "";
        static char vidFilter[16] = "";
        static char pidFilter[16] = "";
        static std::string lastManufacturer;
        static std::string lastProduct;
        static std::string lastSerial;
        static std::string lastVidFilter;
        static std::string lastPidFilter;

        auto syncHidText = [&]() {
            if (lastManufacturer != config.teensy_hid_manufacturer)
            {
                strncpy(manufacturer, config.teensy_hid_manufacturer.c_str(), sizeof(manufacturer));
                manufacturer[sizeof(manufacturer) - 1] = '\0';
                lastManufacturer = config.teensy_hid_manufacturer;
            }
            if (lastProduct != config.teensy_hid_product)
            {
                strncpy(product, config.teensy_hid_product.c_str(), sizeof(product));
                product[sizeof(product) - 1] = '\0';
                lastProduct = config.teensy_hid_product;
            }
            if (lastSerial != config.teensy_hid_serial)
            {
                strncpy(serial, config.teensy_hid_serial.c_str(), sizeof(serial));
                serial[sizeof(serial) - 1] = '\0';
                lastSerial = config.teensy_hid_serial;
            }
            if (lastVidFilter != config.teensy_hid_vid_filter)
            {
                strncpy(vidFilter, config.teensy_hid_vid_filter.c_str(), sizeof(vidFilter));
                vidFilter[sizeof(vidFilter) - 1] = '\0';
                lastVidFilter = config.teensy_hid_vid_filter;
            }
            if (lastPidFilter != config.teensy_hid_pid_filter)
            {
                strncpy(pidFilter, config.teensy_hid_pid_filter.c_str(), sizeof(pidFilter));
                pidFilter[sizeof(pidFilter) - 1] = '\0';
                lastPidFilter = config.teensy_hid_pid_filter;
            }
        };
        syncHidText();

        ImGui::InputText("Manufacturer", manufacturer, sizeof(manufacturer));
        ImGui::InputText("Product", product, sizeof(product));
        ImGui::InputText("Serial", serial, sizeof(serial));
        ImGui::InputText("VID filter", vidFilter, sizeof(vidFilter));
        ImGui::InputText("PID filter", pidFilter, sizeof(pidFilter));
        bool hidNumericChanged = false;
        hidNumericChanged |= ImGui::SliderInt("Usage page", &config.teensy_hid_usage_page, 1, 0xFFFF);
        hidNumericChanged |= ImGui::SliderInt("Usage ID", &config.teensy_hid_usage_id, 1, 0xFFFF);
        hidNumericChanged |= ImGui::SliderInt("Open index", &config.teensy_hid_open_index, 0, 32);
        hidNumericChanged |= ImGui::SliderInt("Packet timeout ms", &config.teensy_hid_packet_timeout_ms, 0, 100);
        hidNumericChanged |= ImGui::SliderInt("Reconnect ms", &config.teensy_hid_reconnect_interval_ms, 50, 10000);
        if (hidNumericChanged)
        {
            OverlayConfig_MarkDirty();
            input_method_changed.store(true);
        }

        if (ImGui::Button("Save & Reconnect##teensy_hid"))
        {
            config.teensy_hid_manufacturer = manufacturer;
            config.teensy_hid_product = product;
            config.teensy_hid_serial = serial;
            config.teensy_hid_vid_filter = vidFilter;
            config.teensy_hid_pid_filter = pidFilter;
            lastManufacturer = config.teensy_hid_manufacturer;
            lastProduct = config.teensy_hid_product;
            lastSerial = config.teensy_hid_serial;
            lastVidFilter = config.teensy_hid_vid_filter;
            lastPidFilter = config.teensy_hid_pid_filter;
            OverlayConfig_MarkDirty();
            input_method_changed.store(true);
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset HID defaults"))
        {
            config.teensy_hid_manufacturer = "Generic";
            config.teensy_hid_product = "USB HID Mouse";
            config.teensy_hid_serial = "AUTO";
            config.teensy_hid_vid_filter = "AUTO";
            config.teensy_hid_pid_filter = "AUTO";
            config.teensy_hid_usage_page = 0xFFAB;
            config.teensy_hid_usage_id = 0x0200;
            config.teensy_hid_open_index = 0;
            config.teensy_hid_packet_timeout_ms = 2;
            config.teensy_hid_reconnect_interval_ms = 500;
            lastManufacturer.clear();
            lastProduct.clear();
            lastSerial.clear();
            lastVidFilter.clear();
            lastPidFilter.clear();
            syncHidText();
            OverlayConfig_MarkDirty();
            input_method_changed.store(true);
        }
    }
    else if (config.input_method == "RAZER")
    {
        if (razerControl && razerControl->isOpen())
            ImGui::TextColored(ImVec4(0, 255, 0, 255), "Razer (direct in-process) connected");
        else
            ImGui::TextColored(ImVec4(255, 0, 0, 255), "Razer (direct) not connected");

        ImGui::TextWrapped("Uses in-process Razer Synapse driver interface (RZCONTROL). On recent Synapse versions a physical Razer device is often required for the driver to expose the control device.");
    }
    else if (config.input_method == "DIRECT")
    {
        // DIRECT is a research/architecture slot per the stealth-first plan.
        // Real kernel-level "direct driver injection" is high-risk on modern ACs
        // (driver load telemetry). The slot exists so the architecture stays uniform.
        // Initial implementation is a safe stub (isOpen()==false).
        if (activeMouseInputOwner && activeMouseInputOwner->isOpen())
            ImGui::TextColored(ImVec4(0, 255, 0, 255), "DIRECT active");
        else
            ImGui::TextColored(ImVec4(255, 165, 0, 255), "DIRECT (research slot) — inactive");

        ImGui::TextWrapped("Direct driver injection slot. On many 2025-2026 ACs a custom kernel driver often carries HIGHER detection risk than the inlined RAZER path or hardware backends. Only enable after current research confirms a lower-signal technique for your targets. Most implementations require admin + a kernel component.");
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
    if (OverlayUI::BeginSection("State Estimator", "mouse_section_state_estimator"))
    {
        ImGui::Checkbox("Runtime latency sweep", &config.runtime_latency_sweep_enabled);
        const char* estimatorModes[] = { "kalman", "imm" };
        int estimatorModeIndex = (config.estimator_mode == "imm") ? 1 : 0;
        if (ImGui::Combo("Estimator mode", &estimatorModeIndex, estimatorModes, IM_ARRAYSIZE(estimatorModes)))
        {
            config.estimator_mode = estimatorModes[estimatorModeIndex];
        }
        ImGui::Checkbox("Enable Kalman estimator", &config.kalman_enabled);
        ImGui::Checkbox("Seed velocity on acquire", &config.kalman_velocity_seed_enabled);
        ImGui::SliderInt("Acquisition frames", &config.kalman_acquisition_frames, 3, 5);
        ImGui::SliderFloat("Process noise position", &config.kalman_process_noise_position, 0.0001f, 5000.0f, "%.3f");
        ImGui::SliderFloat("Process noise velocity", &config.kalman_process_noise_velocity, 0.0001f, 50000.0f, "%.3f");
        ImGui::SliderFloat("Measurement noise", &config.kalman_measurement_noise, 0.0001f, 5000.0f, "%.3f");
        ImGui::SliderFloat("Velocity damping", &config.kalman_velocity_damping, 0.0f, 3.0f, "%.3f");
        ImGui::SliderFloat("Max velocity (px/s)", &config.kalman_max_velocity, 100.0f, 60000.0f, "%.0f");
        ImGui::SliderInt("Warmup frames", &config.kalman_warmup_frames, 0, 20);
        ImGui::Checkbox("Compensate detection delay", &config.kalman_compensate_detection_delay);
        ImGui::SliderFloat("Additional prediction (ms)", &config.kalman_additional_prediction_ms, -80.0f, 120.0f, "%.1f");
        ImGui::SliderFloat("Reset timeout (s)", &config.kalman_reset_timeout_sec, 0.05f, 3.0f, "%.2f");
        ImGui::Separator();
        ImGui::Checkbox("Ego-motion compensation", &config.ego_motion_compensation_enabled);
        ImGui::SliderFloat("Ego compensation strength", &config.ego_motion_compensation_strength, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Ego max shift (px @640)", &config.ego_motion_compensation_max_shift_px, 1.0f, 128.0f, "%.1f");
        ImGui::SliderInt("Ego max age (ms)", &config.ego_motion_compensation_max_age_ms, 16, 500);
        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Direct Targeting Movement", "mouse_section_direct_targeting"))
    {
        ImGui::SliderFloat("Deadzone (px)", &config.target_deadzone_px, 0.0f, 20.0f, "%.3f");
        ImGui::Checkbox("Target stream", &config.target_stream_enabled);
        if (!config.target_stream_enabled)
            ImGui::BeginDisabled();

        ImGui::SliderFloat("Stream interval (ms)", &config.target_stream_interval_ms, 0.5f, 8.0f, "%.2f");
        ImGui::SliderFloat("Stream sharpness", &config.target_stream_sharpness, 1.0f, 80.0f, "%.2f");
        ImGui::SliderFloat("Max speed (px/s)", &config.target_max_pixel_speed, 50.0f, 20000.0f, "%.0f");
        ImGui::SliderInt("State max age (ms)", &config.target_state_max_age_ms, 16, 500);
        ImGui::SliderFloat("Min stream confidence", &config.target_min_stream_confidence, 0.0f, 0.95f, "%.2f");
        ImGui::SliderFloat("Prediction blend", &config.target_prediction_blend, 0.0f, 0.65f, "%.3f");
        ImGui::SliderFloat("Max prediction lead (px)", &config.target_prediction_max_lead_px, 0.0f, 40.0f, "%.2f");

        if (!config.target_stream_enabled)
            ImGui::EndDisabled();

        ImGui::Separator();
        ImGui::SliderFloat("Fallback max step (px/call)", &config.target_max_pixel_step, 0.25f, 120.0f, "%.2f");
        ImGui::SliderFloat("Fallback output scale", &config.target_output_scale, 0.01f, 3.0f, "%.3f");
        ImGui::Checkbox("Calibrated pixel counts", &config.target_calibrated_pixel_counts_enabled);
        if (!config.target_calibrated_pixel_counts_enabled)
            ImGui::BeginDisabled();

        ImGui::SliderFloat("Counts / px X", &config.target_counts_per_pixel_x, -50.0f, 50.0f, "%.4f");
        ImGui::SliderFloat("Counts / px Y", &config.target_counts_per_pixel_y, -50.0f, 50.0f, "%.4f");

        if (!config.target_calibrated_pixel_counts_enabled)
            ImGui::EndDisabled();

        if (ImGui::Button("Reset direct defaults"))
        {
            config.target_deadzone_px = 0.0f;
            config.target_stream_enabled = true;
            config.target_stream_interval_ms = 1.0f;
            config.target_stream_sharpness = 18.0f;
            config.target_max_pixel_speed = 1800.0f;
            config.target_state_max_age_ms = 120;
            config.target_min_stream_confidence = 0.15f;
            config.target_max_pixel_step = 28.0f;
            config.target_output_scale = 0.28f;
            config.target_calibrated_pixel_counts_enabled = false;
            config.target_counts_per_pixel_x = 0.0f;
            config.target_counts_per_pixel_y = 0.0f;
            config.target_prediction_blend = 0.18f;
            config.target_prediction_max_lead_px = 8.0f;
            OverlayConfig_MarkDirty();
            refreshMouseThread();
        }

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

    if (prev_runtime_latency_sweep_enabled != config.runtime_latency_sweep_enabled ||
        prev_estimator_mode != config.estimator_mode ||
        prev_kalman_enabled != config.kalman_enabled ||
        prev_kalman_process_noise_position != config.kalman_process_noise_position ||
        prev_kalman_process_noise_velocity != config.kalman_process_noise_velocity ||
        prev_kalman_measurement_noise != config.kalman_measurement_noise ||
        prev_kalman_velocity_damping != config.kalman_velocity_damping ||
        prev_kalman_max_velocity != config.kalman_max_velocity ||
        prev_kalman_warmup_frames != config.kalman_warmup_frames ||
        prev_kalman_velocity_seed_enabled != config.kalman_velocity_seed_enabled ||
        prev_kalman_acquisition_frames != config.kalman_acquisition_frames ||
        prev_kalman_compensate_detection_delay != config.kalman_compensate_detection_delay ||
        prev_kalman_additional_prediction_ms != config.kalman_additional_prediction_ms ||
        prev_kalman_reset_timeout_sec != config.kalman_reset_timeout_sec ||
        prev_ego_motion_compensation_enabled != config.ego_motion_compensation_enabled ||
        prev_ego_motion_compensation_strength != config.ego_motion_compensation_strength ||
        prev_ego_motion_compensation_max_shift_px != config.ego_motion_compensation_max_shift_px ||
        prev_ego_motion_compensation_max_age_ms != config.ego_motion_compensation_max_age_ms ||
        prev_target_deadzone_px != config.target_deadzone_px ||
        prev_target_stream_enabled != config.target_stream_enabled ||
        prev_target_stream_interval_ms != config.target_stream_interval_ms ||
        prev_target_stream_sharpness != config.target_stream_sharpness ||
        prev_target_max_pixel_speed != config.target_max_pixel_speed ||
        prev_target_state_max_age_ms != config.target_state_max_age_ms ||
        prev_target_min_stream_confidence != config.target_min_stream_confidence ||
        prev_target_max_pixel_step != config.target_max_pixel_step ||
        prev_target_output_scale != config.target_output_scale ||
        prev_target_calibrated_pixel_counts_enabled != config.target_calibrated_pixel_counts_enabled ||
        prev_target_counts_per_pixel_x != config.target_counts_per_pixel_x ||
        prev_target_counts_per_pixel_y != config.target_counts_per_pixel_y ||
        prev_target_prediction_blend != config.target_prediction_blend ||
        prev_target_prediction_max_lead_px != config.target_prediction_max_lead_px ||
        prev_auto_shoot != config.auto_shoot ||
        prev_bScope_multiplier != config.bScope_multiplier)
    {
        prev_runtime_latency_sweep_enabled = config.runtime_latency_sweep_enabled;
        prev_estimator_mode = config.estimator_mode;
        prev_kalman_enabled = config.kalman_enabled;
        prev_kalman_process_noise_position = config.kalman_process_noise_position;
        prev_kalman_process_noise_velocity = config.kalman_process_noise_velocity;
        prev_kalman_measurement_noise = config.kalman_measurement_noise;
        prev_kalman_velocity_damping = config.kalman_velocity_damping;
        prev_kalman_max_velocity = config.kalman_max_velocity;
        prev_kalman_warmup_frames = config.kalman_warmup_frames;
        prev_kalman_velocity_seed_enabled = config.kalman_velocity_seed_enabled;
        prev_kalman_acquisition_frames = config.kalman_acquisition_frames;
        prev_kalman_compensate_detection_delay = config.kalman_compensate_detection_delay;
        prev_kalman_additional_prediction_ms = config.kalman_additional_prediction_ms;
        prev_kalman_reset_timeout_sec = config.kalman_reset_timeout_sec;
        prev_ego_motion_compensation_enabled = config.ego_motion_compensation_enabled;
        prev_ego_motion_compensation_strength = config.ego_motion_compensation_strength;
        prev_ego_motion_compensation_max_shift_px = config.ego_motion_compensation_max_shift_px;
        prev_ego_motion_compensation_max_age_ms = config.ego_motion_compensation_max_age_ms;
        prev_target_deadzone_px = config.target_deadzone_px;
        prev_target_stream_enabled = config.target_stream_enabled;
        prev_target_stream_interval_ms = config.target_stream_interval_ms;
        prev_target_stream_sharpness = config.target_stream_sharpness;
        prev_target_max_pixel_speed = config.target_max_pixel_speed;
        prev_target_state_max_age_ms = config.target_state_max_age_ms;
        prev_target_min_stream_confidence = config.target_min_stream_confidence;
        prev_target_max_pixel_step = config.target_max_pixel_step;
        prev_target_output_scale = config.target_output_scale;
        prev_target_calibrated_pixel_counts_enabled = config.target_calibrated_pixel_counts_enabled;
        prev_target_counts_per_pixel_x = config.target_counts_per_pixel_x;
        prev_target_counts_per_pixel_y = config.target_counts_per_pixel_y;
        prev_target_prediction_blend = config.target_prediction_blend;
        prev_target_prediction_max_lead_px = config.target_prediction_max_lead_px;
        prev_auto_shoot = config.auto_shoot;
        prev_bScope_multiplier = config.bScope_multiplier;

        refreshMouseThread();
        OverlayConfig_MarkDirty();
    }
}
