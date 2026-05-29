#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cctype>

#include "config.h"
#include "MouseInput.h"
#include "modules/SimpleIni.h"

std::vector<std::string> Config::splitString(const std::string& str, char delimiter)
{
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delimiter))
    {
        while (!item.empty() && (item.front() == ' ' || item.front() == '\t'))
            item.erase(item.begin());
        while (!item.empty() && (item.back() == ' ' || item.back() == '\t'))
            item.pop_back();

        tokens.push_back(item);
    }
    return tokens;
}

std::string Config::joinStrings(const std::vector<std::string>& vec, const std::string& delimiter)
{
    std::ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i)
    {
        if (i != 0) oss << delimiter;
        oss << vec[i];
    }
    return oss.str();
}

bool Config::loadConfig(const std::string& filename)
{
    std::string target = filename.empty() ? "config.ini" : filename;
    std::error_code absEc;
    std::filesystem::path absPath = std::filesystem::absolute(target, absEc);
    config_path = absEc ? target : absPath.string();

    if (!std::filesystem::exists(target))
    {
        std::cerr << "[Config] Config file does not exist, creating default config: " << target << std::endl;

        // Capture
        capture_method = "duplication_api";
        capture_target = "monitor";
        capture_window_title = "";
        udp_ip = "0.0.0.0";
        udp_port = 1234;
        detection_resolution = 320;
        capture_fps = 60;
        monitor_idx = 0;
        circle_mask = true;
        capture_borders = true;
        capture_cursor = true;
        virtual_camera_name = "None";
        virtual_camera_width = 1920;
        virtual_camera_heigth = 1080;

        // Target
        disable_headshot = false;
        body_y_offset = 0.15f;
        head_y_offset = 0.05f;
        auto_aim = false;

        // Mouse
        fovX = 106;
        fovY = 74;
        minSpeedMultiplier = 0.1f;
        maxSpeedMultiplier = 0.1f;

        predictionInterval = 0.01f;
        prediction_futurePositions = 20;
        draw_futurePositions = true;
        runtime_latency_sweep_enabled = false;
        estimator_mode = "kalman";
        kalman_enabled = true;
        kalman_process_noise_position = 40.0f;
        kalman_process_noise_velocity = 1800.0f;
        kalman_measurement_noise = 35.0f;
        kalman_velocity_damping = 0.08f;
        kalman_max_velocity = 20000.0f;
        kalman_warmup_frames = 2;
        kalman_velocity_seed_enabled = true;
        kalman_acquisition_frames = 4;
        kalman_compensate_detection_delay = true;
        kalman_additional_prediction_ms = 0.0f;
        kalman_reset_timeout_sec = 0.5f;
        ego_motion_compensation_enabled = false;
        ego_motion_compensation_strength = 0.80f;
        ego_motion_compensation_max_shift_px = 32.0f;
        ego_motion_compensation_max_age_ms = 120;

        snapRadius = 1.5f;
        nearRadius = 25.0f;
        speedCurveExponent = 3.0f;
        snapBoostFactor = 1.15f;

        easynorecoil = false;
        easynorecoilstrength = 0.0f;
        input_method = "RAZER";

        // Wind mouse
        wind_mouse_enabled = false;
        wind_G = 18.0f;
        wind_W = 15.0f;
        wind_M = 10.0f;
        wind_D = 8.0f;

        // Pure PID mouse control
        pid_actuator_hz = 1000;
        pid_kp = 0.0085f;
        pid_ki = 0.0003f;
        pid_kd = 0.0001f;
        pid_deadzone_px = 0.0f;
        pid_max_pixel_step = 0.80f;
        pid_output_scale = 0.10f;
        pid_min_output_scale = 0.02f;
        pid_max_output_scale = 0.35f;
        pid_size_reference_px = 48.0f;
        pid_size_min_scale = 0.20f;
        pid_size_max_scale = 1.00f;
        pid_precision_radius_scale = 0.012f;
        pid_slowdown_radius_scale = 0.30f;
        pid_overshoot_brake = 0.35f;
        pid_divergence_boost = 0.35f;
        pid_scale_response = 8.0f;
        pid_max_integral = 120.0f;
        pid_max_derivative_term = 0.02f;
        pid_derivative_filter_tau_ms = 18.0f;
        pid_target_loss_timeout_ms = 90.0f;
        pid_feed_forward_enabled = true;
        pid_feed_forward_gain = 0.35f;
        pid_feed_forward_lookahead_ms = 24.0f;
        pid_feed_forward_frame_lookahead = 1;
        pid_feed_forward_max_step = 0.35f;
        pid_feed_forward_min_speed = 20.0f;
        pid_feed_forward_confidence_floor = 0.55f;
        pid_conditional_integration_enabled = true;
        pid_conditional_integration_error_px = 12.0f;
        pid_adaptive_output_scaling_enabled = true;
        pid_adaptive_output_error_scale = 96.0f;
        pid_derivative_smoothing_multiplier = 1.5f;
        pid_perspective_fov_mapping_enabled = false;
        pid_governor_enabled = false;
        pid_governor_model_path = "training/models/pid_governor.onnx";
        pid_governor_blend = 1.0f;
        pid_governor_max_speed_multiple = 5.0f;
        pid_smart_blending_enabled = false;
        pid_smart_blending_aggression = 0.65f;
        pid_smart_blending_near_damping = 0.75f;
        pid_smart_blending_deadzone_px = 0.0f;
        pid_smart_blending_jerk_limit_px = 0.65f;
        pid_smart_blending_confidence_floor = 0.45f;

        // Neural tracker association
        neural_tracker_enabled = false;
        neural_tracker_runtime = "CPU";
        neural_tracker_model_path = "training/models/neural_tracker.onnx";
        neural_tracker_blend = 0.35f;
        neural_tracker_log_enabled = false;
        neural_tracker_debug_enabled = false;
        neural_tracker_log_path = "training/logs/neural_tracker_association.csv";

        // Learned temporal predictor
        temporal_prediction_enabled = false;
        temporal_prediction_model_path = "models/temporal_predictor.onnx";
        temporal_prediction_history_length = 12;
        temporal_prediction_horizon = 16;
        temporal_prediction_interval_frames = 2;
        temporal_prediction_feed_forward_enabled = false;
        temporal_prediction_influence = 0.35f;
        temporal_prediction_adaptive_influence_enabled = false;
        temporal_prediction_adaptive_ema_alpha = 0.62f;
        temporal_prediction_max_lead_px = 45.0f;

        // Neural targeting head
        neural_targeting_enabled = false;
        neural_targeting_model_path = "models/neural_targeting_head.onnx";
        neural_targeting_influence = 0.40f;
        neural_targeting_max_refinement_px = 35.0f;
        neural_targeting_max_iterations = 2;
        neural_control_preset = "Balanced";
        neural_control_telemetry_overlay_enabled = false;
        neural_control_telemetry_logging_enabled = false;
        neural_control_telemetry_log_path = "logs/neural_control_telemetry.csv";
        neural_control_telemetry_log_interval_ms = 250;

        // Arduino
        arduino_baudrate = 115200;
        arduino_port = "COM0";
        arduino_16_bit_mouse = false;
        arduino_enable_keys = false;

        // Teensy 4.1 RawHID generic mouse bridge
        teensy_hid_manufacturer = "Generic";
        teensy_hid_product = "USB HID Mouse";
        teensy_hid_serial = "AUTO";
        teensy_hid_vid_filter = "AUTO";
        teensy_hid_pid_filter = "AUTO";
        teensy_hid_usage_page = 0xFFAB;
        teensy_hid_usage_id = 0x0200;
        teensy_hid_open_index = 0;
        teensy_hid_packet_timeout_ms = 2;
        teensy_hid_reconnect_interval_ms = 500;

        // kmbox_net
        kmbox_net_ip = "10.42.42.42";
        kmbox_net_port = "1984";
        kmbox_net_uuid = "DEADC0DE";

        // kmbox_a
        kmbox_a_pidvid = "";

        // makcu
        makcu_baudrate = 4000000;
        makcu_port = "AUTO";

        // Mouse shooting
        auto_shoot = false;
        bScope_multiplier = 1.0f;

        // AI
#ifdef USE_CUDA
        backend = "TRT";
#else
        backend = "DML";
#endif
        dml_device_id = 0;

#ifdef USE_CUDA
        ai_model = "sunxds_0.5.6.engine";
#else
        ai_model = "sunxds_0.5.6.onnx";
#endif

        confidence_threshold = 0.10f;
        nms_threshold = 0.50f;
        max_detections = 100;
#ifdef USE_CUDA
        export_enable_fp8 = false;
        export_enable_fp16 = true;
#endif
        fixed_input_size = false;

        // CUDA
#ifdef USE_CUDA
        use_cuda_graph = false;
        use_pinned_memory = false;
        gpuMemoryReserveMB = 2048;
        enableGpuExclusiveMode = true;
        capture_use_cuda = true;
#endif

        // System
        cpuCoreReserveCount = 4;
        systemMemoryReserveMB = 2048;

        // Buttons
        button_targeting = splitString("RightMouseButton");
        button_shoot = splitString("LeftMouseButton");
        button_zoom = splitString("RightMouseButton");
        button_exit = splitString("F2");
        button_pause = splitString("F3");
        button_reload_config = splitString("F4");
        button_open_overlay = splitString("Home");
        enable_arrows_settings = false;

        // Overlay
        overlay_opacity = 225;
        overlay_ui_scale = 1.0f;
        overlay_exclude_from_capture = true;

        // Depth
        depth_inference_enabled = true;
        depth_model_path = "depth_anything_v2.engine";
        depth_fps = 100;
        depth_colormap = 18;
        depth_mask_enabled = false;
        depth_mask_fps = 5;
        depth_mask_near_percent = 20;
        depth_mask_expand = 0;
        depth_mask_hold_frames = 0;
        depth_mask_alpha = 90;
        depth_mask_invert = false;
        depth_debug_overlay_enabled = false;

        // Game overlay
        game_overlay_enabled = false;
        game_overlay_max_fps = 0;
        game_overlay_draw_boxes = true;
        game_overlay_draw_future = true;
        game_overlay_draw_wind_tail = true;
        game_overlay_draw_frame = true;
        game_overlay_show_target_correction = true;
        game_overlay_box_a = 255;
        game_overlay_box_r = 0;
        game_overlay_box_g = 255;
        game_overlay_box_b = 0;
        game_overlay_frame_a = 180;
        game_overlay_frame_r = 255;
        game_overlay_frame_g = 255;
        game_overlay_frame_b = 255;
        game_overlay_box_thickness = 2.0f;
        game_overlay_frame_thickness = 1.5f;
        game_overlay_future_point_radius = 5.0f;
        game_overlay_future_alpha_falloff = 1.0f;

        game_overlay_icon_enabled = false;
        game_overlay_icon_path = "icon.png";
        game_overlay_icon_width = 64;
        game_overlay_icon_height = 64;
        game_overlay_icon_offset_x = 0.0f;
        game_overlay_icon_offset_y = 0.0f;
        game_overlay_icon_anchor = "center";
        game_overlay_icon_class = -1;

        // Aim simulation overlay
        aim_sim_enabled = false;
        aim_sim_x = 24;
        aim_sim_y = 24;
        aim_sim_width = 560;
        aim_sim_height = 360;
        aim_sim_fps_min = 90;
        aim_sim_fps_max = 120;
        aim_sim_fps_jitter = 0.15f;
        aim_sim_capture_delay_ms = 6.0f;
        aim_sim_inference_delay_ms = 12.0f;
        aim_sim_use_live_inference = true;
        aim_sim_input_delay_ms = 2.0f;
        aim_sim_extra_delay_ms = 2.0f;
        aim_sim_target_max_speed = 560.0f;
        aim_sim_target_accel = 1850.0f;
        aim_sim_target_stop_chance = 0.25f;
        aim_sim_show_observed = true;
        aim_sim_show_history = true;
        aim_sim_show_kalman_debug = true;

        // Classes
        class_player = 0;
        class_head = 1;

        // Debug
        show_window = true;
        show_fps = false;
        screenshot_button = splitString("None");
        screenshot_delay = 500;
        verbose = false;
        debug_log_file_enabled = false;
        debug_log_file_path = "logs/0BS.log";

        // Game profiles
        game_profiles.clear();
        GameProfile uni;
        uni.name = "UNIFIED";
        uni.sens = 1.0;
        uni.yaw = 0.022;
        uni.pitch = uni.yaw;
        uni.fovScaled = false;
        uni.baseFOV = 0.0;
        game_profiles[uni.name] = uni;
        active_game = uni.name;

        saveConfig(target);
        return true;
    }

    CSimpleIniA ini;
    ini.SetUnicode();
    SI_Error rc = ini.LoadFile(target.c_str());
    if (rc < 0)
    {
        std::cerr << "[Config] Error parsing INI file: " << target << std::endl;
        return false;
    }

    auto get_string = [&](const char* key, const std::string& defval)
    {
        const char* val = ini.GetValue("", key, defval.c_str());
        return val ? std::string(val) : defval;
    };

    auto get_bool = [&](const char* key, bool defval)
        {
            return ini.GetBoolValue("", key, defval);
        };

    auto get_long = [&](const char* key, long defval)
        {
            return (int)ini.GetLongValue("", key, defval);
        };

    auto get_double = [&](const char* key, double defval)
        {
            return ini.GetDoubleValue("", key, defval);
        };

    game_profiles.clear();

    CSimpleIniA::TNamesDepend keys;
    ini.GetAllKeys("Games", keys);

    for (const auto& k : keys)
    {
        std::string name = k.pItem;
        std::string val = ini.GetValue("Games", k.pItem, "");
        auto parts = splitString(val, ',');

        try
        {
            if (parts.size() < 2)
                throw std::runtime_error("not enough values");

            GameProfile gp;
            gp.name = name;
            gp.sens = std::stod(parts[0]);
            gp.yaw = std::stod(parts[1]);
            gp.pitch = parts.size() > 2 ? std::stod(parts[2]) : gp.yaw;
            gp.fovScaled = parts.size() > 3 && (parts[3] == "true" || parts[3] == "1");
            gp.baseFOV = parts.size() > 4 ? std::stod(parts[4]) : 0.0;

            game_profiles[name] = gp;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[Config] Failed to parse profile: " << name
                << " = " << val << " (" << e.what() << ")" << std::endl;
        }
    }

    if (!game_profiles.count("UNIFIED"))
    {
        GameProfile uni;
        uni.name = "UNIFIED";
        uni.sens = 1.0;
        uni.yaw = 0.022;
        uni.pitch = uni.yaw;
        uni.fovScaled = false;
        uni.baseFOV = 0.0;
        game_profiles[uni.name] = uni;
    }

    active_game = get_string("active_game", active_game);
    if (!game_profiles.count(active_game) && !game_profiles.empty())
        active_game = game_profiles.begin()->first;

    // Capture
    capture_method = get_string("capture_method", "duplication_api");
    capture_target = get_string("capture_target", "monitor");
    capture_window_title = get_string("capture_window_title", "");
    udp_ip = get_string("udp_ip", "0.0.0.0");
    udp_port = get_long("udp_port", 1234);
    if (udp_port < 1 || udp_port > 65535)
        udp_port = 1234;
    detection_resolution = get_long("detection_resolution", 320);
    if (detection_resolution != 160 && detection_resolution != 320 && detection_resolution != 640)
        detection_resolution = 320;

    capture_fps = get_long("capture_fps", 60);
    monitor_idx = get_long("monitor_idx", 0);
    circle_mask = get_bool("circle_mask", true);
    capture_borders = get_bool("capture_borders", true);
    capture_cursor = get_bool("capture_cursor", true);
    virtual_camera_name = get_string("virtual_camera_name", "None");
    virtual_camera_width = get_long("virtual_camera_width", 1920);
    virtual_camera_heigth = get_long("virtual_camera_heigth", 1080);

    // Target
    disable_headshot = get_bool("disable_headshot", false);
    body_y_offset = (float)get_double("body_y_offset", 0.15);
    head_y_offset = (float)get_double("head_y_offset", 0.05);
    auto_aim = get_bool("auto_aim", false);

    // Mouse
    fovX = get_long("fovX", 106);
    fovY = get_long("fovY", 74);
    minSpeedMultiplier = (float)get_double("minSpeedMultiplier", 0.1);
    maxSpeedMultiplier = (float)get_double("maxSpeedMultiplier", 0.1);

    predictionInterval = (float)get_double("predictionInterval", 0.01);
    prediction_futurePositions = get_long("prediction_futurePositions", 20);
    draw_futurePositions = get_bool("draw_futurePositions", true);
    runtime_latency_sweep_enabled = get_bool("runtime_latency_sweep_enabled", false);
    estimator_mode = get_string("estimator_mode", "kalman");
    kalman_enabled = get_bool("kalman_enabled", true);
    kalman_process_noise_position = (float)get_double("kalman_process_noise_position", 40.0);
    kalman_process_noise_velocity = (float)get_double("kalman_process_noise_velocity", 1800.0);
    kalman_measurement_noise = (float)get_double("kalman_measurement_noise", 35.0);
    kalman_velocity_damping = (float)get_double("kalman_velocity_damping", 0.08);
    kalman_max_velocity = (float)get_double("kalman_max_velocity", 20000.0);
    kalman_warmup_frames = get_long("kalman_warmup_frames", 2);
    kalman_velocity_seed_enabled = get_bool("kalman_velocity_seed_enabled", true);
    kalman_acquisition_frames = get_long("kalman_acquisition_frames", 4);
    kalman_compensate_detection_delay = get_bool("kalman_compensate_detection_delay", true);
    kalman_additional_prediction_ms = (float)get_double("kalman_additional_prediction_ms", 0.0);
    kalman_reset_timeout_sec = (float)get_double("kalman_reset_timeout_sec", 0.5);
    ego_motion_compensation_enabled = get_bool("ego_motion_compensation_enabled", false);
    ego_motion_compensation_strength = (float)get_double("ego_motion_compensation_strength", 0.80);
    ego_motion_compensation_max_shift_px = (float)get_double("ego_motion_compensation_max_shift_px", 32.0);
    ego_motion_compensation_max_age_ms = get_long("ego_motion_compensation_max_age_ms", 120);
    
    snapRadius = (float)get_double("snapRadius", 1.5);
    nearRadius = (float)get_double("nearRadius", 25.0);
    speedCurveExponent = (float)get_double("speedCurveExponent", 3.0);
    snapBoostFactor = (float)get_double("snapBoostFactor", 1.15);

    easynorecoil = get_bool("easynorecoil", false);
    easynorecoilstrength = (float)get_double("easynorecoilstrength", 0.0);
    input_method = get_string("input_method", "RAZER");

    // Wind mouse
    wind_mouse_enabled = get_bool("wind_mouse_enabled", false);
    wind_G = (float)get_double("wind_G", 18.0f);
    wind_W = (float)get_double("wind_W", 15.0f);
    wind_M = (float)get_double("wind_M", 10.0f);
    wind_D = (float)get_double("wind_D", 8.0f);

    // Pure PID mouse control
    pid_actuator_hz = get_long("pid_actuator_hz", get_long("adaptive_pid_actuator_hz", 1000));
    pid_kp = (float)get_double("pid_kp", get_double("adaptive_pid_kp", 0.0085));
    pid_ki = (float)get_double("pid_ki", get_double("adaptive_pid_ki", 0.0003));
    pid_kd = (float)get_double("pid_kd", get_double("adaptive_pid_kd", 0.0001));
    pid_deadzone_px = (float)get_double("pid_deadzone_px", get_double("adaptive_pid_deadzone", 0.0));
    pid_max_pixel_step = (float)get_double("pid_max_pixel_step", get_double("adaptive_pid_max_pixel_step", 0.80));
    pid_output_scale = (float)get_double("pid_output_scale", 0.10);
    pid_min_output_scale = (float)get_double("pid_min_output_scale", 0.02);
    pid_max_output_scale = (float)get_double("pid_max_output_scale", 0.35);
    pid_size_reference_px = (float)get_double("pid_size_reference_px", 48.0);
    pid_size_min_scale = (float)get_double("pid_size_min_scale", 0.20);
    pid_size_max_scale = (float)get_double("pid_size_max_scale", 1.00);
    pid_precision_radius_scale = (float)get_double("pid_precision_radius_scale", 0.012);
    pid_slowdown_radius_scale = (float)get_double("pid_slowdown_radius_scale", 0.30);
    pid_overshoot_brake = (float)get_double("pid_overshoot_brake", 0.35);
    pid_divergence_boost = (float)get_double("pid_divergence_boost", 0.35);
    pid_scale_response = (float)get_double("pid_scale_response", 8.0);
    pid_max_integral = (float)get_double("pid_max_integral", get_double("adaptive_pid_max_integral", 120.0));
    pid_max_derivative_term = (float)get_double("pid_max_derivative_term", 0.02);
    pid_derivative_filter_tau_ms = (float)get_double("pid_derivative_filter_tau_ms", get_double("adaptive_pid_derivative_filter_tau_ms", 18.0));
    pid_target_loss_timeout_ms = (float)get_double("pid_target_loss_timeout_ms", get_double("adaptive_pid_target_loss_timeout_ms", 90.0));
    pid_feed_forward_enabled = get_bool("pid_feed_forward_enabled", true);
    pid_feed_forward_gain = (float)get_double("pid_feed_forward_gain", 0.35);
    pid_feed_forward_lookahead_ms = (float)get_double("pid_feed_forward_lookahead_ms", 24.0);
    pid_feed_forward_frame_lookahead = get_long("pid_feed_forward_frame_lookahead", 1);
    pid_feed_forward_max_step = (float)get_double("pid_feed_forward_max_step", 0.35);
    pid_feed_forward_min_speed = (float)get_double("pid_feed_forward_min_speed", 20.0);
    pid_feed_forward_confidence_floor = (float)get_double("pid_feed_forward_confidence_floor", 0.55);
    pid_conditional_integration_enabled = get_bool("pid_conditional_integration_enabled", true);
    pid_conditional_integration_error_px = (float)get_double("pid_conditional_integration_error_px", 12.0);
    pid_adaptive_output_scaling_enabled = get_bool("pid_adaptive_output_scaling_enabled", true);
    pid_adaptive_output_error_scale = (float)get_double("pid_adaptive_output_error_scale", 96.0);
    pid_derivative_smoothing_multiplier = (float)get_double("pid_derivative_smoothing_multiplier", 1.5);
    pid_perspective_fov_mapping_enabled = get_bool("pid_perspective_fov_mapping_enabled", false);
    pid_governor_enabled = get_bool("pid_governor_enabled", false);
    pid_governor_model_path = get_string("pid_governor_model_path", "training/models/pid_governor.onnx");
    pid_governor_blend = (float)get_double("pid_governor_blend", 1.0);
    pid_governor_max_speed_multiple = (float)get_double("pid_governor_max_speed_multiple", 5.0);
    pid_smart_blending_enabled = get_bool("pid_smart_blending_enabled", false);
    pid_smart_blending_aggression = (float)get_double("pid_smart_blending_aggression", 0.65);
    pid_smart_blending_near_damping = (float)get_double("pid_smart_blending_near_damping", 0.75);
    pid_smart_blending_deadzone_px = (float)get_double("pid_smart_blending_deadzone_px", 0.0);
    pid_smart_blending_jerk_limit_px = (float)get_double("pid_smart_blending_jerk_limit_px", 0.65);
    pid_smart_blending_confidence_floor = (float)get_double("pid_smart_blending_confidence_floor", 0.45);

    // Neural tracker association
    neural_tracker_enabled = get_bool("neural_tracker_enabled", false);
    neural_tracker_runtime = get_string("neural_tracker_runtime", "CPU");
    neural_tracker_model_path = get_string("neural_tracker_model_path", "training/models/neural_tracker.onnx");
    neural_tracker_blend = (float)get_double("neural_tracker_blend", 0.35);
    neural_tracker_log_enabled = get_bool("neural_tracker_log_enabled", false);
    neural_tracker_debug_enabled = get_bool("neural_tracker_debug_enabled", false);
    neural_tracker_log_path = get_string("neural_tracker_log_path", "training/logs/neural_tracker_association.csv");

    // Learned temporal predictor
    temporal_prediction_enabled = get_bool("temporal_prediction_enabled", false);
    temporal_prediction_model_path = get_string("temporal_prediction_model_path", "models/temporal_predictor.onnx");
    temporal_prediction_history_length = get_long("temporal_prediction_history_length", 12);
    temporal_prediction_horizon = get_long("temporal_prediction_horizon", 16);
    temporal_prediction_interval_frames = get_long("temporal_prediction_interval_frames", 2);
    temporal_prediction_feed_forward_enabled = get_bool("temporal_prediction_feed_forward_enabled", false);
    temporal_prediction_influence = (float)get_double("temporal_prediction_influence", 0.35);
    temporal_prediction_adaptive_influence_enabled = get_bool("temporal_prediction_adaptive_influence_enabled", false);
    temporal_prediction_adaptive_ema_alpha = (float)get_double("temporal_prediction_adaptive_ema_alpha", 0.62);
    temporal_prediction_max_lead_px = (float)get_double("temporal_prediction_max_lead_px", 45.0);

    // Neural targeting head
    neural_targeting_enabled = get_bool("neural_targeting_enabled", false);
    neural_targeting_model_path = get_string("neural_targeting_model_path", "models/neural_targeting_head.onnx");
    neural_targeting_influence = (float)get_double("neural_targeting_influence", 0.40);
    neural_targeting_max_refinement_px = (float)get_double("neural_targeting_max_refinement_px", 35.0);
    neural_targeting_max_iterations = get_long("neural_targeting_max_iterations", 2);
    neural_control_preset = get_string("neural_control_preset", "Balanced");
    neural_control_telemetry_overlay_enabled = get_bool("neural_control_telemetry_overlay_enabled", false);
    neural_control_telemetry_logging_enabled = get_bool("neural_control_telemetry_logging_enabled", false);
    neural_control_telemetry_log_path = get_string("neural_control_telemetry_log_path", "logs/neural_control_telemetry.csv");
    neural_control_telemetry_log_interval_ms = get_long("neural_control_telemetry_log_interval_ms", 250);

    // Arduino
    arduino_baudrate = get_long("arduino_baudrate", 115200);
    arduino_port = get_string("arduino_port", "COM0");
    arduino_16_bit_mouse = get_bool("arduino_16_bit_mouse", false);
    arduino_enable_keys = get_bool("arduino_enable_keys", false);

    // Teensy 4.1 RawHID generic mouse bridge
    teensy_hid_manufacturer = get_string("teensy_hid_manufacturer", "Generic");
    teensy_hid_product = get_string("teensy_hid_product", "USB HID Mouse");
    teensy_hid_serial = get_string("teensy_hid_serial", "AUTO");
    teensy_hid_vid_filter = get_string("teensy_hid_vid_filter", "AUTO");
    teensy_hid_pid_filter = get_string("teensy_hid_pid_filter", "AUTO");
    teensy_hid_usage_page = get_long("teensy_hid_usage_page", 0xFFAB);
    teensy_hid_usage_id = get_long("teensy_hid_usage_id", 0x0200);
    teensy_hid_open_index = get_long("teensy_hid_open_index", 0);
    teensy_hid_packet_timeout_ms = get_long("teensy_hid_packet_timeout_ms", 2);
    teensy_hid_reconnect_interval_ms = get_long("teensy_hid_reconnect_interval_ms", 500);

    // kmbox_net
    kmbox_net_ip = get_string("kmbox_net_ip", "10.42.42.42");
    kmbox_net_port = get_string("kmbox_net_port", "1984");
    kmbox_net_uuid = get_string("kmbox_net_uuid", "DEADC0DE");

    // kmbox_a
    kmbox_a_pidvid = get_string("kmbox_a_pidvid", "");

    // makcu
    makcu_baudrate = get_long("makcu_baudrate", 4000000);
    makcu_port = get_string("makcu_port", "AUTO");

    // Mouse shooting
    auto_shoot = get_bool("auto_shoot", false);
    bScope_multiplier = (float)get_double("bScope_multiplier", 1.2);

    // AI
#ifdef USE_CUDA
    backend = get_string("backend", "TRT");
#else
    backend = get_string("backend", "DML");
#endif

    dml_device_id = get_long("dml_device_id", 0);

#ifdef USE_CUDA
    ai_model = get_string("ai_model", "sunxds_0.8.0.engine");
#else
    ai_model = get_string("ai_model", "sunxds_0.8.0.onnx");
#endif
    confidence_threshold = (float)get_double("confidence_threshold", 0.15);
    nms_threshold = (float)get_double("nms_threshold", 0.50);
    max_detections = get_long("max_detections", 20);
#ifdef USE_CUDA
    export_enable_fp8 = get_bool("export_enable_fp8", true);
    export_enable_fp16 = get_bool("export_enable_fp16", true);
#endif
    // CUDA
#ifdef USE_CUDA
    use_cuda_graph = get_bool("use_cuda_graph", false);
    use_pinned_memory = get_bool("use_pinned_memory", true);
    gpuMemoryReserveMB = get_long("gpuMemoryReserveMB", 2048);
    enableGpuExclusiveMode = get_bool("enableGpuExclusiveMode", true);
    capture_use_cuda = get_bool("capture_use_cuda", true);
#endif

    // System
    cpuCoreReserveCount = get_long("cpuCoreReserveCount", 4);
    systemMemoryReserveMB = get_long("systemMemoryReserveMB", 2048);

    // Buttons
    button_targeting = splitString(get_string("button_targeting", "RightMouseButton"));
    button_shoot = splitString(get_string("button_shoot", "LeftMouseButton"));
    button_zoom = splitString(get_string("button_zoom", "RightMouseButton"));
    button_exit = splitString(get_string("button_exit", "F2"));
    button_pause = splitString(get_string("button_pause", "F3"));
    button_reload_config = splitString(get_string("button_reload_config", "F4"));
    button_open_overlay = splitString(get_string("button_open_overlay", "Home"));
    enable_arrows_settings = get_bool("enable_arrows_settings", false);

    // Overlay
    overlay_opacity = get_long("overlay_opacity", 225);
    overlay_ui_scale = (float)get_double("overlay_ui_scale", 1.0);
    overlay_exclude_from_capture = get_bool("overlay_exclude_from_capture", true);

    // Depth
    depth_inference_enabled = get_bool("depth_inference_enabled", true);
    depth_model_path = get_string("depth_model_path", "depth_anything_v2.engine");
    depth_fps = get_long("depth_fps", 100);
    if (depth_fps < 0) depth_fps = 0;
    depth_colormap = get_long("depth_colormap", 18);
    if (depth_colormap < 0 || depth_colormap > 21) depth_colormap = 18;
    depth_mask_enabled = get_bool("depth_mask_enabled", false);
    depth_mask_fps = get_long("depth_mask_fps", 5);
    if (depth_mask_fps < 0) depth_mask_fps = 0;
    depth_mask_near_percent = get_long("depth_mask_near_percent", 20);
    if (depth_mask_near_percent < 1) depth_mask_near_percent = 1;
    if (depth_mask_near_percent > 100) depth_mask_near_percent = 100;
    depth_mask_expand = get_long("depth_mask_expand", 0);
    if (depth_mask_expand < 0) depth_mask_expand = 0;
    if (depth_mask_expand > 128) depth_mask_expand = 128;
    depth_mask_hold_frames = get_long("depth_mask_hold_frames", 0);
    if (depth_mask_hold_frames < 0) depth_mask_hold_frames = 0;
    if (depth_mask_hold_frames > 120) depth_mask_hold_frames = 120;
    depth_mask_alpha = get_long("depth_mask_alpha", 90);
    if (depth_mask_alpha < 0) depth_mask_alpha = 0;
    if (depth_mask_alpha > 255) depth_mask_alpha = 255;
    depth_mask_invert = get_bool("depth_mask_invert", false);
    depth_debug_overlay_enabled = get_bool("depth_debug_overlay_enabled", false);

    game_overlay_enabled = get_bool("game_overlay_enabled", false);
    game_overlay_max_fps = get_long("game_overlay_max_fps", 0);
    game_overlay_draw_boxes = get_bool("game_overlay_draw_boxes", true);
    game_overlay_draw_future = get_bool("game_overlay_draw_future", true);
    game_overlay_draw_wind_tail = get_bool("game_overlay_draw_wind_tail", true);
    game_overlay_draw_frame = get_bool("game_overlay_draw_frame", true);
    game_overlay_show_target_correction = get_bool("game_overlay_show_target_correction", true);
    game_overlay_box_a = get_long("game_overlay_box_a", 255);
    game_overlay_box_r = get_long("game_overlay_box_r", 0);
    game_overlay_box_g = get_long("game_overlay_box_g", 255);
    game_overlay_box_b = get_long("game_overlay_box_b", 0);
    game_overlay_frame_a = get_long("game_overlay_frame_a", 180);
    game_overlay_frame_r = get_long("game_overlay_frame_r", 255);
    game_overlay_frame_g = get_long("game_overlay_frame_g", 255);
    game_overlay_frame_b = get_long("game_overlay_frame_b", 255);
    game_overlay_box_thickness = (float)get_double("game_overlay_box_thickness", 2.0);
    game_overlay_frame_thickness = (float)get_double("game_overlay_frame_thickness", 1.5);
    game_overlay_future_point_radius = (float)get_double("game_overlay_future_point_radius", 5.0);
    game_overlay_future_alpha_falloff = (float)get_double("game_overlay_future_alpha_falloff", 1.0);
    clampGameOverlayColor();

    game_overlay_icon_enabled = get_bool("game_overlay_icon_enabled", false);
    game_overlay_icon_path = get_string("game_overlay_icon_path", "icon.png");
    game_overlay_icon_width = get_long("game_overlay_icon_width", 64);
    game_overlay_icon_height = get_long("game_overlay_icon_height", 64);
    game_overlay_icon_offset_x = (float)get_double("game_overlay_icon_offset_x", 0.0f);
    game_overlay_icon_offset_y = (float)get_double("game_overlay_icon_offset_y", 0.0f);
    game_overlay_icon_anchor = get_string("game_overlay_icon_anchor", "center");
    game_overlay_icon_class = get_long("game_overlay_icon_class", -1);

    // Aim simulation overlay
    aim_sim_enabled = get_bool("aim_sim_enabled", false);
    aim_sim_x = get_long("aim_sim_x", 24);
    aim_sim_y = get_long("aim_sim_y", 24);
    aim_sim_width = get_long("aim_sim_width", 560);
    aim_sim_height = get_long("aim_sim_height", 360);
    aim_sim_fps_min = get_long("aim_sim_fps_min", 90);
    aim_sim_fps_max = get_long("aim_sim_fps_max", 120);
    aim_sim_fps_jitter = (float)get_double("aim_sim_fps_jitter", 0.15);
    aim_sim_capture_delay_ms = (float)get_double("aim_sim_capture_delay_ms", 6.0);
    aim_sim_inference_delay_ms = (float)get_double("aim_sim_inference_delay_ms", 12.0);
    aim_sim_use_live_inference = get_bool("aim_sim_use_live_inference", true);
    aim_sim_input_delay_ms = (float)get_double("aim_sim_input_delay_ms", 2.0);
    aim_sim_extra_delay_ms = (float)get_double("aim_sim_extra_delay_ms", 2.0);
    aim_sim_target_max_speed = (float)get_double("aim_sim_target_max_speed", 560.0);
    aim_sim_target_accel = (float)get_double("aim_sim_target_accel", 1850.0);
    aim_sim_target_stop_chance = (float)get_double("aim_sim_target_stop_chance", 0.25);
    aim_sim_show_observed = get_bool("aim_sim_show_observed", true);
    aim_sim_show_history = get_bool("aim_sim_show_history", true);
    aim_sim_show_kalman_debug = get_bool("aim_sim_show_kalman_debug", true);

    if (kalman_process_noise_position < 0.0001f) kalman_process_noise_position = 0.0001f;
    if (kalman_process_noise_position > 5000.0f) kalman_process_noise_position = 5000.0f;
    if (kalman_process_noise_velocity < 0.0001f) kalman_process_noise_velocity = 0.0001f;
    if (kalman_process_noise_velocity > 50000.0f) kalman_process_noise_velocity = 50000.0f;
    if (kalman_measurement_noise < 0.0001f) kalman_measurement_noise = 0.0001f;
    if (kalman_measurement_noise > 5000.0f) kalman_measurement_noise = 5000.0f;
    if (kalman_velocity_damping < 0.0f) kalman_velocity_damping = 0.0f;
    if (kalman_velocity_damping > 3.0f) kalman_velocity_damping = 3.0f;
    if (kalman_max_velocity < 100.0f) kalman_max_velocity = 100.0f;
    if (kalman_max_velocity > 60000.0f) kalman_max_velocity = 60000.0f;
    if (kalman_warmup_frames < 0) kalman_warmup_frames = 0;
    if (kalman_warmup_frames > 20) kalman_warmup_frames = 20;
    if (kalman_acquisition_frames < 3) kalman_acquisition_frames = 3;
    if (kalman_acquisition_frames > 5) kalman_acquisition_frames = 5;
    if (kalman_additional_prediction_ms < -80.0f) kalman_additional_prediction_ms = -80.0f;
    if (kalman_additional_prediction_ms > 120.0f) kalman_additional_prediction_ms = 120.0f;
    if (kalman_reset_timeout_sec < 0.05f) kalman_reset_timeout_sec = 0.05f;
    if (kalman_reset_timeout_sec > 3.0f) kalman_reset_timeout_sec = 3.0f;
    if (ego_motion_compensation_strength < 0.0f) ego_motion_compensation_strength = 0.0f;
    if (ego_motion_compensation_strength > 1.0f) ego_motion_compensation_strength = 1.0f;
    if (ego_motion_compensation_max_shift_px < 1.0f) ego_motion_compensation_max_shift_px = 1.0f;
    if (ego_motion_compensation_max_shift_px > 128.0f) ego_motion_compensation_max_shift_px = 128.0f;
    if (ego_motion_compensation_max_age_ms < 16) ego_motion_compensation_max_age_ms = 16;
    if (ego_motion_compensation_max_age_ms > 500) ego_motion_compensation_max_age_ms = 500;
    std::transform(estimator_mode.begin(), estimator_mode.end(), estimator_mode.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (estimator_mode != "kalman" && estimator_mode != "imm")
        estimator_mode = "kalman";

    if (aim_sim_width < 220) aim_sim_width = 220;
    if (aim_sim_width > 1920) aim_sim_width = 1920;
    if (aim_sim_height < 180) aim_sim_height = 180;
    if (aim_sim_height > 1080) aim_sim_height = 1080;

    if (aim_sim_fps_min < 15) aim_sim_fps_min = 15;
    if (aim_sim_fps_min > 360) aim_sim_fps_min = 360;
    if (aim_sim_fps_max < 15) aim_sim_fps_max = 15;
    if (aim_sim_fps_max > 360) aim_sim_fps_max = 360;
    if (aim_sim_fps_min > aim_sim_fps_max)
        std::swap(aim_sim_fps_min, aim_sim_fps_max);

    if (aim_sim_fps_jitter < 0.0f) aim_sim_fps_jitter = 0.0f;
    if (aim_sim_fps_jitter > 0.8f) aim_sim_fps_jitter = 0.8f;
    if (aim_sim_capture_delay_ms < 0.0f) aim_sim_capture_delay_ms = 0.0f;
    if (aim_sim_capture_delay_ms > 80.0f) aim_sim_capture_delay_ms = 80.0f;
    if (aim_sim_inference_delay_ms < 0.0f) aim_sim_inference_delay_ms = 0.0f;
    if (aim_sim_inference_delay_ms > 120.0f) aim_sim_inference_delay_ms = 120.0f;
    if (aim_sim_input_delay_ms < 0.0f) aim_sim_input_delay_ms = 0.0f;
    if (aim_sim_input_delay_ms > 60.0f) aim_sim_input_delay_ms = 60.0f;
    if (aim_sim_extra_delay_ms < 0.0f) aim_sim_extra_delay_ms = 0.0f;
    if (aim_sim_extra_delay_ms > 60.0f) aim_sim_extra_delay_ms = 60.0f;
    if (aim_sim_target_max_speed < 20.0f) aim_sim_target_max_speed = 20.0f;
    if (aim_sim_target_max_speed > 2500.0f) aim_sim_target_max_speed = 2500.0f;
    if (aim_sim_target_accel < 20.0f) aim_sim_target_accel = 20.0f;
    if (aim_sim_target_accel > 10000.0f) aim_sim_target_accel = 10000.0f;
    if (aim_sim_target_stop_chance < 0.0f) aim_sim_target_stop_chance = 0.0f;
    if (aim_sim_target_stop_chance > 0.95f) aim_sim_target_stop_chance = 0.95f;

    if (pid_actuator_hz < 30) pid_actuator_hz = 30;
    if (pid_actuator_hz > 2000) pid_actuator_hz = 2000;
    if (pid_kp < 0.0f) pid_kp = 0.0f;
    if (pid_kp > 5.0f) pid_kp = 5.0f;
    if (pid_ki < 0.0f) pid_ki = 0.0f;
    if (pid_ki > 2.0f) pid_ki = 2.0f;
    if (pid_kd < 0.0f) pid_kd = 0.0f;
    if (pid_kd > 1.0f) pid_kd = 1.0f;
    if (pid_deadzone_px < 0.0f) pid_deadzone_px = 0.0f;
    if (pid_deadzone_px > 10.0f) pid_deadzone_px = 10.0f;
    if (pid_max_pixel_step < 0.01f) pid_max_pixel_step = 0.01f;
    if (pid_max_pixel_step > 80.0f) pid_max_pixel_step = 80.0f;
    if (pid_output_scale < 0.01f) pid_output_scale = 0.01f;
    if (pid_output_scale > 3.0f) pid_output_scale = 3.0f;
    if (pid_min_output_scale < 0.0f) pid_min_output_scale = 0.0f;
    if (pid_min_output_scale > 3.0f) pid_min_output_scale = 3.0f;
    if (pid_max_output_scale < 0.01f) pid_max_output_scale = 0.01f;
    if (pid_max_output_scale > 3.0f) pid_max_output_scale = 3.0f;
    if (pid_min_output_scale > pid_max_output_scale) std::swap(pid_min_output_scale, pid_max_output_scale);
    if (pid_size_reference_px < 1.0f) pid_size_reference_px = 1.0f;
    if (pid_size_reference_px > 640.0f) pid_size_reference_px = 640.0f;
    if (pid_size_min_scale < 0.01f) pid_size_min_scale = 0.01f;
    if (pid_size_min_scale > 2.0f) pid_size_min_scale = 2.0f;
    if (pid_size_max_scale < 0.01f) pid_size_max_scale = 0.01f;
    if (pid_size_max_scale > 2.0f) pid_size_max_scale = 2.0f;
    if (pid_size_min_scale > pid_size_max_scale) std::swap(pid_size_min_scale, pid_size_max_scale);
    if (pid_precision_radius_scale < 0.0f) pid_precision_radius_scale = 0.0f;
    if (pid_precision_radius_scale > 0.25f) pid_precision_radius_scale = 0.25f;
    if (pid_slowdown_radius_scale < 0.01f) pid_slowdown_radius_scale = 0.01f;
    if (pid_slowdown_radius_scale > 2.0f) pid_slowdown_radius_scale = 2.0f;
    if (pid_overshoot_brake < 0.01f) pid_overshoot_brake = 0.01f;
    if (pid_overshoot_brake > 1.0f) pid_overshoot_brake = 1.0f;
    if (pid_divergence_boost < 0.0f) pid_divergence_boost = 0.0f;
    if (pid_divergence_boost > 3.0f) pid_divergence_boost = 3.0f;
    if (pid_scale_response < 0.1f) pid_scale_response = 0.1f;
    if (pid_scale_response > 60.0f) pid_scale_response = 60.0f;
    if (pid_max_integral < 0.0f) pid_max_integral = 0.0f;
    if (pid_max_integral > 10000.0f) pid_max_integral = 10000.0f;
    if (pid_max_derivative_term < 0.0f) pid_max_derivative_term = 0.0f;
    if (pid_max_derivative_term > 20.0f) pid_max_derivative_term = 20.0f;
    if (pid_derivative_filter_tau_ms < 0.0f) pid_derivative_filter_tau_ms = 0.0f;
    if (pid_derivative_filter_tau_ms > 250.0f) pid_derivative_filter_tau_ms = 250.0f;
    if (pid_target_loss_timeout_ms < 10.0f) pid_target_loss_timeout_ms = 10.0f;
    if (pid_target_loss_timeout_ms > 1000.0f) pid_target_loss_timeout_ms = 1000.0f;
    if (pid_feed_forward_gain < 0.0f) pid_feed_forward_gain = 0.0f;
    if (pid_feed_forward_gain > 4.0f) pid_feed_forward_gain = 4.0f;
    if (pid_feed_forward_lookahead_ms < 0.0f) pid_feed_forward_lookahead_ms = 0.0f;
    if (pid_feed_forward_lookahead_ms > 120.0f) pid_feed_forward_lookahead_ms = 120.0f;
    if (pid_feed_forward_frame_lookahead < 0) pid_feed_forward_frame_lookahead = 0;
    if (pid_feed_forward_frame_lookahead > 2) pid_feed_forward_frame_lookahead = 2;
    if (pid_feed_forward_max_step < 0.0f) pid_feed_forward_max_step = 0.0f;
    if (pid_feed_forward_max_step > 5.0f) pid_feed_forward_max_step = 5.0f;
    if (pid_feed_forward_min_speed < 0.0f) pid_feed_forward_min_speed = 0.0f;
    if (pid_feed_forward_min_speed > 3000.0f) pid_feed_forward_min_speed = 3000.0f;
    if (pid_feed_forward_confidence_floor < 0.0f) pid_feed_forward_confidence_floor = 0.0f;
    if (pid_feed_forward_confidence_floor > 1.0f) pid_feed_forward_confidence_floor = 1.0f;
    if (pid_conditional_integration_error_px < 0.0f) pid_conditional_integration_error_px = 0.0f;
    if (pid_conditional_integration_error_px > 240.0f) pid_conditional_integration_error_px = 240.0f;
    if (pid_adaptive_output_error_scale < 1.0f) pid_adaptive_output_error_scale = 1.0f;
    if (pid_adaptive_output_error_scale > 640.0f) pid_adaptive_output_error_scale = 640.0f;
    if (pid_derivative_smoothing_multiplier < 1.0f) pid_derivative_smoothing_multiplier = 1.0f;
    if (pid_derivative_smoothing_multiplier > 6.0f) pid_derivative_smoothing_multiplier = 6.0f;
    if (pid_governor_model_path.empty()) pid_governor_model_path = "training/models/pid_governor.onnx";
    if (pid_governor_blend < 0.0f) pid_governor_blend = 0.0f;
    if (pid_governor_blend > 1.0f) pid_governor_blend = 1.0f;
    if (pid_governor_max_speed_multiple < 1.0f) pid_governor_max_speed_multiple = 1.0f;
    if (pid_governor_max_speed_multiple > 5.0f) pid_governor_max_speed_multiple = 5.0f;
    if (pid_smart_blending_aggression < 0.30f) pid_smart_blending_aggression = 0.30f;
    if (pid_smart_blending_aggression > 1.0f) pid_smart_blending_aggression = 1.0f;
    if (pid_smart_blending_near_damping < 0.0f) pid_smart_blending_near_damping = 0.0f;
    if (pid_smart_blending_near_damping > 1.0f) pid_smart_blending_near_damping = 1.0f;
    if (pid_smart_blending_deadzone_px < 0.0f) pid_smart_blending_deadzone_px = 0.0f;
    if (pid_smart_blending_deadzone_px > 12.0f) pid_smart_blending_deadzone_px = 12.0f;
    if (pid_smart_blending_jerk_limit_px < 0.02f) pid_smart_blending_jerk_limit_px = 0.02f;
    if (pid_smart_blending_jerk_limit_px > 8.0f) pid_smart_blending_jerk_limit_px = 8.0f;
    if (pid_smart_blending_confidence_floor < 0.0f) pid_smart_blending_confidence_floor = 0.0f;
    if (pid_smart_blending_confidence_floor > 1.0f) pid_smart_blending_confidence_floor = 1.0f;
    if (neural_tracker_runtime.empty()) neural_tracker_runtime = "CPU";
    if (neural_tracker_model_path.empty()) neural_tracker_model_path = "training/models/neural_tracker.onnx";
    if (neural_tracker_blend < 0.0f) neural_tracker_blend = 0.0f;
    if (neural_tracker_blend > 1.0f) neural_tracker_blend = 1.0f;
    if (neural_tracker_log_path.empty()) neural_tracker_log_path = "training/logs/neural_tracker_association.csv";
    if (temporal_prediction_model_path.empty()) temporal_prediction_model_path = "models/temporal_predictor.onnx";
    if (temporal_prediction_history_length < 2) temporal_prediction_history_length = 2;
    if (temporal_prediction_history_length > 64) temporal_prediction_history_length = 64;
    if (temporal_prediction_horizon < 1) temporal_prediction_horizon = 1;
    if (temporal_prediction_horizon > 64) temporal_prediction_horizon = 64;
    if (temporal_prediction_interval_frames < 1) temporal_prediction_interval_frames = 1;
    if (temporal_prediction_interval_frames > 16) temporal_prediction_interval_frames = 16;
    if (temporal_prediction_influence < 0.0f) temporal_prediction_influence = 0.0f;
    if (temporal_prediction_influence > 1.0f) temporal_prediction_influence = 1.0f;
    if (temporal_prediction_adaptive_ema_alpha < 0.05f) temporal_prediction_adaptive_ema_alpha = 0.05f;
    if (temporal_prediction_adaptive_ema_alpha > 1.0f) temporal_prediction_adaptive_ema_alpha = 1.0f;
    if (temporal_prediction_max_lead_px < 20.0f) temporal_prediction_max_lead_px = 20.0f;
    if (temporal_prediction_max_lead_px > 80.0f) temporal_prediction_max_lead_px = 80.0f;
    if (neural_targeting_model_path.empty()) neural_targeting_model_path = "models/neural_targeting_head.onnx";
    if (neural_targeting_influence < 0.0f) neural_targeting_influence = 0.0f;
    if (neural_targeting_influence > 1.0f) neural_targeting_influence = 1.0f;
    if (neural_targeting_max_refinement_px < 1.0f) neural_targeting_max_refinement_px = 1.0f;
    if (neural_targeting_max_refinement_px > 80.0f) neural_targeting_max_refinement_px = 80.0f;
    if (neural_targeting_max_iterations < 1) neural_targeting_max_iterations = 1;
    if (neural_targeting_max_iterations > 2) neural_targeting_max_iterations = 2;
    if (neural_control_preset.empty()) neural_control_preset = "Balanced";
    if (neural_control_preset != "Balanced" &&
        neural_control_preset != "Aggressive" &&
        neural_control_preset != "Smooth" &&
        neural_control_preset != "Sniper") neural_control_preset = "Balanced";
    if (neural_control_telemetry_log_path.empty()) neural_control_telemetry_log_path = "logs/neural_control_telemetry.csv";
    if (neural_control_telemetry_log_interval_ms < 50) neural_control_telemetry_log_interval_ms = 50;
    if (neural_control_telemetry_log_interval_ms > 5000) neural_control_telemetry_log_interval_ms = 5000;

    // Classes
    class_player = get_long("class_player", 0);
    class_head = get_long("class_head", 1);

    // Debug window
    show_window = get_bool("show_window", true);
    screenshot_button = splitString(get_string("screenshot_button", "None"));
    screenshot_delay = get_long("screenshot_delay", 500);
    verbose = get_bool("verbose", false);
    debug_log_file_enabled = get_bool("debug_log_file_enabled", false);
    debug_log_file_path = get_string("debug_log_file_path", "logs/0BS.log");
    if (debug_log_file_path.empty()) debug_log_file_path = "logs/0BS.log";

    return validate();
}

bool Config::loadConfigMerged(const std::string& filename)
{
    const std::string target = filename.empty() ? "config.ini" : filename;
    if (!std::filesystem::exists(target))
        return loadConfig(target);

    CSimpleIniA ini;
    ini.SetUnicode();
    SI_Error rc = ini.LoadFile(target.c_str());
    if (rc < 0)
    {
        std::cerr << "[Config] Error parsing INI file for merge: " << target << std::endl;
        return false;
    }

    std::unordered_set<std::string> presentKeys;
    CSimpleIniA::TNamesDepend keys;
    ini.GetAllKeys("", keys);
    for (const auto& k : keys)
        presentKeys.insert(k.pItem);

    Config loaded = *this;
    if (!loaded.loadConfig(target))
        return false;

    auto hasKey = [&](const char* key) {
        return presentKeys.find(key) != presentKeys.end();
    };

#define MERGE_FIELD(keyName, fieldName) \
    do { if (hasKey(keyName)) fieldName = loaded.fieldName; } while (0)

    MERGE_FIELD("capture_method", capture_method);
    MERGE_FIELD("capture_target", capture_target);
    MERGE_FIELD("capture_window_title", capture_window_title);
    MERGE_FIELD("udp_ip", udp_ip);
    MERGE_FIELD("udp_port", udp_port);
    MERGE_FIELD("detection_resolution", detection_resolution);
    MERGE_FIELD("capture_fps", capture_fps);
    MERGE_FIELD("monitor_idx", monitor_idx);
    MERGE_FIELD("circle_mask", circle_mask);
    MERGE_FIELD("capture_borders", capture_borders);
    MERGE_FIELD("capture_cursor", capture_cursor);
    MERGE_FIELD("virtual_camera_name", virtual_camera_name);
    MERGE_FIELD("virtual_camera_width", virtual_camera_width);
    MERGE_FIELD("virtual_camera_heigth", virtual_camera_heigth);

    MERGE_FIELD("disable_headshot", disable_headshot);
    MERGE_FIELD("body_y_offset", body_y_offset);
    MERGE_FIELD("head_y_offset", head_y_offset);
    MERGE_FIELD("auto_aim", auto_aim);

    MERGE_FIELD("fovX", fovX);
    MERGE_FIELD("fovY", fovY);
    MERGE_FIELD("minSpeedMultiplier", minSpeedMultiplier);
    MERGE_FIELD("maxSpeedMultiplier", maxSpeedMultiplier);
    MERGE_FIELD("predictionInterval", predictionInterval);
    MERGE_FIELD("prediction_futurePositions", prediction_futurePositions);
    MERGE_FIELD("draw_futurePositions", draw_futurePositions);
    MERGE_FIELD("runtime_latency_sweep_enabled", runtime_latency_sweep_enabled);
    MERGE_FIELD("estimator_mode", estimator_mode);
    MERGE_FIELD("kalman_enabled", kalman_enabled);
    MERGE_FIELD("kalman_process_noise_position", kalman_process_noise_position);
    MERGE_FIELD("kalman_process_noise_velocity", kalman_process_noise_velocity);
    MERGE_FIELD("kalman_measurement_noise", kalman_measurement_noise);
    MERGE_FIELD("kalman_velocity_damping", kalman_velocity_damping);
    MERGE_FIELD("kalman_max_velocity", kalman_max_velocity);
    MERGE_FIELD("kalman_warmup_frames", kalman_warmup_frames);
    MERGE_FIELD("kalman_velocity_seed_enabled", kalman_velocity_seed_enabled);
    MERGE_FIELD("kalman_acquisition_frames", kalman_acquisition_frames);
    MERGE_FIELD("kalman_compensate_detection_delay", kalman_compensate_detection_delay);
    MERGE_FIELD("kalman_additional_prediction_ms", kalman_additional_prediction_ms);
    MERGE_FIELD("kalman_reset_timeout_sec", kalman_reset_timeout_sec);
    MERGE_FIELD("ego_motion_compensation_enabled", ego_motion_compensation_enabled);
    MERGE_FIELD("ego_motion_compensation_strength", ego_motion_compensation_strength);
    MERGE_FIELD("ego_motion_compensation_max_shift_px", ego_motion_compensation_max_shift_px);
    MERGE_FIELD("ego_motion_compensation_max_age_ms", ego_motion_compensation_max_age_ms);

    MERGE_FIELD("snapRadius", snapRadius);
    MERGE_FIELD("nearRadius", nearRadius);
    MERGE_FIELD("speedCurveExponent", speedCurveExponent);
    MERGE_FIELD("snapBoostFactor", snapBoostFactor);
    MERGE_FIELD("easynorecoil", easynorecoil);
    MERGE_FIELD("easynorecoilstrength", easynorecoilstrength);
    MERGE_FIELD("input_method", input_method);

    MERGE_FIELD("wind_mouse_enabled", wind_mouse_enabled);
    MERGE_FIELD("wind_G", wind_G);
    MERGE_FIELD("wind_W", wind_W);
    MERGE_FIELD("wind_M", wind_M);
    MERGE_FIELD("wind_D", wind_D);

    MERGE_FIELD("pid_actuator_hz", pid_actuator_hz);
    MERGE_FIELD("pid_kp", pid_kp);
    MERGE_FIELD("pid_ki", pid_ki);
    MERGE_FIELD("pid_kd", pid_kd);
    MERGE_FIELD("pid_deadzone_px", pid_deadzone_px);
    MERGE_FIELD("pid_max_pixel_step", pid_max_pixel_step);
    MERGE_FIELD("pid_output_scale", pid_output_scale);
    MERGE_FIELD("pid_min_output_scale", pid_min_output_scale);
    MERGE_FIELD("pid_max_output_scale", pid_max_output_scale);
    MERGE_FIELD("pid_size_reference_px", pid_size_reference_px);
    MERGE_FIELD("pid_size_min_scale", pid_size_min_scale);
    MERGE_FIELD("pid_size_max_scale", pid_size_max_scale);
    MERGE_FIELD("pid_precision_radius_scale", pid_precision_radius_scale);
    MERGE_FIELD("pid_slowdown_radius_scale", pid_slowdown_radius_scale);
    MERGE_FIELD("pid_overshoot_brake", pid_overshoot_brake);
    MERGE_FIELD("pid_divergence_boost", pid_divergence_boost);
    MERGE_FIELD("pid_scale_response", pid_scale_response);
    MERGE_FIELD("pid_max_integral", pid_max_integral);
    MERGE_FIELD("pid_max_derivative_term", pid_max_derivative_term);
    MERGE_FIELD("pid_derivative_filter_tau_ms", pid_derivative_filter_tau_ms);
    MERGE_FIELD("pid_target_loss_timeout_ms", pid_target_loss_timeout_ms);
    MERGE_FIELD("pid_feed_forward_enabled", pid_feed_forward_enabled);
    MERGE_FIELD("pid_feed_forward_gain", pid_feed_forward_gain);
    MERGE_FIELD("pid_feed_forward_lookahead_ms", pid_feed_forward_lookahead_ms);
    MERGE_FIELD("pid_feed_forward_frame_lookahead", pid_feed_forward_frame_lookahead);
    MERGE_FIELD("pid_feed_forward_max_step", pid_feed_forward_max_step);
    MERGE_FIELD("pid_feed_forward_min_speed", pid_feed_forward_min_speed);
    MERGE_FIELD("pid_feed_forward_confidence_floor", pid_feed_forward_confidence_floor);
    MERGE_FIELD("pid_conditional_integration_enabled", pid_conditional_integration_enabled);
    MERGE_FIELD("pid_conditional_integration_error_px", pid_conditional_integration_error_px);
    MERGE_FIELD("pid_adaptive_output_scaling_enabled", pid_adaptive_output_scaling_enabled);
    MERGE_FIELD("pid_adaptive_output_error_scale", pid_adaptive_output_error_scale);
    MERGE_FIELD("pid_derivative_smoothing_multiplier", pid_derivative_smoothing_multiplier);
    MERGE_FIELD("pid_perspective_fov_mapping_enabled", pid_perspective_fov_mapping_enabled);
    MERGE_FIELD("pid_governor_enabled", pid_governor_enabled);
    MERGE_FIELD("pid_governor_model_path", pid_governor_model_path);
    MERGE_FIELD("pid_governor_blend", pid_governor_blend);
    MERGE_FIELD("pid_governor_max_speed_multiple", pid_governor_max_speed_multiple);
    MERGE_FIELD("pid_smart_blending_enabled", pid_smart_blending_enabled);
    MERGE_FIELD("pid_smart_blending_aggression", pid_smart_blending_aggression);
    MERGE_FIELD("pid_smart_blending_near_damping", pid_smart_blending_near_damping);
    MERGE_FIELD("pid_smart_blending_deadzone_px", pid_smart_blending_deadzone_px);
    MERGE_FIELD("pid_smart_blending_jerk_limit_px", pid_smart_blending_jerk_limit_px);
    MERGE_FIELD("pid_smart_blending_confidence_floor", pid_smart_blending_confidence_floor);

    MERGE_FIELD("neural_tracker_enabled", neural_tracker_enabled);
    MERGE_FIELD("neural_tracker_runtime", neural_tracker_runtime);
    MERGE_FIELD("neural_tracker_model_path", neural_tracker_model_path);
    MERGE_FIELD("neural_tracker_blend", neural_tracker_blend);
    MERGE_FIELD("neural_tracker_log_enabled", neural_tracker_log_enabled);
    MERGE_FIELD("neural_tracker_debug_enabled", neural_tracker_debug_enabled);
    MERGE_FIELD("neural_tracker_log_path", neural_tracker_log_path);
    MERGE_FIELD("temporal_prediction_enabled", temporal_prediction_enabled);
    MERGE_FIELD("temporal_prediction_model_path", temporal_prediction_model_path);
    MERGE_FIELD("temporal_prediction_history_length", temporal_prediction_history_length);
    MERGE_FIELD("temporal_prediction_horizon", temporal_prediction_horizon);
    MERGE_FIELD("temporal_prediction_interval_frames", temporal_prediction_interval_frames);
    MERGE_FIELD("temporal_prediction_feed_forward_enabled", temporal_prediction_feed_forward_enabled);
    MERGE_FIELD("temporal_prediction_influence", temporal_prediction_influence);
    MERGE_FIELD("temporal_prediction_adaptive_influence_enabled", temporal_prediction_adaptive_influence_enabled);
    MERGE_FIELD("temporal_prediction_adaptive_ema_alpha", temporal_prediction_adaptive_ema_alpha);
    MERGE_FIELD("temporal_prediction_max_lead_px", temporal_prediction_max_lead_px);
    MERGE_FIELD("neural_targeting_enabled", neural_targeting_enabled);
    MERGE_FIELD("neural_targeting_model_path", neural_targeting_model_path);
    MERGE_FIELD("neural_targeting_influence", neural_targeting_influence);
    MERGE_FIELD("neural_targeting_max_refinement_px", neural_targeting_max_refinement_px);
    MERGE_FIELD("neural_targeting_max_iterations", neural_targeting_max_iterations);
    MERGE_FIELD("neural_control_preset", neural_control_preset);
    MERGE_FIELD("neural_control_telemetry_overlay_enabled", neural_control_telemetry_overlay_enabled);
    MERGE_FIELD("neural_control_telemetry_logging_enabled", neural_control_telemetry_logging_enabled);
    MERGE_FIELD("neural_control_telemetry_log_path", neural_control_telemetry_log_path);
    MERGE_FIELD("neural_control_telemetry_log_interval_ms", neural_control_telemetry_log_interval_ms);

    MERGE_FIELD("arduino_baudrate", arduino_baudrate);
    MERGE_FIELD("arduino_port", arduino_port);
    MERGE_FIELD("arduino_16_bit_mouse", arduino_16_bit_mouse);
    MERGE_FIELD("arduino_enable_keys", arduino_enable_keys);
    MERGE_FIELD("teensy_hid_manufacturer", teensy_hid_manufacturer);
    MERGE_FIELD("teensy_hid_product", teensy_hid_product);
    MERGE_FIELD("teensy_hid_serial", teensy_hid_serial);
    MERGE_FIELD("teensy_hid_vid_filter", teensy_hid_vid_filter);
    MERGE_FIELD("teensy_hid_pid_filter", teensy_hid_pid_filter);
    MERGE_FIELD("teensy_hid_usage_page", teensy_hid_usage_page);
    MERGE_FIELD("teensy_hid_usage_id", teensy_hid_usage_id);
    MERGE_FIELD("teensy_hid_open_index", teensy_hid_open_index);
    MERGE_FIELD("teensy_hid_packet_timeout_ms", teensy_hid_packet_timeout_ms);
    MERGE_FIELD("teensy_hid_reconnect_interval_ms", teensy_hid_reconnect_interval_ms);
    MERGE_FIELD("kmbox_net_ip", kmbox_net_ip);
    MERGE_FIELD("kmbox_net_port", kmbox_net_port);
    MERGE_FIELD("kmbox_net_uuid", kmbox_net_uuid);
    MERGE_FIELD("kmbox_a_pidvid", kmbox_a_pidvid);
    MERGE_FIELD("makcu_baudrate", makcu_baudrate);
    MERGE_FIELD("makcu_port", makcu_port);

    MERGE_FIELD("auto_shoot", auto_shoot);
    MERGE_FIELD("bScope_multiplier", bScope_multiplier);

    MERGE_FIELD("backend", backend);
    MERGE_FIELD("dml_device_id", dml_device_id);
    MERGE_FIELD("ai_model", ai_model);
    MERGE_FIELD("confidence_threshold", confidence_threshold);
    MERGE_FIELD("nms_threshold", nms_threshold);
    MERGE_FIELD("max_detections", max_detections);
#ifdef USE_CUDA
    MERGE_FIELD("export_enable_fp8", export_enable_fp8);
    MERGE_FIELD("export_enable_fp16", export_enable_fp16);
    MERGE_FIELD("use_cuda_graph", use_cuda_graph);
    MERGE_FIELD("use_pinned_memory", use_pinned_memory);
    MERGE_FIELD("gpuMemoryReserveMB", gpuMemoryReserveMB);
    MERGE_FIELD("enableGpuExclusiveMode", enableGpuExclusiveMode);
    MERGE_FIELD("capture_use_cuda", capture_use_cuda);
#endif
    MERGE_FIELD("fixed_input_size", fixed_input_size);

    MERGE_FIELD("cpuCoreReserveCount", cpuCoreReserveCount);
    MERGE_FIELD("systemMemoryReserveMB", systemMemoryReserveMB);

    MERGE_FIELD("button_targeting", button_targeting);
    MERGE_FIELD("button_shoot", button_shoot);
    MERGE_FIELD("button_zoom", button_zoom);
    MERGE_FIELD("button_exit", button_exit);
    MERGE_FIELD("button_pause", button_pause);
    MERGE_FIELD("button_reload_config", button_reload_config);
    MERGE_FIELD("button_open_overlay", button_open_overlay);
    MERGE_FIELD("enable_arrows_settings", enable_arrows_settings);

    MERGE_FIELD("overlay_opacity", overlay_opacity);
    MERGE_FIELD("overlay_ui_scale", overlay_ui_scale);
    MERGE_FIELD("overlay_exclude_from_capture", overlay_exclude_from_capture);

    MERGE_FIELD("depth_inference_enabled", depth_inference_enabled);
    MERGE_FIELD("depth_model_path", depth_model_path);
    MERGE_FIELD("depth_fps", depth_fps);
    MERGE_FIELD("depth_colormap", depth_colormap);
    MERGE_FIELD("depth_mask_enabled", depth_mask_enabled);
    MERGE_FIELD("depth_mask_fps", depth_mask_fps);
    MERGE_FIELD("depth_mask_near_percent", depth_mask_near_percent);
    MERGE_FIELD("depth_mask_expand", depth_mask_expand);
    MERGE_FIELD("depth_mask_hold_frames", depth_mask_hold_frames);
    MERGE_FIELD("depth_mask_alpha", depth_mask_alpha);
    MERGE_FIELD("depth_mask_invert", depth_mask_invert);
    MERGE_FIELD("depth_debug_overlay_enabled", depth_debug_overlay_enabled);

    MERGE_FIELD("game_overlay_enabled", game_overlay_enabled);
    MERGE_FIELD("game_overlay_max_fps", game_overlay_max_fps);
    MERGE_FIELD("game_overlay_draw_boxes", game_overlay_draw_boxes);
    MERGE_FIELD("game_overlay_draw_future", game_overlay_draw_future);
    MERGE_FIELD("game_overlay_draw_wind_tail", game_overlay_draw_wind_tail);
    MERGE_FIELD("game_overlay_draw_frame", game_overlay_draw_frame);
    MERGE_FIELD("game_overlay_show_target_correction", game_overlay_show_target_correction);
    MERGE_FIELD("game_overlay_box_a", game_overlay_box_a);
    MERGE_FIELD("game_overlay_box_r", game_overlay_box_r);
    MERGE_FIELD("game_overlay_box_g", game_overlay_box_g);
    MERGE_FIELD("game_overlay_box_b", game_overlay_box_b);
    MERGE_FIELD("game_overlay_frame_a", game_overlay_frame_a);
    MERGE_FIELD("game_overlay_frame_r", game_overlay_frame_r);
    MERGE_FIELD("game_overlay_frame_g", game_overlay_frame_g);
    MERGE_FIELD("game_overlay_frame_b", game_overlay_frame_b);
    MERGE_FIELD("game_overlay_box_thickness", game_overlay_box_thickness);
    MERGE_FIELD("game_overlay_frame_thickness", game_overlay_frame_thickness);
    MERGE_FIELD("game_overlay_future_point_radius", game_overlay_future_point_radius);
    MERGE_FIELD("game_overlay_future_alpha_falloff", game_overlay_future_alpha_falloff);
    MERGE_FIELD("game_overlay_icon_enabled", game_overlay_icon_enabled);
    MERGE_FIELD("game_overlay_icon_path", game_overlay_icon_path);
    MERGE_FIELD("game_overlay_icon_width", game_overlay_icon_width);
    MERGE_FIELD("game_overlay_icon_height", game_overlay_icon_height);
    MERGE_FIELD("game_overlay_icon_offset_x", game_overlay_icon_offset_x);
    MERGE_FIELD("game_overlay_icon_offset_y", game_overlay_icon_offset_y);
    MERGE_FIELD("game_overlay_icon_anchor", game_overlay_icon_anchor);
    MERGE_FIELD("game_overlay_icon_class", game_overlay_icon_class);

    MERGE_FIELD("aim_sim_enabled", aim_sim_enabled);
    MERGE_FIELD("aim_sim_x", aim_sim_x);
    MERGE_FIELD("aim_sim_y", aim_sim_y);
    MERGE_FIELD("aim_sim_width", aim_sim_width);
    MERGE_FIELD("aim_sim_height", aim_sim_height);
    MERGE_FIELD("aim_sim_fps_min", aim_sim_fps_min);
    MERGE_FIELD("aim_sim_fps_max", aim_sim_fps_max);
    MERGE_FIELD("aim_sim_fps_jitter", aim_sim_fps_jitter);
    MERGE_FIELD("aim_sim_capture_delay_ms", aim_sim_capture_delay_ms);
    MERGE_FIELD("aim_sim_inference_delay_ms", aim_sim_inference_delay_ms);
    MERGE_FIELD("aim_sim_use_live_inference", aim_sim_use_live_inference);
    MERGE_FIELD("aim_sim_input_delay_ms", aim_sim_input_delay_ms);
    MERGE_FIELD("aim_sim_extra_delay_ms", aim_sim_extra_delay_ms);
    MERGE_FIELD("aim_sim_target_max_speed", aim_sim_target_max_speed);
    MERGE_FIELD("aim_sim_target_accel", aim_sim_target_accel);
    MERGE_FIELD("aim_sim_target_stop_chance", aim_sim_target_stop_chance);
    MERGE_FIELD("aim_sim_show_observed", aim_sim_show_observed);
    MERGE_FIELD("aim_sim_show_history", aim_sim_show_history);
    MERGE_FIELD("aim_sim_show_kalman_debug", aim_sim_show_kalman_debug);

    MERGE_FIELD("class_player", class_player);
    MERGE_FIELD("class_head", class_head);

    MERGE_FIELD("show_window", show_window);
    MERGE_FIELD("show_fps", show_fps);
    MERGE_FIELD("screenshot_button", screenshot_button);
    MERGE_FIELD("screenshot_delay", screenshot_delay);
    MERGE_FIELD("verbose", verbose);
    MERGE_FIELD("debug_log_file_enabled", debug_log_file_enabled);
    MERGE_FIELD("debug_log_file_path", debug_log_file_path);

    if (hasKey("active_game"))
        active_game = loaded.active_game;

#undef MERGE_FIELD

    CSimpleIniA::TNamesDepend gameKeys;
    ini.GetAllKeys("Games", gameKeys);
    for (const auto& k : gameKeys)
    {
        std::string name = k.pItem;
        auto it = loaded.game_profiles.find(name);
        if (it != loaded.game_profiles.end())
            game_profiles[name] = it->second;
    }

    config_path = loaded.config_path;
    return validate();
}

bool Config::validate()
{
    std::transform(neural_tracker_runtime.begin(), neural_tracker_runtime.end(), neural_tracker_runtime.begin(),
        [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (neural_tracker_runtime != "CPU"
#ifdef USE_CUDA
        && neural_tracker_runtime != "CUDA"
#endif
        )
    {
        std::cerr << "[Config] Unknown neural_tracker_runtime '" << neural_tracker_runtime
                  << "', using CPU neural tracker runtime." << std::endl;
        neural_tracker_runtime = "CPU";
    }

    if (!ParseMouseInputMethod(input_method))
    {
        std::cerr << "[Config] Unknown input_method '" << input_method
                  << "', falling back to WIN32." << std::endl;
        input_method = "WIN32";
    }

    const char* fallbackBackend = "DML";
#ifdef USE_CUDA
    // Runtime CUDA inference is selected with TRT; keep CUDA as a config alias.
    if (backend == "CUDA")
        backend = "TRT";
    fallbackBackend = "TRT";
#endif

    if (backend != "DML"
#ifdef USE_CUDA
        && backend != "TRT"
#endif
        )
    {
        std::cerr << "[Config] Unknown backend '" << backend
                  << "', falling back to " << fallbackBackend << "." << std::endl;
        backend = fallbackBackend;
    }

    if (teensy_hid_manufacturer.empty()) teensy_hid_manufacturer = "Generic";
    if (teensy_hid_product.empty()) teensy_hid_product = "USB HID Mouse";
    if (teensy_hid_serial.empty()) teensy_hid_serial = "AUTO";
    if (teensy_hid_vid_filter.empty()) teensy_hid_vid_filter = "AUTO";
    if (teensy_hid_pid_filter.empty()) teensy_hid_pid_filter = "AUTO";
    teensy_hid_usage_page = std::clamp(teensy_hid_usage_page, 1, 0xFFFF);
    teensy_hid_usage_id = std::clamp(teensy_hid_usage_id, 1, 0xFFFF);
    teensy_hid_open_index = std::clamp(teensy_hid_open_index, 0, 32);
    teensy_hid_packet_timeout_ms = std::clamp(teensy_hid_packet_timeout_ms, 0, 100);
    teensy_hid_reconnect_interval_ms = std::clamp(teensy_hid_reconnect_interval_ms, 50, 10000);

    if (debug_log_file_path.empty()) debug_log_file_path = "logs/0BS.log";
    return true;
}

bool Config::saveConfig(const std::string& filename)
{
    validate();

    std::string target = filename.empty() ? "config.ini" : filename;
    if (target == "config.ini" && !config_path.empty())
    {
        target = config_path;
    }

    std::ofstream file(target);
    if (!file.is_open())
    {
        std::cerr << "Error opening config for writing: " << target << std::endl;
        return false;
    }

    file << "# 0BS configuration\n\n";

    // Capture
    file << "# Capture\n"
        << "capture_method = " << capture_method << "\n"
        << "capture_target = " << capture_target << "\n"
        << "capture_window_title = " << capture_window_title << "\n"
        << "udp_ip = " << udp_ip << "\n"
        << "udp_port = " << udp_port << "\n"
        << "detection_resolution = " << detection_resolution << "\n"
        << "capture_fps = " << capture_fps << "\n"
        << "monitor_idx = " << monitor_idx << "\n"
        << "circle_mask = " << (circle_mask ? "true" : "false") << "\n"
        << "capture_borders = " << (capture_borders ? "true" : "false") << "\n"
        << "capture_cursor = " << (capture_cursor ? "true" : "false") << "\n"
        << "virtual_camera_name = " << virtual_camera_name << "\n"
        << "virtual_camera_width = " << virtual_camera_width << "\n"
        << "virtual_camera_heigth = " << virtual_camera_heigth << "\n\n";

    // Target
    file << "# Target\n"
        << "disable_headshot = " << (disable_headshot ? "true" : "false") << "\n"
        << std::fixed << std::setprecision(2)
        << "body_y_offset = " << body_y_offset << "\n"
        << "head_y_offset = " << head_y_offset << "\n"
        << "auto_aim = " << (auto_aim ? "true" : "false") << "\n\n";

    // Mouse
    file << "# Mouse move\n"
        << "fovX = " << fovX << "\n"
        << "fovY = " << fovY << "\n"
        << "prediction_futurePositions = " << prediction_futurePositions << "\n"
        << "draw_futurePositions = " << (draw_futurePositions ? "true" : "false") << "\n"
        << "runtime_latency_sweep_enabled = " << (runtime_latency_sweep_enabled ? "true" : "false") << "\n"
        << "estimator_mode = " << estimator_mode << "\n"
        << std::fixed << std::setprecision(3)
        << "kalman_enabled = " << (kalman_enabled ? "true" : "false") << "\n"
        << "kalman_process_noise_position = " << kalman_process_noise_position << "\n"
        << "kalman_process_noise_velocity = " << kalman_process_noise_velocity << "\n"
        << "kalman_measurement_noise = " << kalman_measurement_noise << "\n"
        << "kalman_velocity_damping = " << kalman_velocity_damping << "\n"
        << "kalman_max_velocity = " << kalman_max_velocity << "\n"
        << "kalman_warmup_frames = " << kalman_warmup_frames << "\n"
        << "kalman_velocity_seed_enabled = " << (kalman_velocity_seed_enabled ? "true" : "false") << "\n"
        << "kalman_acquisition_frames = " << kalman_acquisition_frames << "\n"
        << "kalman_compensate_detection_delay = " << (kalman_compensate_detection_delay ? "true" : "false") << "\n"
        << "kalman_additional_prediction_ms = " << kalman_additional_prediction_ms << "\n"
        << "kalman_reset_timeout_sec = " << kalman_reset_timeout_sec << "\n"
        << "ego_motion_compensation_enabled = " << (ego_motion_compensation_enabled ? "true" : "false") << "\n"
        << "ego_motion_compensation_strength = " << ego_motion_compensation_strength << "\n"
        << "ego_motion_compensation_max_shift_px = " << ego_motion_compensation_max_shift_px << "\n"
        << "ego_motion_compensation_max_age_ms = " << ego_motion_compensation_max_age_ms << "\n"
        << "# WIN32, GHUB, RAZER, ARDUINO, TEENSY41, TEENSY41_HID, KMBOX_NET, KMBOX_A, MAKCU\n"
        << "input_method = " << input_method << "\n\n";

    file << "# Pure PID mouse control\n"
        << "pid_actuator_hz = " << pid_actuator_hz << "\n"
        << std::fixed << std::setprecision(4)
        << "pid_kp = " << pid_kp << "\n"
        << "pid_ki = " << pid_ki << "\n"
        << "pid_kd = " << pid_kd << "\n"
        << std::fixed << std::setprecision(3)
        << "pid_deadzone_px = " << pid_deadzone_px << "\n"
        << "pid_max_pixel_step = " << pid_max_pixel_step << "\n"
        << "pid_output_scale = " << pid_output_scale << "\n"
        << "pid_min_output_scale = " << pid_min_output_scale << "\n"
        << "pid_max_output_scale = " << pid_max_output_scale << "\n"
        << "pid_size_reference_px = " << pid_size_reference_px << "\n"
        << "pid_size_min_scale = " << pid_size_min_scale << "\n"
        << "pid_size_max_scale = " << pid_size_max_scale << "\n"
        << "pid_precision_radius_scale = " << pid_precision_radius_scale << "\n"
        << "pid_slowdown_radius_scale = " << pid_slowdown_radius_scale << "\n"
        << "pid_overshoot_brake = " << pid_overshoot_brake << "\n"
        << "pid_divergence_boost = " << pid_divergence_boost << "\n"
        << "pid_scale_response = " << pid_scale_response << "\n"
        << "pid_max_integral = " << pid_max_integral << "\n"
        << "pid_max_derivative_term = " << pid_max_derivative_term << "\n"
        << "pid_derivative_filter_tau_ms = " << pid_derivative_filter_tau_ms << "\n"
        << "pid_target_loss_timeout_ms = " << pid_target_loss_timeout_ms << "\n"
        << "pid_feed_forward_enabled = " << (pid_feed_forward_enabled ? "true" : "false") << "\n"
        << "pid_feed_forward_gain = " << pid_feed_forward_gain << "\n"
        << "pid_feed_forward_lookahead_ms = " << pid_feed_forward_lookahead_ms << "\n"
        << "pid_feed_forward_frame_lookahead = " << pid_feed_forward_frame_lookahead << "\n"
        << "pid_feed_forward_max_step = " << pid_feed_forward_max_step << "\n"
        << "pid_feed_forward_min_speed = " << pid_feed_forward_min_speed << "\n"
        << "pid_feed_forward_confidence_floor = " << pid_feed_forward_confidence_floor << "\n"
        << "pid_conditional_integration_enabled = " << (pid_conditional_integration_enabled ? "true" : "false") << "\n"
        << "pid_conditional_integration_error_px = " << pid_conditional_integration_error_px << "\n"
        << "pid_adaptive_output_scaling_enabled = " << (pid_adaptive_output_scaling_enabled ? "true" : "false") << "\n"
        << "pid_adaptive_output_error_scale = " << pid_adaptive_output_error_scale << "\n"
        << "pid_derivative_smoothing_multiplier = " << pid_derivative_smoothing_multiplier << "\n"
        << "pid_perspective_fov_mapping_enabled = " << (pid_perspective_fov_mapping_enabled ? "true" : "false") << "\n"
        << "pid_governor_enabled = " << (pid_governor_enabled ? "true" : "false") << "\n"
        << "pid_governor_model_path = " << pid_governor_model_path << "\n"
        << "pid_governor_blend = " << pid_governor_blend << "\n"
        << "pid_governor_max_speed_multiple = " << pid_governor_max_speed_multiple << "\n"
        << "pid_smart_blending_enabled = " << (pid_smart_blending_enabled ? "true" : "false") << "\n"
        << "pid_smart_blending_aggression = " << pid_smart_blending_aggression << "\n"
        << "pid_smart_blending_near_damping = " << pid_smart_blending_near_damping << "\n"
        << "pid_smart_blending_deadzone_px = " << pid_smart_blending_deadzone_px << "\n"
        << "pid_smart_blending_jerk_limit_px = " << pid_smart_blending_jerk_limit_px << "\n"
        << "pid_smart_blending_confidence_floor = " << pid_smart_blending_confidence_floor << "\n\n";

    file << "# Neural tracker association\n"
        << "neural_tracker_enabled = " << (neural_tracker_enabled ? "true" : "false") << "\n"
        << "neural_tracker_runtime = " << neural_tracker_runtime << "\n"
        << "neural_tracker_model_path = " << neural_tracker_model_path << "\n"
        << std::fixed << std::setprecision(3)
        << "neural_tracker_blend = " << neural_tracker_blend << "\n"
        << "neural_tracker_log_enabled = " << (neural_tracker_log_enabled ? "true" : "false") << "\n"
        << "neural_tracker_debug_enabled = " << (neural_tracker_debug_enabled ? "true" : "false") << "\n"
        << "neural_tracker_log_path = " << neural_tracker_log_path << "\n\n";

    file << "# Learned temporal predictor\n"
        << "temporal_prediction_enabled = " << (temporal_prediction_enabled ? "true" : "false") << "\n"
        << "temporal_prediction_model_path = " << temporal_prediction_model_path << "\n"
        << "temporal_prediction_history_length = " << temporal_prediction_history_length << "\n"
        << "temporal_prediction_horizon = " << temporal_prediction_horizon << "\n"
        << "temporal_prediction_interval_frames = " << temporal_prediction_interval_frames << "\n"
        << "temporal_prediction_feed_forward_enabled = " << (temporal_prediction_feed_forward_enabled ? "true" : "false") << "\n"
        << std::fixed << std::setprecision(3)
        << "temporal_prediction_influence = " << temporal_prediction_influence << "\n"
        << "temporal_prediction_adaptive_influence_enabled = " << (temporal_prediction_adaptive_influence_enabled ? "true" : "false") << "\n"
        << "temporal_prediction_adaptive_ema_alpha = " << temporal_prediction_adaptive_ema_alpha << "\n"
        << "temporal_prediction_max_lead_px = " << temporal_prediction_max_lead_px << "\n\n";

    file << "# Neural targeting head\n"
        << "neural_targeting_enabled = " << (neural_targeting_enabled ? "true" : "false") << "\n"
        << "neural_targeting_model_path = " << neural_targeting_model_path << "\n"
        << std::fixed << std::setprecision(3)
        << "neural_targeting_influence = " << neural_targeting_influence << "\n"
        << "neural_targeting_max_refinement_px = " << neural_targeting_max_refinement_px << "\n"
        << "neural_targeting_max_iterations = " << neural_targeting_max_iterations << "\n"
        << "neural_control_preset = " << neural_control_preset << "\n"
        << "neural_control_telemetry_overlay_enabled = " << (neural_control_telemetry_overlay_enabled ? "true" : "false") << "\n"
        << "neural_control_telemetry_logging_enabled = " << (neural_control_telemetry_logging_enabled ? "true" : "false") << "\n"
        << "neural_control_telemetry_log_path = " << neural_control_telemetry_log_path << "\n"
        << "neural_control_telemetry_log_interval_ms = " << neural_control_telemetry_log_interval_ms << "\n\n";

    // Arduino
    file << "# Arduino\n"
        << "arduino_baudrate = " << arduino_baudrate << "\n"
        << "arduino_port = " << arduino_port << "\n"
        << "arduino_16_bit_mouse = " << (arduino_16_bit_mouse ? "true" : "false") << "\n"
        << "arduino_enable_keys = " << (arduino_enable_keys ? "true" : "false") << "\n\n";

    file << "# Teensy 4.1 RawHID generic mouse bridge\n"
        << "teensy_hid_manufacturer = " << teensy_hid_manufacturer << "\n"
        << "teensy_hid_product = " << teensy_hid_product << "\n"
        << "teensy_hid_serial = " << teensy_hid_serial << "\n"
        << "teensy_hid_vid_filter = " << teensy_hid_vid_filter << "\n"
        << "teensy_hid_pid_filter = " << teensy_hid_pid_filter << "\n"
        << "teensy_hid_usage_page = " << teensy_hid_usage_page << "\n"
        << "teensy_hid_usage_id = " << teensy_hid_usage_id << "\n"
        << "teensy_hid_open_index = " << teensy_hid_open_index << "\n"
        << "teensy_hid_packet_timeout_ms = " << teensy_hid_packet_timeout_ms << "\n"
        << "teensy_hid_reconnect_interval_ms = " << teensy_hid_reconnect_interval_ms << "\n\n";

    // kmbox_net
    file << "# Kmbox_net\n"
        << "kmbox_net_ip = " << kmbox_net_ip << "\n"
        << "kmbox_net_port = " << kmbox_net_port << "\n"
        << "kmbox_net_uuid = " << kmbox_net_uuid << "\n\n";

    file << "# Kmbox_a\n"
        << "kmbox_a_pidvid = " << kmbox_a_pidvid << "\n\n";

    // makcu
    file << "# Makcu\n"
        << "makcu_baudrate = " << makcu_baudrate << "\n"
		<< "makcu_port = " << makcu_port << "\n\n";

    // Mouse shooting
    file << "# Mouse shooting\n"
        << "auto_shoot = " << (auto_shoot ? "true" : "false") << "\n"
        << std::fixed << std::setprecision(1)
        << "bScope_multiplier = " << bScope_multiplier << "\n\n";

    // AI
    file << "# AI\n"
        << "backend = " << backend << "\n"
        << "dml_device_id = " << dml_device_id << "\n"
        << "ai_model = " << ai_model << "\n"
        << std::fixed << std::setprecision(2)
        << "confidence_threshold = " << confidence_threshold << "\n"
        << "nms_threshold = " << nms_threshold << "\n"
        << std::setprecision(0)
        << "max_detections = " << max_detections << "\n"
#ifdef USE_CUDA
        << "export_enable_fp8 = " << (export_enable_fp8 ? "true" : "false") << "\n"
        << "export_enable_fp16 = " << (export_enable_fp16 ? "true" : "false") << "\n"
#endif
        ;

    // CUDA
#ifdef USE_CUDA
    file << "\n# CUDA\n"
        << "use_cuda_graph = " << (use_cuda_graph ? "true" : "false") << "\n"
        << "use_pinned_memory = " << (use_pinned_memory ? "true" : "false") << "\n"
        << "gpuMemoryReserveMB = " << gpuMemoryReserveMB << "\n"
        << "enableGpuExclusiveMode = " << (enableGpuExclusiveMode ? "true" : "false") << "\n"
        << "capture_use_cuda = " << (capture_use_cuda ? "true" : "false") << "\n\n";
#endif

	// System
    file << "# System\n"
        << "cpuCoreReserveCount = " << cpuCoreReserveCount << "\n"
        << "systemMemoryReserveMB = " << systemMemoryReserveMB << "\n\n";

    // Buttons
    file << "# Buttons\n"
        << "button_targeting = " << joinStrings(button_targeting) << "\n"
        << "button_shoot = " << joinStrings(button_shoot) << "\n"
        << "button_zoom = " << joinStrings(button_zoom) << "\n"
        << "button_exit = " << joinStrings(button_exit) << "\n"
        << "button_pause = " << joinStrings(button_pause) << "\n"
        << "button_reload_config = " << joinStrings(button_reload_config) << "\n"
        << "button_open_overlay = " << joinStrings(button_open_overlay) << "\n"
        << "enable_arrows_settings = " << (enable_arrows_settings ? "true" : "false") << "\n\n";

    // Overlay
    file << "# Overlay\n"
        << "overlay_opacity = " << overlay_opacity << "\n"
        << std::fixed << std::setprecision(2)
        << "overlay_ui_scale = " << overlay_ui_scale << "\n"
        << "overlay_exclude_from_capture = " << (overlay_exclude_from_capture ? "true" : "false") << "\n\n";

    file << "# Depth\n"
        << "depth_inference_enabled = " << (depth_inference_enabled ? "true" : "false") << "\n"
        << "depth_model_path = " << depth_model_path << "\n"
        << "depth_fps = " << depth_fps << "\n"
        << "depth_colormap = " << depth_colormap << "\n"
        << "depth_mask_enabled = " << (depth_mask_enabled ? "true" : "false") << "\n"
        << "depth_mask_fps = " << depth_mask_fps << "\n"
        << "depth_mask_near_percent = " << depth_mask_near_percent << "\n"
        << "depth_mask_expand = " << depth_mask_expand << "\n"
        << "depth_mask_hold_frames = " << depth_mask_hold_frames << "\n"
        << "depth_mask_alpha = " << depth_mask_alpha << "\n"
        << "depth_mask_invert = " << (depth_mask_invert ? "true" : "false") << "\n"
        << "depth_debug_overlay_enabled = " << (depth_debug_overlay_enabled ? "true" : "false") << "\n\n";

    file << "# Game Overlay\n"
        << "game_overlay_enabled = " << (game_overlay_enabled ? "true" : "false") << "\n"
        << "game_overlay_max_fps = " << game_overlay_max_fps << "\n"
        << "game_overlay_draw_boxes = " << (game_overlay_draw_boxes ? "true" : "false") << "\n"
        << "game_overlay_draw_future = " << (game_overlay_draw_future ? "true" : "false") << "\n"
        << "game_overlay_draw_wind_tail = " << (game_overlay_draw_wind_tail ? "true" : "false") << "\n"
        << "game_overlay_draw_frame = " << (game_overlay_draw_frame ? "true" : "false") << "\n"
        << "game_overlay_show_target_correction = " << (game_overlay_show_target_correction ? "true" : "false") << "\n"
        << "game_overlay_box_a = " << game_overlay_box_a << "\n"
        << "game_overlay_box_r = " << game_overlay_box_r << "\n"
        << "game_overlay_box_g = " << game_overlay_box_g << "\n"
        << "game_overlay_box_b = " << game_overlay_box_b << "\n"
        << "game_overlay_frame_a = " << game_overlay_frame_a << "\n"
        << "game_overlay_frame_r = " << game_overlay_frame_r << "\n"
        << "game_overlay_frame_g = " << game_overlay_frame_g << "\n"
        << "game_overlay_frame_b = " << game_overlay_frame_b << "\n"
        << std::fixed << std::setprecision(2)
        << "game_overlay_box_thickness = " << game_overlay_box_thickness << "\n"
        << "game_overlay_frame_thickness = " << game_overlay_frame_thickness << "\n"
        << "game_overlay_future_point_radius = " << game_overlay_future_point_radius << "\n"
        << "game_overlay_future_alpha_falloff = " << game_overlay_future_alpha_falloff << "\n\n";

    file << "game_overlay_icon_enabled = " << (game_overlay_icon_enabled ? "true" : "false") << "\n"
        << "game_overlay_icon_path = " << game_overlay_icon_path << "\n"
        << "game_overlay_icon_width = " << game_overlay_icon_width << "\n"
        << "game_overlay_icon_height = " << game_overlay_icon_height << "\n"
        << std::fixed << std::setprecision(2)
        << "game_overlay_icon_offset_x = " << game_overlay_icon_offset_x << "\n"
        << std::fixed << std::setprecision(2)
        << "game_overlay_icon_offset_y = " << game_overlay_icon_offset_y << "\n"
        << "game_overlay_icon_anchor = " << game_overlay_icon_anchor << "\n"
        << "game_overlay_icon_class = " << game_overlay_icon_class << "\n\n";

    file << "# Aim Simulation Overlay\n"
        << "aim_sim_enabled = " << (aim_sim_enabled ? "true" : "false") << "\n"
        << "aim_sim_x = " << aim_sim_x << "\n"
        << "aim_sim_y = " << aim_sim_y << "\n"
        << "aim_sim_width = " << aim_sim_width << "\n"
        << "aim_sim_height = " << aim_sim_height << "\n"
        << "aim_sim_fps_min = " << aim_sim_fps_min << "\n"
        << "aim_sim_fps_max = " << aim_sim_fps_max << "\n"
        << std::fixed << std::setprecision(3)
        << "aim_sim_fps_jitter = " << aim_sim_fps_jitter << "\n"
        << std::fixed << std::setprecision(2)
        << "aim_sim_capture_delay_ms = " << aim_sim_capture_delay_ms << "\n"
        << "aim_sim_inference_delay_ms = " << aim_sim_inference_delay_ms << "\n"
        << "aim_sim_use_live_inference = " << (aim_sim_use_live_inference ? "true" : "false") << "\n"
        << "aim_sim_input_delay_ms = " << aim_sim_input_delay_ms << "\n"
        << "aim_sim_extra_delay_ms = " << aim_sim_extra_delay_ms << "\n"
        << "aim_sim_target_max_speed = " << aim_sim_target_max_speed << "\n"
        << "aim_sim_target_accel = " << aim_sim_target_accel << "\n"
        << "aim_sim_target_stop_chance = " << aim_sim_target_stop_chance << "\n"
        << "aim_sim_show_observed = " << (aim_sim_show_observed ? "true" : "false") << "\n"
        << "aim_sim_show_history = " << (aim_sim_show_history ? "true" : "false") << "\n"
        << "aim_sim_show_kalman_debug = " << (aim_sim_show_kalman_debug ? "true" : "false") << "\n\n";

    // Classes
    file << "# Custom Classes\n"
        << "class_player = " << class_player << "\n"
        << "class_head = " << class_head << "\n\n";

    // Debug
    file << "# Debug\n"
        << "show_window = " << (show_window ? "true" : "false") << "\n"
        << "show_fps = " << (show_fps ? "true" : "false") << "\n"
        << "screenshot_button = " << joinStrings(screenshot_button) << "\n"
        << "screenshot_delay = " << screenshot_delay << "\n"
        << "verbose = " << (verbose ? "true" : "false") << "\n"
        << "debug_log_file_enabled = " << (debug_log_file_enabled ? "true" : "false") << "\n"
        << "debug_log_file_path = " << debug_log_file_path << "\n\n";

    // Active game
    file << "# Active game profile\n";
    file << "active_game = " << active_game << "\n\n";
    file << "[Games]\n";
    for (auto& kv : game_profiles)
    {
        auto & gp = kv.second;
        file << gp.name << " = "
             << gp.sens << "," << gp.yaw;
        file << "," << gp.pitch;
        if (gp.fovScaled)
            file << ",true," << gp.baseFOV;
        file << "\n";
    }

    file.close();
    return true;
}

const Config::GameProfile& Config::currentProfile() const
{
    auto it = game_profiles.find(active_game);
    if (it != game_profiles.end()) return it->second;
    throw std::runtime_error("Active game profile not found: " + active_game);
}

std::pair<double, double> Config::degToCounts(double degX, double degY, double fovNow) const
{
    const auto& gp = currentProfile();
    double scale = (gp.fovScaled && gp.baseFOV > 1.0) ? (fovNow / gp.baseFOV) : 1.0;

    if (gp.sens == 0.0 || gp.yaw == 0.0 || gp.pitch == 0.0)
        return { 0.0, 0.0 };

    double cx = degX / (gp.sens * gp.yaw * scale);
    double cy = degY / (gp.sens * gp.pitch * scale);
    return { cx, cy };
}
