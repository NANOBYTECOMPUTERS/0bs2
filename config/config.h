#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>

class Config
{
public:
    // Capture
    std::string capture_method; // "duplication_api", "winrt", "virtual_camera", "udp_capture"
    std::string capture_target;
    std::string capture_window_title;
    std::string udp_ip;
    int udp_port;
    int detection_resolution;
    int capture_fps;
    int monitor_idx;
    bool circle_mask;
    bool capture_borders;
    bool capture_cursor;
    std::string virtual_camera_name;
    int virtual_camera_width;
    int virtual_camera_heigth;

    // Target
    // Vertical aim offsets are normalized box-space Y values from top (0) to bottom (1).
    static constexpr float kHeadYOffsetMin = 0.05f;
    static constexpr float kHeadYOffsetMax = 0.55f;
    static constexpr float kHeadYOffsetDefault = 0.05f;
    static constexpr float kBodyYOffsetMin = kHeadYOffsetMin;
    static constexpr float kBodyYOffsetMax = 0.90f;
    static constexpr float kBodyYOffsetDefault = 0.15f;

    bool disable_headshot;
    float body_y_offset;
    float head_y_offset;
    bool auto_aim;

    // Mouse
    // Legacy prediction compatibility: deprecated but retained/loadable for old configs
    // and for aim-sim/overlay demo parity. Runtime convergence should use tracker/Kalman.
    float minSpeedMultiplier;
    float maxSpeedMultiplier;

    float predictionInterval;
    int prediction_futurePositions;
    bool draw_futurePositions;
    bool runtime_latency_sweep_enabled;
    std::string estimator_mode; // "kalman" or "imm"; IMM is opt-in and tracker-only in v1.
    bool kalman_enabled;
    float kalman_process_noise_position;
    float kalman_process_noise_velocity;
    float kalman_measurement_noise;
    float kalman_velocity_damping;
    float kalman_max_velocity;
    int kalman_warmup_frames;
    bool kalman_velocity_seed_enabled;
    int kalman_acquisition_frames;
    bool kalman_compensate_detection_delay;
    float kalman_additional_prediction_ms;
    float kalman_reset_timeout_sec;
    bool ego_motion_compensation_enabled;
    float ego_motion_compensation_strength;
    float ego_motion_compensation_max_shift_px;
    int ego_motion_compensation_max_age_ms;

    // Legacy prediction compatibility: deprecated snap-curve knobs kept for config
    // compatibility and the game-overlay aim simulation.
    float snapRadius;
    float nearRadius;
    float speedCurveExponent;
    float snapBoostFactor;

    bool easynorecoil;
    float easynorecoilstrength;
    std::string input_method; // "WIN32", "GHUB", "RAZER", "DIRECT", "ARDUINO", "TEENSY41", "TEENSY41_HID", "KMBOX_NET", "KMBOX_A", "MAKCU"

    // Wind mouse
    bool wind_mouse_enabled;
    float wind_G;
    float wind_W;
    float wind_M;
    float wind_D;

    // Direct targeting movement
    float target_deadzone_px;
    float target_max_pixel_step;
    float target_output_scale;
    bool target_calibrated_pixel_counts_enabled;
    float target_counts_per_pixel_x;
    float target_counts_per_pixel_y;
    float target_prediction_blend;
    float target_prediction_max_lead_px;

    // Tracker identity pipeline
    bool tracker_v2_enabled;
    float tracker_v2_high_confidence;
    float tracker_v2_new_track_confidence;
    int tracker_v2_detector_max_candidates;
    float tracker_v2_box_smoothing_alpha;
    float tracker_v2_box_prediction_alpha;
    bool yolo26_disable_nms;

    // Arduino
    int arduino_baudrate;
    std::string arduino_port;
    bool arduino_16_bit_mouse;
    bool arduino_enable_keys;

    // Teensy 4.1 RawHID generic mouse bridge
    std::string teensy_hid_manufacturer;
    std::string teensy_hid_product;
    std::string teensy_hid_serial;
    std::string teensy_hid_vid_filter;
    std::string teensy_hid_pid_filter;
    int teensy_hid_usage_page;
    int teensy_hid_usage_id;
    int teensy_hid_open_index;
    int teensy_hid_packet_timeout_ms;
    int teensy_hid_reconnect_interval_ms;

    // kmbox_net
    std::string kmbox_net_ip;
    std::string kmbox_net_port;
    std::string kmbox_net_uuid;

    // kmbox_a
    std::string kmbox_a_pidvid; // PIDVID in one field, format: PPPPVVVV

    // makcu
    int makcu_baudrate;
    std::string makcu_port;

    // Mouse shooting
    bool auto_shoot;
    float bScope_multiplier;

    // AI
    std::string backend;
    int dml_device_id;
    std::string ai_model;
    float confidence_threshold;
    float nms_threshold;
    int max_detections;
#ifdef USE_CUDA
    bool export_enable_fp8;
    bool export_enable_fp16;
#endif
    bool fixed_input_size;

    // CUDA
#ifdef USE_CUDA
    bool use_cuda_graph;
    bool use_pinned_memory;
    int gpuMemoryReserveMB;
    bool enableGpuExclusiveMode;
    bool capture_use_cuda;
#endif

    // System
    int cpuCoreReserveCount;
    int systemMemoryReserveMB;

    // Buttons
    std::vector<std::string> button_targeting;
    std::vector<std::string> button_shoot;
    std::vector<std::string> button_zoom;
    std::vector<std::string> button_exit;
    std::vector<std::string> button_pause;
    std::vector<std::string> button_reload_config;
    std::vector<std::string> button_open_overlay;
    bool enable_arrows_settings;

    // Overlay
    int overlay_opacity;
    float overlay_ui_scale;
    bool overlay_exclude_from_capture;

    // Game Overlay
    bool game_overlay_enabled;
    int game_overlay_max_fps;
    bool game_overlay_draw_boxes;
    bool game_overlay_draw_future;
    bool game_overlay_draw_wind_tail;
    bool game_overlay_draw_frame;
    bool game_overlay_show_target_correction;
    int game_overlay_box_a;
    int game_overlay_box_r;
    int game_overlay_box_g;
    int game_overlay_box_b;
    int game_overlay_frame_a;
    int game_overlay_frame_r;
    int game_overlay_frame_g;
    int game_overlay_frame_b;
    float game_overlay_box_thickness;
    float game_overlay_frame_thickness;
    float game_overlay_future_point_radius;
    float game_overlay_future_alpha_falloff;

    bool game_overlay_icon_enabled;
    std::string game_overlay_icon_path;
    int game_overlay_icon_width;
    int game_overlay_icon_height;
    float game_overlay_icon_offset_x;
    float game_overlay_icon_offset_y;
    std::string game_overlay_icon_anchor; // "center", "top", "bottom", "head"
    int game_overlay_icon_class; // -1 = all

    // Aim Simulation Overlay
    bool aim_sim_enabled;
    int aim_sim_x;
    int aim_sim_y;
    int aim_sim_width;
    int aim_sim_height;
    int aim_sim_fps_min;
    int aim_sim_fps_max;
    float aim_sim_fps_jitter;
    float aim_sim_capture_delay_ms;
    float aim_sim_inference_delay_ms;
    bool aim_sim_use_live_inference;
    float aim_sim_input_delay_ms;
    float aim_sim_extra_delay_ms;
    float aim_sim_target_max_speed;
    float aim_sim_target_accel;
    float aim_sim_target_stop_chance;
    bool aim_sim_show_observed;
    bool aim_sim_show_history;
    bool aim_sim_show_kalman_debug;

    void clampGameOverlayColor()
    {
        auto clamp255 = [](int& v) { if (v < 0) v = 0; if (v > 255) v = 255; };
        clamp255(game_overlay_box_a);
        clamp255(game_overlay_box_r);
        clamp255(game_overlay_box_g);
        clamp255(game_overlay_box_b);
        clamp255(game_overlay_frame_a);
        clamp255(game_overlay_frame_r);
        clamp255(game_overlay_frame_g);
        clamp255(game_overlay_frame_b);
    }

    // Classes
    int class_player;
    int class_head;

    // Debug
    bool show_window;
    bool show_fps;
    std::vector<std::string> screenshot_button;
    int screenshot_delay;
    bool verbose;
    bool debug_log_file_enabled;
    std::string debug_log_file_path;

    bool validate();
    bool loadConfig(const std::string& filename = "config.ini");
    bool loadConfigMerged(const std::string& filename = "config.ini");
    bool saveConfig(const std::string& filename = "config.ini");

    std::string joinStrings(const std::vector<std::string>& vec, const std::string& delimiter = ",");
private:
    std::vector<std::string> splitString(const std::string& str, char delimiter = ',');
    std::string config_path;
};

#endif // CONFIG_H
