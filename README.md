# 0BS GUI and Config Setting Reference

Generated on 2026-05-16 from `x64\DML\config.ini` and the ImGui source files under `overlay/`.

This reference is ordered by the GUI sidebar tabs in `overlay/overlay.cpp`. It lists every GUI slider and every activate/deactivate checkbox, then documents config.ini settings that are not editable in the current DML GUI or are hidden/loadable config keys.

## Searchable Directory

- PDF: `docs/settings-reference/0BS-settings-reference.pdf`
- Search index: `docs/settings-reference/settings-index.csv`
- Directory README: `docs/settings-reference/README.md`

Search by GUI label, config key, tab, section, or source file. Rows marked `CUDA GUI / DML config-only` appear as GUI controls only in CUDA builds; the current `x64/DML/config.ini` keeps those values as config-only settings.

## Perfect Aim v1.0

Perfect Aim v1.0 keeps all neural systems advisory only and default OFF. The runtime path remains:

`Video Frame -> Detector -> Tracker -> Kalman -> Temporal Predictor -> Neural Targeting Head -> Adaptive Influence + SmartBlender -> PID/Governor -> Mouse Output`

PID/Kalman remains the convergence owner. Temporal prediction and the neural targeting head can only add bounded feed-forward/refinement offsets; they do not replace the tracked target, modify the PID observation point, or write directly to final actuator deltas.

The Neural tab exposes Balanced, Aggressive, Smooth, and Sniper presets. Presets tune prediction influence, neural refinement range, and SmartBlender damping/jerk limits while retaining opt-in master toggles. Optional telemetry can show current adaptive influence, confidence, predicted lead, neural refinement, and SmartBlender jitter/oscillation state on the game overlay, or write a throttled CSV log for tuning.

## GUI Tab Order

| Order | Tab | Notes |
| --- | --- | --- |
| 1 | Capture | Adjustable controls documented below |
| 2 | Target | Adjustable controls documented below |
| 3 | Mouse | Adjustable controls documented below |
| 4 | Neural | Adjustable controls documented below |
| 5 | AI | Adjustable controls documented below |
| 6 | Buttons | Adjustable controls documented below |
| 7 | Overlay | Adjustable controls documented below |
| 8 | Game Overlay | Adjustable controls documented below |
| 9 | Stats | Read-only monitor tab |
| 10 | Debug | Adjustable controls documented below |

## Capture Tab

### Sliders

| Section | Slider | Config key | Range | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| General Capture | Capture FPS | capture_fps | 0-240 | 70 | Limits capture loop FPS. 0 disables the cap path and the GUI warns at 0 or 61+. |
| Capture Preview | Debug scale | preview_debug_scale | 0.1x-2.0x | n/a | Preview-only scale slider. It is not saved to config.ini. |
| Virtual Camera | Virtual camera width | virtual_camera_width | 128-3840 | 1920 | Applies when capture_method is virtual_camera. |
| Virtual Camera | Virtual camera heigth | virtual_camera_heigth | 128-2160 | 1080 | Spelling follows the existing config key. Applies when capture_method is virtual_camera. |

### Activate/Deactivate

| Section | Control | Config key | Values | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| General Capture | Circle mask | circle_mask | true/false | true | Applies the circular capture mask. |
| General Capture | Use CUDA Direct Capture | capture_use_cuda | true/false | n/a | CUDA/TensorRT builds only. Available with duplication_api capture. |
| Capture Preview | Show Preview Window | show_window | true/false | true | Shows the capture preview/debug frame inside the Capture tab. |
| WinRT | Capture Borders | capture_borders | true/false | true | WinRT capture only. Disabled automatically on unsupported Windows builds. |
| WinRT | Capture Cursor | capture_cursor | true/false | true | WinRT capture only. Includes the cursor in captured frames. |

### Other GUI-Exposed Config Controls

| Section | Control | Config key | Type/options | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| General Capture | Detection Resolution | detection_resolution | 160, 320, 640 | 640 | Changes detector input resolution and restarts/reloads dependent paths. |
| General Capture | Capture method | capture_method | duplication_api, winrt, virtual_camera, udp_capture | duplication_api | Selects frame source. |
| WinRT | Capture target (WinRT) | capture_target | monitor, window | monitor | WinRT only. |
| WinRT | Window title contains | capture_window_title | text |  | Used when WinRT target is window. |
| Monitor Capture | Capture monitor | monitor_idx | monitor index | 0 | Monitor list is built from active displays. |
| Virtual Camera | Virtual camera | virtual_camera_name | available cameras | None | Filtered list of DirectShow video input devices. |
| UDP Capture | UDP IP | udp_ip | IPv4/string | 0.0.0.0 | Applied with the Apply UDP Settings button. |
| UDP Capture | UDP Port | udp_port | 1-65535 | 1234 | Applied with the Apply UDP Settings button. |

## Target Tab

### Sliders

| Section | Slider | Config key | Range | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Offsets | Approximate Body Y Offset | body_y_offset | 0.00-1.00 | 0.06 | Vertical aim point used for body-class detections. |
| Offsets | Approximate Head Y Offset | head_y_offset | 0.00-1.00 | 0.38 | Vertical aim point used for head-class detections. |

### Activate/Deactivate

| Section | Control | Config key | Values | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Targeting | Disable Headshot | disable_headshot | true/false | false | Prevents the head class from being selected as the aim target. |
| Targeting | Auto Aim | auto_aim | true/false | false | Allows target selection/movement logic to act automatically while held conditions are met. |

## Mouse Tab

### Sliders

| Section | Slider | Config key | Range | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| FOV | FOV X | fovX | 10-120 | 70 | Horizontal field of view for the mouse controller. |
| FOV | FOV Y | fovY | 10-120 | 70 | Vertical field of view for the mouse controller. |
| State Estimator | Acquisition frames | kalman_acquisition_frames | 3-5 | n/a | Frames used to ramp prediction weight after target acquisition. |
| State Estimator | Process noise position | kalman_process_noise_position | 0.0001-5000 | n/a | Position process noise. Higher values adapt faster to position changes. |
| State Estimator | Process noise velocity | kalman_process_noise_velocity | 0.0001-50000 | n/a | Velocity process noise. Higher values adapt faster to speed changes. |
| State Estimator | Measurement noise | kalman_measurement_noise | 0.0001-5000 | n/a | Measurement noise. Higher values trust detections less. |
| State Estimator | Velocity damping | kalman_velocity_damping | 0.0-3.0 | n/a | Damps velocity estimates to reduce runaway prediction. |
| State Estimator | Max velocity (px/s) | kalman_max_velocity | 100-60000 | n/a | Maximum allowed tracked velocity. |
| State Estimator | Warmup frames | kalman_warmup_frames | 0-20 | n/a | Legacy warmup frames when latency sweep is off. |
| State Estimator | Additional prediction (ms) | kalman_additional_prediction_ms | -80.0-120.0 | n/a | Manual extra prediction offset in milliseconds. |
| State Estimator | Reset timeout (s) | kalman_reset_timeout_sec | 0.05-3.0 | n/a | Time without stable observations before estimator state resets. |
| Pure PID Movement | Actuator Hz | pid_actuator_hz | 30-2000 | 2000 | PID update rate used by the mouse actuator. |
| Pure PID Movement | Kp | pid_kp | 0.0000-1.5000 | 0.0115 | Proportional PID gain. |
| Pure PID Movement | Ki | pid_ki | 0.0000-0.5000 | 0.0003 | Integral PID gain. |
| Pure PID Movement | Kd | pid_kd | 0.0000-0.2500 | 0.0001 | Derivative PID gain. |
| Pure PID Movement | Deadzone (px) | pid_deadzone_px | 0.000-10.000 | 1.500 | Minimum error radius before movement is applied. |
| Pure PID Movement | Max step (px/tick) | pid_max_pixel_step | 0.010-20.000 | 20.000 | Maximum movement step per PID tick from the GUI. Runtime clamp allows up to 80. |
| Pure PID Movement | Output scale | pid_output_scale | 0.010-3.000 | 0.209 | Base multiplier applied to PID output. |
| Pure PID Movement | Min output scale | pid_min_output_scale | 0.000-3.000 | 0.100 | Lower bound for adaptive output scaling. |
| Pure PID Movement | Max output scale | pid_max_output_scale | 0.010-3.000 | 0.509 | Upper bound for adaptive output scaling. |
| Pure PID Movement | Size reference (px) | pid_size_reference_px | 1.0-240.0 | 48.000 | Target size used as the neutral reference for size scaling. |
| Pure PID Movement | Small target scale | pid_size_min_scale | 0.010-1.000 | 0.200 | Minimum scale when targets are smaller than the reference. |
| Pure PID Movement | Large target scale | pid_size_max_scale | 0.050-2.000 | 1.000 | Maximum scale when targets are larger than the reference. |
| Pure PID Movement | Precision radius / size | pid_precision_radius_scale | 0.0000-0.1000 | 0.020 | Target-size-relative radius where movement is considered precise enough. |
| Pure PID Movement | Slowdown radius / size | pid_slowdown_radius_scale | 0.010-1.000 | 0.300 | Target-size-relative radius where scaling slows approach. |
| Pure PID Movement | Overshoot brake | pid_overshoot_brake | 0.010-1.000 | 0.350 | Reduces movement after crossing past the target. |
| Pure PID Movement | Divergence boost | pid_divergence_boost | 0.000-2.000 | 0.350 | Boosts output when the target error is growing. |
| Pure PID Movement | Scale response | pid_scale_response | 0.1-40.0 | 8.000 | Responsiveness of adaptive output scale changes. |
| Pure PID Movement | Max integral | pid_max_integral | 0.0-10000.0 | 120.000 | Caps integral accumulation. |
| Pure PID Movement | Max derivative term | pid_max_derivative_term | 0.000-5.000 | 0.020 | Caps derivative contribution from a single axis. |
| Pure PID Movement | Derivative filter (ms) | pid_derivative_filter_tau_ms | 0.0-250.0 | 18.000 | Smoothing time constant for derivative filtering. |
| Pure PID Movement | Target timeout (ms) | pid_target_loss_timeout_ms | 10.0-1000.0 | 90.000 | Time without target observation before PID state resets. |
| Pure PID Movement | Feed-forward gain | pid_feed_forward_gain | 0.000-4.000 | 0.201 | Strength of feed-forward movement. |
| Pure PID Movement | Feed-forward lookahead (ms) | pid_feed_forward_lookahead_ms | 0.0-120.0 | 16.800 | Prediction lookahead horizon. |
| Pure PID Movement | Feed-forward frame lookahead | pid_feed_forward_frame_lookahead | 0-2 | n/a | Additional frame-based velocity lead used when latency sweep is enabled. |
| Pure PID Movement | Feed-forward max step (px/tick) | pid_feed_forward_max_step | 0.000-5.000 | 5.000 | Caps feed-forward contribution per tick. |
| Pure PID Movement | Feed-forward min speed (px/s) | pid_feed_forward_min_speed | 0.0-3000.0 | 3000.000 | Minimum target speed required before feed-forward contributes. |
| Pure PID Movement | Feed-forward confidence floor | pid_feed_forward_confidence_floor | 0.000-1.000 | 0.527 | Minimum observation confidence for feed-forward contribution. |
| Pure PID Movement | Integration error limit (px) | pid_conditional_integration_error_px | 0.0-240.0 | n/a | Maximum per-axis error allowed for conditional integral accumulation. |
| Pure PID Movement | Adaptive error scale (px) | pid_adaptive_output_error_scale | 1.0-640.0 | n/a | Error distance used as the adaptive output scaling reference. |
| Pure PID Movement | Derivative smoothing multiplier | pid_derivative_smoothing_multiplier | 1.00-6.00 | n/a | Multiplies derivative filter tau when latency sweep is enabled. |
| Pure PID Movement | Governor blend | pid_governor_blend | 0.00-1.00 | 0.200 | Blend strength for governor-generated PID scales. |
| Pure PID Movement | Governor max speed multiple | pid_governor_max_speed_multiple | 1.00-5.00 | 1.000 | Maximum speed multiplier allowed when governor output is active. |
| Game Profile | Sensitivity | Games.<profile>.sens | 0.0010-10.0000 | profile/local | Editable for custom profiles. UNIFIED is shown read-only. |
| Game Profile | Yaw | Games.<profile>.yaw | 0.0010-0.1000 | profile/local | Horizontal degree-to-count conversion for a custom profile. |
| Game Profile | Pitch | Games.<profile>.pitch | 0.0010-0.1000 | profile/local | Vertical degree-to-count conversion for a custom profile. |
| Game Profile | Base FOV | Games.<profile>.baseFOV | 10.0-180.0 | profile/local | Shown only when FOV Scaled is enabled on a custom profile. |
| Auto Shoot | bScope Multiplier | bScope_multiplier | 0.5-2.0 | 1.0 | Multiplier used by auto-shoot scope timing/behavior. Disabled in UI until Auto Shoot is on. |

### Activate/Deactivate

| Section | Control | Config key | Values | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| State Estimator | Runtime latency sweep | runtime_latency_sweep_enabled | true/false | n/a | Enables side-by-side latency sweep behavior for estimator and PID experiments. |
| State Estimator | Enable Kalman estimator | kalman_enabled | true/false | n/a | Enables Kalman filtering for target motion estimation. |
| State Estimator | Seed velocity on acquire | kalman_velocity_seed_enabled | true/false | n/a | Seeds velocity from early measurement deltas during acquisition. |
| State Estimator | Compensate detection delay | kalman_compensate_detection_delay | true/false | n/a | Accounts for detector latency in prediction. |
| Pure PID Movement | Feed-forward prediction | pid_feed_forward_enabled | true/false | true | Adds velocity-based prediction ahead of pure PID output. |
| Pure PID Movement | Conditional integration | pid_conditional_integration_enabled | true/false | n/a | Prevents integral windup when latency sweep is enabled and error/output conditions are unsafe. |
| Pure PID Movement | Adaptive output scaling | pid_adaptive_output_scaling_enabled | true/false | n/a | Enables error-magnitude output scaling when latency sweep is enabled. |
| Pure PID Movement | Enable PID governor | pid_governor_enabled | true/false | true | Enables the ONNX PID governor when a compatible model can be loaded. |
| Game Profile | FOV Scaled | Games.<profile>.fovScaled | true/false | profile/local | When enabled, the profile also uses a Base FOV value. |
| Auto Shoot | Auto Shoot | auto_shoot | true/false | false | Enables automatic shooting behavior. |
| Input Method | Arduino 16-bit Mouse | arduino_16_bit_mouse | true/false | false | Arduino input only. Sends wider mouse movement values. |
| Input Method | Arduino Enable Keys | arduino_enable_keys | true/false | false | Arduino input only. Enables keyboard key output through Arduino. |

### Other GUI-Exposed Config Controls

| Section | Control | Config key | Type/options | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Game Profile | Active Game Profile | active_game | profile names | UNIFIED | Selects the active game profile. |
| Manage Profiles | Game profile rows | Games.<profile> | name = sens,yaw,pitch[,true,baseFOV] | profile/local | Custom profiles can be added/deleted; UNIFIED is read-only. |
| Input Method | Mouse Input Method | input_method | WIN32, GHUB, RAZER, ARDUINO, TEENSY41, KMBOX_NET, KMBOX_A, MAKCU | RAZER | Changes active mouse backend. |
| Input Method | Arduino/Teensy Port | arduino_port | COM1-COM30 | COM0 | Arduino and Teensy 4.1 serial input. |
| Input Method | Arduino/Teensy Baudrate | arduino_baudrate | 9600-115200 | 115200 | Arduino and Teensy 4.1 serial input. |
| Input Method | Kmbox Net IP | kmbox_net_ip | text | 10.42.42.42 | Saved with Save & Reconnect. |
| Input Method | Kmbox Net Port | kmbox_net_port | text | 1984 | Saved with Save & Reconnect. |
| Input Method | Kmbox Net UUID | kmbox_net_uuid | text | DEADC0DE | Saved with Save & Reconnect. |
| Input Method | Kmbox A PIDVID | kmbox_a_pidvid | PPPPVVVV |  | Saved with Save & Reconnect. |
| Input Method | Makcu Port | makcu_port | AUTO, COM1-COM30 | AUTO | Makcu input only. |
| Input Method | Makcu Baudrate | makcu_baudrate | 115200-4000000 | 4000000 | Makcu input only. |

## Neural Tab

### Sliders

| Section | Slider | Config key | Range | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Neural Tracker | Association blend | neural_tracker_blend | 0.00-1.00 | 1.000 | Blend strength between regular and neural association output. |

### Activate/Deactivate

| Section | Control | Config key | Values | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Neural Tracker | Enable neural association | neural_tracker_enabled | true/false | true | Enables learned association/tracking helper logic. |

### Other GUI-Exposed Config Controls

| Section | Control | Config key | Type/options | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Neural Tracker | Association model | neural_tracker_model_path | path | training/models/neural_tracker.onnx | Path to the neural tracker ONNX file. |

## AI Tab

### Sliders

| Section | Slider | Config key | Range | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Detection | Confidence Threshold | confidence_threshold | 0.01-1.00 | 0.10 | Minimum detector confidence accepted as a detection. |
| Detection | NMS Threshold | nms_threshold | 0.00-1.00 | 0.50 | Non-max suppression overlap threshold. |
| Detection | Max Detections | max_detections | 1-100 | 100 | Maximum detections retained after model output processing. |
| Depth Runtime | Depth FPS | depth_fps | 0-120 | 100 | CUDA builds only. 0 disables depth debug update throttling path. |
| Depth Runtime | Depth Mask FPS | depth_mask_fps | 1-30 | 5 | CUDA builds only. Mask update frequency. |
| Depth Mask | Depth Mask Near % | depth_mask_near_percent | 1-100 | 20 | CUDA builds only. Percentile cutoff for the near-depth mask. |
| Depth Mask | Depth Mask Expand (px) | depth_mask_expand | 0-128 | 0 | CUDA builds only. Expands the generated mask in pixels. |
| Depth Mask | Depth Mask Hold Frames | depth_mask_hold_frames | 0-120 | 0 | CUDA builds only. Reuses a recent mask for this many detector frames. |
| Depth Mask | Depth Mask Alpha | depth_mask_alpha | 0-255 | 90 | CUDA builds only. Visual alpha for the depth mask overlay. |

### Activate/Deactivate

| Section | Control | Config key | Values | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Depth Inference | Enable Depth Inference | depth_inference_enabled | true/false | true | CUDA builds only. DML build shows a requires-CUDA message. |
| Depth Mask | Enable Depth Mask | depth_mask_enabled | true/false | false | CUDA builds only. Masks detections/capture based on estimated depth. |
| Depth Mask | Depth Mask Invert | depth_mask_invert | true/false | false | CUDA builds only. Flips near/far mask selection. |
| Depth Mask | Depth Debug Overlay (Game) | depth_debug_overlay_enabled | true/false | false | CUDA builds only. Shows depth debug output in the game overlay. |

### Other GUI-Exposed Config Controls

| Section | Control | Config key | Type/options | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Model | Model | ai_model | models folder | sunxds_0.8.2_DML.onnx | Selects detector model from available files. |
| Backend | Backend | backend | TRT, DML | DML | CUDA builds only; DML build does not expose this combo. |
| Depth Inference | Depth model | depth_model_path | models/depth | depth_anything_v2.engine | CUDA builds only. |
| Depth Mask | Depth colormap | depth_colormap | 0-21 OpenCV colormap index | 18 | CUDA builds only. |

## Buttons Tab

### Sliders

No saved GUI sliders in this tab.

### Activate/Deactivate

| Section | Control | Config key | Values | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Arrow Key Options | Enable arrows keys options | enable_arrows_settings | true/false | false | Allows arrow-key adjustment of target offsets. |

### Other GUI-Exposed Config Controls

| Section | Control | Config key | Type/options | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Button Bindings | Targeting Buttons | button_targeting | key list | RightMouseButton | Supports multiple bindings with + and -. |
| Button Bindings | Shoot Buttons | button_shoot | key list | LeftMouseButton | Supports multiple bindings with + and -. |
| Button Bindings | Zoom Buttons | button_zoom | key list | RightMouseButton | Supports multiple bindings with + and -. |
| Button Bindings | Exit Buttons | button_exit | key list | F2 | Supports multiple bindings with + and -. |
| Button Bindings | Pause Buttons | button_pause | key list | F3 | Supports multiple bindings with + and -. |
| Button Bindings | Reload Config Buttons | button_reload_config | key list | F4 | Supports multiple bindings with + and -. |
| Button Bindings | Overlay Buttons | button_open_overlay | key list | Home | Supports multiple bindings with + and -. |

## Overlay Tab

### Sliders

| Section | Slider | Config key | Range | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Visual | Overlay Opacity | overlay_opacity | 220-255 | 225 | Window opacity. The UI clamps to a readable minimum of 220. |
| Visual | UI Fine Scale | overlay_ui_scale | 0.85-1.35 | 1.00 | Fine-grained scaling for the overlay UI. |

### Activate/Deactivate

| Section | Control | Config key | Values | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Capture Privacy | Hide Overlays From Recording | overlay_exclude_from_capture | true/false | true | Applies capture exclusion so overlay windows are hidden from supported recording/capture paths. |

### Other GUI-Exposed Config Controls

| Section | Control | Config key | Type/options | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Config | Save Config | config.ini | writes config.ini | n/a | Immediately writes the current runtime config. |
| Config | Load Config | config.ini | reads config.ini | n/a | Merges config.ini into current runtime values and refreshes runtime-sensitive paths. |

## Game Overlay Tab

### Sliders

| Section | Slider | Config key | Range | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| General | Overlay Max FPS (0 = uncapped) | game_overlay_max_fps | 0-256 | 0 | Caps game overlay render rate. 0 is uncapped. |
| Box Color | A | game_overlay_box_a | 0-255 | 255 | Detection box alpha. |
| Box Color | R | game_overlay_box_r | 0-255 | 0 | Detection box red channel. |
| Box Color | G | game_overlay_box_g | 0-255 | 255 | Detection box green channel. |
| Box Color | B | game_overlay_box_b | 0-255 | 0 | Detection box blue channel. |
| Box Color | Box Thickness | game_overlay_box_thickness | 0.5-10.0 | 2.00 | Detection box line thickness. |
| Capture Frame | A | game_overlay_frame_a | 0-255 | 180 | Capture frame alpha. |
| Capture Frame | R | game_overlay_frame_r | 0-255 | 255 | Capture frame red channel. |
| Capture Frame | G | game_overlay_frame_g | 0-255 | 255 | Capture frame green channel. |
| Capture Frame | B | game_overlay_frame_b | 0-255 | 255 | Capture frame blue channel. |
| Capture Frame | Frame Thickness | game_overlay_frame_thickness | 0.5-10.0 | 1.50 | Capture frame line thickness. |
| Future Point Style | Point Radius | game_overlay_future_point_radius | 1.0-20.0 | 5.00 | Radius of future-position points. |
| Future Point Style | Point Step Alpha Falloff | game_overlay_future_alpha_falloff | 0.10-5.00 | 1.00 | Controls how quickly future-point alpha fades along the path. |
| Icon Overlay | Icon Width | game_overlay_icon_width | 4-512 | 64 | Icon width in pixels. |
| Icon Overlay | Icon Height | game_overlay_icon_height | 4-512 | 64 | Icon height in pixels. |
| Icon Overlay | Icon Offset X | game_overlay_icon_offset_x | -500.0-500.0 | 0.00 | Horizontal icon offset from the selected anchor. |
| Icon Overlay | Icon Offset Y | game_overlay_icon_offset_y | -500.0-500.0 | 0.00 | Vertical icon offset from the selected anchor. |
| Aim Simulation | Sim X | aim_sim_x | -3000-3000 | 24 | Aim simulation window X position. |
| Aim Simulation | Sim Y | aim_sim_y | -3000-3000 | 24 | Aim simulation window Y position. |
| Aim Simulation | Sim Width | aim_sim_width | 220-1600 | 560 | Aim simulation window width. |
| Aim Simulation | Sim Height | aim_sim_height | 180-1000 | 360 | Aim simulation window height. |
| Aim Simulation | Sim FPS Min | aim_sim_fps_min | 15-360 | 90 | Minimum simulated frame rate. |
| Aim Simulation | Sim FPS Max | aim_sim_fps_max | 15-360 | 120 | Maximum simulated frame rate. The GUI keeps min and max ordered. |
| Aim Simulation | FPS Jitter | aim_sim_fps_jitter | 0.000-0.800 | 0.150 | Randomized FPS variance for the simulation. |
| Aim Simulation | Capture Delay (ms) | aim_sim_capture_delay_ms | 0.0-80.0 | 6.00 | Simulated capture delay. |
| Aim Simulation | Inference Delay (ms) | aim_sim_inference_delay_ms | 0.0-120.0 | 12.00 | Manual simulated inference delay when not using live timing. |
| Aim Simulation | Input Delay (ms) | aim_sim_input_delay_ms | 0.0-60.0 | 2.00 | Simulated input delay. |
| Aim Simulation | Extra Delay (ms) | aim_sim_extra_delay_ms | 0.0-60.0 | 2.00 | Additional simulated processing delay. |
| Aim Simulation | Target Max Speed | aim_sim_target_max_speed | 20-2500 | 560.00 | Maximum simulated target speed. |
| Aim Simulation | Target Accel | aim_sim_target_accel | 20-10000 | 1850.00 | Simulated target acceleration. |
| Aim Simulation | Target Stop Chance | aim_sim_target_stop_chance | 0.00-0.95 | 0.25 | Probability that the simulated target pauses during a retarget. |

### Activate/Deactivate

| Section | Control | Config key | Values | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| General | Enable | game_overlay_enabled | true/false | false | Enables the separate game overlay renderer. |
| General | Draw Detection Boxes | game_overlay_draw_boxes | true/false | true | Shows detection boxes in the game overlay. |
| General | Draw Future Positions | game_overlay_draw_future | true/false | true | Shows predicted future target points. |
| General | Draw Wind Debug Tail | game_overlay_draw_wind_tail | true/false | true | Shows the wind-mouse debug trail when wind motion is active. |
| General | Show Target Correction | game_overlay_show_target_correction | true/false | true | Shows correction indicators for target prediction/aim adjustment. |
| Capture Frame | Draw Capture Frame | game_overlay_draw_frame | true/false | true | Draws the capture frame rectangle. |
| Icon Overlay | Enable Icon Overlay | game_overlay_icon_enabled | true/false | false | Enables drawing an icon relative to detections. |
| Aim Simulation | Enable Aim Simulation Window | aim_sim_enabled | true/false | false | Shows the aim simulation overlay window. |
| Aim Simulation | Use Live Inference Delay | aim_sim_use_live_inference | true/false | true | Uses live backend timing for inference delay instead of the manual value. |
| Aim Simulation | Show Delayed Observation | aim_sim_show_observed | true/false | true | Shows delayed target observation in the simulation. |
| Aim Simulation | Show Trajectory History | aim_sim_show_history | true/false | true | Shows historical target/aim trails. |
| Aim Simulation | Show Kalman Debug | aim_sim_show_kalman_debug | true/false | true | Shows Kalman-related debug visuals in the simulation. |

### Other GUI-Exposed Config Controls

| Section | Control | Config key | Type/options | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Icon Overlay | Icon Path | game_overlay_icon_path | path | icon.png | Image path for icon overlay. |
| Icon Overlay | Icon Class (-1 = all) | game_overlay_icon_class | -1 or class id | -1 | Restricts icon to one class when >= 0. |
| Icon Overlay | Icon Anchor | game_overlay_icon_anchor | center, top, bottom, head | center | Anchor point used for icon placement. |

## Stats Tab

The Stats tab is read-only in the current GUI. It displays timing graphs, capture FPS, capture details, and CUDA/depth status where available.

## Debug Tab

### Sliders

No saved GUI sliders in this tab.

### Activate/Deactivate

| Section | Control | Config key | Values | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Screenshot Buttons | Verbose console output | verbose | true/false | false | Enables extra console logging. |
| Log File | Enable log file | debug_log_file_enabled | true/false | false | Writes logs to debug_log_file_path when enabled. |
| Neural Diagnostics | Log neural tracker associations | neural_tracker_log_enabled | true/false | false | Writes neural association diagnostics to neural_tracker_log_path. |
| Neural Diagnostics | Show neural tracker debug | neural_tracker_debug_enabled | true/false | false | Shows neural tracker debug output in the game overlay path. |

### Other GUI-Exposed Config Controls

| Section | Control | Config key | Type/options | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Screenshot Buttons | Screenshot buttons | screenshot_button | key list | None | Supports multiple bindings with + and -. |
| Screenshot Buttons | Screenshot delay | screenshot_delay | step 50/500 | 500 | Delay before screenshot capture. |
| Log File | Log file path | debug_log_file_path | path | logs/0BS.log | Used when Enable log file is on. |
| Neural Diagnostics | Neural tracker log | neural_tracker_log_path | path | training/logs/neural_tracker_association.csv | CSV path for neural association diagnostics. |

## Config.ini Settings Not In The GUI

These settings are editable by changing `config.ini` directly. Some are build-gated: CUDA rows are emitted only by CUDA builds, and depth rows are GUI-editable only when the app is compiled with `USE_CUDA`.

| Section | Setting | Current/default | Range/options | Visibility | Guidance | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Mouse prediction | prediction_futurePositions | 20 | integer | Config only | Number of future target positions to retain/use for prediction visualization. | Saved in current config.ini, but no direct GUI control exists. |
| Mouse prediction | draw_futurePositions | true | true/false | Config only | Enables drawing future positions in preview/debug paths. | Separate from game_overlay_draw_future. |
| PID governor | pid_governor_model_path | training/models/pid_governor.onnx | path | Config only | Model path for the ONNX PID governor. | Relative paths are resolved against current/exe parent locations. |
| AI backend | backend | DML | DML or TRT | DML config-only / CUDA GUI | Selects DirectML or TensorRT backend. | GUI combo exists only in CUDA builds. |
| AI backend | dml_device_id | 0 | integer adapter id | Config only | DirectML adapter index used by ONNX Runtime DML. | Not exposed in the GUI. |
| System reserves | cpuCoreReserveCount | 4 | integer | Config only | CPU cores reserved away from worker assignment. | Not exposed in the GUI. |
| System reserves | systemMemoryReserveMB | 2048 | MB | Config only | System memory reserve used by runtime resource planning. | Not exposed in the GUI. |
| Custom classes | class_player | 0 | integer class id | Config only | Detector class id treated as player/body. | Not exposed in the GUI. |
| Custom classes | class_head | 1 | integer class id | Config only | Detector class id treated as head. | Not exposed in the GUI. |
| Debug | show_fps | false | true/false | Config only | Legacy/debug FPS display flag. | Saved in config.ini but not exposed in the current GUI. |
| Game profiles | Games.UNIFIED | 1.00,0.02,0.02 | sens,yaw,pitch[,true,baseFOV] | Config only | Default profile row used for degree-to-count conversion. | UNIFIED is shown read-only in the GUI. |
| Depth | depth_inference_enabled | true | true/false | DML config-only / CUDA GUI | Enables depth inference runtime. | In the DML build this remains config-only; CUDA builds expose it in AI -> Depth. |
| Depth | depth_model_path | depth_anything_v2.engine | path | DML config-only / CUDA GUI | Depth model or engine path. | In the DML build this remains config-only; CUDA builds expose it in AI -> Depth. |
| Depth | depth_fps | 100 | 0-120 | DML config-only / CUDA GUI | Depth debug inference rate. | In the DML build this remains config-only; CUDA builds expose it in AI -> Depth. |
| Depth | depth_colormap | 18 | 0-21 | DML config-only / CUDA GUI | OpenCV colormap index for depth debug view. | In the DML build this remains config-only; CUDA builds expose it in AI -> Depth. |
| Depth | depth_mask_enabled | false | true/false | DML config-only / CUDA GUI | Enables depth mask generation. | In the DML build this remains config-only; CUDA builds expose it in AI -> Depth. |
| Depth | depth_mask_fps | 5 | 1-30 GUI, >=0 load clamp | DML config-only / CUDA GUI | Depth mask update frequency. | In the DML build this remains config-only; CUDA builds expose it in AI -> Depth. |
| Depth | depth_mask_near_percent | 20 | 1-100 | DML config-only / CUDA GUI | Near-depth percentile threshold for mask generation. | In the DML build this remains config-only; CUDA builds expose it in AI -> Depth. |
| Depth | depth_mask_expand | 0 | 0-128 px | DML config-only / CUDA GUI | Expands the mask after generation. | In the DML build this remains config-only; CUDA builds expose it in AI -> Depth. |
| Depth | depth_mask_hold_frames | 0 | 0-120 | DML config-only / CUDA GUI | Frames to hold/reuse the last mask. | In the DML build this remains config-only; CUDA builds expose it in AI -> Depth. |
| Depth | depth_mask_alpha | 90 | 0-255 | DML config-only / CUDA GUI | Depth mask overlay alpha. | In the DML build this remains config-only; CUDA builds expose it in AI -> Depth. |
| Depth | depth_mask_invert | false | true/false | DML config-only / CUDA GUI | Inverts near/far mask selection. | In the DML build this remains config-only; CUDA builds expose it in AI -> Depth. |
| Depth | depth_debug_overlay_enabled | false | true/false | DML config-only / CUDA GUI | Shows depth debug overlay in the game overlay. | In the DML build this remains config-only; CUDA builds expose it in AI -> Depth. |
| Advanced mouse | minSpeedMultiplier | 0.1 | float | Loadable hidden key | Legacy minimum movement speed multiplier. | Loadable from config.ini if added manually; saveConfig does not currently emit it. |
| Advanced mouse | maxSpeedMultiplier | 0.1 | float | Loadable hidden key | Legacy maximum movement speed multiplier. | Loadable from config.ini if added manually; saveConfig does not currently emit it. |
| Advanced mouse | predictionInterval | 0.01 | seconds | Loadable hidden key | Prediction sample interval used by legacy/auxiliary prediction logic. | Loadable from config.ini if added manually; saveConfig does not currently emit it. |
| Legacy snap | snapRadius | 1.5 | float | Config only | Legacy snap radius. | Loadable hidden key; no GUI control. |
| Legacy snap | nearRadius | 25.0 | float | Config only | Legacy near-target radius. | Loadable hidden key; no GUI control. |
| Legacy snap | speedCurveExponent | 3.0 | float | Config only | Legacy speed curve exponent. | Loadable hidden key; no GUI control. |
| Legacy snap | snapBoostFactor | 1.15 | float | Config only | Legacy snap boost multiplier. | Loadable hidden key; no GUI control. |
| Legacy recoil | easynorecoil | false | true/false | Config only | Legacy recoil helper flag. | Loadable hidden key; no GUI control and not emitted by saveConfig. |
| Legacy recoil | easynorecoilstrength | 0.0 | float | Config only | Legacy recoil helper strength. | Loadable hidden key; no GUI control and not emitted by saveConfig. |
| Wind mouse | wind_mouse_enabled | false | true/false | Config only | Enables wind-style motion noise in supported runtime/simulation paths. | Loadable hidden key; no GUI control. |
| Wind mouse | wind_G | 18.0 | float | Config only | Wind mouse gravity parameter. | Loadable hidden key; no GUI control. |
| Wind mouse | wind_W | 15.0 | float | Config only | Wind mouse wind parameter. | Loadable hidden key; no GUI control. |
| Wind mouse | wind_M | 10.0 | float | Config only | Wind mouse max-step parameter. | Loadable hidden key; no GUI control. |
| Wind mouse | wind_D | 8.0 | float | Config only | Wind mouse damping/distance parameter. | Loadable hidden key; no GUI control. |
| CUDA | export_enable_fp8 | false | true/false | CUDA config-only | Enables FP8 TensorRT export where supported. | CUDA builds only; no GUI checkbox. |
| CUDA | export_enable_fp16 | true | true/false | CUDA config-only | Enables FP16 TensorRT export where supported. | CUDA builds only; no GUI checkbox. |
| CUDA | use_cuda_graph | false | true/false | CUDA config-only | Enables CUDA graph execution path. | CUDA builds only; no GUI checkbox. |
| CUDA | use_pinned_memory | true | true/false | CUDA config-only | Uses pinned host memory for CUDA paths. | CUDA builds only; no GUI checkbox. |
| CUDA | gpuMemoryReserveMB | 2048 | MB | CUDA config-only | GPU memory reserve for CUDA resource planning. | CUDA builds only; no GUI control. |
| CUDA | enableGpuExclusiveMode | true | true/false | CUDA config-only | Attempts GPU exclusive mode at startup. | CUDA builds only; no GUI checkbox. |

## Source Map

| Area | Source |
| --- | --- |
| Tab order | overlay/overlay.cpp |
| Capture | overlay/draw_capture.cpp and overlay/draw_debug.cpp |
| Target | overlay/draw_target.cpp |
| Mouse | overlay/draw_mouse.cpp |
| Neural | overlay/draw_neural.cpp |
| AI and Depth | overlay/draw_ai.cpp and overlay/draw_depth.cpp |
| Buttons | overlay/draw_buttons.cpp |
| Overlay | overlay/draw_overlay.cpp |
| Game Overlay | overlay/draw_game_overlay.cpp |
| Debug | overlay/draw_debug.cpp |
| Config load/save/defaults | config/config.cpp and config/config.h |
