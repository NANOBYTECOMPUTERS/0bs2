# 0BS GUI and Config Setting Reference

Generated on 2026-07-23 from `dist\0BS\config.ini` and the ImGui source files under `overlay/`.

This reference is ordered by the GUI sidebar tabs in `overlay/overlay.cpp`. It lists every GUI slider and every activate/deactivate checkbox, then documents config.ini settings that are not editable in the current DML GUI or are hidden/loadable config keys.

## Searchable Directory

- PDF: `docs/settings-reference/0BS-settings-reference.pdf`
- Search index: `docs/settings-reference/settings-index.csv`
- Directory README: `docs/settings-reference/README.md`
- Program flow chart: `docs/program-flow.mmd`

Search by GUI label, config key, tab, section, or source file. Rows marked `CUDA GUI / DML config-only` appear as GUI controls only in CUDA builds; the current `x64/DML/config.ini` keeps those values as config-only settings.

## Repository Reference

Private backup/update repository: `NANOBYTECOMPUTERS/0bs2`.

This reference covers the OpenCV 5 CUDA / TensorRT 11.1 project state after neural, depth, PID, feed-forward, and smart-blending code paths were removed. Local dependency drops under `modules/` and `extras/` are intentionally excluded from Git and should be restored through the documented build/dependency scripts.

## Targeting Pipeline

The runtime path is now fully deterministic:

`Video Frame -> Detector -> Tracker -> State Estimator (Kalman/IMM) -> Direct Targeting Movement -> Mouse Output`

The tracker aim point and Kalman prediction own convergence. Selecting IMM changes only the InnerAim tracker estimator; detections and tracker observations feed the targeting pipeline directly.

Ego-motion compensation is an opt-in tracker stabilizer. It subtracts bounded, recent emitted mouse/view motion from tracker priors only; raw detector boxes and final mouse output remain unchanged.

## Legacy Prediction Compatibility

Deprecated but retained: `minSpeedMultiplier`, `maxSpeedMultiplier`, `predictionInterval`, `snapRadius`, `nearRadius`, `speedCurveExponent`, `snapBoostFactor`, `kalman_warmup_frames`, and `kalman_acquisition_frames` remain loadable for old configs and aim-simulation parity.

Removal gate: remove these only after a behavior baseline proves behavior stays identical across the live tracker, game overlay simulation, and config merge path.

## GUI Tab Order

| Order | Tab | Notes |
| --- | --- | --- |
| 1 | Capture | Adjustable controls documented below |
| 2 | Target | Adjustable controls documented below |
| 3 | Mouse | Adjustable controls documented below |
| 4 | AI | Adjustable controls documented below |
| 5 | Buttons | Adjustable controls documented below |
| 6 | Overlay | Adjustable controls documented below |
| 7 | Game Overlay | Adjustable controls documented below |
| 8 | Stats | Read-only monitor tab |
| 9 | Debug | Adjustable controls documented below |

## Capture Tab

### Sliders

| Section | Slider | Config key | Range | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| General Capture | Capture FPS | capture_fps | 0-240 | n/a | Limits capture loop FPS. 0 disables the cap path and the GUI warns at 0 or 61+. |
| Capture Preview | Debug scale | preview_debug_scale | 0.1x-2.0x | n/a | Preview-only scale slider. It is not saved to config.ini. |
| Virtual Camera | Virtual camera width | virtual_camera_width | 128-3840 | n/a | Applies when capture_method is virtual_camera. |
| Virtual Camera | Virtual camera heigth | virtual_camera_heigth | 128-2160 | n/a | Spelling follows the existing config key. Applies when capture_method is virtual_camera. |

### Activate/Deactivate

| Section | Control | Config key | Values | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| General Capture | Circle mask | circle_mask | true/false | n/a | Applies the circular capture mask. |
| General Capture | Use CUDA Direct Capture | capture_use_cuda | true/false | n/a | CUDA/TensorRT builds only. Available with duplication_api capture. |
| Capture Preview | Show Preview Window | show_window | true/false | n/a | Shows the capture preview/debug frame inside the Capture tab. |
| WinRT | Capture Borders | capture_borders | true/false | n/a | WinRT capture only. Disabled automatically on unsupported Windows builds. |
| WinRT | Capture Cursor | capture_cursor | true/false | n/a | WinRT capture only. Includes the cursor in captured frames. |

### Other GUI-Exposed Config Controls

| Section | Control | Config key | Type/options | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| General Capture | Detection Resolution | detection_resolution | 160, 320, 640 | n/a | Changes detector input resolution and restarts/reloads dependent paths. |
| General Capture | Capture method | capture_method | duplication_api, winrt, virtual_camera, udp_capture | n/a | Selects frame source. |
| WinRT | Capture target (WinRT) | capture_target | monitor, window | n/a | WinRT only. |
| WinRT | Window title contains | capture_window_title | text | n/a | Used when WinRT target is window. |
| Monitor Capture | Capture monitor | monitor_idx | monitor index | n/a | Monitor list is built from active displays. |
| Virtual Camera | Virtual camera | virtual_camera_name | available cameras | n/a | Filtered list of DirectShow video input devices. |
| UDP Capture | UDP IP | udp_ip | IPv4/string | n/a | Applied with the Apply UDP Settings button. |
| UDP Capture | UDP Port | udp_port | 1-65535 | n/a | Applied with the Apply UDP Settings button. |

## Target Tab

### Sliders

| Section | Slider | Config key | Range | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Offsets | Approximate Body Y Offset | body_y_offset | 0.05-0.90 | n/a | Vertical aim point used for body-class detections. |
| Offsets | Approximate Head Y Offset | head_y_offset | 0.05-0.55 | n/a | Vertical aim point used for head-class detections. |

### Activate/Deactivate

| Section | Control | Config key | Values | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Targeting | Disable Headshot | disable_headshot | true/false | n/a | Prevents the head class from being selected as the aim target. |
| Targeting | Auto Aim | auto_aim | true/false | n/a | Allows target selection/movement logic to act automatically while held conditions are met. |

## Mouse Tab

### Sliders

| Section | Slider | Config key | Range | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| State Estimator | Acquisition frames | kalman_acquisition_frames | 3-5 | n/a | Frames used to ramp prediction weight after target acquisition. |
| State Estimator | Process noise position | kalman_process_noise_position | 0.0001-5000 | n/a | Position process noise. Higher values adapt faster to position changes. |
| State Estimator | Process noise velocity | kalman_process_noise_velocity | 0.0001-50000 | n/a | Velocity process noise. Higher values adapt faster to speed changes. |
| State Estimator | Measurement noise | kalman_measurement_noise | 0.0001-5000 | n/a | Measurement noise. Higher values trust detections less. |
| State Estimator | Velocity damping | kalman_velocity_damping | 0.0-3.0 | n/a | Damps velocity estimates to reduce runaway prediction. |
| State Estimator | Max velocity (px/s) | kalman_max_velocity | 100-60000 | n/a | Maximum allowed tracked velocity. |
| State Estimator | Warmup frames | kalman_warmup_frames | 0-20 | n/a | Legacy warmup frames when latency sweep is off. |
| State Estimator | Additional prediction (ms) | kalman_additional_prediction_ms | -80.0-120.0 | n/a | Manual extra prediction offset in milliseconds. |
| State Estimator | Reset timeout (s) | kalman_reset_timeout_sec | 0.05-3.0 | n/a | Time without stable observations before estimator state resets. |
| State Estimator | Ego compensation strength | ego_motion_compensation_strength | 0.0-1.0 | n/a | Fraction of recent emitted motion subtracted from tracker priors/history to reduce camera-motion jitter. |
| State Estimator | Ego max shift (px @640) | ego_motion_compensation_max_shift_px | 1-128 | n/a | Per-frame compensation clamp at 640 detection resolution; scales with detection_resolution. |
| State Estimator | Ego max age (ms) | ego_motion_compensation_max_age_ms | 16-500 | n/a | Drops stale emitted-motion samples so compensation fails closed. |
| Direct Targeting Movement | Deadzone (px) | target_deadzone_px | 0.000-20.000 | n/a | Minimum target error radius before stream movement is emitted. |
| Direct Targeting Movement | Stream interval (ms) | target_stream_interval_ms | 0.50-8.00 | n/a | Paced target-stream wake interval. New tracker observations wake the stream immediately. |
| Direct Targeting Movement | Stream sharpness | target_stream_sharpness | 1.00-80.00 | n/a | Time-based convergence factor for the target stream. |
| Direct Targeting Movement | Max speed (px/s) | target_max_pixel_speed | 50-20000 | n/a | Maximum pixel-space stream speed before count conversion. |
| Direct Targeting Movement | State max age (ms) | target_state_max_age_ms | 16-500 | n/a | Drops stale target-motion snapshots before streaming output. |
| Direct Targeting Movement | Min stream confidence | target_min_stream_confidence | 0.00-0.95 | n/a | Minimum confidence required for the target stream to emit movement. |
| Direct Targeting Movement | Prediction blend | target_prediction_blend | 0.000-0.650 | n/a | Fraction of bounded short-horizon predicted aim mixed into the observed tracker aim point. |
| Direct Targeting Movement | Max prediction lead (px) | target_prediction_max_lead_px | 0.00-40.00 | n/a | Maximum pixel distance the mouse path may lead the current tracker aim point. |
| Direct Targeting Movement | Governor strength | target_convergence_governor_strength | 0.000-1.000 | n/a | How strongly the convergence governor can damp or release stream gain from live overshoot, jitter, lock, and ambiguity signals. |
| Direct Targeting Movement | Governor min gain | target_convergence_governor_min_gain | 0.050-1.000 | n/a | Lower clamp for governor-scaled stream sharpness and max-step output. |
| Direct Targeting Movement | Governor max gain | target_convergence_governor_max_gain | 1.000-2.000 | n/a | Upper clamp for governor-scaled stream sharpness and max-step output. |
| Direct Targeting Movement | Fallback max step (px/call) | target_max_pixel_step | 0.25-120.00 | n/a | Legacy per-call clamp for non-stream direct movement helpers and aim-simulation parity. |
| Direct Targeting Movement | Fallback output scale | target_output_scale | 0.010-3.000 | n/a | Legacy per-call scale for non-stream direct movement helpers and aim-simulation parity. |
| Direct Targeting Movement | Signal window samples | target_signal_window_samples | 64-2048 | n/a | Rolling sample window used for live diagnostic averages and frequency/lag estimates. |
| Direct Targeting Movement | Signal log interval (ms) | target_signal_log_interval_ms | 1.0-1000.0 | n/a | Minimum interval between CSV diagnostic rows. |
| Direct Targeting Movement | Counts / px X | target_counts_per_pixel_x | 0.0000-50.0000 | n/a | Horizontal calibrated mouse counts per screen-space pixel magnitude. |
| Direct Targeting Movement | Counts / px Y | target_counts_per_pixel_y | 0.0000-50.0000 | n/a | Vertical calibrated mouse counts per screen-space pixel magnitude. |
| Auto Shoot | bScope Multiplier | bScope_multiplier | 0.5-2.0 | n/a | Multiplier used by auto-shoot scope timing/behavior. Disabled in UI until Auto Shoot is on. |

### Activate/Deactivate

| Section | Control | Config key | Values | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| State Estimator | Runtime latency sweep | runtime_latency_sweep_enabled | true/false | n/a | Enables side-by-side latency sweep behavior for estimator comparisons. |
| State Estimator | Enable Kalman estimator | kalman_enabled | true/false | n/a | Enables Kalman filtering for target motion estimation. |
| State Estimator | Seed velocity on acquire | kalman_velocity_seed_enabled | true/false | n/a | Seeds velocity from early measurement deltas during acquisition. |
| State Estimator | Compensate detection delay | kalman_compensate_detection_delay | true/false | n/a | Accounts for detector latency in prediction. |
| State Estimator | Ego-motion compensation | ego_motion_compensation_enabled | true/false | n/a | Opt-in tracker-prior compensation from emitted mouse/view motion. Raw detections and final mouse output are unchanged. |
| Direct Targeting Movement | Target stream | target_stream_enabled | true/false | n/a | Enables the paced target-motion stream used by the live targeting path. |
| Direct Targeting Movement | Convergence governor | target_convergence_governor_enabled | true/false | n/a | Enables the optional convergence governor that scales stream gain from live target-lock quality and movement feedback. |
| Direct Targeting Movement | Target stream debug | target_stream_debug_enabled | true/false | n/a | Shows live target-stream status, prediction, pixel delta, count output, and block reason in the Mouse tab. |
| Direct Targeting Movement | Target signal diagnostics | target_signal_diagnostics_enabled | true/false | n/a | Shows passive tuning diagnostics for stream cadence, error, output counts, trajectory quality, frequency content, lag, and stability. |
| Direct Targeting Movement | Signal CSV logging | target_signal_logging_enabled | true/false | n/a | Writes target signal samples to CSV for offline or automatic tuning analysis. |
| Direct Targeting Movement | Calibrated pixel counts | target_calibrated_pixel_counts_enabled | true/false | n/a | Uses measured counts-per-pixel gains for direct targeting output instead of the default 1:1 pixel-count fallback. |
| Auto Shoot | Auto Shoot | auto_shoot | true/false | n/a | Enables automatic shooting behavior. |
| Input Method | Arduino 16-bit Mouse | arduino_16_bit_mouse | true/false | n/a | Arduino input only. Sends wider mouse movement values. |
| Input Method | Arduino Enable Keys | arduino_enable_keys | true/false | n/a | Arduino input only. Enables keyboard key output through Arduino. |

### Other GUI-Exposed Config Controls

| Section | Control | Config key | Type/options | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Input Method | Mouse Input Method | input_method | WIN32, GHUB, RAZER, DIRECT, ARDUINO, TEENSY41, TEENSY41_HID, KMBOX_NET, KMBOX_A, MAKCU | n/a | Changes active mouse backend. |
| Input Method | Arduino/Teensy Port | arduino_port | COM1-COM30 | n/a | Arduino and Teensy 4.1 serial input. |
| Input Method | Arduino/Teensy Baudrate | arduino_baudrate | 9600-115200 | n/a | Arduino and Teensy 4.1 serial input. |
| Input Method | Kmbox Net IP | kmbox_net_ip | text | n/a | Saved with Save & Reconnect. |
| Input Method | Kmbox Net Port | kmbox_net_port | text | n/a | Saved with Save & Reconnect. |
| Input Method | Kmbox Net UUID | kmbox_net_uuid | text | n/a | Saved with Save & Reconnect. |
| Input Method | Kmbox A PIDVID | kmbox_a_pidvid | PPPPVVVV | n/a | Saved with Save & Reconnect. |
| Input Method | Makcu Port | makcu_port | AUTO, COM1-COM30 | n/a | Makcu input only. |
| Input Method | Makcu Baudrate | makcu_baudrate | 115200-4000000 | n/a | Makcu input only. |

## AI Tab

### Sliders

| Section | Slider | Config key | Range | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Detection | Confidence Threshold | confidence_threshold | 0.01-1.00 | n/a | Minimum detector confidence accepted as a detection. |
| Detection | NMS Threshold | nms_threshold | 0.00-1.00 | n/a | Non-max suppression overlap threshold. |
| Detection | Max Detections | max_detections | 1-100 | n/a | Maximum detections retained after model output processing. |

### Activate/Deactivate

No activate/deactivate checkboxes in this tab.

### Other GUI-Exposed Config Controls

| Section | Control | Config key | Type/options | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Model | Model | ai_model | models folder | n/a | Selects detector model from available files. |
| Backend | Backend | backend | TRT, DML | n/a | CUDA builds only; DML build does not expose this combo. |

## Buttons Tab

### Sliders

No saved GUI sliders in this tab.

### Activate/Deactivate

| Section | Control | Config key | Values | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Arrow Key Options | Enable arrows keys options | enable_arrows_settings | true/false | n/a | Allows arrow-key adjustment of target offsets. |

### Other GUI-Exposed Config Controls

| Section | Control | Config key | Type/options | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Button Bindings | Targeting Buttons | button_targeting | key list | n/a | Supports multiple bindings with + and -. |
| Button Bindings | Shoot Buttons | button_shoot | key list | n/a | Supports multiple bindings with + and -. |
| Button Bindings | Zoom Buttons | button_zoom | key list | n/a | Supports multiple bindings with + and -. |
| Button Bindings | Exit Buttons | button_exit | key list | n/a | Supports multiple bindings with + and -. |
| Button Bindings | Pause Buttons | button_pause | key list | n/a | Supports multiple bindings with + and -. |
| Button Bindings | Reload Config Buttons | button_reload_config | key list | n/a | Supports multiple bindings with + and -. |
| Button Bindings | Overlay Buttons | button_open_overlay | key list | n/a | Supports multiple bindings with + and -. |

## Overlay Tab

### Sliders

| Section | Slider | Config key | Range | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Visual | Overlay Opacity | overlay_opacity | 220-255 | n/a | Window opacity. The UI clamps to a readable minimum of 220. |
| Visual | UI Fine Scale | overlay_ui_scale | 0.85-1.35 | n/a | Fine-grained scaling for the overlay UI. |

### Activate/Deactivate

| Section | Control | Config key | Values | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Capture Privacy | Hide Overlays From Recording | overlay_exclude_from_capture | true/false | n/a | Applies capture exclusion so overlay windows are hidden from supported recording/capture paths. |

### Other GUI-Exposed Config Controls

| Section | Control | Config key | Type/options | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Config | Save Config | config.ini | writes config.ini | n/a | Immediately writes the current runtime config. |
| Config | Load Config | config.ini | reads config.ini | n/a | Merges config.ini into current runtime values and refreshes runtime-sensitive paths. |

## Game Overlay Tab

### Sliders

| Section | Slider | Config key | Range | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| General | Overlay Max FPS (0 = uncapped) | game_overlay_max_fps | 0-256 | n/a | Caps game overlay render rate. 0 is uncapped. |
| Box Color | A | game_overlay_box_a | 0-255 | n/a | Detection box alpha. |
| Box Color | R | game_overlay_box_r | 0-255 | n/a | Detection box red channel. |
| Box Color | G | game_overlay_box_g | 0-255 | n/a | Detection box green channel. |
| Box Color | B | game_overlay_box_b | 0-255 | n/a | Detection box blue channel. |
| Box Color | Box Thickness | game_overlay_box_thickness | 0.5-10.0 | n/a | Detection box line thickness. |
| Capture Frame | A | game_overlay_frame_a | 0-255 | n/a | Capture frame alpha. |
| Capture Frame | R | game_overlay_frame_r | 0-255 | n/a | Capture frame red channel. |
| Capture Frame | G | game_overlay_frame_g | 0-255 | n/a | Capture frame green channel. |
| Capture Frame | B | game_overlay_frame_b | 0-255 | n/a | Capture frame blue channel. |
| Capture Frame | Frame Thickness | game_overlay_frame_thickness | 0.5-10.0 | n/a | Capture frame line thickness. |
| Future Point Style | Point Radius | game_overlay_future_point_radius | 1.0-20.0 | n/a | Radius of future-position points. |
| Future Point Style | Point Step Alpha Falloff | game_overlay_future_alpha_falloff | 0.10-5.00 | n/a | Controls how quickly future-point alpha fades along the path. |
| Icon Overlay | Icon Width | game_overlay_icon_width | 4-512 | n/a | Icon width in pixels. |
| Icon Overlay | Icon Height | game_overlay_icon_height | 4-512 | n/a | Icon height in pixels. |
| Icon Overlay | Icon Offset X | game_overlay_icon_offset_x | -500.0-500.0 | n/a | Horizontal icon offset from the selected anchor. |
| Icon Overlay | Icon Offset Y | game_overlay_icon_offset_y | -500.0-500.0 | n/a | Vertical icon offset from the selected anchor. |
| Aim Simulation | Sim X | aim_sim_x | -3000-3000 | n/a | Aim simulation window X position. |
| Aim Simulation | Sim Y | aim_sim_y | -3000-3000 | n/a | Aim simulation window Y position. |
| Aim Simulation | Sim Width | aim_sim_width | 220-1600 | n/a | Aim simulation window width. |
| Aim Simulation | Sim Height | aim_sim_height | 180-1000 | n/a | Aim simulation window height. |
| Aim Simulation | Sim FPS Min | aim_sim_fps_min | 15-360 | n/a | Minimum simulated frame rate. |
| Aim Simulation | Sim FPS Max | aim_sim_fps_max | 15-360 | n/a | Maximum simulated frame rate. The GUI keeps min and max ordered. |
| Aim Simulation | FPS Jitter | aim_sim_fps_jitter | 0.000-0.800 | n/a | Randomized FPS variance for the simulation. |
| Aim Simulation | Capture Delay (ms) | aim_sim_capture_delay_ms | 0.0-80.0 | n/a | Simulated capture delay. |
| Aim Simulation | Inference Delay (ms) | aim_sim_inference_delay_ms | 0.0-120.0 | n/a | Manual simulated inference delay when not using live timing. |
| Aim Simulation | Input Delay (ms) | aim_sim_input_delay_ms | 0.0-60.0 | n/a | Simulated input delay. |
| Aim Simulation | Extra Delay (ms) | aim_sim_extra_delay_ms | 0.0-60.0 | n/a | Additional simulated processing delay. |
| Aim Simulation | Target Max Speed | aim_sim_target_max_speed | 20-2500 | n/a | Maximum simulated target speed. |
| Aim Simulation | Target Accel | aim_sim_target_accel | 20-10000 | n/a | Simulated target acceleration. |
| Aim Simulation | Target Stop Chance | aim_sim_target_stop_chance | 0.00-0.95 | n/a | Probability that the simulated target pauses during a retarget. |

### Activate/Deactivate

| Section | Control | Config key | Values | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| General | Enable | game_overlay_enabled | true/false | n/a | Enables the separate game overlay renderer. |
| General | Draw Detection Boxes | game_overlay_draw_boxes | true/false | n/a | Shows detection boxes in the game overlay. |
| General | Draw Future Positions | game_overlay_draw_future | true/false | n/a | Shows predicted future target points. |
| General | Draw Wind Debug Tail | game_overlay_draw_wind_tail | true/false | n/a | Shows the wind-mouse debug trail when wind motion is active. |
| General | Show Target Correction | game_overlay_show_target_correction | true/false | n/a | Shows correction indicators for target prediction/aim adjustment. |
| Capture Frame | Draw Capture Frame | game_overlay_draw_frame | true/false | n/a | Draws the capture frame rectangle. |
| Icon Overlay | Enable Icon Overlay | game_overlay_icon_enabled | true/false | n/a | Enables drawing an icon relative to detections. |
| Aim Simulation | Enable Aim Simulation Window | aim_sim_enabled | true/false | n/a | Shows the aim simulation overlay window. |
| Aim Simulation | Use Live Inference Delay | aim_sim_use_live_inference | true/false | n/a | Uses live backend timing for inference delay instead of the manual value. |
| Aim Simulation | Show Delayed Observation | aim_sim_show_observed | true/false | n/a | Shows delayed target observation in the simulation. |
| Aim Simulation | Show Trajectory History | aim_sim_show_history | true/false | n/a | Shows historical target/aim trails. |
| Aim Simulation | Show Kalman Debug | aim_sim_show_kalman_debug | true/false | n/a | Shows Kalman-related debug visuals in the simulation. |

### Other GUI-Exposed Config Controls

| Section | Control | Config key | Type/options | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Icon Overlay | Icon Path | game_overlay_icon_path | path | n/a | Image path for icon overlay. |
| Icon Overlay | Icon Class (-1 = all) | game_overlay_icon_class | -1 or class id | n/a | Restricts icon to one class when >= 0. |
| Icon Overlay | Icon Anchor | game_overlay_icon_anchor | center, top, bottom, head | n/a | Anchor point used for icon placement. |

## Stats Tab

The Stats tab is read-only in the current GUI. It displays timing graphs, capture FPS, capture details, and CUDA status where available.

## Debug Tab

### Sliders

No saved GUI sliders in this tab.

### Activate/Deactivate

| Section | Control | Config key | Values | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Screenshot Buttons | Verbose console output | verbose | true/false | n/a | Enables extra console logging. |
| Log File | Enable log file | debug_log_file_enabled | true/false | n/a | Writes logs to debug_log_file_path when enabled. |

### Other GUI-Exposed Config Controls

| Section | Control | Config key | Type/options | Current | Notes |
| --- | --- | --- | --- | --- | --- |
| Screenshot Buttons | Screenshot buttons | screenshot_button | key list | n/a | Supports multiple bindings with + and -. |
| Screenshot Buttons | Screenshot delay | screenshot_delay | step 50/500 | n/a | Delay before screenshot capture. |
| Log File | Log file path | debug_log_file_path | path | n/a | Used when Enable log file is on. |

## Config.ini Settings Not In The GUI

These settings are editable by changing `config.ini` directly. Some are build-gated: CUDA rows are emitted only by CUDA builds.

| Section | Setting | Current/default | Range/options | Visibility | Guidance | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Mouse prediction | prediction_futurePositions | 20 | integer | Config only | Number of future target positions to retain/use for prediction visualization. | Saved in current config.ini, but no direct GUI control exists. |
| Mouse prediction | draw_futurePositions | true | true/false | Config only | Enables drawing future positions in preview/debug paths. | Separate from game_overlay_draw_future. |
| AI backend | backend | DML | DML or TRT | DML config-only / CUDA GUI | Selects DirectML or TensorRT backend. | GUI combo exists only in CUDA builds. |
| AI backend | dml_device_id | 0 | integer adapter id | Config only | DirectML adapter index used by ONNX Runtime DML. | Not exposed in the GUI. |
| System reserves | cpuCoreReserveCount | 4 | integer | Config only | CPU cores reserved away from worker assignment. | Not exposed in the GUI. |
| System reserves | systemMemoryReserveMB | 2048 | MB | Config only | System memory reserve used by runtime resource planning. | Not exposed in the GUI. |
| Custom classes | class_player | 0 | integer class id | Config only | Detector class id treated as player/body. | Not exposed in the GUI. |
| Custom classes | class_head | 1 | integer class id | Config only | Detector class id treated as head. | Not exposed in the GUI. |
| Debug | show_fps | false | true/false | Config only | Legacy/debug FPS display flag. | Saved in config.ini but not exposed in the current GUI. |
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
| Buttons | overlay/draw_buttons.cpp |
| Overlay | overlay/draw_overlay.cpp |
| Game Overlay | overlay/draw_game_overlay.cpp |
| Debug | overlay/draw_debug.cpp |
| Config load/save/defaults | config/config.cpp and config/config.h |
