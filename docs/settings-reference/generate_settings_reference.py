from __future__ import annotations

import csv
import datetime as dt
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs" / "settings-reference"
PRIMARY_CONFIG = ROOT / "x64" / "DML" / "config.ini"
FALLBACK_CONFIG = ROOT / "dist" / "0BS" / "config.ini"


def parse_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    section = ""
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if section == "Games":
            values[f"Games.{key}"] = value
        else:
            values[key] = value
    return values


CONFIG_PATH = PRIMARY_CONFIG if PRIMARY_CONFIG.exists() else FALLBACK_CONFIG
CONFIG_VALUES = parse_config(CONFIG_PATH)


def cfg(key: str, default: str = "n/a") -> str:
    return CONFIG_VALUES.get(key, default)


def row(
    tab: str,
    section: str,
    label: str,
    key: str,
    kind: str,
    value_range: str,
    note: str,
    source: str,
    visibility: str = "GUI",
) -> dict[str, str]:
    return {
        "tab": tab,
        "section": section,
        "label": label,
        "key": key,
        "kind": kind,
        "range": value_range,
        "note": note,
        "source": source,
        "visibility": visibility,
        "current": cfg(key, "profile/local" if key.startswith("Games.") else "n/a"),
    }


GUI_CONTROLS: list[dict[str, str]] = [
    # Capture
    row("Capture", "General Capture", "Capture FPS", "capture_fps", "Slider", "0-240", "Limits capture loop FPS. 0 disables the cap path and the GUI warns at 0 or 61+.", "overlay/draw_capture.cpp"),
    row("Capture", "General Capture", "Circle mask", "circle_mask", "Activate/deactivate", "true/false", "Applies the circular capture mask.", "overlay/draw_capture.cpp"),
    row("Capture", "General Capture", "Use CUDA Direct Capture", "capture_use_cuda", "Activate/deactivate", "true/false", "CUDA/TensorRT builds only. Available with duplication_api capture.", "overlay/draw_capture.cpp", "CUDA GUI"),
    row("Capture", "Capture Preview", "Show Preview Window", "show_window", "Activate/deactivate", "true/false", "Shows the capture preview/debug frame inside the Capture tab.", "overlay/draw_debug.cpp"),
    row("Capture", "Capture Preview", "Debug scale", "preview_debug_scale", "Slider", "0.1x-2.0x", "Preview-only scale slider. It is not saved to config.ini.", "overlay/draw_debug.cpp"),
    row("Capture", "WinRT", "Capture Borders", "capture_borders", "Activate/deactivate", "true/false", "WinRT capture only. Disabled automatically on unsupported Windows builds.", "overlay/draw_capture.cpp"),
    row("Capture", "WinRT", "Capture Cursor", "capture_cursor", "Activate/deactivate", "true/false", "WinRT capture only. Includes the cursor in captured frames.", "overlay/draw_capture.cpp"),
    row("Capture", "Virtual Camera", "Virtual camera width", "virtual_camera_width", "Slider", "128-3840", "Applies when capture_method is virtual_camera.", "overlay/draw_capture.cpp"),
    row("Capture", "Virtual Camera", "Virtual camera heigth", "virtual_camera_heigth", "Slider", "128-2160", "Spelling follows the existing config key. Applies when capture_method is virtual_camera.", "overlay/draw_capture.cpp"),
    # Target
    row("Target", "Targeting", "Disable Headshot", "disable_headshot", "Activate/deactivate", "true/false", "Prevents the head class from being selected as the aim target.", "overlay/draw_target.cpp"),
    row("Target", "Targeting", "Auto Aim", "auto_aim", "Activate/deactivate", "true/false", "Allows target selection/movement logic to act automatically while held conditions are met.", "overlay/draw_target.cpp"),
    row("Target", "Offsets", "Approximate Body Y Offset", "body_y_offset", "Slider", "0.05-0.90", "Vertical aim point used for body-class detections.", "overlay/draw_target.cpp"),
    row("Target", "Offsets", "Approximate Head Y Offset", "head_y_offset", "Slider", "0.05-0.55", "Vertical aim point used for head-class detections.", "overlay/draw_target.cpp"),
    # Mouse
    row("Mouse", "State Estimator", "Runtime latency sweep", "runtime_latency_sweep_enabled", "Activate/deactivate", "true/false", "Enables side-by-side latency sweep behavior for estimator comparisons.", "overlay/draw_mouse.cpp"),
    row("Mouse", "State Estimator", "Enable Kalman estimator", "kalman_enabled", "Activate/deactivate", "true/false", "Enables Kalman filtering for target motion estimation.", "overlay/draw_mouse.cpp"),
    row("Mouse", "State Estimator", "Seed velocity on acquire", "kalman_velocity_seed_enabled", "Activate/deactivate", "true/false", "Seeds velocity from early measurement deltas during acquisition.", "overlay/draw_mouse.cpp"),
    row("Mouse", "State Estimator", "Acquisition frames", "kalman_acquisition_frames", "Slider", "3-5", "Frames used to ramp prediction weight after target acquisition.", "overlay/draw_mouse.cpp"),
    row("Mouse", "State Estimator", "Process noise position", "kalman_process_noise_position", "Slider", "0.0001-5000", "Position process noise. Higher values adapt faster to position changes.", "overlay/draw_mouse.cpp"),
    row("Mouse", "State Estimator", "Process noise velocity", "kalman_process_noise_velocity", "Slider", "0.0001-50000", "Velocity process noise. Higher values adapt faster to speed changes.", "overlay/draw_mouse.cpp"),
    row("Mouse", "State Estimator", "Measurement noise", "kalman_measurement_noise", "Slider", "0.0001-5000", "Measurement noise. Higher values trust detections less.", "overlay/draw_mouse.cpp"),
    row("Mouse", "State Estimator", "Velocity damping", "kalman_velocity_damping", "Slider", "0.0-3.0", "Damps velocity estimates to reduce runaway prediction.", "overlay/draw_mouse.cpp"),
    row("Mouse", "State Estimator", "Max velocity (px/s)", "kalman_max_velocity", "Slider", "100-60000", "Maximum allowed tracked velocity.", "overlay/draw_mouse.cpp"),
    row("Mouse", "State Estimator", "Warmup frames", "kalman_warmup_frames", "Slider", "0-20", "Legacy warmup frames when latency sweep is off.", "overlay/draw_mouse.cpp"),
    row("Mouse", "State Estimator", "Compensate detection delay", "kalman_compensate_detection_delay", "Activate/deactivate", "true/false", "Accounts for detector latency in prediction.", "overlay/draw_mouse.cpp"),
    row("Mouse", "State Estimator", "Additional prediction (ms)", "kalman_additional_prediction_ms", "Slider", "-80.0-120.0", "Manual extra prediction offset in milliseconds.", "overlay/draw_mouse.cpp"),
    row("Mouse", "State Estimator", "Reset timeout (s)", "kalman_reset_timeout_sec", "Slider", "0.05-3.0", "Time without stable observations before estimator state resets.", "overlay/draw_mouse.cpp"),
    row("Mouse", "State Estimator", "Ego-motion compensation", "ego_motion_compensation_enabled", "Activate/deactivate", "true/false", "Opt-in tracker-prior compensation from emitted mouse/view motion. Raw detections and final mouse output are unchanged.", "overlay/draw_mouse.cpp"),
    row("Mouse", "State Estimator", "Ego compensation strength", "ego_motion_compensation_strength", "Slider", "0.0-1.0", "Fraction of recent emitted motion subtracted from tracker priors/history to reduce camera-motion jitter.", "overlay/draw_mouse.cpp"),
    row("Mouse", "State Estimator", "Ego max shift (px @640)", "ego_motion_compensation_max_shift_px", "Slider", "1-128", "Per-frame compensation clamp at 640 detection resolution; scales with detection_resolution.", "overlay/draw_mouse.cpp"),
    row("Mouse", "State Estimator", "Ego max age (ms)", "ego_motion_compensation_max_age_ms", "Slider", "16-500", "Drops stale emitted-motion samples so compensation fails closed.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Direct Targeting Movement", "Deadzone (px)", "target_deadzone_px", "Slider", "0.000-20.000", "Minimum target error radius before stream movement is emitted.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Direct Targeting Movement", "Stream interval (ms)", "target_stream_interval_ms", "Slider", "0.50-8.00", "Paced target-stream wake interval. New tracker observations wake the stream immediately.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Direct Targeting Movement", "Stream sharpness", "target_stream_sharpness", "Slider", "1.00-80.00", "Time-based convergence factor for the target stream.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Direct Targeting Movement", "Max speed (px/s)", "target_max_pixel_speed", "Slider", "50-20000", "Maximum pixel-space stream speed before count conversion.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Direct Targeting Movement", "State max age (ms)", "target_state_max_age_ms", "Slider", "16-500", "Drops stale target-motion snapshots before streaming output.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Direct Targeting Movement", "Min stream confidence", "target_min_stream_confidence", "Slider", "0.00-0.95", "Minimum confidence required for the target stream to emit movement.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Direct Targeting Movement", "Prediction blend", "target_prediction_blend", "Slider", "0.000-0.650", "Fraction of bounded short-horizon predicted aim mixed into the observed tracker aim point.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Direct Targeting Movement", "Max prediction lead (px)", "target_prediction_max_lead_px", "Slider", "0.00-40.00", "Maximum pixel distance the mouse path may lead the current tracker aim point.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Direct Targeting Movement", "Fallback max step (px/call)", "target_max_pixel_step", "Slider", "0.25-120.00", "Legacy per-call clamp for non-stream direct movement helpers and aim-simulation parity.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Direct Targeting Movement", "Fallback output scale", "target_output_scale", "Slider", "0.010-3.000", "Legacy per-call scale for non-stream direct movement helpers and aim-simulation parity.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Direct Targeting Movement", "Target stream", "target_stream_enabled", "Activate/deactivate", "true/false", "Enables the paced target-motion stream used by the live targeting path.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Direct Targeting Movement", "Target stream debug", "target_stream_debug_enabled", "Activate/deactivate", "true/false", "Shows live target-stream status, prediction, pixel delta, count output, and block reason in the Mouse tab.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Direct Targeting Movement", "Target signal diagnostics", "target_signal_diagnostics_enabled", "Activate/deactivate", "true/false", "Shows passive tuning diagnostics for stream cadence, error, output counts, trajectory quality, frequency content, lag, and stability.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Direct Targeting Movement", "Signal CSV logging", "target_signal_logging_enabled", "Activate/deactivate", "true/false", "Writes target signal samples to CSV for offline or automatic tuning analysis.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Direct Targeting Movement", "Signal window samples", "target_signal_window_samples", "Slider", "64-2048", "Rolling sample window used for live diagnostic averages and frequency/lag estimates.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Direct Targeting Movement", "Signal log interval (ms)", "target_signal_log_interval_ms", "Slider", "1.0-1000.0", "Minimum interval between CSV diagnostic rows.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Direct Targeting Movement", "Signal log file", "target_signal_log_file_path", "Input text", "path", "CSV output path for target signal diagnostics.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Direct Targeting Movement", "Calibrated pixel counts", "target_calibrated_pixel_counts_enabled", "Activate/deactivate", "true/false", "Uses measured counts-per-pixel gains for direct targeting output instead of the default 1:1 pixel-count fallback.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Direct Targeting Movement", "Counts / px X", "target_counts_per_pixel_x", "Slider", "0.0000-50.0000", "Horizontal calibrated mouse counts per screen-space pixel magnitude.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Direct Targeting Movement", "Counts / px Y", "target_counts_per_pixel_y", "Slider", "0.0000-50.0000", "Vertical calibrated mouse counts per screen-space pixel magnitude.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Auto Shoot", "Auto Shoot", "auto_shoot", "Activate/deactivate", "true/false", "Enables automatic shooting behavior.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Auto Shoot", "bScope Multiplier", "bScope_multiplier", "Slider", "0.5-2.0", "Multiplier used by auto-shoot scope timing/behavior. Disabled in UI until Auto Shoot is on.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Input Method", "Arduino 16-bit Mouse", "arduino_16_bit_mouse", "Activate/deactivate", "true/false", "Arduino input only. Sends wider mouse movement values.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Input Method", "Arduino Enable Keys", "arduino_enable_keys", "Activate/deactivate", "true/false", "Arduino input only. Enables keyboard key output through Arduino.", "overlay/draw_mouse.cpp"),
    # AI
    row("AI", "Detection", "Confidence Threshold", "confidence_threshold", "Slider", "0.01-1.00", "Minimum detector confidence accepted as a detection.", "overlay/draw_ai.cpp"),
    row("AI", "Detection", "NMS Threshold", "nms_threshold", "Slider", "0.00-1.00", "Non-max suppression overlap threshold.", "overlay/draw_ai.cpp"),
    row("AI", "Detection", "Max Detections", "max_detections", "Slider", "1-100", "Maximum detections retained after model output processing.", "overlay/draw_ai.cpp"),
    # Buttons
    row("Buttons", "Arrow Key Options", "Enable arrows keys options", "enable_arrows_settings", "Activate/deactivate", "true/false", "Allows arrow-key adjustment of target offsets.", "overlay/draw_buttons.cpp"),
    # Overlay
    row("Overlay", "Visual", "Overlay Opacity", "overlay_opacity", "Slider", "220-255", "Window opacity. The UI clamps to a readable minimum of 220.", "overlay/draw_overlay.cpp"),
    row("Overlay", "Visual", "UI Fine Scale", "overlay_ui_scale", "Slider", "0.85-1.35", "Fine-grained scaling for the overlay UI.", "overlay/draw_overlay.cpp"),
    row("Overlay", "Capture Privacy", "Hide Overlays From Recording", "overlay_exclude_from_capture", "Activate/deactivate", "true/false", "Applies capture exclusion so overlay windows are hidden from supported recording/capture paths.", "overlay/draw_overlay.cpp"),
    # Game Overlay
    row("Game Overlay", "General", "Enable", "game_overlay_enabled", "Activate/deactivate", "true/false", "Enables the separate game overlay renderer.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "General", "Overlay Max FPS (0 = uncapped)", "game_overlay_max_fps", "Slider", "0-256", "Caps game overlay render rate. 0 is uncapped.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "General", "Draw Detection Boxes", "game_overlay_draw_boxes", "Activate/deactivate", "true/false", "Shows detection boxes in the game overlay.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "General", "Draw Future Positions", "game_overlay_draw_future", "Activate/deactivate", "true/false", "Shows predicted future target points.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "General", "Draw Wind Debug Tail", "game_overlay_draw_wind_tail", "Activate/deactivate", "true/false", "Shows the wind-mouse debug trail when wind motion is active.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "General", "Show Target Correction", "game_overlay_show_target_correction", "Activate/deactivate", "true/false", "Shows correction indicators for target prediction/aim adjustment.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Box Color", "A", "game_overlay_box_a", "Slider", "0-255", "Detection box alpha.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Box Color", "R", "game_overlay_box_r", "Slider", "0-255", "Detection box red channel.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Box Color", "G", "game_overlay_box_g", "Slider", "0-255", "Detection box green channel.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Box Color", "B", "game_overlay_box_b", "Slider", "0-255", "Detection box blue channel.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Box Color", "Box Thickness", "game_overlay_box_thickness", "Slider", "0.5-10.0", "Detection box line thickness.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Capture Frame", "Draw Capture Frame", "game_overlay_draw_frame", "Activate/deactivate", "true/false", "Draws the capture frame rectangle.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Capture Frame", "A", "game_overlay_frame_a", "Slider", "0-255", "Capture frame alpha.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Capture Frame", "R", "game_overlay_frame_r", "Slider", "0-255", "Capture frame red channel.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Capture Frame", "G", "game_overlay_frame_g", "Slider", "0-255", "Capture frame green channel.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Capture Frame", "B", "game_overlay_frame_b", "Slider", "0-255", "Capture frame blue channel.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Capture Frame", "Frame Thickness", "game_overlay_frame_thickness", "Slider", "0.5-10.0", "Capture frame line thickness.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Future Point Style", "Point Radius", "game_overlay_future_point_radius", "Slider", "1.0-20.0", "Radius of future-position points.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Future Point Style", "Point Step Alpha Falloff", "game_overlay_future_alpha_falloff", "Slider", "0.10-5.00", "Controls how quickly future-point alpha fades along the path.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Icon Overlay", "Enable Icon Overlay", "game_overlay_icon_enabled", "Activate/deactivate", "true/false", "Enables drawing an icon relative to detections.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Icon Overlay", "Icon Width", "game_overlay_icon_width", "Slider", "4-512", "Icon width in pixels.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Icon Overlay", "Icon Height", "game_overlay_icon_height", "Slider", "4-512", "Icon height in pixels.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Icon Overlay", "Icon Offset X", "game_overlay_icon_offset_x", "Slider", "-500.0-500.0", "Horizontal icon offset from the selected anchor.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Icon Overlay", "Icon Offset Y", "game_overlay_icon_offset_y", "Slider", "-500.0-500.0", "Vertical icon offset from the selected anchor.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Aim Simulation", "Enable Aim Simulation Window", "aim_sim_enabled", "Activate/deactivate", "true/false", "Shows the aim simulation overlay window.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Aim Simulation", "Sim X", "aim_sim_x", "Slider", "-3000-3000", "Aim simulation window X position.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Aim Simulation", "Sim Y", "aim_sim_y", "Slider", "-3000-3000", "Aim simulation window Y position.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Aim Simulation", "Sim Width", "aim_sim_width", "Slider", "220-1600", "Aim simulation window width.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Aim Simulation", "Sim Height", "aim_sim_height", "Slider", "180-1000", "Aim simulation window height.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Aim Simulation", "Sim FPS Min", "aim_sim_fps_min", "Slider", "15-360", "Minimum simulated frame rate.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Aim Simulation", "Sim FPS Max", "aim_sim_fps_max", "Slider", "15-360", "Maximum simulated frame rate. The GUI keeps min and max ordered.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Aim Simulation", "FPS Jitter", "aim_sim_fps_jitter", "Slider", "0.000-0.800", "Randomized FPS variance for the simulation.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Aim Simulation", "Capture Delay (ms)", "aim_sim_capture_delay_ms", "Slider", "0.0-80.0", "Simulated capture delay.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Aim Simulation", "Use Live Inference Delay", "aim_sim_use_live_inference", "Activate/deactivate", "true/false", "Uses live backend timing for inference delay instead of the manual value.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Aim Simulation", "Inference Delay (ms)", "aim_sim_inference_delay_ms", "Slider", "0.0-120.0", "Manual simulated inference delay when not using live timing.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Aim Simulation", "Input Delay (ms)", "aim_sim_input_delay_ms", "Slider", "0.0-60.0", "Simulated input delay.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Aim Simulation", "Extra Delay (ms)", "aim_sim_extra_delay_ms", "Slider", "0.0-60.0", "Additional simulated processing delay.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Aim Simulation", "Target Max Speed", "aim_sim_target_max_speed", "Slider", "20-2500", "Maximum simulated target speed.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Aim Simulation", "Target Accel", "aim_sim_target_accel", "Slider", "20-10000", "Simulated target acceleration.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Aim Simulation", "Target Stop Chance", "aim_sim_target_stop_chance", "Slider", "0.00-0.95", "Probability that the simulated target pauses during a retarget.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Aim Simulation", "Show Delayed Observation", "aim_sim_show_observed", "Activate/deactivate", "true/false", "Shows delayed target observation in the simulation.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Aim Simulation", "Show Trajectory History", "aim_sim_show_history", "Activate/deactivate", "true/false", "Shows historical target/aim trails.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Aim Simulation", "Show Kalman Debug", "aim_sim_show_kalman_debug", "Activate/deactivate", "true/false", "Shows Kalman-related debug visuals in the simulation.", "overlay/draw_game_overlay.cpp"),
    # Debug
    row("Debug", "Screenshot Buttons", "Verbose console output", "verbose", "Activate/deactivate", "true/false", "Enables extra console logging.", "overlay/draw_debug.cpp"),
    row("Debug", "Log File", "Enable log file", "debug_log_file_enabled", "Activate/deactivate", "true/false", "Writes logs to debug_log_file_path when enabled.", "overlay/draw_debug.cpp"),
]


OTHER_GUI: list[dict[str, str]] = [
    row("Capture", "General Capture", "Detection Resolution", "detection_resolution", "Combo", "160, 320, 640", "Changes detector input resolution and restarts/reloads dependent paths.", "overlay/draw_capture.cpp"),
    row("Capture", "General Capture", "Capture method", "capture_method", "Combo", "duplication_api, winrt, virtual_camera, udp_capture", "Selects frame source.", "overlay/draw_capture.cpp"),
    row("Capture", "WinRT", "Capture target (WinRT)", "capture_target", "Combo", "monitor, window", "WinRT only.", "overlay/draw_capture.cpp"),
    row("Capture", "WinRT", "Window title contains", "capture_window_title", "Input text", "text", "Used when WinRT target is window.", "overlay/draw_capture.cpp"),
    row("Capture", "Monitor Capture", "Capture monitor", "monitor_idx", "Combo", "monitor index", "Monitor list is built from active displays.", "overlay/draw_capture.cpp"),
    row("Capture", "Virtual Camera", "Virtual camera", "virtual_camera_name", "Combo", "available cameras", "Filtered list of DirectShow video input devices.", "overlay/draw_capture.cpp"),
    row("Capture", "UDP Capture", "UDP IP", "udp_ip", "Input text", "IPv4/string", "Applied with the Apply UDP Settings button.", "overlay/draw_capture.cpp"),
    row("Capture", "UDP Capture", "UDP Port", "udp_port", "Input int", "1-65535", "Applied with the Apply UDP Settings button.", "overlay/draw_capture.cpp"),
    row("Mouse", "Input Method", "Mouse Input Method", "input_method", "Combo", "WIN32, GHUB, RAZER, DIRECT, ARDUINO, TEENSY41, TEENSY41_HID, KMBOX_NET, KMBOX_A, MAKCU", "Changes active mouse backend.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Input Method", "Arduino/Teensy Port", "arduino_port", "Combo", "COM1-COM30", "Arduino and Teensy 4.1 serial input.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Input Method", "Arduino/Teensy Baudrate", "arduino_baudrate", "Combo", "9600-115200", "Arduino and Teensy 4.1 serial input.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Input Method", "Kmbox Net IP", "kmbox_net_ip", "Input text", "text", "Saved with Save & Reconnect.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Input Method", "Kmbox Net Port", "kmbox_net_port", "Input text", "text", "Saved with Save & Reconnect.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Input Method", "Kmbox Net UUID", "kmbox_net_uuid", "Input text", "text", "Saved with Save & Reconnect.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Input Method", "Kmbox A PIDVID", "kmbox_a_pidvid", "Input text", "PPPPVVVV", "Saved with Save & Reconnect.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Input Method", "Makcu Port", "makcu_port", "Combo", "AUTO, COM1-COM30", "Makcu input only.", "overlay/draw_mouse.cpp"),
    row("Mouse", "Input Method", "Makcu Baudrate", "makcu_baudrate", "Combo", "115200-4000000", "Makcu input only.", "overlay/draw_mouse.cpp"),
    row("AI", "Model", "Model", "ai_model", "Combo", "models folder", "Selects detector model from available files.", "overlay/draw_ai.cpp"),
    row("AI", "Backend", "Backend", "backend", "Combo", "TRT, DML", "CUDA builds only; DML build does not expose this combo.", "overlay/draw_ai.cpp", "CUDA GUI / DML config-only"),
    row("Buttons", "Button Bindings", "Targeting Buttons", "button_targeting", "Combo rows", "key list", "Supports multiple bindings with + and -.", "overlay/draw_buttons.cpp"),
    row("Buttons", "Button Bindings", "Shoot Buttons", "button_shoot", "Combo rows", "key list", "Supports multiple bindings with + and -.", "overlay/draw_buttons.cpp"),
    row("Buttons", "Button Bindings", "Zoom Buttons", "button_zoom", "Combo rows", "key list", "Supports multiple bindings with + and -.", "overlay/draw_buttons.cpp"),
    row("Buttons", "Button Bindings", "Exit Buttons", "button_exit", "Combo rows", "key list", "Supports multiple bindings with + and -.", "overlay/draw_buttons.cpp"),
    row("Buttons", "Button Bindings", "Pause Buttons", "button_pause", "Combo rows", "key list", "Supports multiple bindings with + and -.", "overlay/draw_buttons.cpp"),
    row("Buttons", "Button Bindings", "Reload Config Buttons", "button_reload_config", "Combo rows", "key list", "Supports multiple bindings with + and -.", "overlay/draw_buttons.cpp"),
    row("Buttons", "Button Bindings", "Overlay Buttons", "button_open_overlay", "Combo rows", "key list", "Supports multiple bindings with + and -.", "overlay/draw_buttons.cpp"),
    row("Overlay", "Config", "Save Config", "config.ini", "Button", "writes config.ini", "Immediately writes the current runtime config.", "overlay/draw_overlay.cpp"),
    row("Overlay", "Config", "Load Config", "config.ini", "Button", "reads config.ini", "Merges config.ini into current runtime values and refreshes runtime-sensitive paths.", "overlay/draw_overlay.cpp"),
    row("Game Overlay", "Icon Overlay", "Icon Path", "game_overlay_icon_path", "Input text/browse", "path", "Image path for icon overlay.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Icon Overlay", "Icon Class (-1 = all)", "game_overlay_icon_class", "Input int", "-1 or class id", "Restricts icon to one class when >= 0.", "overlay/draw_game_overlay.cpp"),
    row("Game Overlay", "Icon Overlay", "Icon Anchor", "game_overlay_icon_anchor", "Combo", "center, top, bottom, head", "Anchor point used for icon placement.", "overlay/draw_game_overlay.cpp"),
    row("Debug", "Screenshot Buttons", "Screenshot buttons", "screenshot_button", "Combo rows", "key list", "Supports multiple bindings with + and -.", "overlay/draw_debug.cpp"),
    row("Debug", "Screenshot Buttons", "Screenshot delay", "screenshot_delay", "Input int", "step 50/500", "Delay before screenshot capture.", "overlay/draw_debug.cpp"),
    row("Debug", "Log File", "Log file path", "debug_log_file_path", "Input text", "path", "Used when Enable log file is on.", "overlay/draw_debug.cpp"),
]


def config_only(
    section: str,
    key: str,
    default: str,
    value_range: str,
    guidance: str,
    note: str,
    source: str = "config/config.cpp",
    visibility: str = "Config only",
) -> dict[str, str]:
    return {
        "section": section,
        "key": key,
        "default": default,
        "range": value_range,
        "guidance": guidance,
        "note": note,
        "source": source,
        "visibility": visibility,
        "current": cfg(key, default),
    }


CONFIG_ONLY: list[dict[str, str]] = [
    config_only("Mouse prediction", "prediction_futurePositions", "20", "integer", "Number of future target positions to retain/use for prediction visualization.", "Saved in current config.ini, but no direct GUI control exists."),
    config_only("Mouse prediction", "draw_futurePositions", "true", "true/false", "Enables drawing future positions in preview/debug paths.", "Separate from game_overlay_draw_future."),
    config_only("AI backend", "backend", "DML", "DML or TRT", "Selects DirectML or TensorRT backend.", "GUI combo exists only in CUDA builds.", visibility="DML config-only / CUDA GUI"),
    config_only("AI backend", "dml_device_id", "0", "integer adapter id", "DirectML adapter index used by ONNX Runtime DML.", "Not exposed in the GUI."),
    config_only("System reserves", "cpuCoreReserveCount", "4", "integer", "CPU cores reserved away from worker assignment.", "Not exposed in the GUI."),
    config_only("System reserves", "systemMemoryReserveMB", "2048", "MB", "System memory reserve used by runtime resource planning.", "Not exposed in the GUI."),
    config_only("Custom classes", "class_player", "0", "integer class id", "Detector class id treated as player/body.", "Not exposed in the GUI."),
    config_only("Custom classes", "class_head", "1", "integer class id", "Detector class id treated as head.", "Not exposed in the GUI."),
    config_only("Debug", "show_fps", "false", "true/false", "Legacy/debug FPS display flag.", "Saved in config.ini but not exposed in the current GUI."),
    # Loadable but not emitted by saveConfig in this build.
    config_only("Advanced mouse", "minSpeedMultiplier", "0.1", "float", "Legacy minimum movement speed multiplier.", "Loadable from config.ini if added manually; saveConfig does not currently emit it.", visibility="Loadable hidden key"),
    config_only("Advanced mouse", "maxSpeedMultiplier", "0.1", "float", "Legacy maximum movement speed multiplier.", "Loadable from config.ini if added manually; saveConfig does not currently emit it.", visibility="Loadable hidden key"),
    config_only("Advanced mouse", "predictionInterval", "0.01", "seconds", "Prediction sample interval used by legacy/auxiliary prediction logic.", "Loadable from config.ini if added manually; saveConfig does not currently emit it.", visibility="Loadable hidden key"),
    config_only("Legacy snap", "snapRadius", "1.5", "float", "Legacy snap radius.", "Loadable hidden key; no GUI control."),
    config_only("Legacy snap", "nearRadius", "25.0", "float", "Legacy near-target radius.", "Loadable hidden key; no GUI control."),
    config_only("Legacy snap", "speedCurveExponent", "3.0", "float", "Legacy speed curve exponent.", "Loadable hidden key; no GUI control."),
    config_only("Legacy snap", "snapBoostFactor", "1.15", "float", "Legacy snap boost multiplier.", "Loadable hidden key; no GUI control."),
    config_only("Legacy recoil", "easynorecoil", "false", "true/false", "Legacy recoil helper flag.", "Loadable hidden key; no GUI control and not emitted by saveConfig."),
    config_only("Legacy recoil", "easynorecoilstrength", "0.0", "float", "Legacy recoil helper strength.", "Loadable hidden key; no GUI control and not emitted by saveConfig."),
    config_only("Wind mouse", "wind_mouse_enabled", "false", "true/false", "Enables wind-style motion noise in supported runtime/simulation paths.", "Loadable hidden key; no GUI control."),
    config_only("Wind mouse", "wind_G", "18.0", "float", "Wind mouse gravity parameter.", "Loadable hidden key; no GUI control."),
    config_only("Wind mouse", "wind_W", "15.0", "float", "Wind mouse wind parameter.", "Loadable hidden key; no GUI control."),
    config_only("Wind mouse", "wind_M", "10.0", "float", "Wind mouse max-step parameter.", "Loadable hidden key; no GUI control."),
    config_only("Wind mouse", "wind_D", "8.0", "float", "Wind mouse damping/distance parameter.", "Loadable hidden key; no GUI control."),
    # CUDA-only config keys not present in the DML config file.
    config_only("CUDA", "export_enable_fp8", "false", "true/false", "Enables FP8 TensorRT export where supported.", "CUDA builds only; no GUI checkbox.", visibility="CUDA config-only"),
    config_only("CUDA", "export_enable_fp16", "true", "true/false", "Enables FP16 TensorRT export where supported.", "CUDA builds only; no GUI checkbox.", visibility="CUDA config-only"),
    config_only("CUDA", "use_cuda_graph", "false", "true/false", "Enables CUDA graph execution path.", "CUDA builds only; no GUI checkbox.", visibility="CUDA config-only"),
    config_only("CUDA", "use_pinned_memory", "true", "true/false", "Uses pinned host memory for CUDA paths.", "CUDA builds only; no GUI checkbox.", visibility="CUDA config-only"),
    config_only("CUDA", "gpuMemoryReserveMB", "2048", "MB", "GPU memory reserve for CUDA resource planning.", "CUDA builds only; no GUI control.", visibility="CUDA config-only"),
    config_only("CUDA", "enableGpuExclusiveMode", "true", "true/false", "Attempts GPU exclusive mode at startup.", "CUDA builds only; no GUI checkbox.", visibility="CUDA config-only"),
]


TAB_ORDER = [
    "Capture",
    "Target",
    "Mouse",
    "AI",
    "Buttons",
    "Overlay",
    "Game Overlay",
    "Stats",
    "Debug",
]


def esc_cell(value: object) -> str:
    text = str(value).replace("\n", " ").strip()
    return text.replace("|", "\\|")


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for values in rows:
        lines.append("| " + " | ".join(esc_cell(v) for v in values) + " |")
    return "\n".join(lines)


def controls_for(tab: str, kind: str | None = None) -> list[dict[str, str]]:
    rows = [item for item in GUI_CONTROLS if item["tab"] == tab]
    if kind:
        rows = [item for item in rows if item["kind"] == kind]
    return rows


def other_controls_for(tab: str) -> list[dict[str, str]]:
    return [item for item in OTHER_GUI if item["tab"] == tab]


def generate_markdown() -> str:
    today = dt.date.today().isoformat()
    out: list[str] = []
    out.append("# 0BS GUI and Config Setting Reference")
    out.append("")
    out.append(f"Generated on {today} from `{CONFIG_PATH.relative_to(ROOT)}` and the ImGui source files under `overlay/`.")
    out.append("")
    out.append("This reference is ordered by the GUI sidebar tabs in `overlay/overlay.cpp`. It lists every GUI slider and every activate/deactivate checkbox, then documents config.ini settings that are not editable in the current DML GUI or are hidden/loadable config keys.")
    out.append("")
    out.append("## Searchable Directory")
    out.append("")
    out.append("- PDF: `docs/settings-reference/0BS-settings-reference.pdf`")
    out.append("- Search index: `docs/settings-reference/settings-index.csv`")
    out.append("- Directory README: `docs/settings-reference/README.md`")
    out.append("- Program flow chart: `docs/program-flow.mmd`")
    out.append("")
    out.append("Search by GUI label, config key, tab, section, or source file. Rows marked `CUDA GUI / DML config-only` appear as GUI controls only in CUDA builds; the current `x64/DML/config.ini` keeps those values as config-only settings.")
    out.append("")
    out.append("## Repository Reference")
    out.append("")
    out.append("Private backup/update repository: `NANOBYTECOMPUTERS/0bs2`.")
    out.append("")
    out.append("This reference covers the OpenCV 5 CUDA / TensorRT 11.1 project state after neural, depth, PID, feed-forward, and smart-blending code paths were removed. Local dependency drops under `modules/` and `extras/` are intentionally excluded from Git and should be restored through the documented build/dependency scripts.")
    out.append("")
    out.append("## Targeting Pipeline")
    out.append("")
    out.append("The runtime path is now fully deterministic:")
    out.append("")
    out.append("`Video Frame -> Detector -> Tracker -> State Estimator (Kalman/IMM) -> Direct Targeting Movement -> Mouse Output`")
    out.append("")
    out.append("The tracker aim point and Kalman prediction own convergence. Selecting IMM changes only the InnerAim tracker estimator; detections and tracker observations feed the targeting pipeline directly.")
    out.append("")
    out.append("Ego-motion compensation is an opt-in tracker stabilizer. It subtracts bounded, recent emitted mouse/view motion from tracker priors only; raw detector boxes and final mouse output remain unchanged.")
    out.append("")
    out.append("## Legacy Prediction Compatibility")
    out.append("")
    out.append("Deprecated but retained: `minSpeedMultiplier`, `maxSpeedMultiplier`, `predictionInterval`, `snapRadius`, `nearRadius`, `speedCurveExponent`, `snapBoostFactor`, `kalman_warmup_frames`, and `kalman_acquisition_frames` remain loadable for old configs and aim-simulation parity.")
    out.append("")
    out.append("Removal gate: remove these only after a behavior baseline proves behavior stays identical across the live tracker, game overlay simulation, and config merge path.")
    out.append("")
    out.append("## GUI Tab Order")
    out.append("")
    out.append(md_table(["Order", "Tab", "Notes"], [[i + 1, tab, "Read-only monitor tab" if tab == "Stats" else "Adjustable controls documented below"] for i, tab in enumerate(TAB_ORDER)]))
    out.append("")

    for tab in TAB_ORDER:
        out.append(f"## {tab} Tab")
        if tab == "Stats":
            out.append("")
            out.append("The Stats tab is read-only in the current GUI. It displays timing graphs, capture FPS, capture details, and CUDA status where available.")
            out.append("")
            continue

        sliders = controls_for(tab, "Slider")
        toggles = controls_for(tab, "Activate/deactivate")
        others = other_controls_for(tab)

        out.append("")
        out.append("### Sliders")
        out.append("")
        if sliders:
            out.append(md_table(["Section", "Slider", "Config key", "Range", "Current", "Notes"], [[r["section"], r["label"], r["key"], r["range"], r["current"], r["note"]] for r in sliders]))
        else:
            out.append("No saved GUI sliders in this tab.")
        out.append("")
        out.append("### Activate/Deactivate")
        out.append("")
        if toggles:
            out.append(md_table(["Section", "Control", "Config key", "Values", "Current", "Notes"], [[r["section"], r["label"], r["key"], r["range"], r["current"], r["note"]] for r in toggles]))
        else:
            out.append("No activate/deactivate checkboxes in this tab.")
        out.append("")
        if others:
            out.append("### Other GUI-Exposed Config Controls")
            out.append("")
            out.append(md_table(["Section", "Control", "Config key", "Type/options", "Current", "Notes"], [[r["section"], r["label"], r["key"], r["range"], r["current"], r["note"]] for r in others]))
            out.append("")

    out.append("## Config.ini Settings Not In The GUI")
    out.append("")
    out.append("These settings are editable by changing `config.ini` directly. Some are build-gated: CUDA rows are emitted only by CUDA builds.")
    out.append("")
    out.append(md_table(
        ["Section", "Setting", "Current/default", "Range/options", "Visibility", "Guidance", "Notes"],
        [[r["section"], r["key"], r["current"], r["range"], r["visibility"], r["guidance"], r["note"]] for r in CONFIG_ONLY],
    ))
    out.append("")
    out.append("## Source Map")
    out.append("")
    out.append(md_table(
        ["Area", "Source"],
        [
            ["Tab order", "overlay/overlay.cpp"],
            ["Capture", "overlay/draw_capture.cpp and overlay/draw_debug.cpp"],
            ["Target", "overlay/draw_target.cpp"],
            ["Mouse", "overlay/draw_mouse.cpp"],
            ["Buttons", "overlay/draw_buttons.cpp"],
            ["Overlay", "overlay/draw_overlay.cpp"],
            ["Game Overlay", "overlay/draw_game_overlay.cpp"],
            ["Debug", "overlay/draw_debug.cpp"],
            ["Config load/save/defaults", "config/config.cpp and config/config.h"],
        ],
    ))
    out.append("")
    return "\n".join(out)


def write_csv(path: Path) -> None:
    rows: list[dict[str, str]] = []
    for item in GUI_CONTROLS + OTHER_GUI:
        rows.append({
            "category": item["kind"],
            "tab": item["tab"],
            "section": item["section"],
            "label": item["label"],
            "config_key": item["key"],
            "range_or_options": item["range"],
            "current_or_default": item["current"],
            "visibility": item["visibility"],
            "notes": item["note"],
            "source": item["source"],
        })
    for item in CONFIG_ONLY:
        rows.append({
            "category": "Config-only setting",
            "tab": "",
            "section": item["section"],
            "label": item["key"],
            "config_key": item["key"],
            "range_or_options": item["range"],
            "current_or_default": item["current"],
            "visibility": item["visibility"],
            "notes": f"{item['guidance']} {item['note']}",
            "source": item["source"],
        })

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def directory_markdown() -> str:
    return "\n".join([
        "# Settings Reference Directory",
        "",
        "This directory contains the generated searchable setting reference.",
        "",
        "- Full Markdown reference: `../../README.md`",
        "- Searchable PDF: `0BS-settings-reference.pdf`",
        "- Searchable CSV index: `settings-index.csv`",
        "- Generator: `generate_settings_reference.py`",
        "",
        "The CSV index is the fastest way to search by config key, GUI label, source file, tab, or visibility.",
        "",
    ])


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    safe = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", safe)
    return Paragraph(safe, style)


def pdf_table(headers: list[str], rows: list[list[object]], widths: list[float]) -> Table:
    styles = getSampleStyleSheet()
    cell = ParagraphStyle(
        "Cell",
        parent=styles["BodyText"],
        fontSize=6.7,
        leading=7.7,
        alignment=TA_LEFT,
    )
    head = ParagraphStyle(
        "Head",
        parent=cell,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )
    data = [[paragraph(str(h), head) for h in headers]]
    for row_values in rows:
        data.append([paragraph(str(v), cell) for v in row_values])
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return table


def write_pdf(path: Path) -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title", parent=styles["Title"], fontSize=20, leading=24)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15, leading=18, spaceBefore=10, spaceAfter=6)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, leading=13, spaceBefore=7, spaceAfter=4)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=8.5, leading=10.5)

    doc = SimpleDocTemplate(
        str(path),
        pagesize=landscape(letter),
        rightMargin=0.35 * inch,
        leftMargin=0.35 * inch,
        topMargin=0.35 * inch,
        bottomMargin=0.35 * inch,
        title="0BS GUI and Config Setting Reference",
        author="Codex",
    )
    story: list[object] = []
    story.append(paragraph("0BS GUI and Config Setting Reference", title))
    story.append(paragraph(f"Generated on {dt.date.today().isoformat()} from {CONFIG_PATH.relative_to(ROOT)}.", body))
    story.append(Spacer(1, 0.1 * inch))
    story.append(paragraph("Searchable Directory", h1))
    story.append(paragraph("Use PDF search for any GUI label, config key, tab, section, or source file. The companion CSV at docs/settings-reference/settings-index.csv contains the same searchable directory in row form.", body))
    story.append(paragraph("Repository Reference", h1))
    story.append(paragraph("Private backup/update repository: NANOBYTECOMPUTERS/0bs2.", body))
    story.append(paragraph("This reference covers the OpenCV 5 CUDA / TensorRT 11.1 project state after neural, depth, PID, feed-forward, and smart-blending code paths were removed. Local dependency drops under modules/ and extras/ are intentionally excluded from Git and should be restored through the documented build/dependency scripts.", body))
    story.append(paragraph("Targeting Pipeline", h1))
    story.append(paragraph("The runtime path is now detector, tracker, Kalman/IMM state estimation, target-motion state publishing, a paced mouse stream, calibrated pixel-to-count conversion, and mouse output. Detections and tracker observations feed targeting directly without relying on the visual game overlay.", body))
    story.append(paragraph("Ego-motion compensation subtracts bounded recent emitted mouse/view motion from tracker priors only. It does not modify raw detections or final mouse output.", body))
    story.append(paragraph("Legacy Prediction Compatibility", h1))
    story.append(paragraph("Deprecated but retained: minSpeedMultiplier, maxSpeedMultiplier, predictionInterval, snapRadius, nearRadius, speedCurveExponent, snapBoostFactor, kalman_warmup_frames, and kalman_acquisition_frames remain loadable for old configs and aim-simulation parity.", body))
    story.append(paragraph("Removal gate: remove these only after a behavior baseline proves behavior stays identical across the live tracker, game overlay simulation, and config merge path.", body))
    story.append(Spacer(1, 0.08 * inch))
    story.append(pdf_table(
        ["Order", "Tab", "Notes"],
        [[i + 1, tab, "Read-only monitor tab" if tab == "Stats" else "Adjustable controls documented below"] for i, tab in enumerate(TAB_ORDER)],
        [0.5 * inch, 1.4 * inch, 7.5 * inch],
    ))
    story.append(PageBreak())

    for tab in TAB_ORDER:
        story.append(paragraph(f"{tab} Tab", h1))
        if tab == "Stats":
            story.append(paragraph("The Stats tab is read-only in the current GUI. It displays timing graphs, capture FPS, capture details, and CUDA status where available.", body))
            story.append(PageBreak())
            continue

        sliders = controls_for(tab, "Slider")
        toggles = controls_for(tab, "Activate/deactivate")
        others = other_controls_for(tab)

        story.append(paragraph("Sliders", h2))
        if sliders:
            story.append(pdf_table(
                ["Section", "Slider", "Config key", "Range", "Current", "Notes"],
                [[r["section"], r["label"], r["key"], r["range"], r["current"], r["note"]] for r in sliders],
                [1.25 * inch, 1.7 * inch, 1.85 * inch, 1.0 * inch, 0.9 * inch, 3.1 * inch],
            ))
        else:
            story.append(paragraph("No saved GUI sliders in this tab.", body))
        story.append(Spacer(1, 0.08 * inch))
        story.append(paragraph("Activate/Deactivate", h2))
        if toggles:
            story.append(pdf_table(
                ["Section", "Control", "Config key", "Values", "Current", "Notes"],
                [[r["section"], r["label"], r["key"], r["range"], r["current"], r["note"]] for r in toggles],
                [1.25 * inch, 1.8 * inch, 1.8 * inch, 0.85 * inch, 0.75 * inch, 3.3 * inch],
            ))
        else:
            story.append(paragraph("No activate/deactivate checkboxes in this tab.", body))
        if others:
            story.append(Spacer(1, 0.08 * inch))
            story.append(paragraph("Other GUI-Exposed Config Controls", h2))
            story.append(pdf_table(
                ["Section", "Control", "Config key", "Type/options", "Current", "Notes"],
                [[r["section"], r["label"], r["key"], r["range"], r["current"], r["note"]] for r in others],
                [1.25 * inch, 1.7 * inch, 1.8 * inch, 1.25 * inch, 0.8 * inch, 3.0 * inch],
            ))
        story.append(PageBreak())

    story.append(paragraph("Config.ini Settings Not In The GUI", h1))
    story.append(paragraph("These settings are editable by changing config.ini directly. Some are build-gated: CUDA rows are emitted only by CUDA builds.", body))
    story.append(Spacer(1, 0.08 * inch))
    story.append(pdf_table(
        ["Section", "Setting", "Current/default", "Range/options", "Visibility", "Guidance", "Notes"],
        [[r["section"], r["key"], r["current"], r["range"], r["visibility"], r["guidance"], r["note"]] for r in CONFIG_ONLY],
        [1.05 * inch, 1.65 * inch, 1.15 * inch, 1.05 * inch, 1.25 * inch, 2.05 * inch, 2.15 * inch],
    ))
    story.append(PageBreak())
    story.append(paragraph("Source Map", h1))
    story.append(pdf_table(
        ["Area", "Source"],
        [
            ["Tab order", "overlay/overlay.cpp"],
            ["Capture", "overlay/draw_capture.cpp and overlay/draw_debug.cpp"],
            ["Target", "overlay/draw_target.cpp"],
            ["Mouse", "overlay/draw_mouse.cpp"],
            ["Buttons", "overlay/draw_buttons.cpp"],
            ["Overlay", "overlay/draw_overlay.cpp"],
            ["Game Overlay", "overlay/draw_game_overlay.cpp"],
            ["Debug", "overlay/draw_debug.cpp"],
            ["Config load/save/defaults", "config/config.cpp and config/config.h"],
        ],
        [2.2 * inch, 6.6 * inch],
    ))

    doc.build(story)


def validate_coverage() -> None:
    documented = {item["key"] for item in GUI_CONTROLS + OTHER_GUI}
    documented.update(item["key"] for item in CONFIG_ONLY)
    documented.add("preview_debug_scale")

    ignored_prefixes = {"Games.<profile>"}
    missing = []
    for key in CONFIG_VALUES:
        if key in documented:
            continue
        if any(key.startswith(prefix) for prefix in ignored_prefixes):
            continue
        missing.append(key)
    if missing:
        raise RuntimeError("Undocumented config keys: " + ", ".join(sorted(missing)))


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    validate_coverage()

    readme = generate_markdown()
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    (DOCS_DIR / "README.md").write_text(directory_markdown(), encoding="utf-8")
    write_csv(DOCS_DIR / "settings-index.csv")
    write_pdf(DOCS_DIR / "0BS-settings-reference.pdf")
    print("Wrote README.md")
    print("Wrote docs/settings-reference/README.md")
    print("Wrote docs/settings-reference/settings-index.csv")
    print("Wrote docs/settings-reference/0BS-settings-reference.pdf")


if __name__ == "__main__":
    main()
