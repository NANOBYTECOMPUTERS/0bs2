#!/usr/bin/env python3
"""
Simple GUI for training the four neural models used by 0BS.

- Suggested good defaults (including LightGBM distillation where recommended)
- All fields are editable
- Live log output
- Launches the real train_*.py scripts via subprocess

Run with:
    python training/train_gui.py
"""

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from pathlib import Path
from typing import Dict, List, Any, Optional


def ensure_package(package: str, import_name: str = None, install_name: str = None) -> bool:
    """
    Check if a package is importable.
    If not, ask the user and attempt to pip install it.
    Returns True if the package is available after this call.
    """
    if import_name is None:
        import_name = package.split('[')[0]  # handle extras like lightgbm[sklearn]
    if install_name is None:
        install_name = package

    try:
        __import__(import_name)
        return True
    except ImportError:
        pass

    # GUI context - ask nicely
    if messagebox.askyesno(
        "Missing Dependency",
        f"The package '{package}' is required for this feature but is not installed.\n\n"
        f"Would you like the GUI to install it now using pip?"
    ):
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", install_name, "--quiet"
            ])
            # Re-attempt import
            __import__(import_name)
            messagebox.showinfo("Success", f"'{package}' was installed successfully.")
            return True
        except Exception as e:
            messagebox.showerror(
                "Installation Failed",
                f"Failed to install '{package}'.\n\n"
                f"You can try manually: pip install {install_name}\n\n"
                f"Error: {e}"
            )
            return False
    return False


# ---------------------------------------------------------------------------
# Suggested defaults (kept in sync with recent best practices + README)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_EVENT_PROCESS_FINISHED = "process_finished"
LOG_EVENT_REALWORLD_FINISHED = "realworld_process_finished"

DEFAULTS = {
    "neural_tracker": {
        "dataset": "training/data/neural_tracker_dataset.csv",
        "output": "neural_models/neural_tracker.pt",
        "metadata": "neural_models/neural_tracker.json",
        "epochs": 30,
        "batch_size": 512,
        "hidden": 56,
        "learning_rate": 0.00045,
        "weight_decay": 0.00012,
        "validation_fraction": 0.16,
        "patience": 8,
        "seed": 1337,
        "use_gbm_teacher": True,      # Recommended
        "distill_weight": 0.20,
        "gbm_boosting_rounds": 250,
        "gbm_learning_rate": 0.05,
        "gbm_max_depth": 6,
        "ranking_loss_weight": 0.20,
        "ranking_margin": 0.15,
    },
    "pid_governor": {
        "dataset": "training/data/pid_governor_dataset.csv",
        "output": "neural_models/pid_governor.pt",
        "metadata": "neural_models/pid_governor.json",
        "epochs": 30,
        "batch_size": 512,
        "hidden": 72,
        "learning_rate": 0.0009,
        "weight_decay": 0.00015,
        "validation_fraction": 0.18,
        "patience": 7,
        "seed": 1337,
        "device": "auto",
        "use_gbm_teacher": True,      # Strongly recommended
        "distill_weight": 0.55,
        "gbm_boosting_rounds": 300,
        "gbm_learning_rate": 0.04,
        "gbm_max_depth": 7,
    },
    "temporal": {
        "dataset": "training/datasets/temporal_tracks_v2.npz",
        "output": "neural_models/temporal_predictor.pt",
        "onnx_output": "neural_models/temporal_predictor.onnx",
        "metadata": "neural_models/temporal_predictor.json",
        "epochs": 60,
        "batch_size": 256,
        "hidden_size": 160,
        "model_type": "gru",          # or "transformer"
        "history_length": 12,
        "prediction_horizon": 16,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "smoothness_weight": 0.008,
        "velocity_weight": 0.10,
        "near_horizon_frames": 8,
        "near_horizon_weight": 3.5,
        "latency_focus_start_frame": 3,
        "latency_focus_end_frame": 6,
        "latency_focus_weight": 0.75,
        "patience": 10,
        "seed": 1337,
        "device": "auto",
        "auto_generate_dataset": True,
        "auto_generate_samples": 65536,
        "jitter_px": 0.55,
        "camera_shake_px": 0.75,
        "occlusion_probability": 0.12,
        "max_speed_px_s": 1800.0,
    },
    "neural_targeting": {
        "dataset": "training/datasets/neural_targeting_tracks.npz",
        "output": "neural_models/neural_targeting_head.pt",
        "onnx_output": "neural_models/neural_targeting_head.onnx",
        "metadata": "neural_models/neural_targeting_head.json",
        "epochs": 35,
        "batch_size": 256,
        "hidden_size": 192,
        "prediction_horizon": 16,
        "learning_rate": 0.0008,
        "weight_decay": 0.0001,
        "confidence_weight": 0.18,
        "near_lock_penalty_weight": 0.035,
        "max_refinement_px": 35.0,
        "patience": 8,
        "seed": 4242,
        "device": "auto",
        "auto_generate_dataset": True,
    },
}


class TrainingGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("0BS Training GUI")
        self.geometry("1100x780")
        self.minsize(980, 680)

        self.current_process = None
        self.log_queue: queue.Queue = queue.Queue()
        self.last_trainer_key: Optional[str] = None
        self.last_output_path: Optional[Path] = None
        self.log_window = None
        self.log_dialog_text = None

        self._create_widgets()
        self._poll_log_queue()
        self._create_menu()

        # Track progress state
        self.current_max_epochs = 0
        self.current_epoch = 0

        # For the Real-World Fine-Tune chaining logic
        self._pending_next_cmd = None
        self._pending_next_label = None
        self._pending_pipeline = None
        self._pending_pipeline_label = None

        # Friendly reminder
        self.after(800, lambda: self._append_log(
            "Tip: For best results on Neural Tracker and PID Governor, enable LightGBM distillation.\n"
            "Install once with:  pip install lightgbm\n\n"
        ))

    def _create_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save Current Tab as Preset", command=self._save_preset)
        file_menu.add_command(label="Load Preset into Current Tab", command=self._load_preset)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Launch TensorBoard", command=self._launch_tensorboard)
        tools_menu.add_command(label="Open Models Folder", command=self._open_models_folder)
        tools_menu.add_command(label="Open Runs Folder", command=self._open_runs_folder)
        tools_menu.add_command(label="Show Training Log", command=self._show_training_log_dialog)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        self.config(menu=menubar)

    def _apply_dark_theme(self):
        """Apply a simple dark theme using ttk styles + manual widget colors."""
        style = ttk.Style(self)

        # Base colors
        bg = "#1e1e1e"
        fg = "#d4d4d4"
        accent = "#007acc"
        frame_bg = "#252526"

        # Try to use a dark base theme if available (clam is usually good)
        try:
            style.theme_use("clam")
        except:
            pass

        # General styling
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TLabelframe", background=bg, foreground=fg)
        style.configure("TLabelframe.Label", background=bg, foreground=fg)

        style.configure("TButton", background=frame_bg, foreground=fg)
        style.map("TButton",
                  background=[("active", accent)],
                  foreground=[("active", "white")])

        style.configure("TNotebook", background=bg)
        style.configure("TNotebook.Tab", background=frame_bg, foreground=fg)
        style.map("TNotebook.Tab",
                  background=[("selected", accent)],
                  foreground=[("selected", "white")])

        style.configure("Horizontal.TProgressbar",
                        background=accent,
                        troughcolor=frame_bg,
                        bordercolor=frame_bg)

        # Apply to root
        self.configure(bg=bg)

        # Log text widget (ScrolledText is a tk widget, not ttk)
        if hasattr(self, "log_text"):
            self.log_text.configure(
                bg="#1e1e1e",
                fg="#d4d4d4",
                insertbackground="#d4d4d4",
                selectbackground=accent,
                selectforeground="white"
            )

        # Also style progress label if it exists
        if hasattr(self, "progress_label"):
            self.progress_label.configure(background=bg, foreground=fg)

    def _create_widgets(self):
        # Top banner
        banner = ttk.Frame(self, padding=6)
        banner.pack(fill="x")
        ttk.Label(banner, text="0BS Neural Model Trainer  •  Suggested defaults are pre-filled and fully editable",
                  font=("Segoe UI", 11, "bold")).pack(side="left")

        ttk.Button(banner, text="Launch TensorBoard", command=self._launch_tensorboard).pack(side="right", padx=4)
        ttk.Button(banner, text="Open models folder", command=self._open_models_folder).pack(side="right", padx=4)
        ttk.Button(banner, text="Open runs folder", command=self._open_runs_folder).pack(side="right")
        ttk.Button(banner, text="Show Training Log", command=self._show_training_log_dialog).pack(side="right", padx=(8, 4))
        ttk.Button(banner, text="Check Environment", command=self._show_environment_check).pack(side="right", padx=(8, 4))

        # Notebook with one tab per trainer
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=(4, 4))

        self.tabs: Dict[str, Dict[str, Any]] = {}

        self._create_tab("neural_tracker", "Neural Tracker (Association)")
        self._create_tab("pid_governor", "PID Governor (40-feature)")
        self._create_tab("temporal", "Temporal Predictor")
        self._create_tab("neural_targeting", "Neural Targeting Head")

        # Special workflow tab for the real-world data fine-tuning pipeline
        self._create_realworld_finetune_tab()

        # Bottom log area
        log_frame = ttk.LabelFrame(self, text="Training Log", padding=6)
        log_frame.pack(fill="both", expand=False, padx=8, pady=(0, 8))

        # Progress bar row (helpful extra)
        progress_frame = ttk.Frame(log_frame)
        progress_frame.pack(fill="x", pady=(0, 4))

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, maximum=100, length=400
        )
        self.progress_bar.pack(side="left", padx=(0, 8))

        self.progress_label = ttk.Label(progress_frame, text="Idle")
        self.progress_label.pack(side="left")

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=16, font=("Consolas", 9), wrap="word", state="disabled"
        )
        self.log_text.pack(fill="both", expand=True)

        btns = ttk.Frame(log_frame)
        btns.pack(fill="x", pady=(4, 0))
        ttk.Button(btns, text="Clear Log", command=self._clear_log).pack(side="left")
        ttk.Button(btns, text="Show Training Log", command=self._show_training_log_dialog).pack(side="left", padx=6)
        ttk.Button(btns, text="Stop Current Training", command=self._stop_training).pack(side="left", padx=6)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(btns, textvariable=self.status_var).pack(side="right")

        # Apply dark theme after all widgets (including progress bar and log) exist
        self._apply_dark_theme()

        # Gentle startup dependency check (non-blocking)
        self.after(1500, lambda: self._startup_dependency_check())

    def _show_training_log_dialog(self, raise_window: bool = True):
        if self.log_window is not None:
            try:
                if self.log_window.winfo_exists():
                    self.log_window.deiconify()
                    if raise_window:
                        self._raise_log_window()
                    return
            except tk.TclError:
                self.log_window = None
                self.log_dialog_text = None

        win = tk.Toplevel(self)
        win.title("Training Log")
        win.geometry("900x520")
        win.minsize(650, 320)
        win.protocol("WM_DELETE_WINDOW", self._hide_training_log_dialog)

        frame = ttk.Frame(win, padding=8)
        frame.pack(fill="both", expand=True)

        self.log_dialog_text = scrolledtext.ScrolledText(
            frame,
            height=24,
            font=("Consolas", 9),
            wrap="word",
            state="disabled",
        )
        self.log_dialog_text.pack(fill="both", expand=True)
        self.log_dialog_text.configure(
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="#d4d4d4",
            selectbackground="#007acc",
            selectforeground="white",
        )

        button_row = ttk.Frame(frame)
        button_row.pack(fill="x", pady=(6, 0))
        ttk.Button(button_row, text="Clear Log", command=self._clear_log).pack(side="left")
        ttk.Button(button_row, text="Hide", command=self._hide_training_log_dialog).pack(side="right")

        self.log_window = win
        if hasattr(self, "log_text"):
            existing = self.log_text.get("1.0", "end-1c")
            if existing:
                self._write_log_widget(self.log_dialog_text, existing)

        if raise_window:
            self._raise_log_window()

    def _hide_training_log_dialog(self):
        if self.log_window is None:
            return
        try:
            self.log_window.withdraw()
        except tk.TclError:
            self.log_window = None
            self.log_dialog_text = None

    def _raise_log_window(self):
        if self.log_window is None:
            return
        try:
            self.log_window.lift()
            self.log_window.focus_force()
            self.log_window.attributes("-topmost", True)
            self.log_window.after(300, lambda: self.log_window.attributes("-topmost", False))
        except tk.TclError:
            pass

    def _startup_dependency_check(self):
        """Lightweight check on launch so user knows if something important is missing."""
        missing = []
        for pkg, imp in [("torch", "torch"), ("numpy", "numpy")]:
            try:
                __import__(imp)
            except ImportError:
                missing.append(pkg)

        if missing:
            self._append_log(f"[Startup] Note: Missing recommended packages: {', '.join(missing)}\n")
            self._append_log("You can install them from the Environment Check dialog or when starting training.\n\n")

    def _create_tab(self, key: str, title: str):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text=title)

        defaults = DEFAULTS[key]
        entries: Dict[str, tk.Variable] = {}

        # Two-column layout
        left = ttk.Frame(frame)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right = ttk.Frame(frame)
        right.pack(side="left", fill="y")

        row = 0
        for param, default in defaults.items():
            var = self._make_var(default)
            entries[param] = var

            lbl = ttk.Label(left, text=param.replace("_", " ").title() + ":")
            lbl.grid(row=row, column=0, sticky="w", pady=2)

            if isinstance(default, bool):
                w = ttk.Checkbutton(left, variable=var)
            elif param == "model_type":
                w = ttk.Combobox(left, textvariable=var, values=["gru", "transformer"], width=18, state="readonly")
            elif param == "device":
                w = ttk.Combobox(left, textvariable=var, values=["auto", "cpu", "cuda"], width=10, state="readonly")
            else:
                w = ttk.Entry(left, textvariable=var, width=42)
            w.grid(row=row, column=1, sticky="we", pady=2, padx=4)
            row += 1

        # Train button on the right
        train_btn = ttk.Button(
            right, text=f"▶ Start {title.split('(')[0].strip()} Training",
            command=lambda k=key, e=entries: self._start_training(k, e)
        )
        train_btn.pack(pady=20, fill="x")

        ttk.Label(right, text="Tip:", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tips = {
            "neural_tracker": "Enable GBM distillation for best association quality.",
            "pid_governor": "GBM distillation is strongly recommended. Use the button below to regenerate the dataset after code changes.",
            "temporal": "GRU is usually the best balance of speed and accuracy.",
            "neural_targeting": "Keep max_refinement_px in sync with runtime config.",
        }
        ttk.Label(right, text=tips.get(key, ""), wraplength=220, justify="left").pack(anchor="w", pady=(2, 12))

        ttk.Button(right, text="Reset to Suggested Defaults",
                   command=lambda k=key, e=entries: self._reset_defaults(k, e)).pack(fill="x", pady=4)

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=6)
        ttk.Button(right, text="Save Preset", command=self._save_preset).pack(fill="x", pady=2)
        ttk.Button(right, text="Load Preset", command=self._load_preset).pack(fill="x", pady=2)

        if key in ("temporal", "neural_targeting"):
            ttk.Separator(right, orient="horizontal").pack(fill="x", pady=8)
            ttk.Label(right, text="Real-World Data", font=("Segoe UI", 9, "bold")).pack(anchor="w")
            ttk.Button(right, text="Open Real-World Fine-Tune Tab",
                       command=lambda: self.notebook.select(self._find_realworld_tab_index())).pack(fill="x", pady=2)

        if key == "pid_governor":
            ttk.Separator(right, orient="horizontal").pack(fill="x", pady=6)
            ttk.Button(right, text="Regenerate PID Dataset",
                       command=self._regenerate_pid_dataset).pack(fill="x", pady=2)

        self.tabs[key] = {"frame": frame, "entries": entries}

    def _make_var(self, default: Any) -> tk.Variable:
        if isinstance(default, bool):
            return tk.BooleanVar(value=default)
        if isinstance(default, int):
            return tk.IntVar(value=default)
        if isinstance(default, float):
            return tk.DoubleVar(value=default)
        return tk.StringVar(value=str(default))

    def _reset_defaults(self, key: str, entries: Dict[str, tk.Variable]):
        for name, var in entries.items():
            val = DEFAULTS[key][name]
            if isinstance(var, tk.BooleanVar):
                var.set(bool(val))
            else:
                var.set(val)

    def _start_training(self, key: str, entries: Dict[str, tk.Variable]):
        if self.current_process and self.current_process.poll() is None:
            messagebox.showwarning("Training running", "A training process is already running.")
            return

        self._show_training_log_dialog()

        # Auto-install critical dependencies if missing (helpful extra)
        if key in ("neural_tracker", "pid_governor"):
            use_gbm = entries.get("use_gbm_teacher")
            if use_gbm and use_gbm.get():
                ensure_package("lightgbm", install_name="lightgbm")

        # Core packages
        ensure_package("torch")
        ensure_package("numpy")

        script_map = {
            "neural_tracker": "train_neural_tracker.py",
            "pid_governor": "train_pid_governor.py",
            "temporal": "train_temporal.py",
            "neural_targeting": "train_neural_targeting.py",
        }
        script = REPO_ROOT / "training" / script_map[key]

        cmd = [sys.executable, "-u", str(script)]

        for name, var in entries.items():
            flag = name.replace("_", "-")   # argparse uses kebab-case (e.g. --hidden-size)
            val = var.get()
            if isinstance(val, bool):
                if val:
                    cmd.append(f"--{flag}")
            else:
                cmd.append(f"--{flag}")
                cmd.append(str(val))

        self._append_log(f"\n{'='*70}\n")
        self._append_log(f"Starting: {key}\nCommand: {' '.join(cmd)}\n{'='*70}\n\n")
        self.status_var.set(f"Training {key} ...")

        # Remember for post-training actions (auto-open folder, etc.)
        self.last_trainer_key = key
        self.last_output_path = self._guess_output_path(key, entries)

        # Reset progress bar and capture max epochs (helpful extra)
        self.current_epoch = 0
        self.progress_var.set(0)
        try:
            epochs_val = entries.get("epochs")
            self.current_max_epochs = int(epochs_val.get()) if epochs_val else 0
        except:
            self.current_max_epochs = 0

        if self.current_max_epochs > 0:
            self.progress_label.config(text=f"Epoch 0/{self.current_max_epochs}")
        else:
            self.progress_label.config(text="Running...")

        # Launch
        try:
            self.current_process = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except Exception as e:
            self._append_log(f"Failed to start process: {e}\n")
            self.status_var.set("Error starting training")
            return

        # Stream in background thread
        threading.Thread(target=self._stream_output, args=(self.current_process,), daemon=True).start()

    def _stream_output(self, proc: subprocess.Popen):
        try:
            for line in proc.stdout:
                self.log_queue.put(line)
        except Exception as e:
            self.log_queue.put(f"[stream error] {e}\n")
        finally:
            code = proc.wait()
            self.log_queue.put(f"\n[Process finished with exit code {code}]\n")
            self.log_queue.put((LOG_EVENT_PROCESS_FINISHED, code))

    def _poll_log_queue(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                self._handle_log_queue_item(item)
        except queue.Empty:
            pass
        self.after(80, self._poll_log_queue)

    def _handle_log_queue_item(self, item):
        if isinstance(item, tuple) and item:
            event = item[0]
            if event == LOG_EVENT_PROCESS_FINISHED:
                self._handle_process_finished(int(item[1]))
                return
            if event == LOG_EVENT_REALWORLD_FINISHED:
                self._handle_realworld_process_finished(int(item[1]), bool(item[2]))
                return

        self._append_log(str(item))

    def _handle_process_finished(self, code: int):
        if code == 0:
            self.status_var.set("Finished successfully")
        elif code == 2:
            self.status_var.set("Failed (argument error - see log above)")
        else:
            self.status_var.set(f"Failed (exit {code})")

        # Helpful extra: auto-open the output folder on success.
        self.after(600, lambda c=code: self._maybe_auto_open_output(c))

        # Reset progress UI after a short delay.
        if code == 0:
            self.after(1500, self._reset_progress_ui)

    def _handle_realworld_process_finished(self, code: int, chain_next: bool):
        if code == 0 and chain_next:
            self.after(400, self._execute_pending_next)
            return

        if code == 0:
            self.status_var.set("Finished successfully")
            self.after(1200, self._reset_progress_ui)
        else:
            self.status_var.set(f"Failed (exit {code})")
            self._pending_next_cmd = None
            self._pending_pipeline = None

    def _parse_epoch_progress(self, text: str):
        """Very simple epoch parser for progress bar."""
        import re
        match = re.search(r"epoch=(\d+)", text, re.IGNORECASE)
        if not match:
            return

        try:
            epoch_num = int(match.group(1))
        except:
            return

        self.current_epoch = max(self.current_epoch, epoch_num)

        if self.current_max_epochs > 0:
            pct = min(100, (self.current_epoch / self.current_max_epochs) * 100)
            self.progress_var.set(pct)
            self.progress_label.config(
                text=f"Epoch {self.current_epoch}/{self.current_max_epochs}  ({pct:.0f}%)"
            )
        else:
            self.progress_label.config(text=f"Epoch {self.current_epoch}")

    def _append_log(self, text: str):
        self._write_log_widget(self.log_text, text)
        if self.log_dialog_text is not None:
            self._write_log_widget(self.log_dialog_text, text)

        # Parse epoch progress (helpful extra)
        self._parse_epoch_progress(text)

    def _write_log_widget(self, widget, text: str):
        try:
            widget.configure(state="normal")
            widget.insert("end", text)
            widget.see("end")
            widget.configure(state="disabled")
        except tk.TclError:
            if widget is self.log_dialog_text:
                self.log_dialog_text = None

    def _clear_log(self):
        for widget in (getattr(self, "log_text", None), getattr(self, "log_dialog_text", None)):
            if widget is None:
                continue
            try:
                widget.configure(state="normal")
                widget.delete("1.0", "end")
                widget.configure(state="disabled")
            except tk.TclError:
                if widget is self.log_dialog_text:
                    self.log_dialog_text = None

    def _stop_training(self):
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()
            self._append_log("\n[Requested stop]\n")
            self.status_var.set("Stopping...")
        else:
            self.status_var.set("No training running")

    def _open_models_folder(self):
        folder = REPO_ROOT / "training" / "models"
        folder.mkdir(parents=True, exist_ok=True)
        self._open_folder(folder)

    def _open_runs_folder(self):
        folder = REPO_ROOT / "training" / "runs"
        folder.mkdir(parents=True, exist_ok=True)
        self._open_folder(folder)

    def _open_folder(self, path: Path):
        try:
            if os.name == "nt":
                os.startfile(str(path))
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder: {e}")

    def _guess_output_path(self, key: str, entries: Dict[str, tk.Variable]) -> Optional[Path]:
        """Best-effort guess of the main output directory/file for this trainer."""
        try:
            if key in ("neural_tracker", "pid_governor"):
                out = entries.get("output")
                if out:
                    p = Path(out.get())
                    return p.parent if p.suffix else p
            elif key in ("temporal", "neural_targeting"):
                out = entries.get("output")
                if out:
                    return Path(out.get()).parent
            # Fallbacks
            if key == "pid_governor":
                return REPO_ROOT / "training" / "models"
            return REPO_ROOT / "training" / "models"
        except Exception:
            return REPO_ROOT / "training" / "models"

    def _launch_tensorboard(self):
        """Launch TensorBoard pointing at the common runs directory."""
        runs_dir = REPO_ROOT / "training" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Try to launch tensorboard in a new console so it doesn't block the GUI
            cmd = [sys.executable, "-m", "tensorboard", "--logdir", str(runs_dir)]
            if os.name == "nt":
                subprocess.Popen(["cmd", "/c", "start", "cmd", "/k"] + cmd, cwd=str(REPO_ROOT))
            else:
                subprocess.Popen(cmd, cwd=str(REPO_ROOT))
            self._append_log(f"[TensorBoard] Launched for: {runs_dir}\n")
        except Exception as e:
            messagebox.showerror(
                "TensorBoard Error",
                f"Could not launch TensorBoard.\nMake sure tensorboard is installed.\n\nError: {e}"
            )

    def _maybe_auto_open_output(self, exit_code: int):
        """If training succeeded, open the most relevant output folder."""
        if exit_code != 0:
            return
        if not self.last_output_path:
            return

        try:
            self.last_output_path.mkdir(parents=True, exist_ok=True)
            self._open_folder(self.last_output_path)
            self._append_log(f"[Auto-open] Opened output folder: {self.last_output_path}\n")
        except Exception as e:
            self._append_log(f"[Auto-open] Failed to open folder: {e}\n")

    def _reset_progress_ui(self):
        self.progress_var.set(0)
        self.progress_label.config(text="Idle")
        self.current_max_epochs = 0
        self.current_epoch = 0

    def _regenerate_pid_dataset(self):
        """Regenerate the PID Governor training dataset using the local generator."""
        if self.current_process and self.current_process.poll() is None:
            messagebox.showwarning("Process running", "Another process is already running.")
            return

        self._show_training_log_dialog()

        gen_script = REPO_ROOT / "training" / "generate_pid_dataset.py"
        output_path = REPO_ROOT / "training" / "data" / "pid_governor_dataset.csv"

        # Try to find a reasonable config.ini
        config_candidates = [
            REPO_ROOT / "x64" / "DML" / "config.ini",
            REPO_ROOT / "x64" / "CUDA" / "config.ini",
            REPO_ROOT / "config.ini",
        ]
        config_path = None
        for cand in config_candidates:
            if cand.exists():
                config_path = cand
                break

        if not config_path:
            config_path = filedialog.askopenfilename(
                title="Select config.ini for PID generation",
                filetypes=[("INI files", "*.ini")],
                initialdir=str(REPO_ROOT)
            )
            if not config_path:
                return
            config_path = Path(config_path)

        cmd = [
            sys.executable, "-u", str(gen_script),
            "--config", str(config_path),
            "--output", str(output_path),
            "--episodes-per-profile", "32",
            "--steps-per-episode", "120",
            "--seed", "42"
        ]

        self._append_log(f"\n{'='*70}\n")
        self._append_log("Starting PID Governor dataset regeneration...\n")
        self._append_log(f"Using config: {config_path}\n")
        self._append_log(f"Command: {' '.join(cmd)}\n{'='*70}\n\n")
        self.status_var.set("Regenerating PID dataset...")

        try:
            self.current_process = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except Exception as e:
            self._append_log(f"Failed to start generator: {e}\n")
            self.status_var.set("Error starting regeneration")
            return

        threading.Thread(target=self._stream_output, args=(self.current_process,), daemon=True).start()

    def _show_environment_check(self):
        """Show a small dialog with availability of important packages (helpful extra)."""
        win = tk.Toplevel(self)
        win.title("Environment Check")
        win.geometry("420x320")
        win.resizable(False, False)

        try:
            win.configure(bg="#1e1e1e")
        except:
            pass

        frame = ttk.Frame(win, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Package Status", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 8))

        checks = []

        # Python
        checks.append(("Python", f"{sys.version.split()[0]}", True))

        # torch
        try:
            import torch
            cuda = torch.cuda.is_available()
            ver = torch.__version__
            extra = f" + CUDA {torch.version.cuda}" if cuda else " (CPU only)"
            checks.append(("torch", f"{ver}{extra}", True))
        except Exception as e:
            checks.append(("torch", str(e), False))

        # lightgbm (important for new distillation features)
        try:
            import lightgbm
            checks.append(("lightgbm", lightgbm.__version__, True))
        except Exception:
            checks.append(("lightgbm", "Not installed (recommended for distillation)", False))

        # numpy
        try:
            import numpy
            checks.append(("numpy", numpy.__version__, True))
        except Exception:
            checks.append(("numpy", "Not installed", False))

        # onnx (nice to have)
        try:
            import onnx
            checks.append(("onnx", onnx.__version__, True))
        except Exception:
            checks.append(("onnx", "Not installed (optional)", False))

        for name, status, ok in checks:
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=2)
            color = "#4ec9b0" if ok else "#f48771"
            ttk.Label(row, text=f"● {name}:", foreground=color, width=14).pack(side="left")
            ttk.Label(row, text=status).pack(side="left", fill="x", expand=True)

            if not ok and name in ("lightgbm", "numpy", "torch"):
                install_name = name
                if name == "lightgbm":
                    install_name = "lightgbm"
                btn = ttk.Button(row, text="Install", width=8,
                                 command=lambda p=install_name, w=win: self._install_from_dialog(p, w))
                btn.pack(side="right")

        ttk.Button(frame, text="Close", command=win.destroy).pack(pady=12)

    def _install_from_dialog(self, package: str, parent_window):
        """Install a package from the environment dialog and refresh it."""
        parent_window.destroy()
        if ensure_package(package):
            # Re-open the dialog after successful install
            self.after(300, self._show_environment_check)

    # ------------------------------------------------------------------
    # Preset helpers (Save / Load current tab settings)
    # ------------------------------------------------------------------
    def _get_current_tab_key(self) -> Optional[str]:
        try:
            tab_id = self.notebook.select()
            for key, data in self.tabs.items():
                if str(data["frame"]) == tab_id:
                    return key
        except Exception:
            pass
        return None

    def _save_preset(self):
        key = self._get_current_tab_key()
        if not key:
            return

        entries = self.tabs[key]["entries"]
        data = {}
        for name, var in entries.items():
            data[name] = var.get()

        preset_dir = REPO_ROOT / "training" / "presets"
        preset_dir.mkdir(parents=True, exist_ok=True)

        filename = filedialog.asksaveasfilename(
            initialdir=str(preset_dir),
            defaultextension=".json",
            filetypes=[("JSON Preset", "*.json")],
            title=f"Save preset for {key}"
        )
        if not filename:
            return

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump({"trainer": key, "values": data}, f, indent=2)
            self._append_log(f"[Preset] Saved: {filename}\n")
        except Exception as e:
            messagebox.showerror("Save Preset Error", str(e))

    def _load_preset(self):
        key = self._get_current_tab_key()
        if not key:
            return

        preset_dir = REPO_ROOT / "training" / "presets"
        preset_dir.mkdir(parents=True, exist_ok=True)

        filename = filedialog.askopenfilename(
            initialdir=str(preset_dir),
            filetypes=[("JSON Preset", "*.json")],
            title="Load preset"
        )
        if not filename:
            return

        try:
            with open(filename, "r", encoding="utf-8") as f:
                payload = json.load(f)

            if payload.get("trainer") != key:
                if not messagebox.askyesno(
                    "Preset Mismatch",
                    f"This preset was saved for '{payload.get('trainer')}'.\n"
                    f"Load anyway into current tab '{key}'?"
                ):
                    return

            values = payload.get("values", {})
            entries = self.tabs[key]["entries"]
            for name, var in entries.items():
                if name in values:
                    val = values[name]
                    if isinstance(var, tk.BooleanVar):
                        var.set(bool(val))
                    else:
                        var.set(val)

            self._append_log(f"[Preset] Loaded: {filename}\n")
        except Exception as e:
            messagebox.showerror("Load Preset Error", str(e))

    # ------------------------------------------------------------------
    # Real-World Fine-Tuning Tab (the big one from the conversation)
    # ------------------------------------------------------------------
    def _create_realworld_finetune_tab(self):
        """Dedicated rich tab with fully editable options for the convert + fine-tune workflow."""
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Real-World Fine-Tune")

        # Store references so we can read values later
        self.rw_vars: Dict[str, tk.Variable] = {}

        # Helper to create labeled row
        def add_row(parent, label, default, key, width=48):
            var = self._make_var(default)
            self.rw_vars[key] = var
            ttk.Label(parent, text=label + ":").grid(row=parent.grid_size()[1], column=0, sticky="w", pady=3)
            entry = ttk.Entry(parent, textvariable=var, width=width)
            entry.grid(row=parent.grid_size()[1]-1, column=1, sticky="we", pady=3, padx=6)
            return var

        # === Top controls: Logs + Convert ===
        convert_frame = ttk.LabelFrame(frame, text="Step 1: Convert Real-World Logs", padding=8)
        convert_frame.pack(fill="x", pady=(0, 10))

        rowf = ttk.Frame(convert_frame)
        rowf.pack(fill="x")
        add_row(rowf, "Raw logs directory", "training/datasets/real_world", "rw_logs_dir", 52)

        rowf2 = ttk.Frame(convert_frame)
        rowf2.pack(fill="x")
        add_row(rowf2, "Output: Temporal dataset (.npz)", "training/datasets/temporal_tracks_realworld.npz", "rw_temporal_out", 52)

        rowf3 = ttk.Frame(convert_frame)
        rowf3.pack(fill="x")
        add_row(rowf3, "Output: Targeting dataset (.npz)", "training/datasets/neural_targeting_tracks_realworld.npz", "rw_targeting_out", 52)

        rowf4 = ttk.Frame(convert_frame)
        rowf4.pack(fill="x")
        self.rw_vars["rw_mix_ratio"] = self._make_var(0.65)
        ttk.Label(rowf4, text="Real-world mix ratio (0.0-1.0):").grid(row=0, column=0, sticky="w")
        mix_entry = ttk.Entry(rowf4, textvariable=self.rw_vars["rw_mix_ratio"], width=8)
        mix_entry.grid(row=0, column=1, padx=6)
        ttk.Label(rowf4, text="(higher = more real data, 0.6-0.7 recommended)").grid(row=0, column=2, sticky="w", padx=8)

        # Synthetic mix options (editable)
        synth_row = ttk.Frame(convert_frame)
        synth_row.pack(fill="x", pady=(6, 0))
        self.rw_vars["rw_use_synth_mix"] = self._make_var(True)
        ttk.Checkbutton(synth_row, text="Mix with synthetic data during conversion", variable=self.rw_vars["rw_use_synth_mix"]).pack(side="left")

        self.rw_vars["rw_synth_temporal"] = self._make_var("training/datasets/temporal_tracks.npz")
        ttk.Entry(synth_row, textvariable=self.rw_vars["rw_synth_temporal"], width=42).pack(side="left", padx=4)
        ttk.Label(synth_row, text="Synthetic temporal").pack(side="left")

        synth_row2 = ttk.Frame(convert_frame)
        synth_row2.pack(fill="x")
        self.rw_vars["rw_synth_targeting"] = self._make_var("training/datasets/neural_targeting_tracks.npz")
        ttk.Entry(synth_row2, textvariable=self.rw_vars["rw_synth_targeting"], width=42).pack(side="left", padx=4)
        ttk.Label(synth_row2, text="Synthetic targeting").pack(side="left")

        ttk.Button(convert_frame, text="▶ Run Convert Only",
                   command=self._run_realworld_convert).pack(pady=8, anchor="w")

        # === Fine-tune Temporal ===
        temp_frame = ttk.LabelFrame(frame, text="Step 2: Fine-Tune Temporal Predictor", padding=8)
        temp_frame.pack(fill="x", pady=(0, 10))

        trow = ttk.Frame(temp_frame)
        trow.pack(fill="x")
        self.rw_vars["rw_temp_checkpoint"] = self._make_var("neural_models/temporal_predictor.pt")
        ttk.Label(trow, text="Base checkpoint:").pack(side="left")
        ttk.Entry(trow, textvariable=self.rw_vars["rw_temp_checkpoint"], width=44).pack(side="left", padx=4)

        trow2 = ttk.Frame(temp_frame)
        trow2.pack(fill="x", pady=3)
        self.rw_vars["rw_temp_epochs"] = self._make_var(25)
        ttk.Label(trow2, text="Epochs:").pack(side="left")
        ttk.Entry(trow2, textvariable=self.rw_vars["rw_temp_epochs"], width=6).pack(side="left", padx=4)

        self.rw_vars["rw_temp_lr"] = self._make_var(5e-5)
        ttk.Label(trow2, text="Learning rate:").pack(side="left", padx=(12, 0))
        ttk.Entry(trow2, textvariable=self.rw_vars["rw_temp_lr"], width=10).pack(side="left", padx=4)

        self.rw_vars["rw_temp_train_mix"] = self._make_var(0.65)
        ttk.Label(trow2, text="Train mix ratio:").pack(side="left", padx=(12, 0))
        ttk.Entry(trow2, textvariable=self.rw_vars["rw_temp_train_mix"], width=8).pack(side="left", padx=4)

        trow3 = ttk.Frame(temp_frame)
        trow3.pack(fill="x")
        self.rw_vars["rw_temp_out_pt"] = self._make_var("neural_models/temporal_predictor_realworld.pt")
        ttk.Label(trow3, text="Output .pt:").pack(side="left")
        ttk.Entry(trow3, textvariable=self.rw_vars["rw_temp_out_pt"], width=44).pack(side="left", padx=4)

        trow4 = ttk.Frame(temp_frame)
        trow4.pack(fill="x")
        self.rw_vars["rw_temp_out_onnx"] = self._make_var("neural_models/temporal_predictor_realworld.onnx")
        ttk.Label(trow4, text="Output .onnx:").pack(side="left")
        ttk.Entry(trow4, textvariable=self.rw_vars["rw_temp_out_onnx"], width=44).pack(side="left", padx=4)

        ttk.Button(temp_frame, text="▶ Fine-Tune Temporal Only",
                   command=self._run_realworld_finetune_temporal).pack(pady=6, anchor="w")

        # === Fine-tune Targeting ===
        targ_frame = ttk.LabelFrame(frame, text="Step 3: Fine-Tune Neural Targeting Head", padding=8)
        targ_frame.pack(fill="x", pady=(0, 10))

        tarow = ttk.Frame(targ_frame)
        tarow.pack(fill="x")
        self.rw_vars["rw_targ_checkpoint"] = self._make_var("neural_models/neural_targeting_head.pt")
        ttk.Label(tarow, text="Base checkpoint:").pack(side="left")
        ttk.Entry(tarow, textvariable=self.rw_vars["rw_targ_checkpoint"], width=44).pack(side="left", padx=4)

        tarow2 = ttk.Frame(targ_frame)
        tarow2.pack(fill="x", pady=3)
        self.rw_vars["rw_targ_epochs"] = self._make_var(20)
        ttk.Label(tarow2, text="Epochs:").pack(side="left")
        ttk.Entry(tarow2, textvariable=self.rw_vars["rw_targ_epochs"], width=6).pack(side="left", padx=4)

        self.rw_vars["rw_targ_lr"] = self._make_var(5e-5)
        ttk.Label(tarow2, text="Learning rate:").pack(side="left", padx=(12, 0))
        ttk.Entry(tarow2, textvariable=self.rw_vars["rw_targ_lr"], width=10).pack(side="left", padx=4)

        self.rw_vars["rw_targ_train_mix"] = self._make_var(0.60)
        ttk.Label(tarow2, text="Train mix ratio:").pack(side="left", padx=(12, 0))
        ttk.Entry(tarow2, textvariable=self.rw_vars["rw_targ_train_mix"], width=8).pack(side="left", padx=4)

        tarow3 = ttk.Frame(targ_frame)
        tarow3.pack(fill="x")
        self.rw_vars["rw_targ_out_pt"] = self._make_var("neural_models/neural_targeting_head_realworld.pt")
        ttk.Label(tarow3, text="Output .pt:").pack(side="left")
        ttk.Entry(tarow3, textvariable=self.rw_vars["rw_targ_out_pt"], width=44).pack(side="left", padx=4)

        tarow4 = ttk.Frame(targ_frame)
        tarow4.pack(fill="x")
        self.rw_vars["rw_targ_out_onnx"] = self._make_var("neural_models/neural_targeting_head_realworld.onnx")
        ttk.Label(tarow4, text="Output .onnx:").pack(side="left")
        ttk.Entry(tarow4, textvariable=self.rw_vars["rw_targ_out_onnx"], width=44).pack(side="left", padx=4)

        ttk.Button(targ_frame, text="▶ Fine-Tune Targeting Only",
                   command=self._run_realworld_finetune_targeting).pack(pady=6, anchor="w")

        # === Big pipeline buttons ===
        action_frame = ttk.Frame(frame)
        action_frame.pack(fill="x", pady=(12, 4))

        ttk.Button(action_frame, text="▶▶ Run Full Pipeline (Convert + Both Fine-Tunes)",
                   command=self._run_full_realworld_pipeline).pack(side="left", padx=4, pady=4)

        ttk.Button(action_frame, text="Run Convert + Temporal",
                   command=lambda: self._run_realworld_convert_then("temporal")).pack(side="left", padx=4)

        ttk.Button(action_frame, text="Run Convert + Targeting",
                   command=lambda: self._run_realworld_convert_then("targeting")).pack(side="left", padx=4)

        ttk.Button(action_frame, text="Reset to Recommended Defaults",
                   command=self._reset_realworld_defaults).pack(side="right", padx=4)

        # Helpful note
        note = ttk.Label(frame, text="After training, copy the new *_realworld.onnx files into your game config (Neural tab → Real-World Data section).",
                         foreground="#888888", wraplength=900)
        note.pack(anchor="w", pady=(8, 0))

    def _reset_realworld_defaults(self):
        defaults = {
            "rw_logs_dir": "training/datasets/real_world",
            "rw_temporal_out": "training/datasets/temporal_tracks_realworld.npz",
            "rw_targeting_out": "training/datasets/neural_targeting_tracks_realworld.npz",
            "rw_mix_ratio": 0.65,
            "rw_use_synth_mix": True,
            "rw_synth_temporal": "training/datasets/temporal_tracks.npz",
            "rw_synth_targeting": "training/datasets/neural_targeting_tracks.npz",
            "rw_temp_checkpoint": "neural_models/temporal_predictor.pt",
            "rw_temp_epochs": 25,
            "rw_temp_lr": 5e-5,
            "rw_temp_train_mix": 0.65,
            "rw_temp_out_pt": "neural_models/temporal_predictor_realworld.pt",
            "rw_temp_out_onnx": "neural_models/temporal_predictor_realworld.onnx",
            "rw_targ_checkpoint": "neural_models/neural_targeting_head.pt",
            "rw_targ_epochs": 20,
            "rw_targ_lr": 5e-5,
            "rw_targ_train_mix": 0.60,
            "rw_targ_out_pt": "neural_models/neural_targeting_head_realworld.pt",
            "rw_targ_out_onnx": "neural_models/neural_targeting_head_realworld.onnx",
        }
        for k, v in defaults.items():
            if k in self.rw_vars:
                self.rw_vars[k].set(v)
        self._append_log("[Real-World] Reset to recommended defaults.\n")

    def _build_convert_cmd(self) -> list:
        cmd = [sys.executable, "-u", "training/data_gen/convert_real_world_logs.py",
               "--input", self.rw_vars["rw_logs_dir"].get(),
               "--temporal-output", self.rw_vars["rw_temporal_out"].get(),
               "--targeting-output", self.rw_vars["rw_targeting_out"].get(),
               "--mix-ratio", str(self.rw_vars["rw_mix_ratio"].get())]

        if self.rw_vars["rw_use_synth_mix"].get():
            cmd += ["--synthetic-temporal", self.rw_vars["rw_synth_temporal"].get(),
                    "--synthetic-targeting", self.rw_vars["rw_synth_targeting"].get()]
        return cmd

    def _build_finetune_temporal_cmd(self) -> list:
        return [
            sys.executable, "-u", "training/train_temporal.py",
            "--fine-tune",
            "--checkpoint", self.rw_vars["rw_temp_checkpoint"].get(),
            "--real-world-data", self.rw_vars["rw_temporal_out"].get(),
            "--mix-ratio", str(self.rw_vars["rw_temp_train_mix"].get()),
            "--epochs", str(int(self.rw_vars["rw_temp_epochs"].get())),
            "--learning-rate", str(self.rw_vars["rw_temp_lr"].get()),
            "--output", self.rw_vars["rw_temp_out_pt"].get(),
            "--onnx-output", self.rw_vars["rw_temp_out_onnx"].get(),
        ]

    def _build_finetune_targeting_cmd(self) -> list:
        return [
            sys.executable, "-u", "training/train_neural_targeting.py",
            "--fine-tune",
            "--checkpoint", self.rw_vars["rw_targ_checkpoint"].get(),
            "--real-world-data", self.rw_vars["rw_targeting_out"].get(),
            "--mix-ratio", str(self.rw_vars["rw_targ_train_mix"].get()),
            "--epochs", str(int(self.rw_vars["rw_targ_epochs"].get())),
            "--learning-rate", str(self.rw_vars["rw_targ_lr"].get()),
            "--output", self.rw_vars["rw_targ_out_pt"].get(),
            "--onnx-output", self.rw_vars["rw_targ_out_onnx"].get(),
        ]

    def _run_realworld_convert(self):
        if self.current_process and self.current_process.poll() is None:
            messagebox.showwarning("Busy", "Another process is already running.")
            return
        cmd = self._build_convert_cmd()
        self._launch_rw_command(cmd, "Real-World Convert", max_epochs=None)

    def _run_realworld_finetune_temporal(self):
        if self.current_process and self.current_process.poll() is None:
            messagebox.showwarning("Busy", "Another process is already running.")
            return
        # Ensure convert output exists as a sanity hint
        out_path = Path(self.rw_vars["rw_temporal_out"].get())
        if not out_path.exists():
            self._append_log(f"[Warning] Converted dataset not found at {out_path}. Run convert first.\n")
        cmd = self._build_finetune_temporal_cmd()
        epochs = int(self.rw_vars["rw_temp_epochs"].get())
        self.last_output_path = Path(self.rw_vars["rw_temp_out_pt"].get()).parent
        self._launch_rw_command(cmd, "Temporal Fine-Tune (Real-World)", max_epochs=epochs)

    def _run_realworld_finetune_targeting(self):
        if self.current_process and self.current_process.poll() is None:
            messagebox.showwarning("Busy", "Another process is already running.")
            return
        out_path = Path(self.rw_vars["rw_targeting_out"].get())
        if not out_path.exists():
            self._append_log(f"[Warning] Converted dataset not found at {out_path}. Run convert first.\n")
        cmd = self._build_finetune_targeting_cmd()
        epochs = int(self.rw_vars["rw_targ_epochs"].get())
        self.last_output_path = Path(self.rw_vars["rw_targ_out_pt"].get()).parent
        self._launch_rw_command(cmd, "Targeting Fine-Tune (Real-World)", max_epochs=epochs)

    def _run_realworld_convert_then(self, which: str):
        """Run convert, then immediately chain the requested fine-tune."""
        if self.current_process and self.current_process.poll() is None:
            messagebox.showwarning("Busy", "Another process is already running.")
            return
        convert_cmd = self._build_convert_cmd()
        if which == "temporal":
            next_cmd = self._build_finetune_temporal_cmd()
            label = "Convert + Temporal Fine-Tune"
        else:
            next_cmd = self._build_finetune_targeting_cmd()
            label = "Convert + Targeting Fine-Tune"

        self._append_log(f"\n{'='*70}\nStarting chained pipeline: {label}\n{'='*70}\n")
        # Launch convert; on success we will auto-launch the next one via a small hack in the stream handler
        self._pending_next_cmd = next_cmd
        self._pending_next_label = label
        self._launch_rw_command(convert_cmd, label, chain_next=True)  # convert has no epochs

    def _run_full_realworld_pipeline(self):
        """Run convert → temporal fine-tune → targeting fine-tune in sequence."""
        if self.current_process and self.current_process.poll() is None:
            messagebox.showwarning("Busy", "Another process is already running.")
            return

        convert_cmd = self._build_convert_cmd()
        temp_cmd = self._build_finetune_temporal_cmd()
        targ_cmd = self._build_finetune_targeting_cmd()

        self._append_log(f"\n{'='*70}\nStarting FULL REAL-WORLD PIPELINE\n{'='*70}\n")
        self._pending_pipeline = [temp_cmd, targ_cmd]
        self._pending_pipeline_label = "Full Real-World Fine-Tuning Pipeline"
        self._launch_rw_command(convert_cmd, "Full Pipeline (Convert)", chain_next=True)  # first step is convert

    def _launch_rw_command(self, cmd: list, label: str, chain_next: bool = False, max_epochs: int | None = None):
        self._show_training_log_dialog()
        self._append_log(f"\n{'='*70}\n{label}\nCommand: {' '.join(cmd)}\n{'='*70}\n\n")
        self.status_var.set(f"Running: {label}")

        # Initialize progress UI when we know the epoch count (for fine-tune steps)
        if max_epochs is not None and max_epochs > 0:
            self.current_epoch = 0
            self.current_max_epochs = max_epochs
            self.progress_var.set(0)
            self.progress_label.config(text=f"Epoch 0/{max_epochs}")
        elif "Convert" in label:
            self.progress_var.set(0)
            self.progress_label.config(text="Converting...")

        try:
            self.current_process = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except Exception as e:
            self._append_log(f"Failed to start: {e}\n")
            self.status_var.set("Error")
            return

        # Use a wrapper stream that can chain the next command
        threading.Thread(target=self._stream_rw_output, args=(self.current_process, chain_next), daemon=True).start()

    def _stream_rw_output(self, proc: subprocess.Popen, chain_next: bool):
        try:
            for line in proc.stdout:
                self.log_queue.put(line)
        except Exception as e:
            self.log_queue.put(f"[stream error] {e}\n")
        finally:
            code = proc.wait()
            self.log_queue.put(f"\n[Process finished with exit code {code}]\n")
            self.log_queue.put((LOG_EVENT_REALWORLD_FINISHED, code, chain_next))

    def _find_realworld_tab_index(self):
        try:
            for i in range(self.notebook.index("end")):
                if "Real-World" in self.notebook.tab(i, "text"):
                    return i
        except Exception:
            pass
        return 0

    def _execute_pending_next(self):
        if getattr(self, "_pending_pipeline", None):
            # Full pipeline mode
            next_cmd = self._pending_pipeline.pop(0)
            remaining = len(self._pending_pipeline)
            label = f"Pipeline step ({3 - remaining}/3)"

            # Figure out epochs for this step
            max_ep = None
            if "train_temporal" in " ".join(next_cmd):
                max_ep = int(self.rw_vars["rw_temp_epochs"].get())
            elif "train_neural_targeting" in " ".join(next_cmd):
                max_ep = int(self.rw_vars["rw_targ_epochs"].get())

            self._append_log(f"\n[Pipeline] Launching next step: {label}\n")
            self._launch_rw_command(next_cmd, label, chain_next=remaining > 0, max_epochs=max_ep)
            if remaining == 0:
                self._pending_pipeline = None
        elif getattr(self, "_pending_next_cmd", None):
            cmd = self._pending_next_cmd
            label = getattr(self, "_pending_next_label", "Next step")
            self._pending_next_cmd = None
            self._pending_next_label = None

            # Try to infer epochs for chained fine-tunes
            max_ep = None
            if "train_temporal" in " ".join(cmd):
                max_ep = int(self.rw_vars["rw_temp_epochs"].get())
            elif "train_neural_targeting" in " ".join(cmd):
                max_ep = int(self.rw_vars["rw_targ_epochs"].get())

            self._launch_rw_command(cmd, label, max_epochs=max_ep)

if __name__ == "__main__":
    app = TrainingGUI()
    app.mainloop()
