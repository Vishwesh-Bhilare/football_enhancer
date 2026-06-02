"""
Professional desktop GUI for SceneForge.

The GUI wraps the existing OpenCV player-selection window and renderers with a
single control panel, progress bar, ETA estimator, status indicators, and live
logs. It intentionally uses only the Python standard library so it can run in
this project without adding another dependency.
"""

from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

APP_TITLE = "SceneForge Studio"
DEFAULT_SELECTION_FILE = "selection.json"
PROGRESS_RE = re.compile(r"(?:Processing\s+)?(?P<percent>\d+(?:\.\d+)?)%.*?Frame\s+(?P<frame>\d+)/(?:\s*)?(?P<total>\d+)")
FRAME_RE = re.compile(r"Frame\s+(?P<frame>\d+)(?:/(?P<total>\d+))?")


class SceneForgeGUI(tk.Tk):
    """Desktop front end for selecting players and rendering videos."""

    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1080x760")
        self.minsize(980, 680)

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar(value="result.mp4")
        self.selection_path = tk.StringVar(value=DEFAULT_SELECTION_FILE)
        self.skip_selection = tk.BooleanVar(value=False)
        self.renderer = tk.StringVar(value="render_video.py")
        self.status_text = tk.StringVar(value="Ready")
        self.progress_text = tk.StringVar(value="Idle")
        self.eta_text = tk.StringVar(value="ETA: --")
        self.elapsed_text = tk.StringVar(value="Elapsed: 00:00")
        self.selection_status = tk.StringVar(value="Selection: not checked")
        self.model_status = tk.StringVar(value="Model: not checked")

        self._process: subprocess.Popen[str] | None = None
        self._worker: threading.Thread | None = None
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._started_at: float | None = None
        self._last_percent = 0.0
        self._render_after_selection = False

        self._configure_style()
        self._build_layout()
        self._check_environment()
        self.after(100, self._drain_queue)
        self.after(1000, self._tick_elapsed)

    def _configure_style(self) -> None:
        self.configure(bg="#0f172a")
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background="#0f172a")
        style.configure("Card.TFrame", background="#111827", relief="flat")
        style.configure("Header.TLabel", background="#0f172a", foreground="#f8fafc", font=("Segoe UI", 24, "bold"))
        style.configure("Subheader.TLabel", background="#0f172a", foreground="#94a3b8", font=("Segoe UI", 11))
        style.configure("CardTitle.TLabel", background="#111827", foreground="#e5e7eb", font=("Segoe UI", 13, "bold"))
        style.configure("TLabel", background="#111827", foreground="#cbd5e1", font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background="#111827", foreground="#94a3b8", font=("Segoe UI", 9))
        style.configure("Status.TLabel", background="#111827", foreground="#38bdf8", font=("Segoe UI", 10, "bold"))
        style.configure("TButton", font=("Segoe UI", 10), padding=8)
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=10)
        style.configure("Danger.TButton", font=("Segoe UI", 10, "bold"), padding=10)
        style.configure("TCheckbutton", background="#111827", foreground="#cbd5e1", font=("Segoe UI", 10))
        style.configure("TRadiobutton", background="#111827", foreground="#cbd5e1", font=("Segoe UI", 10))
        style.configure("Horizontal.TProgressbar", troughcolor="#1f2937", background="#38bdf8", bordercolor="#1f2937", lightcolor="#38bdf8", darkcolor="#0284c7")

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=24)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 18))
        ttk.Label(header, text="SceneForge Studio", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Professional workflow for selecting players, rendering removals, and monitoring progress with ETA.",
            style="Subheader.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)

        self._build_project_card(left)
        self._build_workflow_card(left)
        self._build_status_card(left)
        self._build_progress_card(right)
        self._build_log_card(right)

    def _card(self, parent: tk.Widget, title: str) -> ttk.Frame:
        card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        card.pack(fill="x", pady=(0, 14))
        ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w", pady=(0, 12))
        return card

    def _build_project_card(self, parent: tk.Widget) -> None:
        card = self._card(parent, "Project files")
        self._path_row(card, "Input video", self.input_path, self._browse_input)
        self._path_row(card, "Output video", self.output_path, self._browse_output)
        self._path_row(card, "Selection file", self.selection_path, self._browse_selection)

    def _path_row(self, parent: tk.Widget, label: str, variable: tk.StringVar, command) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=6)
        ttk.Label(row, text=label, width=14).pack(side="left")
        entry = ttk.Entry(row, textvariable=variable)
        entry.pack(side="left", fill="x", expand=True, padx=(8, 8))
        ttk.Button(row, text="Browse", command=command).pack(side="right")

    def _build_workflow_card(self, parent: tk.Widget) -> None:
        card = self._card(parent, "Workflow")
        ttk.Checkbutton(
            card,
            text="Skip player selection and use the existing selection file",
            variable=self.skip_selection,
        ).pack(anchor="w", pady=(0, 10))

        renderer_box = ttk.LabelFrame(card, text="Render engine")
        renderer_box.pack(fill="x", pady=(0, 12))
        ttk.Radiobutton(renderer_box, text="Advanced renderer (SAM + temporal smoothing)", variable=self.renderer, value="render_video.py").pack(anchor="w", padx=10, pady=4)
        ttk.Radiobutton(renderer_box, text="Fast batch renderer", variable=self.renderer, value="batch_render.py").pack(anchor="w", padx=10, pady=4)

        buttons = ttk.Frame(card, style="Card.TFrame")
        buttons.pack(fill="x", pady=(4, 0))
        self.pipeline_button = ttk.Button(buttons, text="Run Full Workflow", command=self.run_pipeline, style="Accent.TButton")
        self.pipeline_button.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.select_button = ttk.Button(buttons, text="Selection Only", command=self.run_selection, style="TButton")
        self.select_button.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.render_button = ttk.Button(buttons, text="Render Only", command=self.run_render, style="TButton")
        self.render_button.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop_process, style="Danger.TButton", state="disabled")
        self.stop_button.pack(side="right")

    def _build_status_card(self, parent: tk.Widget) -> None:
        card = self._card(parent, "Readiness")
        ttk.Label(card, textvariable=self.selection_status, style="Status.TLabel").pack(anchor="w", pady=3)
        ttk.Label(card, textvariable=self.model_status, style="Status.TLabel").pack(anchor="w", pady=3)
        ttk.Label(card, textvariable=self.status_text, style="Muted.TLabel", wraplength=390).pack(anchor="w", pady=(8, 0))
        ttk.Button(card, text="Refresh checks", command=self._check_environment).pack(anchor="e", pady=(12, 0))

    def _build_progress_card(self, parent: tk.Widget) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        card.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        card.columnconfigure(0, weight=1)
        ttk.Label(card, text="Render progress", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(card, mode="determinate", maximum=100)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(14, 8))
        metrics = ttk.Frame(card, style="Card.TFrame")
        metrics.grid(row=2, column=0, sticky="ew")
        ttk.Label(metrics, textvariable=self.progress_text).pack(side="left")
        ttk.Label(metrics, textvariable=self.elapsed_text).pack(side="left", padx=(24, 0))
        ttk.Label(metrics, textvariable=self.eta_text).pack(side="right")

    def _build_log_card(self, parent: tk.Widget) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        card.grid(row=1, column=0, sticky="nsew")
        card.rowconfigure(1, weight=1)
        card.columnconfigure(0, weight=1)
        ttk.Label(card, text="Live activity log", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 12))
        log_frame = ttk.Frame(card, style="Card.TFrame")
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(log_frame, bg="#020617", fg="#d1d5db", insertbackground="#f8fafc", relief="flat", wrap="word", font=("Consolas", 10))
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)
        ttk.Button(card, text="Clear log", command=lambda: self.log.delete("1.0", "end")).grid(row=2, column=0, sticky="e", pady=(12, 0))

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(title="Choose input video", filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")])
        if path:
            self.input_path.set(path)
            if self.output_path.get() == "result.mp4":
                stem = Path(path).with_suffix("").name
                self.output_path.set(f"{stem}_sceneforge.mp4")

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(title="Choose output video", defaultextension=".mp4", filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")])
        if path:
            self.output_path.set(path)

    def _browse_selection(self) -> None:
        path = filedialog.askopenfilename(title="Choose selection JSON", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if path:
            self.selection_path.set(path)
            self._check_environment()

    def run_pipeline(self) -> None:
        if self.skip_selection.get():
            self.run_render()
            return
        self._render_after_selection = True
        self.run_selection()

    def run_selection(self) -> None:
        if not self._validate_input(require_selection=False):
            self._render_after_selection = False
            return
        self._start_command([sys.executable, "main.py", "--input", self.input_path.get()], "Opening player selection UI...")

    def run_render(self) -> None:
        if not self._validate_input(require_selection=True):
            return
        selection = Path(self.selection_path.get())
        if selection.name != DEFAULT_SELECTION_FILE or selection.parent != Path.cwd():
            Path(DEFAULT_SELECTION_FILE).write_text(selection.read_text(), encoding="utf-8")
            self._append_log(f"Copied {selection} to {DEFAULT_SELECTION_FILE} for the renderer.\n")
        self._last_percent = 0.0
        self.progress.configure(mode="determinate")
        self.progress["value"] = 0
        self.progress_text.set("0.0%")
        self.eta_text.set("ETA: calculating...")
        self._start_command([
            sys.executable,
            self.renderer.get(),
            "--input",
            self.input_path.get(),
            "--output",
            self.output_path.get(),
        ], "Rendering video...")

    def stop_process(self) -> None:
        if self._process and self._process.poll() is None:
            self._append_log("\nStopping process...\n")
            self._process.terminate()

    def _start_command(self, command: list[str], status: str) -> None:
        if self._process and self._process.poll() is None:
            messagebox.showwarning(APP_TITLE, "A task is already running.")
            return
        self._started_at = time.monotonic()
        self.status_text.set(status)
        self.progress.configure(mode="indeterminate")
        if "render" not in " ".join(command):
            self.progress.start(12)
            self.progress_text.set("Waiting for selection window")
            self.eta_text.set("ETA: user controlled")
        self._set_running(True)
        self._append_log(f"\n$ {' '.join(command)}\n")
        self._worker = threading.Thread(target=self._run_subprocess, args=(command,), daemon=True)
        self._worker.start()

    def _run_subprocess(self, command: list[str]) -> None:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=Path.cwd(),
                env=env,
            )
            assert self._process.stdout is not None
            for line in self._process.stdout:
                self._queue.put(("log", line))
                progress = self._parse_progress(line)
                if progress is not None:
                    self._queue.put(("progress", progress))
            return_code = self._process.wait()
            self._queue.put(("done", return_code))
        except Exception as exc:  # pragma: no cover - defensive GUI reporting
            self._queue.put(("error", str(exc)))

    def _parse_progress(self, text: str) -> dict[str, float | int] | None:
        match = PROGRESS_RE.search(text)
        if match:
            return {
                "percent": float(match.group("percent")),
                "frame": int(match.group("frame")),
                "total": int(match.group("total")),
            }
        match = FRAME_RE.search(text)
        if match and match.group("total"):
            frame = int(match.group("frame"))
            total = int(match.group("total"))
            percent = (frame / total * 100) if total else 0.0
            return {"percent": percent, "frame": frame, "total": total}
        return None

    def _drain_queue(self) -> None:
        try:
            while True:
                event, payload = self._queue.get_nowait()
                if event == "log":
                    self._append_log(str(payload))
                elif event == "progress":
                    self._update_progress(payload)  # type: ignore[arg-type]
                elif event == "done":
                    self._finish_process(int(payload))
                elif event == "error":
                    self._finish_process(1, str(payload))
        except queue.Empty:
            pass
        self.after(100, self._drain_queue)

    def _update_progress(self, payload: dict[str, float | int]) -> None:
        percent = max(0.0, min(100.0, float(payload["percent"])))
        self._last_percent = percent
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress["value"] = percent
        frame = int(payload["frame"])
        total = int(payload["total"])
        self.progress_text.set(f"{percent:5.1f}% · Frame {frame}/{total}")
        self.eta_text.set(f"ETA: {self._estimate_eta(percent)}")

    def _estimate_eta(self, percent: float) -> str:
        if not self._started_at or percent <= 0:
            return "calculating..."
        elapsed = time.monotonic() - self._started_at
        remaining = elapsed * (100 - percent) / percent
        return self._format_duration(remaining)

    def _tick_elapsed(self) -> None:
        if self._started_at and self._process and self._process.poll() is None:
            elapsed = time.monotonic() - self._started_at
            self.elapsed_text.set(f"Elapsed: {self._format_duration(elapsed)}")
            if self._last_percent > 0:
                self.eta_text.set(f"ETA: {self._estimate_eta(self._last_percent)}")
        self.after(1000, self._tick_elapsed)

    def _finish_process(self, return_code: int, error: str | None = None) -> None:
        self.progress.stop()
        self._set_running(False)
        should_render_next = self._render_after_selection and return_code == 0
        self._render_after_selection = False
        if return_code == 0:
            self.progress.configure(mode="determinate")
            if self._last_percent > 0:
                self.progress["value"] = 100
                self.progress_text.set("100.0% · Complete")
                self.eta_text.set("ETA: done")
            self.status_text.set("Task completed successfully.")
            self._append_log("\nTask completed successfully.\n")
        else:
            self.status_text.set("Task failed or was stopped. See the activity log for details.")
            self._append_log(f"\nTask exited with code {return_code}. {error or ''}\n")
        self._process = None
        self._started_at = None
        self._last_percent = 0.0
        self._check_environment()
        if should_render_next:
            self.after(250, self.run_render)

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        self.pipeline_button.configure(state=state)
        self.select_button.configure(state=state)
        self.render_button.configure(state=state)
        self.stop_button.configure(state="normal" if running else "disabled")

    def _validate_input(self, require_selection: bool) -> bool:
        if not self.input_path.get() or not Path(self.input_path.get()).exists():
            messagebox.showerror(APP_TITLE, "Choose a valid input video first.")
            return False
        if require_selection and not Path(self.selection_path.get()).exists():
            messagebox.showerror(APP_TITLE, "Create or choose a selection.json file before rendering.")
            return False
        if require_selection and not self.output_path.get():
            messagebox.showerror(APP_TITLE, "Choose an output video path before rendering.")
            return False
        return True

    def _check_environment(self) -> None:
        selection = Path(self.selection_path.get())
        self.selection_status.set("Selection: ready" if selection.exists() else "Selection: missing")
        model = Path("yolov8m-seg.pt")
        self.model_status.set("Model: yolov8m-seg.pt found" if model.exists() else "Model: yolov8m-seg.pt not found (download may be required)")

    def _append_log(self, text: str) -> None:
        self.log.insert("end", text)
        self.log.see("end")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        minutes, secs = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"


def main() -> None:
    app = SceneForgeGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
