import csv
import os
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

DEFAULT_INPUT_DIR = r"D:\\math-db-5\\pdf_inbox"
DEFAULT_OUT_BASE = r"D:\\math-db-5\\pdf_cutter_output\\books"


class DropFolderGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Run Drop Folder")
        self.root.geometry("900x600")

        self.proc: subprocess.Popen | None = None
        self.log_path: Path | None = None
        self.err_pos = 0
        self.summary_path: Path | None = None
        self._poll_job: str | None = None

        self.input_var = tk.StringVar(value=DEFAULT_INPUT_DIR)
        self.out_var = tk.StringVar(value=DEFAULT_OUT_BASE)
        self.max_pdfs_var = tk.StringVar(value="")
        self.render_zoom_var = tk.StringVar(value="2.0")
        self.force_var = tk.BooleanVar(value=False)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=4)
        ttk.Label(row1, text="Input folder", width=14).pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.input_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(row1, text="Browse", command=self.browse_input).pack(side=tk.LEFT)

        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=4)
        ttk.Label(row2, text="Output folder", width=14).pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.out_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(row2, text="Browse", command=self.browse_output).pack(side=tk.LEFT)

        row3 = ttk.Frame(frame)
        row3.pack(fill=tk.X, pady=4)
        ttk.Label(row3, text="max_pdfs", width=14).pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.max_pdfs_var, width=12).pack(side=tk.LEFT, padx=4)
        ttk.Label(row3, text="render_zoom").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(row3, textvariable=self.render_zoom_var, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(row3, text="force", variable=self.force_var).pack(side=tk.LEFT, padx=10)

        row4 = ttk.Frame(frame)
        row4.pack(fill=tk.X, pady=8)
        self.run_btn = ttk.Button(row4, text="Run", command=self.run_batch)
        self.run_btn.pack(side=tk.LEFT)
        self.open_btn = ttk.Button(row4, text="Open Output", command=self.open_output, state=tk.NORMAL)
        self.open_btn.pack(side=tk.LEFT, padx=8)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(frame, textvariable=self.status_var).pack(anchor=tk.W, pady=(0, 6))

        self.log_text = tk.Text(frame, wrap="word", height=25)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def browse_input(self) -> None:
        path = filedialog.askdirectory(initialdir=self.input_var.get() or DEFAULT_INPUT_DIR)
        if path:
            self.input_var.set(path)

    def browse_output(self) -> None:
        path = filedialog.askdirectory(initialdir=self.out_var.get() or DEFAULT_OUT_BASE)
        if path:
            self.out_var.set(path)

    def append_log(self, text: str) -> None:
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)

    def validate(self) -> tuple[Path, Path, str, str]:
        input_dir = Path(self.input_var.get().strip())
        out_base = Path(self.out_var.get().strip())
        max_pdfs = self.max_pdfs_var.get().strip()
        render_zoom = self.render_zoom_var.get().strip()

        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"input_dir not found: {input_dir}")
        if max_pdfs:
            if not max_pdfs.isdigit() or int(max_pdfs) < 1:
                raise ValueError("max_pdfs must be empty or positive integer")
        if not render_zoom:
            raise ValueError("render_zoom is required")
        try:
            zoom_value = float(render_zoom)
        except Exception:
            raise ValueError("render_zoom must be a float value")
        if zoom_value <= 0:
            raise ValueError("render_zoom must be > 0")
        return input_dir, out_base, max_pdfs, str(zoom_value)

    def run_batch(self) -> None:
        if self.proc and self.proc.poll() is None:
            return

        try:
            input_dir, out_base, max_pdfs, render_zoom = self.validate()
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        self.log_text.delete("1.0", tk.END)
        self.summary_path = out_base / "batch_summary.csv"
        self.log_path = Path.cwd() / "batch_err.log"
        self.err_pos = 0

        pyexe = Path(".venv311") / "Scripts" / "python.exe"
        if not pyexe.exists():
            messagebox.showerror("Missing python", f"not found: {pyexe}")
            return

        cmd = (
            f'chcp 65001>nul & "{pyexe}" run_drop_folder.py '
            f'--input_dir "{input_dir}" --out_base "{out_base}" --render_zoom {render_zoom}'
        )
        if max_pdfs:
            cmd += f" --max_pdfs {max_pdfs}"
        if self.force_var.get():
            cmd += " --force"
        cmd += " 1> batch_out.log 2> batch_err.log"

        full_cmd = f'cmd /c "{cmd}"'

        try:
            self.proc = subprocess.Popen(full_cmd, shell=True)
        except Exception as exc:
            messagebox.showerror("Run failed", str(exc))
            return

        self.run_btn.config(state=tk.DISABLED)
        self.status_var.set("Running...")
        self.append_log(f"$ {full_cmd}\n")
        self._schedule_poll()

    def _schedule_poll(self) -> None:
        self._poll_job = self.root.after(500, self.poll_process)

    def poll_process(self) -> None:
        self.read_err_tail()

        if self.proc and self.proc.poll() is None:
            self._schedule_poll()
            return

        rc = self.proc.returncode if self.proc else -1
        self.run_btn.config(state=tk.NORMAL)

        if self.summary_path and self.summary_path.exists():
            summary_text = self.summarize_batch_status(self.summary_path)
            self.status_var.set(f"Done (rc={rc}) summary={self.summary_path} {summary_text}")
            self.append_log(f"\n[done] rc={rc} summary={self.summary_path} {summary_text}\n")
        else:
            self.status_var.set(f"Done (rc={rc}) summary not found")
            self.append_log(f"\n[done] rc={rc} summary not found\n")

    def summarize_batch_status(self, summary_csv: Path) -> str:
        counts: dict[str, int] = {}
        try:
            with open(summary_csv, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    status = (row.get("status") or "").strip() or "unknown"
                    counts[status] = counts.get(status, 0) + 1
        except Exception:
            return "(status=unreadable)"
        if not counts:
            return "(status=empty)"
        parts = [f"{k}:{v}" for k, v in sorted(counts.items())]
        return f"({' '.join(parts)})"

    def read_err_tail(self) -> None:
        if not self.log_path:
            return
        if not self.log_path.exists():
            return

        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self.err_pos)
                chunk = f.read()
                self.err_pos = f.tell()
        except Exception as exc:
            self.append_log(f"[log read error] {exc}\n")
            return

        if chunk:
            self.append_log(chunk)

    def open_output(self) -> None:
        out_dir = Path(self.out_var.get().strip())
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            os.startfile(str(out_dir))
        except Exception as exc:
            messagebox.showerror("Open output failed", str(exc))

    def stop_process(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def on_close(self) -> None:
        if self.proc and self.proc.poll() is None:
            ok = messagebox.askyesno("Exit", "A run is in progress. Terminate and close?")
            if not ok:
                return
            self.stop_process()

        if self._poll_job is not None:
            self.root.after_cancel(self._poll_job)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    DropFolderGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
