import tkinter as tk
from pathlib import Path
from pdf_cutter_experiment_gui import PDFCutterApp, PageTask

root = tk.Tk(); root.withdraw()
app = PDFCutterApp(root)
app.full_profile_var.set(False)
app.save_runner_logs_var.set(False)

pages_dir = Path(r"D:\math-db-5\pdf_cutter_output\_test50")
page_paths = sorted(pages_dir.glob("P*.png"))
tasks = [PageTask(page_number=i+1, total_pages=len(page_paths), page_png_path=p) for i, p in enumerate(page_paths)]

out_root = Path(r"D:\math-db-5\pdf_cutter_output\_test50_gui_from_jsonl")
crops_dir = out_root / "crops"
errors_dir = out_root / "errors"
crops_dir.mkdir(parents=True, exist_ok=True)
errors_dir.mkdir(parents=True, exist_ok=True)

# JSONL (runner output)
jsonl = Path(r"D:\math-db-5\runner_out_test50.jsonl")

# best-effort inject path into app if the GUI expects it somewhere
for name in ["runner_out_path", "runner_jsonl_path", "runner_jsonl", "jsonl_path", "out_jsonl_path"]:
    if hasattr(app, name):
        setattr(app, name, jsonl)

# run batch using the GUI's available entrypoint
if hasattr(app, "run_isolation_batch"):
    app.run_isolation_batch(tasks, out_root)
elif hasattr(app, "process_tasks_batch"):
    app.process_tasks_batch(tasks, out_root)
else:
    saved = 0
    errs = 0
    for t in tasks:
        try:
            r = app.process_page(t, out_root)
            if isinstance(r, dict) and r.get("saved"):
                saved += int(r["saved"])
        except Exception:
            errs += 1
    print("fallback_batch_saved=", saved, "errs=", errs)

print("DONE out_root=", out_root)
