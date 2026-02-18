import json
from pathlib import Path
from PIL import Image
import tkinter as tk

from pdf_cutter_experiment_gui import PDFCutterApp, PageTask

pages_dir = Path(r"D:\math-db-5\pdf_cutter_output\_test50")
jsonl_path = Path(r"D:\math-db-5\runner_out_test50.jsonl")

out_root = Path(r"D:\math-db-5\pdf_cutter_output\_test50_gui_from_jsonl")
crops_dir = out_root / "crops"
errors_dir = out_root / "errors"
crops_dir.mkdir(parents=True, exist_ok=True)
errors_dir.mkdir(parents=True, exist_ok=True)

page_paths = sorted(pages_dir.glob("P*.png"))
tasks = [PageTask(page_number=i+1, total_pages=len(page_paths), page_png_path=p) for i, p in enumerate(page_paths)]
task_by_name = {t.page_png_path.name: t for t in tasks}

root = tk.Tk(); root.withdraw()
app = PDFCutterApp(root)
app.full_profile_var.set(False)
app.save_runner_logs_var.set(False)

total_saved = 0
total_errors = 0
seen = 0

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        page_file = str(payload.get("page_file", ""))
        if not page_file:
            continue
        task = task_by_name.get(page_file)
        if task is None:
            continue

        img = Image.open(task.page_png_path)
        w, h = img.size

        # anchors/objects 직접 포함(payload min) 케이스 우선
        if isinstance(payload.get("anchors"), list) and isinstance(payload.get("objects"), list):
            anchors = payload.get("anchors", [])
            objects = payload.get("objects", [])
        else:
            # (구형 payload) normalize
            data = {"pp_json": payload.get("pp_json", {}), "pp_obj": payload.get("pp_obj", {}), "pp_meta": payload.get("pp_meta", {})}
            anchors, objects = app._normalize_structure(data, w, h)

        # slice regions
        crops, dropped, errors = app._build_anchor_slice_regions(anchors, objects, w, h)

        if errors > 0 and not crops:
            total_errors += 1
            app._write_page_error(errors_dir, task.page_number, task.page_png_path, "anchor overlap conflict", stage="slice",
                                 extras={"anchors": len(anchors), "objects": len(objects), "errors": errors})
            continue

        saved = 0
        for seq, (qid, x1, y1, x2, y2) in enumerate(crops, start=1):
            crop_img = img.crop((x1, y1, x2, y2))
            out_name = f"P{task.page_number:03d}_Q{seq:03d}_N{qid:04d}.png"
            crop_img.save(crops_dir / out_name)
            saved += 1

        total_saved += saved
        seen += 1

print("DONE pages_seen=", seen, "total_saved=", total_saved, "total_errors=", total_errors, "out_root=", out_root)
