import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_INPUT_DIR = r"D:\math-db-5\pdf_inbox"
DEFAULT_OUT_BASE = r"D:\math-db-5\pdf_cutter_output\books"


def sanitize_doc_prefix(stem: str) -> str:
    s = stem.strip().replace(" ", "_")
    s = re.sub(r"[^\w.-]", "", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s)
    s = s.strip("._")
    return s or "untitled"


def count_crop_pngs(out_root: Path) -> int:
    crops_dir = out_root / "crops"
    if not crops_dir.exists():
        return 0
    return sum(1 for _ in crops_dir.glob("*.png"))


def parse_summary_from_out_log(log_path: Path) -> Dict[str, Any]:
    if not log_path.exists():
        return {}
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return {}
    for line in reversed(lines):
        text = line.strip()
        if not (text.startswith("{") and text.endswith("}")):
            continue
        try:
            obj = json.loads(text)
        except Exception:
            continue
        if isinstance(obj, dict) and "total_saved" in obj:
            return obj
    return {}


def parse_summary_from_replay(out_root: Path) -> Dict[str, Any]:
    summary_path = out_root / "replay_summary.json"
    if not summary_path.exists():
        return {}
    try:
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def run_one_pdf(
    pdf_path: Path,
    out_base: Path,
    profile: str,
    force: bool,
    render_zoom: float,
) -> Dict[str, Any]:
    started = time.perf_counter()
    doc_prefix = sanitize_doc_prefix(pdf_path.stem)
    out_root = out_base / doc_prefix
    logs_dir = out_root / "_logs"
    out_log = logs_dir / "auto_out.log"
    err_log = logs_dir / "auto_err.log"

    record: Dict[str, Any] = {
        "pdf_path": str(pdf_path),
        "out_root": str(out_root),
        "doc_prefix": doc_prefix,
        "status": "fail",
        "total_saved": 0,
        "total_errors": 0,
        "chosen_profile": "",
        "elapsed_sec": 0.0,
        "note": "",
    }

    if not force:
        index_path = out_root / "index.csv"
        if index_path.exists() and count_crop_pngs(out_root) > 0:
            record["status"] = "skip"
            record["note"] = "already_processed"
            record["elapsed_sec"] = round(time.perf_counter() - started, 3)
            return record

    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    pipeline_script = Path(__file__).resolve().parent / "auto_cut_pipeline.py"
    cmd_line = (
        f'chcp 65001>nul & "{sys.executable}" "{pipeline_script}" '
        f'--pdf_path "{pdf_path}" --out_root "{out_root}" --doc_prefix "{doc_prefix}" '
        f'--runner_profile "{profile}" --render_zoom {float(render_zoom)}'
    )
    cmd = f'cmd /c "{cmd_line}"'

    try:
        with open(out_log, "w", encoding="utf-8", newline="\n") as f_out, open(
            err_log, "w", encoding="utf-8", newline="\n"
        ) as f_err:
            proc = subprocess.run(
                cmd,
                shell=True,
                stdout=f_out,
                stderr=f_err,
                check=False,
            )
        note = f"returncode={proc.returncode}"
    except Exception as exc:
        proc = None
        note = f"subprocess_error={exc}"

    summary = parse_summary_from_out_log(out_log)
    if not summary:
        summary = parse_summary_from_replay(out_root)

    total_saved = int(summary.get("total_saved", 0)) if isinstance(summary, dict) else 0
    total_errors = int(summary.get("total_errors", 0)) if isinstance(summary, dict) else 0
    chosen_profile = str(summary.get("chosen_profile", "")) if isinstance(summary, dict) else ""

    record["total_saved"] = total_saved
    record["total_errors"] = total_errors
    record["chosen_profile"] = chosen_profile
    record["status"] = "ok" if (proc is not None and proc.returncode == 0) else "fail"
    record["note"] = note
    record["elapsed_sec"] = round(time.perf_counter() - started, 3)
    return record


def write_batch_summary(summary_csv: Path, rows: List[Dict[str, Any]]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pdf_path",
        "out_root",
        "doc_prefix",
        "status",
        "total_saved",
        "total_errors",
        "chosen_profile",
        "elapsed_sec",
        "note",
    ]
    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out_base", default=DEFAULT_OUT_BASE)
    parser.add_argument("--profile", choices=["fast", "full"], default="fast")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max_pdfs", type=int, default=0)
    parser.add_argument("--sleep_sec", type=float, default=0.0)
    parser.add_argument("--render_zoom", type=float, default=2.0)
    args = parser.parse_args()
    if float(args.render_zoom) <= 0:
        print("[error] --render_zoom must be > 0", file=sys.stderr)
        return 2

    input_dir = Path(args.input_dir).resolve()
    out_base = Path(args.out_base).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[error] missing input_dir: {input_dir}", file=sys.stderr)
        return 2

    pdfs = sorted(input_dir.glob("*.pdf"), key=lambda p: p.name.lower())
    if args.max_pdfs and args.max_pdfs > 0:
        pdfs = pdfs[: args.max_pdfs]

    rows: List[Dict[str, Any]] = []
    for i, pdf in enumerate(pdfs):
        row = run_one_pdf(
            pdf_path=pdf,
            out_base=out_base,
            profile=args.profile,
            force=bool(args.force),
            render_zoom=float(args.render_zoom),
        )
        rows.append(row)
        print(
            f"[book] {i + 1}/{len(pdfs)} status={row['status']} doc_prefix={row['doc_prefix']} "
            f"saved={row['total_saved']} errors={row['total_errors']}"
        )
        if args.sleep_sec and i < len(pdfs) - 1:
            time.sleep(max(0.0, float(args.sleep_sec)))

    summary_csv = out_base / "batch_summary.csv"
    write_batch_summary(summary_csv, rows)
    print(f"[done] batch_summary={summary_csv} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
