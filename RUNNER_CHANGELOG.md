# PDF Cutter Experiment Runner Changes

## What changed and why

- `v3_isolation_runner.py` was finalized as the canonical runner file used by GUI.
  - Added hardening for Windows console encoding (`backslashreplace`) to block cp949 print crashes.
  - Enforced stdout JSON-only behavior with robust emit fallback that always outputs at least one JSON line, even on serialization errors.
  - Expanded JSON converter safety for numpy types, tuple/set, bytes, and `Path`.
  - Kept both CLI modes:
    - Batch mode: `--pages_dir <dir>` emits JSONL (per-page lines with `page_file`).
    - Single mode: `<image_path>` emits one JSON line with `ok` + `page_file`.
  - Added `--warmup {0,1}` (default `1`) to perform one post-init dummy prediction.
  - Added timing metadata fields (`t_init_ms`, `t_predict_ms`, `t_emit_ms`) in output payloads.

- `pdf_cutter_experiment_gui.py` was made more tolerant to mixed/dirty stdout lines from subprocess runners.
  - Subprocess decode path now uses `encoding="utf-8", errors="replace"`.
  - Single/fallback runner parsers now scan stdout lines and accept the last valid JSON line, instead of failing immediately on malformed lines.
  - Batch isolation loop continues on malformed lines (existing behavior maintained), with safer decode configuration.

- Backup file added:
  - `v3_isolation_runner.py.bak`

## Example commands

```bash
# Batch (JSONL)
python v3_isolation_runner.py --pages_dir C:/temp/pages --dpi 250 --warmup 1

# Single image (single JSON line)
python v3_isolation_runner.py C:/temp/pages/P001.png --warmup 1
```
