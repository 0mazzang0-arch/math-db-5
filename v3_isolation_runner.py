import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_new_ir"] = "0"


def _stage(msg: str) -> None:
    print(f"[stage] {msg}", file=sys.stderr, flush=True)


def _default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _emit_line(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, default=_default), flush=True)


def _first_output(output: Any) -> Any:
    if output is None:
        return None
    if isinstance(output, (list, tuple)):
        return output[0] if output else None
    if isinstance(output, dict):
        return output
    try:
        return next(iter(output), None)
    except Exception:
        return output


def _extract_json(first: Any) -> Any:
    if first is None:
        return None
    j = getattr(first, "json", None)
    if callable(j):
        try:
            j = j()
        except Exception:
            j = None
    if isinstance(j, dict):
        return j
    if isinstance(first, dict):
        for k in ("json", "result", "res"):
            cand = first.get(k)
            if isinstance(cand, dict):
                return cand
    for attr in ("result", "res", "data"):
        cand = getattr(first, attr, None)
        if isinstance(cand, dict):
            return cand
    return None


def _extract_first_object_fields(first: Any) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    keys = ["overall_ocr_res", "parsing_res_list", "region_det_res", "layout_det_res", "table_res_list"]
    if isinstance(first, dict):
        for key in keys:
            if key in first:
                fields[key] = first.get(key)
        return fields
    for key in keys:
        try:
            value = getattr(first, key, None)
        except Exception:
            value = None
        if value is not None:
            fields[key] = value
    return fields


def _sorted_page_files(pages_dir: Path) -> List[Path]:
    files = [p for p in pages_dir.glob("P*.png") if p.is_file()]

    def _key(p: Path) -> Any:
        m = re.match(r"^P(\d+)\.png$", p.name)
        if m:
            return (0, int(m.group(1)), p.name)
        return (1, 0, p.name)

    return sorted(files, key=_key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages_dir", required=True)
    parser.add_argument("--dpi", type=int, default=250)
    args = parser.parse_args()

    pages_dir = Path(args.pages_dir)
    if not pages_dir.exists() or not pages_dir.is_dir():
        _emit_line({"ok": False, "page_file": "", "stage": "args", "err": f"invalid pages_dir: {pages_dir}"})
        return

    try:
        import cv2
        import numpy as np
        from paddleocr import PPStructureV3
    except Exception as e:
        _emit_line({"ok": False, "page_file": "", "stage": "imports", "err": str(e)})
        return

    page_files = _sorted_page_files(pages_dir)
    _stage(f"start pages={len(page_files)}")
    if not page_files:
        _emit_line({"ok": False, "page_file": "", "stage": "load_pages", "err": "no P*.png files"})
        return

    try:
        engine = PPStructureV3()
        _stage("init_ok")
    except Exception as e:
        _emit_line({"ok": False, "page_file": "", "stage": "init_engine", "err": str(e)})
        return

    for page_path in page_files:
        page_name = page_path.name
        try:
            _stage(f"predict_start {page_name}")
            raw = cv2.imdecode(np.fromfile(str(page_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if raw is None:
                _emit_line({"ok": False, "page_file": page_name, "stage": "detect_load", "err": "imdecode failed"})
                continue

            try:
                output = engine.predict(input=raw)
            except Exception:
                try:
                    output = engine.predict(input=str(page_path))
                except Exception as e:
                    _emit_line({"ok": False, "page_file": page_name, "stage": "detect_predict", "err": str(e)})
                    continue

            first = _first_output(output)
            pp_json = _extract_json(first)
            if not isinstance(pp_json, dict):
                _emit_line({
                    "ok": False,
                    "page_file": page_name,
                    "stage": "parse_json",
                    "err": "invalid json payload",
                    "first_type": type(first).__name__ if first is not None else "None",
                })
                continue

            pp_obj = _extract_first_object_fields(first)
            pp_meta = {
                "runner_mode": "pages_dir_batch",
                "page_file": page_name,
                "first_type": type(first).__name__ if first is not None else "None",
                "json_keys": sorted(pp_json.keys()),
            }
            _emit_line({"ok": True, "page_file": page_name, "pp_json": pp_json, "pp_obj": pp_obj, "pp_meta": pp_meta})
            _stage(f"predict_done {page_name}")
        except Exception as e:
            _emit_line({"ok": False, "page_file": page_name, "stage": "runner_loop", "err": str(e)})

    _stage("done")


if __name__ == "__main__":
    main()
