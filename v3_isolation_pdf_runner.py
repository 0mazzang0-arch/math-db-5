import json
import os
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


def _emit(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


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


def main() -> None:
    _stage("start")
    if len(sys.argv) < 3:
        _emit({"ok": False, "stage": "args", "err": "usage: v3_isolation_pdf_runner.py <pdf_path> <out_jsonl> [dpi]"})
        return

    pdf_path = Path(sys.argv[1])
    out_jsonl = Path(sys.argv[2])
    dpi = int(sys.argv[3]) if len(sys.argv) >= 4 else 250

    try:
        import fitz
        import cv2
        import numpy as np
        from paddleocr import PPStructureV3
    except Exception as e:
        _emit({"ok": False, "stage": "imports", "err": str(e)})
        return

    try:
        engine = PPStructureV3()
        _stage("init_ok")
    except Exception as e:
        _emit({"ok": False, "stage": "init_engine", "err": str(e)})
        return

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as wf:
        with fitz.open(pdf_path) as doc:
            for idx, page in enumerate(doc):
                page_no = idx + 1
                _stage(f"predict_start page={page_no}")
                try:
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    raw = cv2.imdecode(np.frombuffer(pix.tobytes("png"), dtype=np.uint8), cv2.IMREAD_COLOR)
                    if raw is None:
                        wf.write(json.dumps({"page": page_no, "ok": False, "stage": "detect_load", "err": "imdecode failed"}, ensure_ascii=False) + "\n")
                        continue
                    output = engine.predict(input=raw)
                    first = _first_output(output)
                    pp_json = _extract_json(first)
                    if not isinstance(pp_json, dict):
                        wf.write(json.dumps({"page": page_no, "ok": False, "stage": "parse_json", "err": "invalid json payload"}, ensure_ascii=False) + "\n")
                        continue
                    wf.write(json.dumps({"page": page_no, "ok": True, "pp_json": pp_json, "pp_obj": _extract_first_object_fields(first)}, ensure_ascii=False) + "\n")
                    _stage(f"predict_done page={page_no}")
                except Exception as e:
                    wf.write(json.dumps({"page": page_no, "ok": False, "stage": "predict", "err": str(e)}, ensure_ascii=False) + "\n")

    _stage("done")
    _emit({"ok": True, "stage": "done", "out_jsonl": str(out_jsonl)})


if __name__ == "__main__":
    main()
