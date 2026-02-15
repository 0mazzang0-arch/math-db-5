import json
import os
import sys
from typing import Any, Dict

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_new_ir"] = "0"


def _emit(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


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
    if len(sys.argv) < 2:
        _emit({"ok": False, "stage": "args", "err": "image path required"})
        return

    try:
        import cv2
        import numpy as np
        from paddleocr import PPStructureV3
    except Exception as e:
        _emit({"ok": False, "stage": "imports", "err": str(e)})
        return

    image_path = sys.argv[1]
    try:
        raw = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        _emit({"ok": False, "stage": "detect_load", "err": str(e)})
        return

    if raw is None:
        _emit({"ok": False, "stage": "detect_load", "err": "imdecode failed"})
        return

    try:
        engine = PPStructureV3()
    except Exception as e:
        _emit({"ok": False, "stage": "init_engine", "err": str(e)})
        return

    try:
        output = engine.predict(input=raw)
    except Exception:
        try:
            output = engine.predict(input=image_path)
        except Exception as e:
            _emit({"ok": False, "stage": "detect_predict", "err": str(e)})
            return

    first = _first_output(output)
    pp_json = _extract_json(first)
    if not isinstance(pp_json, dict):
        _emit({
            "ok": False,
            "stage": "parse_json",
            "err": "invalid json payload",
            "first_type": type(first).__name__ if first is not None else "None",
        })
        return

    _emit({"ok": True, "pp_json": pp_json, "pp_obj": _extract_first_object_fields(first)})


if __name__ == "__main__":
    main()
