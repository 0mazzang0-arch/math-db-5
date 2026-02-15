import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_LAST_EMIT_MS = 0.0

# ---------------------------------------------------------------------
# Windows + PaddleOCR(PPStructureV3) 안정화용 환경 변수 (요구사항 고정)
# ---------------------------------------------------------------------
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_new_ir"] = "0"

# ---------------------------------------------------------------------
# IMPORTANT (Windows cp949/mbcs):
# stdout/stderr는 encoding을 바꾸지 않고 error handler만 바꿔서
# print 시 UnicodeEncodeError로 프로세스가 죽지 않게 한다.
# ---------------------------------------------------------------------
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")  # type: ignore[attr-defined]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="backslashreplace")  # type: ignore[attr-defined]
except Exception:
    pass


def _stage(msg: str) -> None:
    # stderr 로그 전용
    print(f"[stage] {msg}", file=sys.stderr, flush=True)


def _converter(obj: Any) -> Any:
    """json.dumps default converter."""
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

    if isinstance(obj, (tuple, set)):
        return list(obj)
    if isinstance(obj, bytes):
        try:
            return {"__bytes_b64__": base64.b64encode(obj).decode("ascii")}
        except Exception:
            return repr(obj)
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def _safe_json_dump(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=_converter)


def _emit_json(payload: Dict[str, Any], fallback_page_file: str = "") -> float:
    """
    stdout에는 JSON 1라인만 출력한다.
    emit 자체가 실패해도 최소 에러 JSON 1라인 출력 보장.
    returns t_emit_ms
    """
    global _LAST_EMIT_MS
    emit_start = time.perf_counter()
    try:
        if "t_emit_ms" not in payload:
            payload["t_emit_ms"] = round(_LAST_EMIT_MS, 3)
        line = _safe_json_dump(payload)
        print(line, flush=True)
    except Exception as e:
        err_payload = {
            "ok": False,
            "stage": "emit",
            "err": str(e),
            "page_file": payload.get("page_file", fallback_page_file) if isinstance(payload, dict) else fallback_page_file,
        }
        try:
            print(json.dumps(err_payload, ensure_ascii=True), flush=True)
        except Exception:
            # 마지막 보루
            sys.stdout.write('{"ok": false, "stage": "emit", "err": "emit failed", "page_file": ""}\n')
            sys.stdout.flush()
    _LAST_EMIT_MS = (time.perf_counter() - emit_start) * 1000.0
    return _LAST_EMIT_MS


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


def _predict_one(engine: Any, page_path: Path, t_init_ms: float) -> Dict[str, Any]:
    page_name = page_path.name
    predict_start = time.perf_counter()
    try:
        import cv2
        import numpy as np

        raw = cv2.imdecode(np.fromfile(str(page_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if raw is None:
            return {
                "ok": False,
                "page_file": page_name,
                "stage": "detect_load",
                "err": "imdecode failed",
                "t_init_ms": round(t_init_ms, 3),
                "t_predict_ms": round((time.perf_counter() - predict_start) * 1000.0, 3),
            }

        try:
            output = engine.predict(input=raw)
        except Exception:
            try:
                output = engine.predict(input=str(page_path))
            except Exception as e:
                return {
                    "ok": False,
                    "page_file": page_name,
                    "stage": "detect_predict",
                    "err": str(e),
                    "t_init_ms": round(t_init_ms, 3),
                    "t_predict_ms": round((time.perf_counter() - predict_start) * 1000.0, 3),
                }

        first = _first_output(output)
        pp_json = _extract_json(first)
        t_predict_ms = (time.perf_counter() - predict_start) * 1000.0
        if not isinstance(pp_json, dict):
            return {
                "ok": False,
                "page_file": page_name,
                "stage": "parse_json",
                "err": "invalid json payload",
                "first_type": type(first).__name__ if first is not None else "None",
                "t_init_ms": round(t_init_ms, 3),
                "t_predict_ms": round(t_predict_ms, 3),
            }

        pp_obj = _extract_first_object_fields(first)
        pp_meta = {
            "runner_mode": "single_or_pages_dir_batch",
            "page_file": page_name,
            "first_type": type(first).__name__ if first is not None else "None",
            "json_keys": sorted(pp_json.keys()),
        }
        return {
            "ok": True,
            "page_file": page_name,
            "pp_json": pp_json,
            "pp_obj": pp_obj,
            "pp_meta": pp_meta,
            "t_init_ms": round(t_init_ms, 3),
            "t_predict_ms": round(t_predict_ms, 3),
        }
    except Exception as e:
        return {
            "ok": False,
            "page_file": page_name,
            "stage": "runner_loop",
            "err": str(e),
            "t_init_ms": round(t_init_ms, 3),
            "t_predict_ms": round((time.perf_counter() - predict_start) * 1000.0, 3),
        }


def _warmup_once(engine: Any, enabled: bool) -> None:
    if not enabled:
        return
    try:
        import numpy as np

        dummy = np.full((64, 64, 3), 255, dtype=np.uint8)
        _stage("warmup_start")
        engine.predict(input=dummy)
        _stage("warmup_done")
    except Exception as e:
        _stage(f"warmup_skip err={e}")


def run_pages_dir(pages_dir: Path, warmup: bool = True) -> None:
    if not pages_dir.exists() or not pages_dir.is_dir():
        _emit_json({"ok": False, "page_file": "", "stage": "args", "err": f"invalid pages_dir: {pages_dir}"})
        return

    try:
        from paddleocr import PPStructureV3
    except Exception as e:
        _emit_json({"ok": False, "page_file": "", "stage": "imports", "err": str(e)})
        return

    page_files = _sorted_page_files(pages_dir)
    _stage(f"start pages={len(page_files)}")
    if not page_files:
        _emit_json({"ok": False, "page_file": "", "stage": "load_pages", "err": "no P*.png files"})
        return

    init_start = time.perf_counter()
    try:
        engine = PPStructureV3()
        t_init_ms = (time.perf_counter() - init_start) * 1000.0
        _stage("init_ok")
    except Exception as e:
        _emit_json({"ok": False, "page_file": "", "stage": "init_engine", "err": str(e)})
        return

    _warmup_once(engine, warmup)

    for page_path in page_files:
        page_name = page_path.name
        _stage(f"predict_start {page_name}")
        payload = _predict_one(engine, page_path, t_init_ms=t_init_ms)
        _emit_json(payload, fallback_page_file=page_name)
        _stage(f"predict_done {page_name}")

    _stage("done")


def run_single_image(image_path: Path, warmup: bool = True) -> None:
    if not image_path.exists() or not image_path.is_file():
        _emit_json({"ok": False, "page_file": image_path.name, "stage": "args", "err": f"invalid image_path: {image_path}"})
        return

    try:
        from paddleocr import PPStructureV3
    except Exception as e:
        _emit_json({"ok": False, "page_file": image_path.name, "stage": "imports", "err": str(e)})
        return

    init_start = time.perf_counter()
    try:
        engine = PPStructureV3()
        t_init_ms = (time.perf_counter() - init_start) * 1000.0
        _stage("init_ok(single)")
    except Exception as e:
        _emit_json({"ok": False, "page_file": image_path.name, "stage": "init_engine", "err": str(e)})
        return

    _warmup_once(engine, warmup)

    _stage(f"predict_start(single) {image_path.name}")
    payload = _predict_one(engine, image_path, t_init_ms=t_init_ms)
    if payload.get("ok"):
        out = {
            "ok": True,
            "page_file": image_path.name,
            "pp_json": payload.get("pp_json", {}),
            "pp_obj": payload.get("pp_obj", {}),
            "pp_meta": payload.get("pp_meta", {}),
            "t_init_ms": payload.get("t_init_ms"),
            "t_predict_ms": payload.get("t_predict_ms"),
        }
    else:
        out = {
            "ok": False,
            "page_file": image_path.name,
            "stage": payload.get("stage", "predict"),
            "err": payload.get("err", "unknown"),
            "t_init_ms": payload.get("t_init_ms"),
            "t_predict_ms": payload.get("t_predict_ms"),
        }
    _emit_json(out, fallback_page_file=image_path.name)
    _stage(f"predict_done(single) {image_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", nargs="?", help="single image path (e.g., P001.png)")
    parser.add_argument("--pages_dir", required=False, help="directory containing P*.png files (batch mode)")
    parser.add_argument("--dpi", type=int, default=250, help="reserved (compat); not used by this runner")
    parser.add_argument("--warmup", type=int, choices=[0, 1], default=1, help="run one warmup predict after init")
    args = parser.parse_args()

    warmup_enabled = bool(args.warmup)

    if args.pages_dir:
        run_pages_dir(Path(args.pages_dir), warmup=warmup_enabled)
        return

    if args.image_path:
        run_single_image(Path(args.image_path), warmup=warmup_enabled)
        return

    _emit_json({"ok": False, "page_file": "", "stage": "args", "err": "usage: v3_isolation_runner.py (--pages_dir DIR) | (image_path)"})


if __name__ == "__main__":
    main()
