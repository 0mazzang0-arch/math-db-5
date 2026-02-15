import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
# - PPStructureV3 결과/로그에 cp949가 못 찍는 유니코드(한자/특수기호 등)가 섞이면
#   stdout print 단계에서 UnicodeEncodeError가 발생할 수 있음.
# - stdout/stderr의 encoding은 건드리지 않고, "encoding error handler"만 바꿔서
#   출력이 절대 죽지 않게 한다. (unencodable char -> \uXXXX 형태로 escape)
# ---------------------------------------------------------------------
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")  # type: ignore[attr-defined]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="backslashreplace")  # type: ignore[attr-defined]
except Exception:
    pass


def _stage(msg: str) -> None:
    # stderr로만 진행 로그(텍스트) 출력
    print(f"[stage] {msg}", file=sys.stderr, flush=True)


def _default(obj: Any) -> Any:
    """
    json.dumps에서 ndarray/np scalar 등이 나오면 터지는 문제 방지.
    - ndarray -> list
    - np.int/np.float -> int/float
    - Path -> str
    """
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
    # JSONL: 1 page == 1 line
    print(json.dumps(payload, ensure_ascii=False, default=_default), flush=True)


def _emit_obj(payload: Dict[str, Any]) -> None:
    # 단일 이미지 모드: 1 JSON object 출력
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


def _predict_one(engine: Any, page_path: Path) -> Dict[str, Any]:
    page_name = page_path.name
    try:
        import cv2
        import numpy as np

        raw = cv2.imdecode(np.fromfile(str(page_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if raw is None:
            return {"ok": False, "page_file": page_name, "stage": "detect_load", "err": "imdecode failed"}

        try:
            output = engine.predict(input=raw)
        except Exception:
            # 일부 환경에서 input=str(path)가 더 안정적인 케이스가 있어 fallback
            try:
                output = engine.predict(input=str(page_path))
            except Exception as e:
                return {"ok": False, "page_file": page_name, "stage": "detect_predict", "err": str(e)}

        first = _first_output(output)
        pp_json = _extract_json(first)
        if not isinstance(pp_json, dict):
            return {
                "ok": False,
                "page_file": page_name,
                "stage": "parse_json",
                "err": "invalid json payload",
                "first_type": type(first).__name__ if first is not None else "None",
            }

        pp_obj = _extract_first_object_fields(first)
        pp_meta = {
            "runner_mode": "single_or_pages_dir_batch",
            "page_file": page_name,
            "first_type": type(first).__name__ if first is not None else "None",
            "json_keys": sorted(pp_json.keys()),
        }
        return {"ok": True, "page_file": page_name, "pp_json": pp_json, "pp_obj": pp_obj, "pp_meta": pp_meta}
    except Exception as e:
        return {"ok": False, "page_file": page_name, "stage": "runner_loop", "err": str(e)}


def run_pages_dir(pages_dir: Path) -> None:
    if not pages_dir.exists() or not pages_dir.is_dir():
        _emit_line({"ok": False, "page_file": "", "stage": "args", "err": f"invalid pages_dir: {pages_dir}"})
        return

    try:
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
        _stage(f"predict_start {page_name}")
        payload = _predict_one(engine, page_path)
        # JSONL 출력은 절대 죽으면 안됨
        _emit_line(payload)
        _stage(f"predict_done {page_name}")

    _stage("done")


def run_single_image(image_path: Path) -> None:
    """
    (호환성) 단일 이미지 입력 모드.
    GUI의 PaddleStructureClient._run_isolation_runner()가 이 모드를 기대한다.
    stdout에는 1개의 JSON object만 출력한다.
    """
    if not image_path.exists() or not image_path.is_file():
        _emit_obj({"ok": False, "stage": "args", "err": f"invalid image_path: {image_path}"})
        return

    try:
        from paddleocr import PPStructureV3
    except Exception as e:
        _emit_obj({"ok": False, "stage": "imports", "err": str(e)})
        return

    try:
        engine = PPStructureV3()
        _stage("init_ok(single)")
    except Exception as e:
        _emit_obj({"ok": False, "stage": "init_engine", "err": str(e)})
        return

    # 단일 이미지라도 payload는 batch와 동일한 스키마로 만들되,
    # PaddleStructureClient는 pp_json만 필수로 보므로 ok/pp_json/pp_obj/pp_meta를 포함한다.
    _stage(f"predict_start(single) {image_path.name}")
    payload = _predict_one(engine, image_path)
    if payload.get("ok"):
        _emit_obj(
            {
                "ok": True,
                "pp_json": payload.get("pp_json", {}),
                "pp_obj": payload.get("pp_obj", {}),
                "pp_meta": payload.get("pp_meta", {}),
            }
        )
    else:
        _emit_obj({"ok": False, "stage": payload.get("stage", "predict"), "err": payload.get("err", "unknown")})
    _stage(f"predict_done(single) {image_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", nargs="?", help="single image path (e.g., P001.png)")
    parser.add_argument("--pages_dir", required=False, help="directory containing P*.png files (batch mode)")
    parser.add_argument("--dpi", type=int, default=250, help="reserved (compat); not used by this runner")
    args = parser.parse_args()

    if args.pages_dir:
        run_pages_dir(Path(args.pages_dir))
        return

    if args.image_path:
        run_single_image(Path(args.image_path))
        return

    _emit_obj({"ok": False, "stage": "args", "err": "usage: v3_isolation_runner.py (--pages_dir DIR) | (image_path)"})


if __name__ == "__main__":
    main()
