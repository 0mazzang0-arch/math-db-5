import argparse
import ast
import base64
import json
import os
import re
import sys
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

_LAST_EMIT_MS = 0.0
ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs"
_ANCHOR_RE = re.compile(r"^\s*([A-C])\s*([0-9]{1,3})?\s*$")
_PAYLOAD_DROP_KEYS = {"img", "image", "dummy", "pixels", "raw", "page_bytes", "file_bytes", "input"}
_PAYLOAD_BASE_KEYS = {
    "ok",
    "page_file",
    "stage",
    "t_init_ms",
    "t_predict_ms",
    "t_page_ms",
    "t_emit_ms",
    "profile",
    "predict_flags",
    "err",
}
_MAX_LIST_LEN = 5000
_MAX_ANCHORS = 200
_MAX_OBJECTS = 200

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_new_ir"] = "0"

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")  # type: ignore[attr-defined]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="backslashreplace")  # type: ignore[attr-defined]
except Exception:
    pass


def _stage(msg: str) -> None:
    print(f"[stage] {msg}", file=sys.stderr, flush=True)


def _converter(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return "[NDARRAY_OMITTED]"
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


def _sanitize_value(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return "[NDARRAY_OMITTED]"
    except Exception:
        pass

    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            if key in _PAYLOAD_DROP_KEYS:
                continue
            if isinstance(v, list) and len(v) > _MAX_LIST_LEN:
                out[key] = "[TRUNCATED]"
                continue
            out[key] = _sanitize_value(v)
        return out

    if isinstance(value, list):
        if len(value) > _MAX_LIST_LEN:
            return "[TRUNCATED]"
        return [_sanitize_value(v) for v in value]

    if isinstance(value, tuple):
        return [_sanitize_value(v) for v in value]

    return value


def _sanitize_payload_for_emit(payload: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in _PAYLOAD_BASE_KEYS:
        if key in payload:
            compact[key] = payload[key]

    anchors = payload.get("anchors")
    if isinstance(anchors, list) and anchors:
        compact["anchors"] = anchors[:_MAX_ANCHORS]

    objects = payload.get("objects")
    if isinstance(objects, list) and objects:
        compact["objects"] = objects[:_MAX_OBJECTS]

    return _sanitize_value(compact)


def _emit_json(payload: Dict[str, Any], fallback_page_file: str = "") -> float:
    global _LAST_EMIT_MS
    emit_start = time.perf_counter()
    try:
        payload.setdefault("page_file", fallback_page_file)
        payload["t_emit_ms"] = round(_LAST_EMIT_MS, 3)
        sanitized_payload = _sanitize_payload_for_emit(payload)
        print(json.dumps(sanitized_payload, ensure_ascii=True, default=_converter), flush=True)
    except Exception as e:
        err_payload = {
            "ok": False,
            "stage": "emit",
            "err": str(e),
            "page_file": payload.get("page_file", fallback_page_file),
            "t_emit_ms": round(_LAST_EMIT_MS, 3),
        }
        try:
            print(json.dumps(_sanitize_payload_for_emit(err_payload), ensure_ascii=True, default=_converter), flush=True)
        except Exception:
            sys.stdout.write('{"ok": false, "stage": "emit", "err": "emit failed", "page_file": "__BATCH__", "t_emit_ms": 0.0}\n')
            sys.stdout.flush()
    _LAST_EMIT_MS = (time.perf_counter() - emit_start) * 1000.0
    return _LAST_EMIT_MS


def _sorted_page_files(pages_dir: Path) -> List[Path]:
    files = [p for p in pages_dir.glob("P*.png") if p.is_file()]

    def _key(p: Path) -> Tuple[int, int, str]:
        m = re.match(r"^P(\d+)\.png$", p.name)
        return (0, int(m.group(1)), p.name) if m else (1, 0, p.name)

    return sorted(files, key=_key)


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


def _extract_first_object_fields(first: Any, keys: List[str] | None = None) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    selected_keys = keys or ["overall_ocr_res", "parsing_res_list", "region_det_res", "layout_det_res", "table_res_list"]
    for key in selected_keys:
        try:
            value = first.get(key) if isinstance(first, dict) else getattr(first, key, None)
        except Exception:
            value = None
        if value is not None:
            fields[key] = value
    return fields


def _sanitize_pp_json(pp_json: Any) -> Dict[str, Any]:
    if not isinstance(pp_json, dict):
        return {}
    result = dict(pp_json)
    res_val = result.get("res")
    if isinstance(res_val, str):
        parsed = None
        try:
            parsed = json.loads(res_val)
        except Exception:
            try:
                parsed = ast.literal_eval(res_val)
            except Exception:
                parsed = None
        if isinstance(parsed, (dict, list)):
            result["res"] = parsed
        else:
            return {}
    return result


def _iter_dicts(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_dicts(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from _iter_dicts(it)


def _extract_bbox(node: Dict[str, Any]) -> List[int]:
    cand = node.get("bbox") or node.get("box") or node.get("rect")
    if isinstance(cand, (list, tuple)) and len(cand) == 4:
        return [int(float(cand[0])), int(float(cand[1])), int(float(cand[2])), int(float(cand[3]))]
    poly = node.get("poly") or node.get("polygon") or node.get("points")
    if isinstance(poly, (list, tuple)) and len(poly) >= 4:
        pts = []
        for p in poly:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    pts.append((float(p[0]), float(p[1])))
                except Exception:
                    pass
        if pts:
            xs = [x for x, _ in pts]
            ys = [y for _, y in pts]
            return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
    return []


def _extract_text(node: Dict[str, Any]) -> str:
    for k in ("text", "transcription", "label", "cls", "type", "content"):
        v = node.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_min_entities(pp_json: Dict[str, Any], pp_obj: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    anchors: List[Dict[str, Any]] = []
    objects: List[Dict[str, Any]] = []
    seen_anchor = set()
    seen_obj = set()

    for src in (pp_obj, pp_json):
        for node in _iter_dicts(src):
            bbox = _extract_bbox(node)
            if not bbox:
                continue
            text = _extract_text(node)
            m = _ANCHOR_RE.match(text.upper()) if text else None
            if m:
                key = tuple(bbox)
                if key not in seen_anchor:
                    seen_anchor.add(key)
                    anchors.append({"id": text, "bbox": bbox, "col": 0})
            low = (text or "").lower()
            obj_type = ""
            if any(t in low for t in ["figure", "fig", "image", "photo"]):
                obj_type = "figure"
            elif any(t in low for t in ["table", "tbl"]):
                obj_type = "table"
            elif node.get("type") in ("figure", "table"):
                obj_type = str(node.get("type"))
            if obj_type:
                key = (obj_type, *bbox)
                if key not in seen_obj:
                    seen_obj.add(key)
                    objects.append({"type": obj_type, "bbox": bbox})

    anchors.sort(key=lambda a: (a.get("bbox", [0, 0, 0, 0])[1], a.get("bbox", [0, 0, 0, 0])[0]))
    return anchors, objects


def _predict_flags(profile: str, force_region_detection: int) -> Dict[str, Any]:
    if profile == "full":
        flags: Dict[str, Any] = {}
    else:
        flags = {
            "use_table_recognition": False,
            "use_formula_recognition": False,
            "use_chart_recognition": False,
            "use_seal_recognition": False,
            "use_region_detection": True,
        }
    if force_region_detection in (0, 1):
        flags["use_region_detection"] = bool(force_region_detection)
    return flags


def _predict_with_fallback(engine: Any, inp: Any, predict_kwargs: Dict[str, Any]) -> Any:
    if not predict_kwargs:
        return engine.predict(input=inp)
    try:
        return engine.predict(input=inp, **predict_kwargs)
    except TypeError:
        return engine.predict(input=inp)
    except Exception as e:
        msg = str(e)
        if "Unknown argument" in msg or "unexpected keyword" in msg:
            return engine.predict(input=inp)
        raise


def _export_full_yaml_if_missing(full_yaml: Path) -> None:
    if full_yaml.exists():
        return
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from paddleocr import PPStructureV3

        PPStructureV3().export_paddlex_config_to_yaml(str(full_yaml))
        _stage(f"exported_full_yaml {full_yaml.name}")
    except Exception as e:
        _stage(f"export_full_yaml_skip err={e}")


def _load_yaml_with_fallback(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        ruby_script = (
            "require 'yaml'; require 'json'; "
            "obj = YAML.safe_load(File.read(ARGV[0]), aliases: true) || {}; "
            "puts JSON.generate(obj)"
        )
        proc = subprocess.run(
            ["ruby", "-e", ruby_script, str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "ruby yaml load failed")
        loaded = json.loads(proc.stdout)
        return loaded if isinstance(loaded, dict) else {}


def _dump_yaml_with_fallback(data: Dict[str, Any], path: Path) -> None:
    try:
        import yaml  # type: ignore

        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")
        return
    except Exception:
        ruby_script = (
            "require 'yaml'; require 'json'; "
            "obj = JSON.parse(STDIN.read); "
            "File.write(ARGV[0], YAML.dump(obj))"
        )
        proc = subprocess.run(
            ["ruby", "-e", ruby_script, str(path)],
            input=json.dumps(data, ensure_ascii=True),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "ruby yaml dump failed")


def _validate_fast_yaml_region_detection(fast_yaml: Path) -> bool:
    try:
        data = _load_yaml_with_fallback(fast_yaml)
    except Exception as e:
        _stage(f"fast_yaml_validate_read_failed err={e}")
        return False

    submodules = data.get("SubModules") if isinstance(data, dict) else None
    region = submodules.get("RegionDetection") if isinstance(submodules, dict) else None
    if not isinstance(region, dict):
        _stage("fast_yaml_validate_missing RegionDetection")
        return False

    required_keys = ["model_name", "module_name", "model_dir"]
    missing = [k for k in required_keys if k not in region]
    if missing:
        _stage(f"fast_yaml_validate_missing_keys {','.join(missing)}")
        return False
    return True


def _make_fast_yaml_from_full(full_yaml: Path, fast_yaml: Path) -> bool:
    if not full_yaml.exists():
        _stage("make_fast_yaml_skip full_yaml_missing")
        return False
    try:
        data = _load_yaml_with_fallback(full_yaml)
        if not isinstance(data, dict) or not data:
            _stage("make_fast_yaml_skip full_yaml_invalid")
            return False

        data["use_table_recognition"] = False
        data["use_formula_recognition"] = False
        data["use_chart_recognition"] = False
        data["use_seal_recognition"] = False
        data["use_region_detection"] = True

        subpipelines = data.get("SubPipelines") if isinstance(data.get("SubPipelines"), dict) else None
        if isinstance(subpipelines, dict):
            doc_prep = subpipelines.get("DocPreprocessor")
            if isinstance(doc_prep, dict):
                if "use_doc_orientation_classify" in doc_prep:
                    doc_prep["use_doc_orientation_classify"] = False
                if "use_doc_unwarping" in doc_prep:
                    doc_prep["use_doc_unwarping"] = False

        _dump_yaml_with_fallback(data, fast_yaml)
        if not _validate_fast_yaml_region_detection(fast_yaml):
            _stage("make_fast_yaml_skip validation_failed")
            return False

        _stage(f"generated_fast_yaml {fast_yaml.name}")
        return True
    except Exception as e:
        _stage(f"make_fast_yaml_skip err={e}")
        return False


def _sanitize_fast_yaml_doc_preprocessor(fast_cfg: Path) -> bool:
    if not fast_cfg.exists():
        return False
    try:
        lines = fast_cfg.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        _stage(f"fast_yaml_read_failed err={e}")
        return False

    changed = False
    out: List[str] = []
    for line in lines:
        if re.match(r"^use_doc_preprocessor:\s*(true|True)\s*$", line):
            out.append("use_doc_preprocessor: false")
            changed = True
            continue
        out.append(line)

    if changed:
        try:
            fast_cfg.write_text("\n".join(out) + "\n", encoding="utf-8")
            _stage("fast_yaml_sanity_fixed use_doc_preprocessor=false")
        except Exception as e:
            _stage(f"fast_yaml_sanity_write_failed err={e}")
            return False
    return changed


def _recover_fast_yaml(full_cfg: Path, fast_cfg: Path, init_err: Exception) -> None:
    _stage(f"fast_init_failed err={init_err}")
    removed = False
    try:
        if fast_cfg.exists():
            fast_cfg.unlink()
            removed = True
    except Exception as e:
        _stage(f"fast_yaml_remove_failed err={e}")
    else:
        _stage(f"fast_yaml_removed={removed}")

    _export_full_yaml_if_missing(full_cfg)
    made = _make_fast_yaml_from_full(full_cfg, fast_cfg)
    if made:
        _sanitize_fast_yaml_doc_preprocessor(fast_cfg)
    _stage(f"fast_yaml_regenerated exists={fast_cfg.exists()}")


def _load_engine(profile: str) -> Any:
    from paddleocr import PPStructureV3

    full_cfg = CONFIG_DIR / "PP-StructureV3_full.yaml"
    fast_cfg = CONFIG_DIR / "PP-StructureV3_fast.yaml"

    if profile == "fast":
        if not fast_cfg.exists():
            _stage("fast_yaml_missing regenerate")
            _export_full_yaml_if_missing(full_cfg)
            created = _make_fast_yaml_from_full(full_cfg, fast_cfg)
            if not created:
                _stage("fast_yaml_generate_failed fallback_full")

        if fast_cfg.exists():
            _sanitize_fast_yaml_doc_preprocessor(fast_cfg)
            if not _validate_fast_yaml_region_detection(fast_cfg):
                _stage("fast_yaml_invalid fallback_full")
            else:
                try:
                    return PPStructureV3(paddlex_config=str(fast_cfg))
                except Exception as e:
                    err_msg = str(e)
                    if "block_region_detection_model" not in err_msg and "doc_preprocessor_pipeline" not in err_msg:
                        raise
                    _recover_fast_yaml(full_cfg, fast_cfg, e)
                    _stage("fast_init_retry")
                    if fast_cfg.exists() and _validate_fast_yaml_region_detection(fast_cfg):
                        return PPStructureV3(paddlex_config=str(fast_cfg))
                    _stage("fast_yaml_retry_invalid fallback_full")

        _export_full_yaml_if_missing(full_cfg)
        if full_cfg.exists():
            _stage("fast_fallback_to_full_config")
            return PPStructureV3(paddlex_config=str(full_cfg))

    elif profile == "full" and not full_cfg.exists():
        _export_full_yaml_if_missing(full_cfg)

    if profile == "full" and full_cfg.exists():
        return PPStructureV3(paddlex_config=str(full_cfg))
    return PPStructureV3()


def _warmup_once(engine: Any, enabled: bool, profile: str, force_region_detection: int) -> None:
    if not enabled:
        return
    try:
        import numpy as np

        dummy = np.full((64, 64, 3), 255, dtype=np.uint8)
        _stage("warmup_start")
        _predict_with_fallback(engine, dummy, _predict_flags(profile, force_region_detection))
        _stage("warmup_done")
    except Exception as e:
        _stage(f"warmup_skip err={e}")


def _predict_one(engine: Any, page_path: Path, t_init_ms: float, profile: str, force_region_detection: int) -> Dict[str, Any]:
    page_name = page_path.name
    page_start = time.perf_counter()
    flags = _predict_flags(profile, force_region_detection)
    try:
        import cv2
        import numpy as np

        raw = cv2.imdecode(np.fromfile(str(page_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if raw is None:
            t_page_ms = (time.perf_counter() - page_start) * 1000.0
            return {
                "ok": False,
                "page_file": page_name,
                "stage": "detect_load",
                "err": "imdecode failed",
                "t_init_ms": round(t_init_ms, 3),
                "t_predict_ms": round(t_page_ms, 3),
                "t_page_ms": round(t_page_ms, 3),
                "predict_flags": flags,
            }

        predict_start = time.perf_counter()
        try:
            output = _predict_with_fallback(engine, raw, flags)
        except Exception:
            output = _predict_with_fallback(engine, str(page_path), flags)

        first = _first_output(output)
        pp_json = _sanitize_pp_json(_extract_json(first))
        t_predict_ms = (time.perf_counter() - predict_start) * 1000.0
        t_page_ms = (time.perf_counter() - page_start) * 1000.0
        if not isinstance(pp_json, dict):
            return {
                "ok": False,
                "page_file": page_name,
                "stage": "parse_json",
                "err": "invalid json payload",
                "t_init_ms": round(t_init_ms, 3),
                "t_predict_ms": round(t_predict_ms, 3),
                "t_page_ms": round(t_page_ms, 3),
                "predict_flags": flags,
            }

        pp_obj = _extract_first_object_fields(first)
        anchors, objects = _extract_min_entities(pp_json, pp_obj)
        return {
            "ok": True,
            "page_file": page_name,
            "stage": "predict_done",
            "anchors": anchors,
            "objects": objects,
            "t_init_ms": round(t_init_ms, 3),
            "t_predict_ms": round(t_predict_ms, 3),
            "t_page_ms": round(t_page_ms, 3),
            "predict_flags": flags,
        }
    except Exception as e:
        t_page_ms = (time.perf_counter() - page_start) * 1000.0
        return {
            "ok": False,
            "page_file": page_name,
            "stage": "runner_loop",
            "err": str(e),
            "t_init_ms": round(t_init_ms, 3),
            "t_predict_ms": round(t_page_ms, 3),
            "t_page_ms": round(t_page_ms, 3),
            "predict_flags": flags,
        }


def run_pages_dir(pages_dir: Path, warmup: bool, profile: str, force_region_detection: int, payload_mode: str) -> int:
    if not pages_dir.exists() or not pages_dir.is_dir():
        _emit_json({"ok": False, "page_file": "__BATCH__", "stage": "args", "err": f"invalid pages_dir: {pages_dir}", "profile": profile})
        return 1

    if payload_mode != "min":
        _stage("payload_mode_forced min")

    page_files = _sorted_page_files(pages_dir)
    _stage(f"start pages={len(page_files)} profile={profile}")
    if not page_files:
        _emit_json({"ok": False, "page_file": "__BATCH__", "stage": "load_pages", "err": "no P*.png files", "profile": profile})
        return 1

    init_start = time.perf_counter()
    try:
        engine = _load_engine(profile)
        t_init_ms = (time.perf_counter() - init_start) * 1000.0
        _stage(f"init_ok t_init_ms={t_init_ms:.1f}")
    except Exception as e:
        _emit_json({"ok": False, "page_file": "__BATCH__", "stage": "init_engine", "err": str(e), "profile": profile})
        return 1

    _warmup_once(engine, warmup, profile, force_region_detection)

    for page_path in page_files:
        payload = _predict_one(
            engine,
            page_path,
            t_init_ms=t_init_ms,
            profile=profile,
            force_region_detection=force_region_detection,
        )
        payload["profile"] = profile
        t_emit_ms = _emit_json(payload, fallback_page_file=page_path.name)
        _stage(
            f"predict_done {page_path.name} t_init_ms={float(payload.get('t_init_ms', 0.0)):.1f} "
            f"t_predict_ms={float(payload.get('t_predict_ms', 0.0)):.1f} t_emit_ms={t_emit_ms:.1f} "
            f"t_page_ms={float(payload.get('t_page_ms', 0.0)):.1f}"
        )

    return 0


def run_single_image(image_path: Path, warmup: bool, profile: str, force_region_detection: int, payload_mode: str) -> int:
    if not image_path.exists() or not image_path.is_file():
        _emit_json({"ok": False, "page_file": image_path.name, "stage": "args", "err": f"invalid image_path: {image_path}", "profile": profile})
        return 1

    if payload_mode != "min":
        _stage("payload_mode_forced min")

    init_start = time.perf_counter()
    try:
        engine = _load_engine(profile)
        t_init_ms = (time.perf_counter() - init_start) * 1000.0
        _stage(f"init_ok(single) t_init_ms={t_init_ms:.1f}")
    except Exception as e:
        _emit_json({"ok": False, "page_file": image_path.name, "stage": "init_engine", "err": str(e), "profile": profile})
        return 1

    _warmup_once(engine, warmup, profile, force_region_detection)
    payload = _predict_one(
        engine,
        image_path,
        t_init_ms=t_init_ms,
        profile=profile,
        force_region_detection=force_region_detection,
    )
    payload["profile"] = profile
    t_emit_ms = _emit_json(payload, fallback_page_file=image_path.name)
    _stage(
        f"predict_done {image_path.name} t_init_ms={float(payload.get('t_init_ms', 0.0)):.1f} "
        f"t_predict_ms={float(payload.get('t_predict_ms', 0.0)):.1f} t_emit_ms={t_emit_ms:.1f} "
        f"t_page_ms={float(payload.get('t_page_ms', 0.0)):.1f}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", nargs="?", help="single image path (e.g., P001.png)")
    parser.add_argument("--pages_dir", required=False, help="directory containing P*.png files (batch mode)")
    parser.add_argument("--dpi", type=int, default=250, help="reserved")
    parser.add_argument("--warmup", type=int, choices=[0, 1], default=1)
    parser.add_argument("--profile", choices=["fast", "full"], default="fast")
    parser.add_argument("--force_region_detection", type=int, choices=[-1, 0, 1], default=-1)
    parser.add_argument("--payload", choices=["min", "full"], default="min")
    args = parser.parse_args()

    if args.pages_dir:
        return run_pages_dir(
            Path(args.pages_dir),
            warmup=bool(args.warmup),
            profile=args.profile,
            force_region_detection=args.force_region_detection,
            payload_mode=args.payload,
        )
    if args.image_path:
        return run_single_image(
            Path(args.image_path),
            warmup=bool(args.warmup),
            profile=args.profile,
            force_region_detection=args.force_region_detection,
            payload_mode=args.payload,
        )

    _emit_json({"ok": False, "page_file": "__BATCH__", "stage": "args", "err": "usage: v3_isolation_runner.py (--pages_dir DIR) | (image_path)", "profile": args.profile})
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
