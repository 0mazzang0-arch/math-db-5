import argparse
import ast
import base64
import json
import os
import re
import sys
import time
import traceback
import subprocess
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

_LAST_EMIT_MS = 0.0
ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs"
_ANCHOR_RE = re.compile(r"^\s*([A-C])\s*([0-9]{1,3})?\s*$")
_LEADING_NUM_RE = re.compile(r"^\s*(\d{4})(?=[\s\.\)\]]|$)")
_ANCHOR_4DIGIT_RELAXED_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")
_LABEL_LIKE_TEXTS = {"text", "footer", "formula", "number", "header", "image", "table", "figure", "caption", "title", "reference", "equation", "aside_text", "footnote"}
_PAYLOAD_DROP_KEYS = {"img", "image", "dummy", "pixels", "raw", "page_bytes", "file_bytes", "input"}
_MIN_PAYLOAD_KEYS = {
    "ok",
    "page_file",
    "profile",
    "stage",
    "t_init_ms",
    "t_predict_ms",
    "t_page_ms",
    "t_emit_ms",
    "predict_flags",
}
_MAX_LIST_LEN = 2000
_MAX_DICT_KEYS = 200
_MAX_DUMP_CHARS = 200 * 1024
_MAX_MIN_ANCHORS = 50
_SKIP_HEAVY_KEYS = {"img", "image", "pixels", "raw", "mask", "score_map", "heatmap", "bitmap", "input"}
_QUIET = False
_JSONL_OUT_PATH = ""

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_new_ir"] = "0"
os.environ["FLAGS_logtostderr"] = "1"
os.environ["GLOG_logtostderr"] = "1"

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")  # type: ignore[attr-defined]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="backslashreplace")  # type: ignore[attr-defined]
except Exception:
    pass


def _stage(msg: str) -> None:
    if _QUIET and not any(tok in msg.lower() for tok in ("error", "failed", "fatal")):
        return
    print(f"[stage] {msg}", file=sys.stderr, flush=True)


def _write_jsonl_line(line: str) -> None:
    if _JSONL_OUT_PATH:
        with open(_JSONL_OUT_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


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


def _safe_debug_json(value: Any) -> str:
    try:
        return json.dumps(_sanitize_value(value), ensure_ascii=True, default=_converter)
    except Exception:
        try:
            return str(value)
        except Exception:
            return '"<unprintable>"'


def _to_builtin_int_map(data: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not isinstance(data, dict):
        return out
    for k, v in data.items():
        try:
            out[str(k)] = int(v)
        except Exception:
            continue
    return out


def _sanitize_value(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return "[NDARRAY_OMITTED]"
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
    except Exception:
        pass

    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= _MAX_DICT_KEYS:
                out["__truncated_keys__"] = "[TRUNCATED]"
                break
            key = str(k)
            if key in _PAYLOAD_DROP_KEYS:
                continue
            out[key] = _sanitize_value(v)
        return out

    if isinstance(value, list):
        if len(value) > _MAX_LIST_LEN:
            return [_sanitize_value(v) for v in value[:_MAX_LIST_LEN]] + ["[TRUNCATED]"]
        return [_sanitize_value(v) for v in value]

    if isinstance(value, tuple):
        as_list = list(value)
        if len(as_list) > _MAX_LIST_LEN:
            as_list = as_list[:_MAX_LIST_LEN] + ["[TRUNCATED]"]
        return [_sanitize_value(v) for v in as_list]

    return value


def _build_min_anchors(anchors: Any) -> List[Dict[str, Any]]:
    if not isinstance(anchors, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in anchors[:_MAX_MIN_ANCHORS]:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        try:
            bbox_i = [int(float(v)) for v in bbox]
        except Exception:
            continue
        text = item.get("text") or item.get("id") or ""
        text_s = str(text).strip()
        n_found = _find_anchor_number(text_s)
        m = _ANCHOR_RE.match(text_s.upper()) if text_s else None
        n_val = n_found if n_found is not None else -1
        if n_val < 0 and m and m.group(2):
            try:
                n_val = int(m.group(2))
            except Exception:
                n_val = -1
        out.append({"n": n_val, "bbox": bbox_i, "text": text_s})
    return out

def _build_min_payload(payload: Dict[str, Any], fallback_page_file: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"page_file": payload.get("page_file", fallback_page_file)}
    for key in _MIN_PAYLOAD_KEYS:
        if key in payload:
            out[key] = payload[key]

    anchors_min = _build_min_anchors(payload.get("anchors"))
    out["anchors"] = anchors_min
    out["anchors_count"] = len(anchors_min)

    objects_min = payload.get("objects")
    if isinstance(objects_min, list):
        out["objects"] = _sanitize_value(objects_min[:_MAX_MIN_ANCHORS])
    else:
        out["objects"] = []

    out.setdefault("stage", payload.get("stage", "predict_done"))
    out.setdefault("profile", payload.get("profile", "fast"))
    out.setdefault("ok", bool(payload.get("ok", False)))
    if not bool(out.get("ok", False)):
        if "err" in payload:
            out["err"] = str(payload.get("err"))
        if "exc_type" in payload:
            out["exc_type"] = str(payload.get("exc_type"))
    return out


def _truncate_heavy_top_level(payload: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for k, v in payload.items():
        if isinstance(v, (list, dict, tuple)):
            compact[k] = "[TRUNCATED]"
        else:
            compact[k] = v
    return compact


def _emit_json(payload: Dict[str, Any], fallback_page_file: str = "", payload_mode: str = "min") -> float:
    global _LAST_EMIT_MS
    emit_start = time.perf_counter()
    try:
        payload.setdefault("page_file", fallback_page_file)
        payload["t_emit_ms"] = round(_LAST_EMIT_MS, 3)
        if payload_mode == "min":
            out_payload = _build_min_payload(payload, fallback_page_file)
            dumped = json.dumps(out_payload, ensure_ascii=True, default=_converter)
        else:
            out_payload = _sanitize_value(payload)
            dumped = json.dumps(out_payload, ensure_ascii=True, default=_converter)
            if len(dumped) > _MAX_DUMP_CHARS:
                out_payload = _truncate_heavy_top_level(out_payload)
                dumped = json.dumps(out_payload, ensure_ascii=True, default=_converter)

        _write_jsonl_line(dumped)
        if (not _QUIET) or (not bool(out_payload.get("ok", True))):
            print(dumped, flush=True)
    except Exception as e:
        err_payload = {
            "ok": False,
            "stage": "emit",
            "page_file": payload.get("page_file", fallback_page_file),
            "profile": payload.get("profile", "fast"),
            "t_emit_ms": round(_LAST_EMIT_MS, 3),
        }
        try:
            dumped_err = json.dumps(_build_min_payload(err_payload, fallback_page_file), ensure_ascii=True, default=_converter)
            _write_jsonl_line(dumped_err)
            print(dumped_err, flush=True)
        except Exception:
            sys.stdout.write('{"ok": false, "stage": "emit", "page_file": "__BATCH__", "profile": "fast", "t_emit_ms": 0.0}\n')
            sys.stdout.flush()
        _stage(f"emit_error err={e}")
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


def _bbox_valid(bbox: Any) -> bool:
    return isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)


def _first_non_none(node: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    for key in keys:
        if key in node:
            value = node.get(key)
            if value is not None:
                return value
    return None


def _extract_bbox(node: Dict[str, Any]) -> List[int]:
    cand = _first_non_none(
        node,
        ("bbox", "dt_bbox", "dt_box", "dt_boxes", "rec_box", "rec_boxes", "box", "boxes", "rect"),
    )
    if hasattr(cand, "tolist"):
        try:
            cand = cand.tolist()
        except Exception:
            pass

    if isinstance(cand, dict):
        if all(k in cand for k in ("x", "y", "w", "h")):
            try:
                x, y, w, h = float(cand["x"]), float(cand["y"]), float(cand["w"]), float(cand["h"])
                return [int(x), int(y), int(x + w), int(y + h)]
            except Exception:
                pass
        if all(k in cand for k in ("left", "top", "right", "bottom")):
            try:
                return [int(float(cand["left"])), int(float(cand["top"])), int(float(cand["right"])), int(float(cand["bottom"]))]
            except Exception:
                pass

    if isinstance(cand, (list, tuple)) and len(cand) == 4 and all(not isinstance(x, (list, tuple, dict)) for x in cand):
        try:
            return [int(float(cand[0])), int(float(cand[1])), int(float(cand[2])), int(float(cand[3]))]
        except Exception:
            pass

    poly = _first_non_none(
        node,
        ("poly", "dt_poly", "dt_polys", "rec_poly", "rec_polys", "polygon", "points", "polys"),
    )
    return _poly_to_bbox(poly)

def _extract_text(node: Dict[str, Any]) -> str:
    for k in ("text", "rec_text", "transcription", "ocr_text", "ocr", "line"):
        v = node.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _poly_to_bbox(poly: Any) -> List[int]:
    if hasattr(poly, "tolist"):
        try:
            poly = poly.tolist()
        except Exception:
            pass

    if isinstance(poly, dict):
        nested = None
        for key in ("points", "poly", "polygon", "dt_polys", "dt_poly", "polys", "boxes", "box"):
            if key in poly:
                val = poly.get(key)
                if val is not None:
                    nested = val
                    break
        if nested is not None:
            poly = nested

    if hasattr(poly, "tolist"):
        try:
            poly = poly.tolist()
        except Exception:
            pass

    if not isinstance(poly, (list, tuple)):
        return []

    if all(not isinstance(p, (list, tuple, dict)) for p in poly):
        nums = []
        for p in poly:
            try:
                nums.append(float(p))
            except Exception:
                pass
        if len(nums) == 4:
            return [int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])]
        if len(nums) >= 8 and len(nums) % 2 == 0:
            xs = nums[0::2]
            ys = nums[1::2]
            return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

    pts: List[Tuple[float, float]] = []
    for p in poly:
        if hasattr(p, "tolist"):
            try:
                p = p.tolist()
            except Exception:
                pass
        if isinstance(p, (list, tuple)) and len(p) >= 2 and not isinstance(p[0], (list, tuple, dict)):
            try:
                pts.append((float(p[0]), float(p[1])))
            except Exception:
                pass
    if not pts:
        return []
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

def _box_to_bbox(box: Any) -> List[int]:
    if hasattr(box, "tolist"):
        try:
            box = box.tolist()
        except Exception:
            pass
    if isinstance(box, dict):
        return _extract_bbox(box)
    if isinstance(box, (list, tuple)):
        if len(box) == 4 and all(not isinstance(v, (list, tuple, dict)) for v in box):
            try:
                return [int(float(box[0])), int(float(box[1])), int(float(box[2])), int(float(box[3]))]
            except Exception:
                pass
        return _poly_to_bbox(box)
    return []


def _is_poly_like(value: Any) -> bool:
    pb = _poly_to_bbox(value)
    if _bbox_valid(pb):
        return True
    bb = _box_to_bbox(value)
    return _bbox_valid(bb)

def _iter_ocr_items(obj: Any, inherited_bbox: List[int] | None = None, zip_key_counts: Dict[str, int] | None = None):
    if isinstance(obj, dict):
        extracted_bbox = _extract_bbox(obj)
        cur_bbox = extracted_bbox if _bbox_valid(extracted_bbox) else (inherited_bbox if _bbox_valid(inherited_bbox) else [])

        text = _extract_text(obj)
        if text:
            yield text, cur_bbox

        def _arr(name: str) -> Any:
            v = obj.get(name)
            if hasattr(v, "tolist"):
                try:
                    v = v.tolist()
                except Exception:
                    pass
            return v

        text_keys = ("rec_texts", "rec_text", "rec_res", "texts", "text")
        box_keys = ("rec_boxes", "dt_boxes", "boxes", "bboxes", "dt_bboxes")
        poly_keys = ("rec_polys", "dt_polys", "polys", "dt_poly")

        text_arr = None
        text_key = ""
        for k in text_keys:
            v = _arr(k)
            if isinstance(v, (list, tuple)):
                if zip_key_counts is not None:
                    try:
                        zip_key_counts[k] = int(len(v))
                    except Exception:
                        pass
                if len(v) > 0 and all(isinstance(x, str) for x in v):
                    text_arr = v
                    text_key = k
                    break

        pair_consumed = False
        paired_keys: set[str] = set()
        if text_arr is not None:
            box_arr = None
            box_key = ""
            for k in box_keys:
                v = _arr(k)
                if isinstance(v, (list, tuple)):
                    if zip_key_counts is not None:
                        try:
                            zip_key_counts[k] = int(len(v))
                        except Exception:
                            pass
                    if len(v) > 0:
                        box_arr = v
                        box_key = k
                        break

            poly_arr = None
            poly_key = ""
            for k in poly_keys:
                v = _arr(k)
                if isinstance(v, (list, tuple)):
                    if zip_key_counts is not None:
                        try:
                            zip_key_counts[k] = int(len(v))
                        except Exception:
                            pass
                    if len(v) > 0:
                        poly_arr = v
                        poly_key = k
                        break

            if isinstance(box_arr, (list, tuple)) and len(box_arr) == len(text_arr):
                for txt, box in zip(text_arr, box_arr):
                    if isinstance(txt, str) and txt.strip():
                        bb = _box_to_bbox(box)
                        if not _bbox_valid(bb):
                            bb = cur_bbox
                        yield txt, bb
                pair_consumed = True
                paired_keys.update({text_key, box_key})
            elif isinstance(poly_arr, (list, tuple)) and len(poly_arr) == len(text_arr):
                for txt, poly in zip(text_arr, poly_arr):
                    if isinstance(txt, str) and txt.strip():
                        pb = _poly_to_bbox(poly)
                        if not _bbox_valid(pb):
                            pb = cur_bbox
                        yield txt, pb
                pair_consumed = True
                paired_keys.update({text_key, poly_key})

        skip_keys = {
            "dt_polys", "polys", "boxes", "dt_boxes", "rec_text", "rec_texts", "rec_res", "texts", "text",
            "rec_boxes", "bboxes", "dt_bboxes", "rec_polys", "dt_poly", "rec_poly",
        } | _SKIP_HEAVY_KEYS
        if pair_consumed:
            skip_keys |= paired_keys
        for k, v in obj.items():
            if k in skip_keys:
                continue
            yield from _iter_ocr_items(v, cur_bbox, zip_key_counts)
        return

    if isinstance(obj, (list, tuple)):
        paired = False
        if len(obj) >= 2:
            a, b = obj[0], obj[1]
            if _is_poly_like(a):
                pb = _poly_to_bbox(a)
                if not _bbox_valid(pb):
                    pb = _box_to_bbox(a)
                txt = ""
                if isinstance(b, str):
                    txt = b
                elif isinstance(b, (list, tuple)) and len(b) > 0 and isinstance(b[0], str):
                    txt = b[0]
                if txt:
                    if not _bbox_valid(pb):
                        pb = inherited_bbox if _bbox_valid(inherited_bbox) else []
                    yield txt, pb
                    paired = True
            elif _is_poly_like(b):
                pb = _poly_to_bbox(b)
                if not _bbox_valid(pb):
                    pb = _box_to_bbox(b)
                txt = ""
                if isinstance(a, str):
                    txt = a
                elif isinstance(a, (list, tuple)) and len(a) > 0 and isinstance(a[0], str):
                    txt = a[0]
                if txt:
                    if not _bbox_valid(pb):
                        pb = inherited_bbox if _bbox_valid(inherited_bbox) else []
                    yield txt, pb
                    paired = True

        if paired:
            return

        for it in obj:
            yield from _iter_ocr_items(it, inherited_bbox, zip_key_counts)

def _normalize_anchor_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    return " ".join(t.strip().split())


def _is_label_like_text(text: str) -> bool:
    if not text:
        return True
    low = text.lower().strip()
    return low in _LABEL_LIKE_TEXTS


def _normalize_digit_candidate(text: str) -> str:
    t = _normalize_anchor_text(text)
    core = re.sub(r"[\s\]\[\(\)\.-]", "", t)
    if not core:
        return t
    if any(ch.isalpha() and ch not in "OoIlSB" for ch in core):
        return t
    digitish = sum(ch.isdigit() or ch in "OoIlSB" for ch in core)
    if digitish / max(1, len(core)) < 0.8:
        return t
    trans = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"})
    return t.translate(trans)


def _find_anchor_number(text: str) -> int | None:
    if not text:
        return None
    m = _LEADING_NUM_RE.match(text)
    if m:
        return int(m.group(1))
    prefix = text[:6]
    m2 = _ANCHOR_4DIGIT_RELAXED_RE.search(prefix)
    if m2:
        return int(m2.group(1))
    return None


def _digit_only(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def _build_fragment_anchors(fragments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str], str]:
    if not fragments:
        return [], [], ""

    frags = sorted(fragments, key=lambda f: (f["yc"], f["bbox"][0]))
    lines: List[List[Dict[str, Any]]] = []
    for frag in frags:
        if not lines:
            lines.append([frag])
            continue
        last_line = lines[-1]
        avg_y = sum(it["yc"] for it in last_line) / len(last_line)
        if abs(frag["yc"] - avg_y) <= 14:
            last_line.append(frag)
        else:
            lines.append([frag])

    made: List[Dict[str, Any]] = []
    dbg_parts: List[str] = []
    dbg_joined = ""

    for line in lines:
        line.sort(key=lambda f: f["bbox"][0])
        parts = [f["digits"] for f in line if f.get("digits")]
        if not parts:
            continue
        joined = "".join(parts)
        if 1 <= len(joined) <= 3:
            z = joined.zfill(4)
            n_val = int(z)
            if 1 <= n_val <= 9999:
                xs1=[b["bbox"][0] for b in line]; ys1=[b["bbox"][1] for b in line]
                xs2=[b["bbox"][2] for b in line]; ys2=[b["bbox"][3] for b in line]
                bbox=[min(xs1), min(ys1), max(xs2), max(ys2)]
                made.append({"id": z, "text": z, "n": n_val, "bbox": bbox, "col": 0})
                if not dbg_parts:
                    dbg_parts = parts[:]
                    dbg_joined = z
            continue
        if len(joined) < 4:
            continue

        starts: List[int] = []
        pos = 0
        for part in parts:
            starts.append(pos)
            pos += len(part)

        for m in re.finditer(r"(?<!\d)(\d{4})(?!\d)", joined):
            n_val = int(m.group(1))
            if n_val <= 0:
                continue
            s_idx, e_idx = m.start(1), m.end(1)
            idxs = []
            for i, st in enumerate(starts):
                en = st + len(parts[i])
                if not (en <= s_idx or st >= e_idx):
                    idxs.append(i)
            if not idxs:
                continue
            xs1=[]; ys1=[]; xs2=[]; ys2=[]
            for i in idxs:
                b=line[i]["bbox"]
                xs1.append(b[0]); ys1.append(b[1]); xs2.append(b[2]); ys2.append(b[3])
            bbox=[min(xs1), min(ys1), max(xs2), max(ys2)]
            txt=m.group(1)
            made.append({"id": txt, "text": txt, "n": n_val, "bbox": bbox, "col": 0})
            if not dbg_parts:
                dbg_parts = parts[:]
                dbg_joined = joined

    return made, dbg_parts[:5], dbg_joined


def _extract_min_entities(pp_json: Dict[str, Any], pp_obj: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int, int, int, List[str], List[str], str, Dict[str, int]]:
    anchors: List[Dict[str, Any]] = []
    objects: List[Dict[str, Any]] = []
    seen_anchor = set()
    seen_obj = set()
    candidate_text_count = 0
    digit_candidate_count = 0
    candidate_samples: List[str] = []
    seen_sample = set()
    digit_fragments: List[Dict[str, Any]] = []
    bbox_item_count = 0
    seen_candidate = set()
    zip_key_counts: Dict[str, int] = {}

    ocr_sources: List[Any] = []
    bbox_sources: List[Any] = []
    if isinstance(pp_obj, dict):
        for key in ("overall_ocr_res", "parsing_res_list"):
            if key in pp_obj:
                ocr_sources.append(pp_obj.get(key))
        for key in ("layout_det_res", "region_det_res"):
            if key in pp_obj:
                bbox_sources.append(pp_obj.get(key))
    if isinstance(pp_json, dict):
        if "res" in pp_json:
            ocr_sources.append(pp_json.get("res"))
        ocr_sources.append(pp_json)

    for src in ocr_sources:
        for text_raw, bbox in _iter_ocr_items(src, None, zip_key_counts):
            text = _normalize_digit_candidate(text_raw)
            if _is_label_like_text(text):
                continue
            text = re.sub(r"\s+", "", text)
            if not text:
                continue
            candidate_text_count += 1
            digits = _digit_only(text)
            has_digit = bool(digits)
            if has_digit:
                digit_candidate_count += 1
                if len(candidate_samples) < 5 and text not in seen_sample:
                    seen_sample.add(text)
                    candidate_samples.append(text)

            cand_key = (text, tuple(bbox) if _bbox_valid(bbox) else ())
            if cand_key in seen_candidate:
                continue
            seen_candidate.add(cand_key)

            if not _bbox_valid(bbox):
                continue
            bbox_item_count += 1

            if 1 <= len(digits) <= 3:
                yc = (bbox[1] + bbox[3]) / 2.0
                digit_fragments.append({"digits": digits, "bbox": bbox, "yc": yc})

            n_found = _find_anchor_number(text)
            m_col = _ANCHOR_RE.match(text.upper())
            if n_found is not None or m_col:
                n_val = n_found if n_found is not None else -1
                anchor_text = text
                key = (n_val, *bbox, anchor_text)
                if key not in seen_anchor:
                    seen_anchor.add(key)
                    anchors.append({"id": anchor_text, "text": anchor_text, "n": n_val, "bbox": bbox, "col": 0})

    for src in bbox_sources:
        for node in _iter_dicts(src):
            bbox = _extract_bbox(node)
            if not _bbox_valid(bbox):
                continue
            node_type = str(node.get("type") or "").lower()
            obj_type = ""
            if node_type in ("figure", "table"):
                obj_type = node_type
            if obj_type:
                key = (obj_type, *bbox)
                if key not in seen_obj:
                    seen_obj.add(key)
                    objects.append({"type": obj_type, "bbox": bbox})

    frag_anchors, frag_parts, frag_joined = _build_fragment_anchors(digit_fragments)
    for a in frag_anchors:
        a_bbox = a.get("bbox")
        if not _bbox_valid(a_bbox):
            continue
        key = (a.get("n", -1), *a_bbox, a.get("text", ""))
        if key not in seen_anchor:
            seen_anchor.add(key)
            anchors.append(a)

    anchors.sort(key=lambda a: (a.get("bbox", [0, 0, 0, 0])[1], a.get("bbox", [0, 0, 0, 0])[0]))
    return anchors, objects, candidate_text_count, digit_candidate_count, len(digit_fragments), bbox_item_count, candidate_samples, frag_parts, frag_joined, zip_key_counts


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



def _validate_export_schema(cfg: Any, label: str) -> tuple[bool, str]:
    if not isinstance(cfg, dict):
        return False, f"{label}: root is not dict"
    if "Global" in cfg:
        return False, f"{label}: top-level Global key detected"
    for key in ("pipeline_name", "SubModules", "SubPipelines"):
        if key not in cfg:
            return False, f"{label}: missing {key}"
    if not isinstance(cfg.get("SubModules"), dict):
        return False, f"{label}: SubModules is not dict"
    if not isinstance(cfg.get("SubPipelines"), dict):
        return False, f"{label}: SubPipelines is not dict"
    return True, "ok"


def _is_valid_full_yaml(full_yaml: Path) -> bool:
    if not full_yaml.exists():
        return False
    try:
        cfg = _load_yaml_with_fallback(full_yaml)
    except Exception as e:
        _stage(f"full_yaml_validate_read_failed err={e}")
        return False
    ok, _ = _validate_export_schema(cfg, "full_yaml")
    return ok


def _export_full_yaml_if_missing(full_yaml: Path) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if full_yaml.exists():
        cfg = _load_yaml_with_fallback(full_yaml)
        ok, reason = _validate_export_schema(cfg, "full_yaml")
        if ok:
            return
        _stage(f"full_yaml_invalid err={reason}")
        raise RuntimeError(reason)

    try:
        from paddleocr import PPStructureV3

        PPStructureV3().export_paddlex_config_to_yaml(str(full_yaml))
        _stage(f"exported_full_yaml {full_yaml.name}")
    except Exception as e:
        _stage(f"export_full_yaml_skip err={e}")
        raise

    cfg2 = _load_yaml_with_fallback(full_yaml)
    ok2, reason2 = _validate_export_schema(cfg2, "full_yaml_exported")
    if not ok2:
        _stage(f"full_yaml_export_invalid err={reason2}")
        raise RuntimeError(reason2)


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

    ok_schema, schema_reason = _validate_export_schema(data, "fast_yaml")
    if not ok_schema:
        _stage(f"fast_yaml_validate_schema_failed err={schema_reason}")
        return False

    submodules = data.get("SubModules")
    region = submodules.get("RegionDetection") if isinstance(submodules, dict) else None
    if not isinstance(region, dict):
        _stage("fast_yaml_validate_missing RegionDetection")
        return False

    required_keys = ["model_name", "module_name", "model_dir"]
    missing = [k for k in required_keys if k not in region]
    if missing:
        _stage(f"fast_yaml_validate_missing_keys {','.join(missing)}")
        return False

    if region.get("model_name") != "PP-DocBlockLayout":
        _stage(f"fast_yaml_validate_bad_model_name got={region.get('model_name')}")
        return False

    if not region.get("module_name"):
        _stage("fast_yaml_validate_missing module_name")
        return False
    return True


def _make_fast_yaml_from_full(full_yaml: Path, fast_yaml: Path) -> bool:
    if not full_yaml.exists():
        _stage("make_fast_yaml_skip full_yaml_missing")
        return False
    try:
        data = _load_yaml_with_fallback(full_yaml)
        ok_schema, schema_reason = _validate_export_schema(data, "full_yaml_for_fast")
        if not ok_schema:
            _stage(f"make_fast_yaml_skip schema_invalid err={schema_reason}")
            return False

        # Top-level fast switches.
        data["use_table_recognition"] = False
        data["use_formula_recognition"] = False
        data["use_chart_recognition"] = False
        data["use_seal_recognition"] = False
        data["use_region_detection"] = True

        # Fast tuning via value-only toggles (preserve full export structure).
        data["batch_size"] = 1

        subpipes = data.get("SubPipelines")
        if isinstance(subpipes, dict):
            doc_pre = subpipes.get("DocPreprocessor")
            if isinstance(doc_pre, dict):
                doc_pre["use_doc_orientation_classify"] = False
                doc_pre["use_doc_unwarping"] = False

            general_ocr = subpipes.get("GeneralOCR")
            if isinstance(general_ocr, dict):
                general_ocr["use_textline_orientation"] = False
                g_sub = general_ocr.get("SubModules")
                if isinstance(g_sub, dict):
                    text_det = g_sub.get("TextDetection")
                    if isinstance(text_det, dict):
                        if "limit_side_len" in text_det:
                            text_det["limit_side_len"] = 512
                        text_det["limit_type"] = "min"
                        text_det["max_side_limit"] = 2000

        _dump_yaml_with_fallback(data, fast_yaml)
        if not _validate_fast_yaml_region_detection(fast_yaml):
            _stage("make_fast_yaml_skip validation_failed")
            region = data.get("SubModules", {}).get("RegionDetection") if isinstance(data.get("SubModules"), dict) else {}
            if isinstance(region, dict):
                _stage(f"fast_yaml_region_keys keys={list(region.keys())}")
            _delete_fast_yaml(fast_yaml, "validation_failed")
            return False

        # Immediate load validation for generated fast yaml.
        try:
            from paddleocr import PPStructureV3

            PPStructureV3(paddlex_config=str(fast_yaml))
        except Exception as e:
            region = data.get("SubModules", {}).get("RegionDetection") if isinstance(data.get("SubModules"), dict) else {}
            if isinstance(region, dict):
                _stage(f"fast_yaml_region_keys keys={list(region.keys())}")
            _stage(f"fast_yaml_post_generate_load_failed err={e}")
            _delete_fast_yaml(fast_yaml, "post_generate_load_failed")
            return False

        _stage(f"generated_fast_yaml {fast_yaml.name}")
        return True
    except Exception as e:
        _stage(f"make_fast_yaml_skip err={e}")
        return False


def _delete_fast_yaml(fast_cfg: Path, reason: str) -> None:
    try:
        if not fast_cfg.exists():
            return
        ts = time.strftime("%Y%m%d-%H%M%S")
        safe_reason = re.sub(r"[^A-Za-z0-9_-]+", "_", reason).strip("_") or "unknown"
        bad = fast_cfg.with_name(f"PP-StructureV3_fast.bad.{safe_reason}.{ts}.yaml")
        fast_cfg.rename(bad)
        _stage(f"fast_yaml_preserved_bad file={bad.name}")
    except Exception as e:
        _stage(f"fast_yaml_preserve_failed reason={reason} err={e}")


def _load_engine(profile: str) -> Any:
    from paddleocr import PPStructureV3

    full_cfg = CONFIG_DIR / "PP-StructureV3_full.yaml"
    fast_cfg = CONFIG_DIR / "PP-StructureV3_fast.yaml"

    if profile == "fast":
        if not fast_cfg.exists():
            _stage("fast_yaml_missing regenerate")
            _export_full_yaml_if_missing(full_cfg)
            if not _make_fast_yaml_from_full(full_cfg, fast_cfg):
                _stage("fast_yaml_generate_failed retry_once")
                _delete_fast_yaml(fast_cfg, "generate_failed_first")
                if not _make_fast_yaml_from_full(full_cfg, fast_cfg):
                    raise RuntimeError("fast_yaml_generate_failed")

        if not _validate_fast_yaml_region_detection(fast_cfg):
            _delete_fast_yaml(fast_cfg, "region_validation_failed")
            raise RuntimeError("fast_yaml_invalid_region_detection")

        try:
            return PPStructureV3(paddlex_config=str(fast_cfg))
        except Exception as e:
            _stage(f"fast_yaml_load_failed err={e}")
            _delete_fast_yaml(fast_cfg, "load_failed")

            _stage("fast_yaml_recover regenerate_and_retry")
            _export_full_yaml_if_missing(full_cfg)
            if not _make_fast_yaml_from_full(full_cfg, fast_cfg):
                _stage("fast_yaml_recover_generate_failed retry_once")
                _delete_fast_yaml(fast_cfg, "recover_generate_failed_first")
                if not _make_fast_yaml_from_full(full_cfg, fast_cfg):
                    raise RuntimeError("fast_yaml_recover_generate_failed") from e
            if not _validate_fast_yaml_region_detection(fast_cfg):
                _delete_fast_yaml(fast_cfg, "recover_region_validation_failed")
                raise RuntimeError("fast_yaml_recover_invalid_region_detection") from e

            try:
                _stage("fast_yaml_recover_retry")
                return PPStructureV3(paddlex_config=str(fast_cfg))
            except Exception as retry_err:
                _stage(f"fast_yaml_recover_retry_failed err={retry_err}")
                _delete_fast_yaml(fast_cfg, "recover_retry_load_failed")
                raise RuntimeError(f"fast_init_failed: {retry_err}") from retry_err

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

        pp_obj = _extract_first_object_fields(
            first,
            keys=["overall_ocr_res", "parsing_res_list", "layout_det_res", "region_det_res"],
        )
        anchors, objects, candidate_count, digit_candidate_count, frag_count, bbox_item_count, candidate_samples, frag_parts, frag_joined, zip_key_counts = _extract_min_entities(pp_json, pp_obj)
        payload = {
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
        if pp_obj:
            payload["pp_obj"] = pp_obj
        bbox_keys_debug: List[str] = []
        if bbox_item_count == 0:
            try:
                for node in _iter_dicts(pp_obj):
                    if isinstance(node, dict):
                        keys = [k for k in node.keys() if any(t in str(k).lower() for t in ("bbox", "box", "poly", "points", "dt_", "rec_text", "text"))]
                        if keys:
                            bbox_keys_debug = sorted(list(dict.fromkeys(str(k) for k in keys)))[:8]
                            break
            except Exception:
                bbox_keys_debug = []

        try:
            zip_keys_debug = _to_builtin_int_map(zip_key_counts)
            extra_debug = ""
            if bbox_item_count == 0:
                if zip_keys_debug:
                    extra_debug = f" zip_lens={_safe_debug_json(zip_keys_debug)}"
                elif bbox_keys_debug:
                    extra_debug = f" bbox_keys={_safe_debug_json(bbox_keys_debug)}"

            print(
                f"[debug] {page_name} candidates={int(candidate_count)} digit_candidates={int(digit_candidate_count)} bbox_items={int(bbox_item_count)} frag_count={int(frag_count)} matches={int(len(anchors))} "
                f"sample_texts={_safe_debug_json(candidate_samples)} zip_lens={_safe_debug_json(zip_keys_debug)}{extra_debug}",
                file=sys.stderr,
                flush=True,
            )
        except Exception as dbg_e:
            print(f"[debug] {page_name} debug_emit_failed={dbg_e}", file=sys.stderr, flush=True)
        return payload
    except Exception as e:
        t_page_ms = (time.perf_counter() - page_start) * 1000.0
        exc_type = e.__class__.__name__
        print(f"[error] {page_name} {exc_type}: {e}", file=sys.stderr, flush=True)
        try:
            traceback.print_exc(file=sys.stderr)
        except Exception:
            pass
        return {
            "ok": False,
            "page_file": page_name,
            "stage": "runner_loop",
            "err": str(e),
            "exc_type": exc_type,
            "t_init_ms": round(t_init_ms, 3),
            "t_predict_ms": round(t_page_ms, 3),
            "t_page_ms": round(t_page_ms, 3),
            "predict_flags": flags,
        }


def run_pages_dir(pages_dir: Path, warmup: bool, profile: str, force_region_detection: int, payload_mode: str) -> int:
    if not pages_dir.exists() or not pages_dir.is_dir():
        _emit_json({"ok": False, "page_file": "__BATCH__", "stage": "args", "err": f"invalid pages_dir: {pages_dir}", "profile": profile}, payload_mode=payload_mode)
        return 1

    page_files = _sorted_page_files(pages_dir)
    _stage(f"start pages={len(page_files)} profile={profile}")
    if not page_files:
        _emit_json({"ok": False, "page_file": "__BATCH__", "stage": "load_pages", "err": "no P*.png files", "profile": profile}, payload_mode=payload_mode)
        return 1

    init_start = time.perf_counter()
    try:
        engine = _load_engine(profile)
        t_init_ms = (time.perf_counter() - init_start) * 1000.0
        _stage(f"init_ok t_init_ms={t_init_ms:.1f}")
    except Exception as e:
        _emit_json({"ok": False, "page_file": "__BATCH__", "stage": "init_engine", "err": str(e), "profile": profile}, payload_mode=payload_mode)
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
        t_emit_ms = _emit_json(payload, fallback_page_file=page_path.name, payload_mode=payload_mode)
        _stage(
            f"predict_done {page_path.name} t_init_ms={float(payload.get('t_init_ms', 0.0)):.1f} "
            f"t_predict_ms={float(payload.get('t_predict_ms', 0.0)):.1f} t_emit_ms={t_emit_ms:.1f} "
            f"t_page_ms={float(payload.get('t_page_ms', 0.0)):.1f}"
        )

    return 0


def run_single_image(image_path: Path, warmup: bool, profile: str, force_region_detection: int, payload_mode: str) -> int:
    if not image_path.exists() or not image_path.is_file():
        _emit_json({"ok": False, "page_file": image_path.name, "stage": "args", "err": f"invalid image_path: {image_path}", "profile": profile}, payload_mode=payload_mode)
        return 1

    init_start = time.perf_counter()
    try:
        engine = _load_engine(profile)
        t_init_ms = (time.perf_counter() - init_start) * 1000.0
        _stage(f"init_ok(single) t_init_ms={t_init_ms:.1f}")
    except Exception as e:
        _emit_json({"ok": False, "page_file": image_path.name, "stage": "init_engine", "err": str(e), "profile": profile}, payload_mode=payload_mode)
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
    t_emit_ms = _emit_json(payload, fallback_page_file=image_path.name, payload_mode=payload_mode)
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
    parser.add_argument("--dpi", type=int, default=180, help="PDF render hint; ignored for direct image input")
    parser.add_argument("--warmup", type=int, choices=[0, 1], default=1)
    parser.add_argument("--profile", choices=["fast", "full"], default="fast")
    parser.add_argument("--force_region_detection", type=int, choices=[-1, 0, 1], default=-1)
    parser.add_argument("--payload", choices=["min", "full"], default="min")
    parser.add_argument("--quiet", type=int, choices=[0, 1], default=0, help="reduce stdout/stderr logging")
    parser.add_argument("--jsonl_out", default="", help="optional JSONL file path for payload output")
    args = parser.parse_args()

    global _QUIET, _JSONL_OUT_PATH
    _QUIET = bool(args.quiet)
    _JSONL_OUT_PATH = args.jsonl_out

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

    _emit_json({"ok": False, "page_file": "__BATCH__", "stage": "args", "err": "usage: v3_isolation_runner.py (--pages_dir DIR) | (image_path)", "profile": args.profile}, payload_mode=args.payload)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
