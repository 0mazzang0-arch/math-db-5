import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz
from PIL import Image
import tkinter as tk

import v3_isolation_runner as v3
from pdf_cutter_experiment_gui import PDFCutterApp


PROFILE_ORDER = ["4digit", "3digit0", "digits", "paren", "circle"]
FOUR_DIGIT_RE = re.compile(r"(?<!\d)(0\d{3})(?!\d)")
THREEDIGIT0_RE = re.compile(r"^\s*(0\d{2})\s*$")
DIGITS_TAIL_RE = re.compile(r"^\s*(\d{1,3})\s*([.)])?\s*$")
PAREN_RE = re.compile(r"(?:\(\s*\d{1,3}\s*\)|\b\d{1,3}\s*[.)])")
CIRCLE_RE = re.compile(r"[\u2460-\u2473\u2776-\u277F]")
ALPHANUM_A_RE = re.compile(r"^\s*A\s*(\d{1,3})\s*$", re.IGNORECASE)
STRICT_4DIGIT_RE = re.compile(r"^\s*(0\d{3})\s*$")
EBS_CODE_4DIGIT_RE = re.compile(r"^\[\d{5}-(\d{4})\]$")
ENABLE_EBS_CODE_4DIGIT_EXCEPTION = False


def _render_pdf_to_pages(pdf_path: Path, pages_dir: Path, render_zoom: float) -> List[Path]:
    pages_dir.mkdir(parents=True, exist_ok=True)
    matrix = fitz.Matrix(render_zoom, render_zoom)
    out: List[Path] = []
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc):
            out_path = pages_dir / f"P{idx + 1:03d}.png"
            page.get_pixmap(matrix=matrix, alpha=False).save(out_path)
            out.append(out_path)
    return out


def _sample_pages(page_paths: List[Path], n_each: int = 5) -> List[Path]:
    n = len(page_paths)
    if n == 0:
        return []
    first = list(range(0, min(n_each, n)))
    mid_start = max(0, (n // 2) - (n_each // 2))
    middle = list(range(mid_start, min(n, mid_start + n_each)))
    last = list(range(max(0, n - n_each), n))
    return [page_paths[i] for i in sorted(set(first + middle + last))]


def _ocr_items_from_payload(payload: Dict[str, Any], max_items: int = 400) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    pp_obj = payload.get("pp_obj", {}) if isinstance(payload, dict) else {}
    pp_json = payload.get("pp_json", {}) if isinstance(payload, dict) else {}
    sources: List[Any] = []
    if isinstance(pp_obj, dict):
        for k in ("overall_ocr_res", "parsing_res_list", "layout_det_res", "region_det_res"):
            if k in pp_obj:
                sources.append(pp_obj.get(k))
    if isinstance(pp_json, dict):
        if "res" in pp_json:
            sources.append(pp_json.get("res"))
        sources.append(pp_json)
    for src in sources:
        for txt, bbox in v3._iter_ocr_items(src, None, None):
            if not isinstance(txt, str):
                continue
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            try:
                bb = [int(float(v)) for v in bbox]
            except Exception:
                continue
            t = txt.strip()
            if not t:
                continue
            key = (t, tuple(bb))
            if key in seen:
                continue
            seen.add(key)
            out.append({"text": t, "bbox": bb})
            if len(out) >= max_items:
                return out
    return out


def _score_profile(ocr_items: List[Dict[str, Any]], anchors_4digit_count: int = 0) -> Dict[str, int]:
    s = {"4digit": 0, "3digit0": 0, "digits": 0, "paren": 0, "circle": 0}
    s["4digit"] += int(max(0, anchors_4digit_count)) * 6
    for it in ocr_items:
        t = str(it.get("text", ""))
        if len(t) > 20:
            continue
        if FOUR_DIGIT_RE.search(t):
            s["4digit"] += 1
        if PAREN_RE.search(t):
            s["paren"] += 1
        if CIRCLE_RE.search(t) and len(t.strip()) <= 6:
            s["circle"] += 1
        m3 = THREEDIGIT0_RE.match(t)
        if m3:
            s["3digit0"] += 2
        md = DIGITS_TAIL_RE.match(t)
        if md:
            digits = md.group(1)
            if len(digits) == 3 and digits.startswith("0"):
                pass
            else:
                s["digits"] += 1
        ma = ALPHANUM_A_RE.match(t)
        if ma:
            s["digits"] += 2
    return s


def _circle_values(text: str) -> List[int]:
    vals: List[int] = []
    for ch in text:
        code = ord(ch)
        if 0x2460 <= code <= 0x2473:
            vals.append(code - 0x245F)
        elif 0x2776 <= code <= 0x277F:
            vals.append(code - 0x2775)
    return vals


def _circle_guard_stats(ocr_items: List[Dict[str, Any]]) -> Dict[str, int]:
    small = 0
    large = 0
    uniq = set()
    for it in ocr_items:
        t = str(it.get("text", ""))
        for v in _circle_values(t):
            uniq.add(v)
            if 1 <= v <= 5:
                small += 1
            elif v >= 6:
                large += 1
    raw = len(uniq)
    effective = 0 if (large == 0 and small >= 10) else raw
    return {"small": small, "large": large, "raw": raw, "effective": effective}


def _choose_profile(scores: Dict[str, int]) -> str:
    return sorted(PROFILE_ORDER, key=lambda p: (-int(scores.get(p, 0)), PROFILE_ORDER.index(p)))[0]


def _infer_col(bbox: List[int], page_w: int) -> int:
    cx = (float(bbox[0]) + float(bbox[2])) / 2.0
    return 0 if cx < (float(page_w) * 0.5) else 1


def _new_excluded_counts() -> Dict[str, int]:
    return {
        "excluded_math": 0,
        "excluded_year": 0,
        "excluded_code": 0,
        "excluded_not_leftmargin": 0,
        "excluded_header": 0,
        "excluded_small": 0,
        "excluded_pattern": 0,
        "excluded_4digit_not_fullmatch": 0,
        "excluded_4digit_decimal_or_dot": 0,
        "excluded_4digit_code": 0,
        "excluded_seq_outlier": 0,
    }


def _digit_str(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def _has_math_noise(text: str, allow_digit_tail: bool = False) -> bool:
    t = (text or "").strip()
    low = t.lower()
    if any(tok in low for tok in ("log", "lim", "sin", "cos", "tan")):
        return True
    if any(ch in t for ch in ("/", "^", "=", "+", "×", "*", "%", "±")):
        return True
    if "." in t and not allow_digit_tail:
        return True
    return False


def _context_bonus(ocr_items: List[Dict[str, Any]], bbox: List[int], page_h: int) -> float:
    cx = (float(bbox[0]) + float(bbox[2])) * 0.5
    cy = (float(bbox[1]) + float(bbox[3])) * 0.5
    ry = max(80.0, float(page_h) * 0.04)
    rx = 260.0
    keywords = ("문제", "다음", "구하", "값", "옳", "보기")
    bonus = 0.0
    for it in ocr_items:
        if not isinstance(it, dict):
            continue
        t = str(it.get("text", "")).strip()
        b = it.get("bbox")
        if not t or not (isinstance(b, list) and len(b) == 4):
            continue
        tcx = (float(b[0]) + float(b[2])) * 0.5
        tcy = (float(b[1]) + float(b[3])) * 0.5
        if abs(tcx - cx) <= rx and abs(tcy - cy) <= ry:
            if any(k in t for k in keywords):
                bonus += 4.0
    return min(20.0, bonus)


def _candidate_score(bbox: List[int], col_left: float, col_w: float, page_h: int, ocr_items: List[Dict[str, Any]]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    h = max(1.0, y2 - y1)
    margin_dist = max(0.0, x1 - col_left)
    margin_score = max(0.0, 25.0 * (1.0 - (margin_dist / max(1.0, 0.18 * col_w))))
    target_h = max(10.0, float(page_h) * 0.02)
    size_score = min(25.0, (h / target_h) * 25.0)
    y_mid = (y1 + y2) * 0.5
    y_score = 15.0 if (0.12 * page_h) <= y_mid <= (0.92 * page_h) else 4.0
    return margin_score + size_score + y_score + _context_bonus(ocr_items, [int(x1), int(y1), int(x2), int(y2)], page_h)


def _topk_by_col(cands: List[Dict[str, Any]], k: int = 40) -> List[Dict[str, Any]]:
    by_col: Dict[int, List[Dict[str, Any]]] = {0: [], 1: []}
    for c in cands:
        by_col[1 if int(c.get("col", 0)) == 1 else 0].append(c)
    out: List[Dict[str, Any]] = []
    for col in (0, 1):
        arr = by_col[col]
        arr.sort(key=lambda x: (float(x.get("score", 0.0)), -int(x["bbox"][1])), reverse=True)
        out.extend(arr[:k])
    out.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    return out


def _drop_seq_outliers_4digit(anchors: List[Dict[str, Any]], gap: int = 5) -> Tuple[List[Dict[str, Any]], List[int], List[Tuple[int, int, int]], int]:
    if not anchors:
        return [], [], [], 0
    arr = []
    for a in anchors:
        try:
            n = int(a.get("n"))
        except Exception:
            continue
        arr.append((n, a))
    if not arr:
        return anchors[:], [], [], 0
    arr.sort(key=lambda x: x[0])
    clusters: List[List[Tuple[int, Dict[str, Any]]]] = []
    cur: List[Tuple[int, Dict[str, Any]]] = [arr[0]]
    for i in range(1, len(arr)):
        prev_n = arr[i - 1][0]
        n = arr[i][0]
        if (n - prev_n) <= int(gap):
            cur.append(arr[i])
        else:
            clusters.append(cur)
            cur = [arr[i]]
    clusters.append(cur)
    max_cluster_size = max(len(c) for c in clusters)
    if max_cluster_size < 3:
        cluster_ranges = [(c[0][0], c[-1][0], len(c)) for c in clusters]
        return [a for _, a in arr], [], cluster_ranges, max_cluster_size
    dropped: List[int] = []
    kept: List[Dict[str, Any]] = []
    for c in clusters:
        if len(c) == 1:
            dropped.append(int(c[0][0]))
        else:
            kept.extend([a for _, a in c])
    kept.sort(key=lambda x: (int(x.get("n", 0)), int(x.get("bbox", [0, 0, 0, 0])[1]), int(x.get("bbox", [0, 0, 0, 0])[0])))
    cluster_ranges = [(c[0][0], c[-1][0], len(c)) for c in clusters]
    return kept, dropped, cluster_ranges, max_cluster_size


def _dedup(anchors: List[Dict[str, Any]], id_pat: str = r"\d+") -> List[Dict[str, Any]]:
    keep: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for a in anchors:
        tid = str(a.get("id", ""))
        bbox = a.get("bbox")
        if not re.fullmatch(id_pat, tid):
            continue
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        key = (tid, int(a.get("col", 0)))
        prev = keep.get(key)
        if prev is None or int(bbox[1]) < int(prev["bbox"][1]):
            keep[key] = {
                "id": tid,
                "text": tid,
                "n": int(tid),
                "bbox": [int(float(v)) for v in bbox],
                "col": 1 if int(a.get("col", 0)) == 1 else 0,
            }
    out = list(keep.values())
    out.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    return out


def _anchors_4digit(payload: Dict[str, Any], page_w: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    excluded = _new_excluded_counts()
    anchors = payload.get("anchors", [])
    if isinstance(anchors, list):
        for a in anchors:
            if not isinstance(a, dict):
                continue
            text = str(a.get("text", "")).strip()
            bbox = a.get("bbox")
            if not re.fullmatch(r"\d{4}", text):
                continue
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            out.append(
                {
                    "id": text,
                    "text": text,
                    "n": int(text),
                    "bbox": [int(float(v)) for v in bbox],
                    "col": int(a.get("col", _infer_col(bbox, page_w))),
                }
            )
    if out:
        kept = _dedup(out, id_pat=r"\d{4}")
        return kept, {"mode": "4digit", "pass1_raw": len(anchors), "pass1_kept": len(kept), "pass2_used": False, "excluded_counts": excluded}
    for it in payload.get("ocr_items", []):
        if not isinstance(it, dict):
            continue
        txt = str(it.get("text", "")).strip()
        bbox = it.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        # 4digit profile must be full-token match only.
        m = STRICT_4DIGIT_RE.match(txt)
        if m:
            t = m.group(1)
        else:
            if "." in txt:
                excluded["excluded_4digit_decimal_or_dot"] += 1
                continue
            if "[" in txt or "]" in txt or "-" in txt:
                if ENABLE_EBS_CODE_4DIGIT_EXCEPTION:
                    m_code = EBS_CODE_4DIGIT_RE.match(txt)
                    if not m_code:
                        excluded["excluded_4digit_code"] += 1
                        continue
                    t = m_code.group(1)
                else:
                    excluded["excluded_4digit_code"] += 1
                    continue
            else:
                excluded["excluded_4digit_not_fullmatch"] += 1
                continue
        out.append({"id": t, "text": t, "n": int(t), "bbox": bbox, "col": _infer_col(bbox, page_w)})
    kept = _dedup(out, id_pat=r"\d{4}")
    return kept, {
        "mode": "4digit",
        "pass1_raw": len(payload.get("ocr_items", []) if isinstance(payload.get("ocr_items"), list) else []),
        "pass1_kept": len(kept),
        "pass2_used": False,
        "excluded_counts": excluded,
    }


def _numeric_candidates(
    payload: Dict[str, Any],
    page_w: int,
    page_h: int,
    mode: str,
    pass_level: int = 1,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    cands: List[Dict[str, Any]] = []
    excluded = _new_excluded_counts()
    col_w = max(1.0, float(page_w) * 0.5)
    left_margin_thr = float(col_w) * 0.18
    if pass_level == 1:
        min_h = max(10.0, float(page_h) * 0.004)
        header_y = float(page_h) * 0.06
        use_header_cut = True
    else:
        min_h = max(8.0, float(page_h) * 0.003)
        header_y = 0.0
        use_header_cut = False
    ocr_items = payload.get("ocr_items", []) if isinstance(payload.get("ocr_items"), list) else []
    for it in payload.get("ocr_items", []):
        if not isinstance(it, dict):
            continue
        txt = str(it.get("text", "")).strip()
        bbox = it.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        x1, y1, x2, y2 = [int(float(v)) for v in bbox]
        if "[" in txt or "]" in txt or "-" in txt:
            excluded["excluded_code"] += 1
            continue
        if use_header_cut and y1 < header_y:
            excluded["excluded_header"] += 1
            continue
        if (y2 - y1) < min_h:
            excluded["excluded_small"] += 1
            continue
        col = _infer_col([x1, y1, x2, y2], page_w)
        col_left = 0.0 if col == 0 else float(page_w) * 0.5
        if (float(x1) - col_left) > left_margin_thr:
            excluded["excluded_not_leftmargin"] += 1
            continue

        if mode == "3digit0":
            if _has_math_noise(txt, allow_digit_tail=False):
                excluded["excluded_math"] += 1
                continue
            m = THREEDIGIT0_RE.match(txt)
            if not m:
                excluded["excluded_pattern"] += 1
                continue
            tid = m.group(1)
            n_val = int(tid)
        else:
            allow_tail = bool(DIGITS_TAIL_RE.match(txt))
            if _has_math_noise(txt, allow_digit_tail=allow_tail):
                excluded["excluded_math"] += 1
                continue
            m = DIGITS_TAIL_RE.match(txt)
            ma = ALPHANUM_A_RE.match(txt)
            if m:
                raw = m.group(1)
                if len(raw) == 3 and raw.startswith("0"):
                    excluded["excluded_pattern"] += 1
                    continue
                try:
                    n_val = int(raw)
                except Exception:
                    excluded["excluded_pattern"] += 1
                    continue
                if n_val <= 0:
                    excluded["excluded_pattern"] += 1
                    continue
            elif ma:
                try:
                    n_val = int(ma.group(1))
                except Exception:
                    excluded["excluded_pattern"] += 1
                    continue
                if n_val <= 0:
                    excluded["excluded_pattern"] += 1
                    continue
            else:
                excluded["excluded_pattern"] += 1
                continue
            if 2000 <= int(n_val) <= 2099:
                excluded["excluded_year"] += 1
                continue
            dstr = _digit_str(txt)
            if len(dstr) == 8:
                excluded["excluded_year"] += 1
                continue
            tid = f"{n_val:03d}"
        score = _candidate_score([x1, y1, x2, y2], col_left, col_w, page_h, ocr_items)
        cands.append({"id": tid, "text": tid, "n": int(n_val), "bbox": [x1, y1, x2, y2], "col": col, "text_original": txt, "score": score})
    cands = _topk_by_col(cands, 40)
    return cands, excluded


def _anchors_3digit0(payload: Dict[str, Any], page_w: int, page_h: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw1, ex1 = _numeric_candidates(payload, page_w, page_h, "3digit0", pass_level=1)
    keep1 = _dedup(raw1, id_pat=r"\d{3}")
    dbg: Dict[str, Any] = {
        "mode": "3digit0",
        "pass1_raw": len(raw1),
        "pass1_kept": len(keep1),
        "pass1_min_h": max(10.0, float(page_h) * 0.004),
        "pass1_header_cut": float(page_h) * 0.06,
        "pass2_used": False,
        "excluded_counts": ex1,
    }
    if keep1:
        return keep1, dbg
    raw2, ex2 = _numeric_candidates(payload, page_w, page_h, "3digit0", pass_level=2)
    keep2 = _dedup(raw2, id_pat=r"\d{3}")
    dbg.update(
        {
            "pass2_used": True,
            "pass2_raw": len(raw2),
            "pass2_kept": len(keep2),
            "pass2_min_h": max(8.0, float(page_h) * 0.003),
            "pass2_header_cut": 0.0,
            "excluded_counts_pass2": ex2,
        }
    )
    return keep2, dbg


def _anchors_digits(payload: Dict[str, Any], page_w: int, page_h: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw1, ex1 = _numeric_candidates(payload, page_w, page_h, "digits", pass_level=1)
    keep1 = _dedup(raw1, id_pat=r"\d{3}")
    dbg: Dict[str, Any] = {
        "mode": "digits",
        "pass1_raw": len(raw1),
        "pass1_kept": len(keep1),
        "pass1_min_h": max(10.0, float(page_h) * 0.004),
        "pass1_header_cut": float(page_h) * 0.06,
        "pass2_used": False,
        "excluded_counts": ex1,
    }
    if keep1:
        return keep1, dbg
    raw2, ex2 = _numeric_candidates(payload, page_w, page_h, "digits", pass_level=2)
    keep2 = _dedup(raw2, id_pat=r"\d{3}")
    dbg.update(
        {
            "pass2_used": True,
            "pass2_raw": len(raw2),
            "pass2_kept": len(keep2),
            "pass2_min_h": max(8.0, float(page_h) * 0.003),
            "pass2_header_cut": 0.0,
            "excluded_counts_pass2": ex2,
        }
    )
    return keep2, dbg


def _anchors_seq(payload: Dict[str, Any], page_w: int, mode: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if mode == "circle":
        cstats = _circle_guard_stats(payload.get("ocr_items", []) if isinstance(payload.get("ocr_items"), list) else [])
        if cstats["large"] == 0 and cstats["small"] >= 10:
            return [], {
                "mode": "circle",
                "circle_guard": True,
                "circle_small": cstats["small"],
                "circle_large": cstats["large"],
                "score_circle_raw": cstats["raw"],
                "score_circle_effective": cstats["effective"],
                "pass1_raw": 0,
                "pass1_kept": 0,
                "pass2_used": False,
            }

    cands: List[Dict[str, Any]] = []
    excluded = _new_excluded_counts()
    ocr_items = payload.get("ocr_items", []) if isinstance(payload.get("ocr_items"), list) else []
    page_h_guess = 3000
    for it in ocr_items[:20]:
        b = it.get("bbox") if isinstance(it, dict) else None
        if isinstance(b, list) and len(b) == 4:
            try:
                page_h_guess = max(page_h_guess, int(float(b[3])) + 1)
            except Exception:
                pass
    col_w = max(1.0, float(page_w) * 0.5)
    left_margin_thr = float(col_w) * 0.18
    min_h = max(10.0, float(page_h_guess) * 0.004)
    header_y = float(page_h_guess) * 0.06

    for it in payload.get("ocr_items", []):
        if not isinstance(it, dict):
            continue
        txt = str(it.get("text", "")).strip()
        bbox = it.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        x1, y1, x2, y2 = [int(float(v)) for v in bbox]
        if "[" in txt or "]" in txt or "-" in txt:
            excluded["excluded_code"] += 1
            continue
        if y1 < header_y:
            excluded["excluded_header"] += 1
            continue
        if (y2 - y1) < min_h:
            excluded["excluded_small"] += 1
            continue
        col = _infer_col([x1, y1, x2, y2], page_w)
        col_left = 0.0 if col == 0 else float(page_w) * 0.5
        if (float(x1) - col_left) > left_margin_thr:
            excluded["excluded_not_leftmargin"] += 1
            continue
        if _has_math_noise(txt, allow_digit_tail=(mode == "paren")):
            excluded["excluded_math"] += 1
            continue
        if mode == "paren":
            if not PAREN_RE.search(txt):
                excluded["excluded_pattern"] += 1
                continue
        else:
            if not CIRCLE_RE.search(txt):
                excluded["excluded_pattern"] += 1
                continue
        score = _candidate_score([x1, y1, x2, y2], col_left, col_w, page_h_guess, ocr_items)
        cands.append({"text_original": txt, "bbox": [x1, y1, x2, y2], "col": col, "score": score})
    cands = _topk_by_col(cands, 40)
    out: List[Dict[str, Any]] = []
    for idx, c in enumerate(cands, start=1):
        tid = f"{idx:04d}"
        out.append(
            {
                "id": tid,
                "text": tid,
                "n": idx,
                "bbox": [int(float(v)) for v in c["bbox"]],
                "col": c["col"],
                "text_original": c["text_original"],
            }
        )
    kept = _dedup(out, id_pat=r"\d{4}")
    dbg = {
        "mode": mode,
        "pass1_raw": len(cands),
        "pass1_kept": len(kept),
        "pass2_used": False,
        "excluded_counts": excluded,
    }
    if mode == "circle":
        cstats = _circle_guard_stats(payload.get("ocr_items", []) if isinstance(payload.get("ocr_items"), list) else [])
        dbg.update(
            {
                "circle_guard": False,
                "circle_small": cstats["small"],
                "circle_large": cstats["large"],
                "score_circle_raw": cstats["raw"],
                "score_circle_effective": cstats["effective"],
            }
        )
    return kept, dbg


def _anchors_by_profile(profile: str, payload: Dict[str, Any], page_w: int, page_h: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if profile == "4digit":
        return _anchors_4digit(payload, page_w)
    if profile == "3digit0":
        return _anchors_3digit0(payload, page_w, page_h)
    if profile == "digits":
        return _anchors_digits(payload, page_w, page_h)
    if profile == "paren":
        return _anchors_seq(payload, page_w, "paren")
    return _anchors_seq(payload, page_w, "circle")


def _safe_objects(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    objs = payload.get("objects", [])
    if not isinstance(objs, list):
        return out
    for o in objs:
        if not isinstance(o, dict):
            continue
        bbox = o.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        bb = [int(float(v)) for v in bbox]
        out.append({"id": str(o.get("id", f"obj:{bb[1]}:{bb[0]}")), "type": str(o.get("type", "")), "bbox": bb})
    return out


def _estimate_saved(app: PDFCutterApp, anchors: List[Dict[str, Any]], objects: List[Dict[str, Any]], w: int, h: int) -> int:
    if not anchors:
        return 0
    crops, _, errors = app._build_anchor_slice_regions(anchors, objects, w, h)
    if errors > 0 and not crops:
        return 0
    return len(crops)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", default="")
    parser.add_argument("--pages_dir", default="")
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--doc_prefix", required=True)
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--render_zoom", type=float, default=2.0)
    parser.add_argument("--runner_profile", choices=["fast", "full"], default="fast")
    args = parser.parse_args()
    if float(args.render_zoom) <= 0:
        print("[error] --render_zoom must be > 0", file=sys.stderr)
        return 2

    if not args.pdf_path and not args.pages_dir:
        print("[error] need --pdf_path or --pages_dir", file=sys.stderr)
        return 2

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    pages_dir = Path(args.pages_dir).resolve() if args.pages_dir else (out_root / "_pages_all")

    if args.pdf_path:
        pdf_path = Path(args.pdf_path).resolve()
        if not pdf_path.exists():
            print(f"[error] missing pdf: {pdf_path}", file=sys.stderr)
            return 2
        print(f"[config] render_zoom={float(args.render_zoom):.3f}")
        page_paths = _render_pdf_to_pages(pdf_path, pages_dir, float(args.render_zoom))
    else:
        page_paths = sorted(Path(args.pages_dir).resolve().glob("P*.png"))
    if not page_paths:
        print("[error] no pages", file=sys.stderr)
        return 2

    init_start = time.perf_counter()
    engine, _cfg = v3._load_engine(args.runner_profile)
    t_init_ms = (time.perf_counter() - init_start) * 1000.0

    preds: Dict[str, Dict[str, Any]] = {}
    for p in page_paths:
        preds[p.name] = v3._predict_one(engine, p, t_init_ms, args.runner_profile, 1)

    sample_pages = _sample_pages(page_paths, 5)
    score_total = {"4digit": 0, "3digit0": 0, "digits": 0, "paren": 0, "circle": 0}
    pre_circle_small = 0
    pre_circle_large = 0
    pre_circle_raw = 0
    pre_circle_effective = 0
    for p in sample_pages:
        payload = preds.get(p.name, {})
        if not isinstance(payload, dict) or not payload.get("ok"):
            continue
        anchors_4digit = 0
        if isinstance(payload.get("anchors"), list):
            anchors_4digit = sum(1 for a in payload.get("anchors") if isinstance(a, dict) and re.fullmatch(r"\d{4}", str(a.get("text", ""))))
        ocr_items = _ocr_items_from_payload(payload)
        s = _score_profile(ocr_items, anchors_4digit_count=anchors_4digit)
        cstats = _circle_guard_stats(ocr_items)
        s["circle"] = cstats["effective"]
        pre_circle_small += cstats["small"]
        pre_circle_large += cstats["large"]
        pre_circle_raw += cstats["raw"]
        pre_circle_effective += cstats["effective"]
        for k in score_total:
            score_total[k] += int(s.get(k, 0))
    chosen = _choose_profile(score_total)
    print(
        f"[start] preflight scores=4digit:{score_total['4digit']} 3digit0:{score_total['3digit0']} "
        f"digits:{score_total['digits']} paren:{score_total['paren']} circle:{score_total['circle']} "
        f"circle_small={pre_circle_small} circle_large={pre_circle_large} "
        f"score_circle_raw={pre_circle_raw} score_circle_effective={pre_circle_effective} chosen={chosen}",
        flush=True,
    )

    root = tk.Tk()
    root.withdraw()
    app = PDFCutterApp(root)
    app.full_profile_var.set(False)
    app.save_runner_logs_var.set(False)

    rows: List[Dict[str, Any]] = []
    per_profile_used = Counter()
    total_errors = 0
    circle_guard_triggered_pages = 0
    excluded_reason_total: Counter[str] = Counter()

    for p in page_paths:
        payload = preds.get(p.name, {})
        if not isinstance(payload, dict) or not payload.get("ok"):
            rows.append({"ok": False, "page_file": p.name, "stage": "runner", "err": str(payload.get("err", "predict failed")), "anchors": [], "objects": []})
            total_errors += 1
            print(f"[page] {p.name} profile_used=none anchors=0 saved=0 retry=0 ok=false", flush=True)
            continue

        with Image.open(p) as img:
            w, h = img.size
        payload_work = dict(payload)
        payload_work["ocr_items"] = _ocr_items_from_payload(payload_work)
        objects = _safe_objects(payload_work)

        order = [chosen] + [x for x in PROFILE_ORDER if x != chosen]
        best: Dict[str, Any] | None = None
        for try_idx, prof in enumerate(order):
            anchors, dbg_info = _anchors_by_profile(prof, payload_work, w, h)
            if prof == "4digit":
                kept_4d, dropped_4d, clusters_4d, max_cs = _drop_seq_outliers_4digit(anchors, gap=5)
                if dropped_4d:
                    exc4 = dbg_info.get("excluded_counts", {}) if isinstance(dbg_info, dict) else {}
                    if not isinstance(exc4, dict):
                        exc4 = {}
                    exc4["excluded_seq_outlier"] = int(exc4.get("excluded_seq_outlier", 0)) + len(dropped_4d)
                    dbg_info["excluded_counts"] = exc4
                    dbg_info["dropped_seq_outliers"] = dropped_4d
                    dbg_info["kept_clusters"] = clusters_4d
                    dbg_info["max_cluster_size"] = int(max_cs)
                    print(
                        f"[debug] {p.name} dropped_seq_outliers={[f'{n:04d}' for n in dropped_4d]} kept_clusters={clusters_4d}",
                        flush=True,
                    )
                anchors = kept_4d
            saved_est = _estimate_saved(app, anchors, objects, w, h)
            exc = dbg_info.get("excluded_counts", {}) if isinstance(dbg_info, dict) else {}
            exc_line = ""
            if isinstance(exc, dict) and exc:
                exc_line = (
                    f" excluded_decimal={int(exc.get('excluded_math',0))}"
                    f" excluded_year={int(exc.get('excluded_year',0))}"
                    f" excluded_code={int(exc.get('excluded_code',0))}"
                    f" excluded_not_leftmargin={int(exc.get('excluded_not_leftmargin',0))}"
                    f" excluded_header={int(exc.get('excluded_header',0))}"
                    f" excluded_small={int(exc.get('excluded_small',0))}"
                    f" excluded_pattern={int(exc.get('excluded_pattern',0))}"
                    f" excluded_4digit_not_fullmatch={int(exc.get('excluded_4digit_not_fullmatch',0))}"
                    f" excluded_4digit_decimal_or_dot={int(exc.get('excluded_4digit_decimal_or_dot',0))}"
                    f" excluded_4digit_code={int(exc.get('excluded_4digit_code',0))}"
                )
            print(
                f"[debug] {p.name} profile={prof} candidates={int(dbg_info.get('pass1_raw',0))} "
                f"after_filter={int(dbg_info.get('pass1_kept',0))} final_anchors={len(anchors)}{exc_line}",
                flush=True,
            )
            if best is None or saved_est > int(best["saved_est"]):
                best = {"profile_used": prof, "retry_count": try_idx, "anchors": anchors, "saved_est": saved_est, "dbg": dbg_info}
            if anchors and saved_est > 0:
                best = {"profile_used": prof, "retry_count": try_idx, "anchors": anchors, "saved_est": saved_est, "dbg": dbg_info}
                break

        assert best is not None
        if bool(best.get("dbg", {}).get("circle_guard", False)):
            circle_guard_triggered_pages += 1
            dbg = best.get("dbg", {})
            print(
                f"[debug] {p.name} circle_guard_triggered=yes circle_small={dbg.get('circle_small',0)} "
                f"circle_large={dbg.get('circle_large',0)} score_circle_raw={dbg.get('score_circle_raw',0)} "
                f"score_circle_effective={dbg.get('score_circle_effective',0)}",
                flush=True,
            )
        best_exc = best.get("dbg", {}).get("excluded_counts", {})
        if isinstance(best_exc, dict):
            for k, v in best_exc.items():
                excluded_reason_total[str(k)] += int(v)
        per_profile_used[best["profile_used"]] += 1
        if best["profile_used"] in {"digits", "3digit0"} and len(best["anchors"]) == 0:
            dbg = best.get("dbg", {})
            print(
                f"[debug] {p.name} {best['profile_used']} anchors=0 "
                f"pass1_raw={dbg.get('pass1_raw', 0)} pass1_kept={dbg.get('pass1_kept', 0)} "
                f"pass1_min_h={dbg.get('pass1_min_h', 0)} pass1_header_cut={dbg.get('pass1_header_cut', 0)} "
                f"pass2_used={str(dbg.get('pass2_used', False)).lower()} "
                f"pass2_raw={dbg.get('pass2_raw', 0)} pass2_kept={dbg.get('pass2_kept', 0)} "
                f"pass2_min_h={dbg.get('pass2_min_h', 0)} pass2_header_cut={dbg.get('pass2_header_cut', 0)}",
                flush=True,
            )
        rows.append(
            {
                "ok": True,
                "page_file": p.name,
                "stage": "predict_done",
                "anchors": best["anchors"],
                "anchors_count": len(best["anchors"]),
                "objects": objects,
                "profile_used": best["profile_used"],
                "retry_count": int(best["retry_count"]),
                "slice_saved": int(best["saved_est"]),
                "ocr_items": payload_work.get("ocr_items", [])[:300],
            }
        )
        print(
            f"[page] {p.name} profile_used={best['profile_used']} anchors={len(best['anchors'])} saved={int(best['saved_est'])} retry={int(best['retry_count'])} ok=true",
            flush=True,
        )

    jsonl_path = out_root / "runner_out_auto.jsonl"
    _write_jsonl(jsonl_path, rows)

    replay_script = Path(__file__).resolve().parent / "replay_jsonl_sectioned.py"
    replay_cmd = [
        sys.executable,
        str(replay_script),
        "--pages_dir",
        str(pages_dir),
        "--jsonl",
        str(jsonl_path),
        "--out_root",
        str(out_root),
        "--doc_prefix",
        str(args.doc_prefix),
    ]
    replay_proc = subprocess.run(replay_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800, check=False)
    if replay_proc.stdout.strip():
        print(replay_proc.stdout.strip(), flush=True)
    if replay_proc.stderr.strip():
        print(replay_proc.stderr.strip(), file=sys.stderr, flush=True)

    sections_detected = 0
    total_saved = 0
    summary_path = out_root / "replay_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            sections_detected = int(summary.get("sections_detected", 0))
            total_saved = int(summary.get("total_saved", 0))
        except Exception:
            pass

    print(
        f"[end] pages_total={len(page_paths)} total_saved={total_saved} total_errors={total_errors} "
        f"per_profile_used={dict(per_profile_used)} sections_detected={sections_detected} "
        f"circle_guard_triggered_pages={circle_guard_triggered_pages} "
        f"excluded_reason_counts={dict(excluded_reason_total)}",
        flush=True,
    )
    # Machine-readable one-line summary for batch launcher parsing.
    print(
        json.dumps(
            {
                "pages_total": len(page_paths),
                "total_saved": total_saved,
                "total_errors": total_errors,
                "sections_detected": sections_detected,
                "chosen_profile": chosen,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    root.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
