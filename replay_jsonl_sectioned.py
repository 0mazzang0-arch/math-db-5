import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image
import tkinter as tk

from pdf_cutter_experiment_gui import PDFCutterApp


KEYWORDS = ("STEP", "UNIT", "CHAPTER", "LEVEL", "유형", "단원", "개념")
MEANINGLESS_SLUG_TOKENS = {
    "MAPL",
    "SYNERGY",
    "SERIES",
    "YOURMASTERPLAN",
    "NORMAL",
    "BASIC",
    "TOUGH",
    "CHOS",
}


def _load_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            page_file = str(obj.get("page_file", ""))
            if page_file:
                out[page_file] = obj
    return out


def _sanitize_slug(text: str, max_len: int = 36) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = re.sub(r"\s+", "_", t.strip())
    t = re.sub(r"[^0-9A-Za-z가-힣_]+", "", t)
    t = re.sub(r"_+", "_", t).strip("_")
    if not t:
        return ""
    return t[:max_len]


def _pick_section_slug(ocr_items: List[Dict[str, Any]], page_h: int) -> str:
    if page_h <= 0:
        return ""
    top_max = int(page_h * 0.22)
    scored: List[Tuple[int, str]] = []
    for it in ocr_items:
        if not isinstance(it, dict):
            continue
        text = str(it.get("text", "")).strip()
        bbox = it.get("bbox")
        if not text or not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        y1 = int(float(bbox[1]))
        if y1 > top_max:
            continue
        norm = unicodedata.normalize("NFKC", text)
        up = norm.upper()
        score = 0
        if any(k in up for k in KEYWORDS):
            score += 20
        score += min(len(norm), 30)
        score += sum(1 for ch in norm if "가" <= ch <= "힣")
        if re.search(r"\d{1,2}\s*-\s*\d{1,2}", norm):
            score += 8
        scored.append((score, norm))
    if not scored:
        return ""
    scored.sort(key=lambda x: x[0], reverse=True)
    return _sanitize_slug(scored[0][1])


def _is_meaningful_slug(slug: str) -> bool:
    s = (slug or "").strip()
    if not s:
        return False
    up = s.upper()
    if up in MEANINGLESS_SLUG_TOKENS:
        return False
    if any(tok in up for tok in MEANINGLESS_SLUG_TOKENS) and len(up) <= 20:
        return False
    if re.fullmatch(r"S\d{2}", s):
        return False
    if len(s) <= 2:
        return False
    return True


def _bbox_to_str(b: List[int]) -> str:
    return f"[{int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])}]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages_dir", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--doc_prefix", required=True)
    args = parser.parse_args()

    pages_dir = Path(args.pages_dir).resolve()
    jsonl_path = Path(args.jsonl).resolve()
    out_root = Path(args.out_root).resolve()

    crops_dir = out_root / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_root / "index.csv"
    summary_path = out_root / "replay_summary.json"

    payload_by_page = _load_jsonl(jsonl_path)
    page_paths = sorted(pages_dir.glob("P*.png"))

    root = tk.Tk()
    root.withdraw()
    app = PDFCutterApp(root)
    app.full_profile_var.set(False)
    app.save_runner_logs_var.set(False)

    section_idx = 1
    prev_slug = ""
    prev_local = 0

    rows: List[Dict[str, Any]] = []
    total_saved = 0
    total_errors = 0

    for page_no, page_path in enumerate(page_paths, start=1):
        payload = payload_by_page.get(page_path.name)
        if not isinstance(payload, dict) or not payload.get("ok"):
            total_errors += 1
            continue

        anchors = payload.get("anchors", [])
        objects = payload.get("objects", [])
        if not isinstance(anchors, list) or not isinstance(objects, list):
            total_errors += 1
            continue

        with Image.open(page_path) as img:
            w, h = img.size
            crops, dropped, errors = app._build_anchor_slice_regions(anchors, objects, w, h)
            if errors > 0 and not crops:
                total_errors += 1
                continue

            ocr_items = payload.get("ocr_items", [])
            section_slug_meta = _pick_section_slug(ocr_items if isinstance(ocr_items, list) else [], h)
            section_slug_for_name = section_slug_meta if _is_meaningful_slug(section_slug_meta) else f"S{section_idx:02d}"

            curr_local = int(crops[0][0]) if crops else 0
            reset_detected = prev_local >= 10 and curr_local <= 2 and curr_local > 0
            if rows and reset_detected:
                section_idx += 1
                section_slug_for_name = section_slug_meta if _is_meaningful_slug(section_slug_meta) else f"S{section_idx:02d}"
            if not section_slug_for_name:
                section_slug_for_name = f"S{section_idx:02d}"

            for seq, (qid, x1, y1, x2, y2) in enumerate(crops, start=1):
                local_no = max(0, int(qid))
                profile_used = str(payload.get("profile_used", ""))
                if profile_used == "4digit":
                    local_fmt = f"{local_no:04d}"
                elif profile_used in {"3digit0", "digits"}:
                    local_fmt = f"{local_no:03d}"
                else:
                    local_fmt = f"{local_no:03d}"
                file_name = (
                    f"{args.doc_prefix}__S{section_idx:02d}_{section_slug_for_name}__{local_fmt}"
                    f"__P{page_no:03d}Q{seq:02d}.png"
                )
                crop_img = img.crop((x1, y1, x2, y2))
                crop_img.save(crops_dir / file_name)
                total_saved += 1
                rows.append(
                    {
                        "file_name": file_name,
                        "doc_prefix": args.doc_prefix,
                        "section_idx": section_idx,
                        "section_slug": section_slug_meta,
                        "local_no": local_no,
                        "page_file": page_path.name,
                        "page_no": page_no,
                        "q_seq": seq,
                        "bbox": _bbox_to_str([x1, y1, x2, y2]),
                        "profile_used": profile_used,
                        "anchors_count": int(payload.get("anchors_count", len(anchors))),
                    }
                )
            prev_slug = section_slug_meta
            if crops:
                prev_local = int(crops[-1][0])

    with open(index_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_name",
                "doc_prefix",
                "section_idx",
                "section_slug",
                "local_no",
                "page_file",
                "page_no",
                "q_seq",
                "bbox",
                "profile_used",
                "anchors_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    sections_detected = max([int(r["section_idx"]) for r in rows], default=0)
    summary = {
        "pages_total": len(page_paths),
        "total_saved": total_saved,
        "total_errors": total_errors,
        "sections_detected": sections_detected,
        "index_csv": str(index_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    root.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
