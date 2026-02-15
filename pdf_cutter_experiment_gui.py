import base64
import io
import json
import os
import queue
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import google.generativeai as genai
from PIL import Image
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import requests


try:
    from config import GOOGLE_API_KEYS
except Exception:
    GOOGLE_API_KEYS = []


APP_TITLE = "PDF Cutter Experiment GUI"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "pdf_cutter_output"
DEFAULT_DPI = 250
DEFAULT_WORKERS = 6
MIN_WORKERS = 1
MAX_WORKERS = 8
MIN_DPI = 200
MAX_DPI = 300


# [1/3] PROMPT 변수를 이걸로 덮어쓰세요
# [1단계] PROMPT 변수 교체
PROMPT = """
You are a strict JSON emitter for detecting Korean math QUESTION items on a single PDF page image.
Return JSON only with this exact schema:
{"page_index": <int>, "items":[{"id":<int>,"kind":"MC"|"SA","bbox":[x1,y1,x2,y2]}, ...]}

Hard rules:
- Detect ONLY items that have a printed 4-digit question number (e.g., 0005, 0020, 0534). The id MUST equal that number as an integer ("0020"->20).
- Each item must contain EXACTLY ONE such 4-digit number, and that number must be visible inside the bbox.
- If the item contains multiple-choice markers like ①②③④⑤, set kind="MC". Otherwise set kind="SA".
- The bbox must include the entire question content (stem + any figures/graphs/tables + choices if present).
- DO NOT include theory/concept explanation boxes, definitions, summaries, headers/footers, page numbers, difficulty labels, or blank areas.
- Never output partial strips (thin bands). If unsure, do not output an item.
- JSON ONLY. No markdown, no code fences, no explanation.
""".strip()



@dataclass
class PageTask:
    page_number: int
    total_pages: int
    page_png_path: Path


class GeminiBBoxClient:
    def __init__(self) -> None:
        # 네가 쓰는 모델로 고정
        self.model_name = "gemini-3-flash-preview"  # <-- 핵심 변경
        key = GOOGLE_API_KEYS[0] if GOOGLE_API_KEYS else os.environ.get("GOOGLE_API_KEY", "")
        self.api_key = key.strip()
        self.enabled = bool(self.api_key)

    def _endpoint(self) -> str:
        # v1beta generateContent 엔드포인트
        return f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"

    def detect(self, page_index: int, image_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not self.enabled:
            return None, "Gemini API key is missing."

        try:
            raw = image_path.read_bytes()
            b64 = base64.b64encode(raw).decode("utf-8")

            payload = {
                "contents": [{
                    "parts": [
                        {"text": PROMPT.replace("{page_index}", str(page_index))},
                        {"inline_data": {"mime_type": "image/png", "data": b64}}
                    ]
                }],
                # 가능하면 JSON 강제(지원 안 하면 무시될 수 있음)
                "generationConfig": {
                    "temperature": 0.0,
                    "response_mime_type": "application/json"
                }
            }

            res = requests.post(self._endpoint(), headers={"Content-Type": "application/json"}, json=payload, timeout=120)

            if res.status_code != 200:
                return None, f"{res.status_code} {res.text}"

            data = res.json()

            # 모델 응답 텍스트 추출(후보 0번)
            text = ""
            try:
                text = data["candidates"][0]["content"]["parts"][0].get("text", "")
            except Exception:
                pass

            if not text:
                return None, f"Empty model output: {str(data)[:200]}"

            # JSON만 오도록 시켰으니 파싱
            try:
                return json.loads(text), None
            except Exception:
                # 혹시 앞뒤로 잡문이 섞이면 JSON 블록만 뽑아 파싱 시도
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if not m:
                    return None, f"JSON parse failed: {text[:200]}"
                return json.loads(m.group(0)), None

        except Exception as e:
            return None, f"Exception: {e}"

# =========================================================
# [GPT-5.3 Codex] 후처리 유틸 함수 모음 (여기에 붙여넣으세요)
# =========================================================
def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def _area(b):
    x1,y1,x2,y2 = b
    return max(0, x2-x1) * max(0, y2-y1)

def _iou(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter <= 0: return 0.0
    ua = _area(a) + _area(b) - inter
    return inter / ua if ua > 0 else 0.0

def _contains(big, small, ratio=0.8):
    bx1,by1,bx2,by2 = big
    sx1,sy1,sx2,sy2 = small
    ix1, iy1 = max(bx1,sx1), max(by1,sy1)
    ix2, iy2 = min(bx2,sx2), min(by2,sy2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    sa = _area(small)
    return sa > 0 and (inter / sa) >= ratio

# [1단계] dedup_items 함수를 이걸로 교체하세요
def dedup_items(items, iou_th=0.70):
    items = sorted(items, key=lambda it: (it["bbox"][1], it["bbox"][0]))
    kept = []
    for it in items:
        b = it["bbox"]
        drop = False
        for kt in kept:
            kb = kt["bbox"]
            if _iou(b, kb) >= iou_th or _contains(kb, b, 0.8) or _contains(b, kb, 0.8):
                # 더 큰 영역을 가진 쪽을 살림
                if _area(b) > _area(kb):
                    kt["bbox"] = b
                    kt["id"] = it.get("id", kt.get("id", 0))
                    # [GPT Fix] kind 병합 (둘 중 하나라도 MC면 MC 유지)
                    kt["kind"] = "MC" if (kt.get("kind") == "MC" or it.get("kind") == "MC") else it.get("kind", "SA")
                else:
                    # 기존 kt가 더 크더라도, 새 it가 MC라면 kind는 MC로 업데이트
                    if it.get("kind") == "MC":
                        kt["kind"] = "MC"
                drop = True
                break
        if not drop: kept.append(it)
    return kept

# [2/3] apply_overlap_cut 함수를 이걸로 덮어쓰세요
def apply_overlap_cut(items, H):
    # Y좌표 순 정렬
    items = sorted(items, key=lambda it: it["bbox"][1])
    
    min_gap = max(10, int(0.01 * H))
    # [강화] 문항 최소 높이 기준을 높여서(18%) 너무 가까운 조각 때문에 잘리는 것 방지
    min_question_h = max(180, int(0.18 * H)) 

    for i in range(len(items)-1):
        b = items[i]["bbox"]
        nb = items[i+1]["bbox"]
        y1, y2 = b[1], b[3]
        next_y1 = nb[1]

        # [핵심] 다음 박스가 '진짜 다음 문제'처럼 충분히 멀리 떨어져 있을 때만 자름
        if (next_y1 - y1) >= min_question_h:
            cut_y2 = next_y1 - min_gap
            if cut_y2 < y2:
                items[i]["bbox"][3] = cut_y2
        else:
            # 너무 가까우면(조각/쓰레기일 확률 높음) 자르지 않고 둠
            pass

    return items

# [3/3] final_garbage_filter 함수를 이걸로 덮어쓰세요
# [2단계] final_garbage_filter 함수 교체
def final_garbage_filter(items, W, H):
    # [설정] 기본 필터
    base_min_h = max(100, int(0.08 * H)) 
    min_area = int(0.01 * W * H)
    
    # [설정] 주관식(SA) 전용 엄격 필터 (개념박스 오탐 방지)
    sa_min_h = max(180, int(0.15 * H)) # 주관식은 더 커야 인정

    out = []
    for it in items:
        kind = it.get("kind", "Q") # MC or SA or Q
        x1, y1, x2, y2 = it["bbox"]
        w_box = x2 - x1
        h_box = y2 - y1
        area = w_box * h_box
        
        # 1. 공통 필터 (너무 작으면 버림)
        if area < min_area: continue
        
        # 2. 주관식(SA) 특별 검사
        if kind == "SA":
            # 주관식인데 높이가 너무 낮으면(개념 한줄 등) 버림
            if h_box < sa_min_h: continue
        else:
            # 객관식(MC) 등은 기본 높이만 넘으면 통과
            if h_box < base_min_h: continue

        # 3. 배너(띠) 제거 (공통)
        if w_box > (W * 0.70) and h_box < (H * 0.12):
            continue
            
        out.append(it)
    return out
# =========================================================


class PDFCutterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("980x680")

        self.input_files: List[Path] = []
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.log_queue: "queue.Queue[str]" = queue.Queue()

        self.output_root_var = tk.StringVar(value=str(DEFAULT_OUTPUT_ROOT))
        self.workers_var = tk.IntVar(value=DEFAULT_WORKERS)
        self.dpi_var = tk.IntVar(value=DEFAULT_DPI)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_label_var = tk.StringVar(value="진행률: 0/0")

        self._build_ui()
        self._start_log_pump()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill=tk.BOTH, expand=True)

        inp_frame = ttk.LabelFrame(top, text="입력 선택", padding=10)
        inp_frame.pack(fill=tk.X, pady=4)

        ttk.Button(inp_frame, text="PDF 파일 선택(1개)", command=self.select_pdf_file).pack(side=tk.LEFT, padx=4)
        ttk.Button(inp_frame, text="PDF 폴더 선택(일괄)", command=self.select_pdf_folder).pack(side=tk.LEFT, padx=4)

        self.input_label = ttk.Label(inp_frame, text="선택된 PDF: 0개", width=80)
        self.input_label.pack(side=tk.LEFT, padx=8)

        out_frame = ttk.LabelFrame(top, text="출력 설정", padding=10)
        out_frame.pack(fill=tk.X, pady=4)

        ttk.Label(out_frame, text="출력 루트 폴더:").pack(side=tk.LEFT)
        ttk.Entry(out_frame, textvariable=self.output_root_var, width=70).pack(side=tk.LEFT, padx=6)
        ttk.Button(out_frame, text="변경", command=self.select_output_root).pack(side=tk.LEFT)

        opt_frame = ttk.LabelFrame(top, text="옵션", padding=10)
        opt_frame.pack(fill=tk.X, pady=4)

        ttk.Label(opt_frame, text="병렬 수").pack(side=tk.LEFT)
        ttk.Spinbox(opt_frame, from_=MIN_WORKERS, to=MAX_WORKERS, textvariable=self.workers_var, width=6).pack(side=tk.LEFT, padx=6)
        ttk.Label(opt_frame, text="DPI").pack(side=tk.LEFT)
        ttk.Spinbox(opt_frame, from_=MIN_DPI, to=MAX_DPI, textvariable=self.dpi_var, width=6).pack(side=tk.LEFT, padx=6)

        ctl_frame = ttk.Frame(top)
        ctl_frame.pack(fill=tk.X, pady=6)

        self.start_btn = ttk.Button(ctl_frame, text="Start", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=4)
        self.stop_btn = ttk.Button(ctl_frame, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)

        ttk.Label(ctl_frame, textvariable=self.progress_label_var).pack(side=tk.LEFT, padx=16)

        self.progress = ttk.Progressbar(ctl_frame, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=8)

        log_frame = ttk.LabelFrame(top, text="로그", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=4)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=26)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state=tk.DISABLED)

    def _start_log_pump(self) -> None:
        def pump() -> None:
            try:
                while True:
                    msg = self.log_queue.get_nowait()
                    self._append_log(msg)
            except queue.Empty:
                pass
            self.root.after(120, pump)

        pump()

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def log(self, message: str) -> None:
        self.log_queue.put(message)

    def select_pdf_file(self) -> None:
        f = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if f:
            self.input_files = [Path(f)]
            self.input_label.configure(text=f"선택된 PDF: 1개 ({Path(f).name})")

    def select_pdf_folder(self) -> None:
        d = filedialog.askdirectory()
        if not d:
            return
        pdfs = sorted(Path(d).glob("*.pdf"))
        self.input_files = pdfs
        self.input_label.configure(text=f"선택된 PDF: {len(pdfs)}개 ({Path(d)})")

    def select_output_root(self) -> None:
        d = filedialog.askdirectory()
        if d:
            self.output_root_var.set(d)

    def start(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            self.log("⚠️ 이미 실행 중입니다.")
            return
        if not self.input_files:
            self.log("⚠️ PDF 파일/폴더를 먼저 선택하세요.")
            return

        self.stop_event.clear()
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.progress_var.set(0)
        self.progress_label_var.set("진행률: 0/0")

        self.worker_thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self.worker_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.log("🛑 Stop 요청 수신: 현재 작업 이후 즉시 중단합니다.")

    def _finalize_ui(self) -> None:
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)

    def _run_pipeline(self) -> None:
        try:
            output_root = Path(self.output_root_var.get())
            workers = max(MIN_WORKERS, min(MAX_WORKERS, int(self.workers_var.get())))
            dpi = max(MIN_DPI, min(MAX_DPI, int(self.dpi_var.get())))

            (output_root / "out_pages").mkdir(parents=True, exist_ok=True)
            (output_root / "out_crops").mkdir(parents=True, exist_ok=True)
            (output_root / "errors").mkdir(parents=True, exist_ok=True)

            total_pages_all = self._count_total_pages(self.input_files)
            processed_pages_all = 0

            client = GeminiBBoxClient()

            for pdf_path in self.input_files:
                if self.stop_event.is_set():
                    break

                saved, errs, done_pages = self._process_pdf(
                    pdf_path=pdf_path,
                    output_root=output_root,
                    dpi=dpi,
                    workers=workers,
                    gemini_client=client,
                    total_pages_all=total_pages_all,
                    processed_pages_before=processed_pages_all,
                )
                processed_pages_all += done_pages
                self.log(f"✅ [Done] pdf={pdf_path.stem} total_saved={saved} total_errors={errs}")

            if self.stop_event.is_set():
                self.log("🛑 사용자 중단으로 작업이 종료되었습니다.")
            else:
                self.log("✅ 전체 작업 완료")
        except Exception as e:
            self.log(f"❌ [Fail] P000 err={str(e)[:200]}")
        finally:
            self.root.after(0, self._finalize_ui)

    def _count_total_pages(self, pdf_paths: List[Path]) -> int:
        total = 0
        for p in pdf_paths:
            try:
                with fitz.open(p) as doc:
                    total += len(doc)
            except Exception:
                continue
        return max(total, 1)

    def _process_pdf(
        self,
        pdf_path: Path,
        output_root: Path,
        dpi: int,
        workers: int,
        gemini_client: GeminiBBoxClient,
        total_pages_all: int,
        processed_pages_before: int,
    ) -> Tuple[int, int, int]:
        pdf_stem = pdf_path.stem
        pages_dir = output_root / "out_pages" / pdf_stem
        crops_dir = output_root / "out_crops" / pdf_stem
        errors_dir = output_root / "errors" / pdf_stem
        pages_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)
        errors_dir.mkdir(parents=True, exist_ok=True)

        tasks = self._render_pdf_pages(pdf_path, pages_dir, dpi)

        total_saved = 0
        total_errors = 0
        done_pages = 0

        pending: Dict[Any, PageTask] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            task_iter = iter(tasks)

            for _ in range(min(workers, len(tasks))):
                t = next(task_iter, None)
                if not t:
                    break
                fut = executor.submit(self._process_page, t, crops_dir, errors_dir, gemini_client, pdf_stem)
                pending[fut] = t

            while pending and not self.stop_event.is_set():
                done, _ = wait(pending.keys(), timeout=0.2, return_when=FIRST_COMPLETED)
                for fut in done:
                    task = pending.pop(fut)
                    done_pages += 1
                    processed = processed_pages_before + done_pages
                    self.progress_label_var.set(f"진행률: {processed}/{total_pages_all}")
                    self.progress_var.set((processed / max(1, total_pages_all)) * 100.0)

                    try:
                        saved_count, is_error = fut.result()
                        total_saved += saved_count
                        total_errors += 1 if is_error else 0
                    except Exception as e:
                        total_errors += 1
                        self.log(f"❌ [Fail] P{task.page_number:03d} err={str(e)[:200]}")

                    nxt = next(task_iter, None)
                    if nxt and not self.stop_event.is_set():
                        nf = executor.submit(self._process_page, nxt, crops_dir, errors_dir, gemini_client, pdf_stem)
                        pending[nf] = nxt

            if self.stop_event.is_set():
                for fut in pending:
                    fut.cancel()

        return total_saved, total_errors, done_pages

    def _render_pdf_pages(self, pdf_path: Path, pages_dir: Path, dpi: int) -> List[PageTask]:
        tasks: List[PageTask] = []
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            for idx, page in enumerate(doc):
                page_no = idx + 1
                out_path = pages_dir / f"P{page_no:03d}.png"
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                pix.save(out_path)
                tasks.append(PageTask(page_number=page_no, total_pages=total_pages, page_png_path=out_path))
        return tasks

    def _process_page(
        self,
        task: PageTask,
        crops_dir: Path,
        errors_dir: Path,
        gemini_client: GeminiBBoxClient,
        pdf_stem: str,
    ) -> Tuple[int, bool]:
        if self.stop_event.is_set():
            return 0, False

        img = Image.open(task.page_png_path)
        w, h = img.size

        try:
            data, raw = gemini_client.detect(page_index=task.page_number - 1, image_path=task.page_png_path)
            if data is None:
                self._write_page_error(errors_dir, task.page_number, task.page_png_path, raw or "parse failed")
                self.log(f"❌ [Fail] P{task.page_number:03d} err={(raw or 'json parse fail')[:200]}")
                return 0, True

            items = self._validate_and_normalize_items(data, w, h)
            if items is None:
                raw_text = raw if raw is not None else json.dumps(data, ensure_ascii=False)
                self._write_page_error(errors_dir, task.page_number, task.page_png_path, raw_text)
                self.log(f"❌ [Fail] P{task.page_number:03d} err=invalid fields or bbox")
                return 0, True

            crops, dropped = self._build_crop_regions(items, w, h, task.page_number)

            saved = 0
# 반환값이 6개로 늘었으니 변수 하나 더 받습니다 (kind)
            for seq, (qid, x1, y1, x2, y2, kind) in enumerate(crops, start=1):
                if self.stop_event.is_set():
                    break
                crop_img = img.crop((x1, y1, x2, y2))
                # 파일명에 kind(MC/SA)를 포함시켜서 구분하기 쉽게 함
                # 예: P003_Q001_N0020_MC.png
                out_name = f"P{task.page_number:03d}_Q{seq:03d}_N{qid:03d}_{kind}.png"
                crop_img.save(crops_dir / out_name)
                saved += 1

            self.log(
                f"🧾 [PDF Cut] pdf={pdf_stem} page={task.page_number}/{task.total_pages} "
                f"items={len(items)} saved={saved} dropped={dropped}"
            )
            return saved, False
        except Exception as e:
            self._write_page_error(errors_dir, task.page_number, task.page_png_path, str(e))
            self.log(f"❌ [Fail] P{task.page_number:03d} err={str(e)[:200]}")
            return 0, True
        finally:
            img.close()

    def _write_page_error(self, errors_dir: Path, page_num: int, png_src: Path, raw_text: str) -> None:
        png_dst = errors_dir / f"P{page_num:03d}.png"
        json_dst = errors_dir / f"P{page_num:03d}.json"
        try:
            png_dst.write_bytes(png_src.read_bytes())
        except Exception:
            pass
        payload = {"page": page_num, "raw": raw_text, "timestamp": time.time()}
        json_dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

# [수정] 이 함수를 이걸로 통째로 교체하세요!
    @staticmethod
    def _validate_and_normalize_items(data: Dict[str, Any], w: int, h: int) -> Optional[List[Dict[str, Any]]]:
        if not isinstance(data, dict):
            return None
        items = data.get("items")
        if not isinstance(items, list):
            return None

        norm: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            
            # [Fix 1] "Q"만 찾는 옛날 규칙 삭제 -> MC/SA/Q 모두 허용
            kind = item.get("kind", "Q")
            
            bbox = item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                return None
            try:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                qid = int(item.get("id", 0))
            except Exception:
                return None
            
            # 좌표 유효성 검사
            if x1 >= x2 or y1 >= y2:
                return None
            if x2 < 0 or y2 < 0 or x1 > w or y1 > h:
                return None
            
            # 좌표 클램핑 (이미지 범위 안으로)
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(1, min(w, x2))
            y2 = max(1, min(h, y2))
            
            # [Fix 2] kind 정보도 같이 포장해서 넘겨줌
            norm.append({
                "id": qid if qid >= 0 else 0, 
                "bbox": [x1, y1, x2, y2],
                "kind": kind 
            })

        return norm

# [2단계] 이 함수 내부를 아래 코드로 완전히 교체하세요
    def _build_crop_regions(self, items: List[Dict[str, Any]], w: int, h: int, page_num: int) -> Tuple[List[Tuple[int, int, int, int, int, str]], int]:
        # 1. ID 및 Kind 정제
        clean_items = []
        for it in items:
            try:
                qid = int(it.get("id", 0))
                if qid < 1 or qid > 9999: qid = 0
            except: qid = 0
            
            bbox = [int(v) for v in it["bbox"]]
            kind = it.get("kind", "Q")
            clean_items.append({"id": qid, "bbox": bbox, "kind": kind})

        # 2. 중복 제거 (MC 정보 보존)
        clean_items = dedup_items(clean_items)

        # 3. 조건부 겹침 컷오프
        clean_items = apply_overlap_cut(clean_items, h)

        # 4. 패딩 및 확장 (그림 잘림 방지 로직 복구)
        pad_x = max(30, int(0.015 * w))
        pad_y = max(30, int(0.015 * h))
        
        final_candidates = []
        for it in clean_items:
            x1, y1, x2, y2 = it["bbox"]
            
            # [GPT Fix] 아래 확장 로직: 짧은 문제는 더 많이, 긴 문제는 조금만 확장
            extra_bottom = int(0.10 * h) if (y2 - y1) < int(0.18 * h) else int(0.06 * h)
            
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            # 여기가 핵심: 원래 y2에 pad_y와 extra_bottom을 더함
            cy2 = min(h, y2 + pad_y + extra_bottom)

            final_candidates.append({"id": it["id"], "bbox": [cx1, cy1, cx2, cy2], "kind": it["kind"]})

        # 5. 최종 쓰레기 제거
        final_candidates = final_garbage_filter(final_candidates, w, h)

        # 6. 결과 반환
        out = []
        for it in final_candidates:
            x1, y1, x2, y2 = it["bbox"]
            out.append((it["id"], x1, y1, x2, y2, it["kind"]))

        dropped = len(items) - len(out)
        return out, dropped



def main() -> None:
    root = tk.Tk()
    app = PDFCutterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
