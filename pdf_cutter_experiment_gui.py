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


PROMPT = """
You are a strict JSON emitter for detecting Korean math question regions.
Input is a single PDF page image.
Return JSON only with this exact schema:
{"page_index": <int>, "items":[{"id":<int>,"kind":"Q","bbox":[x1,y1,x2,y2]}, ...]}
Rules:
- Detect only question blocks (kind must be "Q").
- bbox coordinates must be integers in image pixel coordinates.
- page_index should be 0 if unknown.
- No markdown, no code fences, no explanation.
""".strip()


@dataclass
class PageTask:
    page_number: int
    total_pages: int
    page_png_path: Path


class GeminiBBoxClient:
    def __init__(self) -> None:
        self.model_name = "gemini-1.5-flash"
        key = GOOGLE_API_KEYS[0] if GOOGLE_API_KEYS else os.environ.get("GOOGLE_API_KEY", "")
        self.enabled = bool(key)
        if self.enabled:
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel(self.model_name)
        else:
            self.model = None

    def detect(self, page_index: int, image_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not self.enabled or self.model is None:
            return None, "Gemini API key is missing."

        with image_path.open("rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("utf-8")

        response = self.model.generate_content(
            [
                {"text": PROMPT},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": b64,
                    }
                },
            ],
            generation_config={"temperature": 0, "max_output_tokens": 2048},
        )

        text = response.text or ""
        parsed = self._parse_json(text)
        if parsed is None:
            return None, text
        parsed.setdefault("page_index", page_index)
        return parsed, text

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return None
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
        return None


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
            for seq, (qid, x1, y1, x2, y2) in enumerate(crops, start=1):
                if self.stop_event.is_set():
                    break
                crop_img = img.crop((x1, y1, x2, y2))
                out_name = f"P{task.page_number:03d}_Q{seq:03d}_N{qid:03d}.png"
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
            if item.get("kind") != "Q":
                continue
            bbox = item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                return None
            try:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                qid = int(item.get("id", 0))
            except Exception:
                return None
            if x1 >= x2 or y1 >= y2:
                return None
            if x2 < 0 or y2 < 0 or x1 > w or y1 > h:
                return None
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(1, min(w, x2))
            y2 = max(1, min(h, y2))
            norm.append({"id": qid if qid >= 0 else 0, "bbox": [x1, y1, x2, y2]})

        return norm

    def _build_crop_regions(self, items: List[Dict[str, Any]], w: int, h: int, page_num: int) -> Tuple[List[Tuple[int, int, int, int, int]], int]:
        sorted_items = sorted(items, key=lambda it: it["bbox"][1])

        pad_x = max(30, int(0.015 * w))
        pad_y = max(30, int(0.015 * h))
        gap = max(10, int(0.01 * h))

        tmp: List[Dict[str, int]] = []
        for i, it in enumerate(sorted_items):
            x1, y1, x2, y2 = it["bbox"]
            extra_bottom = int(0.10 * h) if (y2 - y1) < int(0.18 * h) else int(0.06 * h)

            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y + extra_bottom)

            if i < len(sorted_items) - 1:
                next_y1 = sorted_items[i + 1]["bbox"][1]
                cy2 = min(cy2, max(cy1 + 1, next_y1 - gap))

            tmp.append({"id": it["id"], "x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2})

        out: List[Tuple[int, int, int, int, int]] = []
        dropped = 0
        min_h = max(100, int(0.06 * h))
        area_limit = int(0.01 * w * h)

        for c in tmp:
            ch = c["y2"] - c["y1"]
            area = (c["x2"] - c["x1"]) * ch
            if ch < min_h:
                dropped += 1
                self.log(f"🗑️ [Drop] P{page_num:03d} reason=height")
                continue
            if area < area_limit:
                dropped += 1
                self.log(f"🗑️ [Drop] P{page_num:03d} reason=area")
                continue
            out.append((c["id"], c["x1"], c["y1"], c["x2"], c["y2"]))

        return out, dropped


def main() -> None:
    root = tk.Tk()
    app = PDFCutterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
