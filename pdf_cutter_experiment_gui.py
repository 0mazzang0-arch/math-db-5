import json
import queue
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_new_ir"] = "0"

try:
    import cv2
except Exception:
    cv2 = None
try:
    import numpy as np
except Exception:
    np = None

try:
    from paddleocr import PPStructureV3
    PADDLE_OCR_AVAILABLE = True
except Exception:
    PPStructureV3 = None
    PADDLE_OCR_AVAILABLE = False

APP_TITLE = "PDF Cutter Experiment GUI"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "pdf_cutter_output"
DEFAULT_DPI = 250
DEFAULT_WORKERS = 6
MIN_WORKERS = 1
MAX_WORKERS = 8
MIN_DPI = 200
MAX_DPI = 300


DEBUG_MODE = True
TRACE_MODE = True
MAX_DEBUG_JSON_CHARS = 10_000
PIN_GUIDE = (
    "Python 3.11 venv에서 paddlepaddle==3.2.0 + paddleocr==3.3.x로 재설치 필요. "
    "명령어: python -m venv .venv && source .venv/bin/activate && "
    "pip install --upgrade pip && pip install paddlepaddle==3.2.0 paddleocr==3.3.0"
)



@dataclass
class PageTask:
    page_number: int
    total_pages: int
    page_png_path: Path


class PaddleStructureClient:
    def __init__(self) -> None:
        self.enabled = bool(PADDLE_OCR_AVAILABLE and cv2 is not None and np is not None)
        self._engine: Optional[Any] = None
        self.paddleocr_version = "unknown"
        self.paddle_version = "unknown"
        self.init_error: Optional[str] = None
        try:
            import paddleocr as _pocr  # type: ignore
            self.paddleocr_version = getattr(_pocr, "__version__", "unknown")
        except Exception:
            pass
        try:
            import paddle  # type: ignore
            self.paddle_version = getattr(paddle, "__version__", "unknown")
        except Exception:
            pass

    def _ensure_engine(self) -> Tuple[bool, Optional[str]]:
        if not self.enabled:
            return False, "PaddleOCR 설치 필요: pip install paddlepaddle paddleocr"
        if self._engine is not None:
            return True, None
        try:
            self._engine = PPStructureV3()
            return True, None
        except Exception as e:
            self.init_error = str(e)
            return False, f"stage=init_engine err={e}"

    @staticmethod
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

    @staticmethod
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
        return None

    @staticmethod
    def _safe_keys(obj: Any) -> List[str]:
        if isinstance(obj, dict):
            return sorted([str(k) for k in obj.keys()])
        return []

    @staticmethod
    def _is_unimplemented_runtime_error(err: Exception) -> bool:
        msg = str(err)
        return (
            "ConvertPirAttribute2RuntimeAttribute" in msg
            or "onednn_instruction" in msg
            or "(Unimplemented)" in msg
        )

    def _run_subprocess_fallback(self, image_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        runner_path = Path(__file__).resolve().parent / "v3_runner_tmp.py"
        script = (
            "import json, os, sys\n"
            "os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'\n"
            "os.environ['FLAGS_use_mkldnn']='0'\n"
            "os.environ['FLAGS_use_onednn']='0'\n"
            "os.environ['FLAGS_enable_mkldnn']='0'\n"
            "os.environ['FLAGS_enable_pir_api']='0'\n"
            "os.environ['FLAGS_enable_new_ir']='0'\n"
            "import cv2, numpy as np\n"
            "from paddleocr import PPStructureV3\n"
            "p=sys.argv[1]\n"
            "raw=cv2.imdecode(np.fromfile(p,dtype=np.uint8), cv2.IMREAD_COLOR)\n"
            "if raw is None:\n"
            "    print(json.dumps({'ok':False,'stage':'detect_load','err':'imdecode failed'},ensure_ascii=False)); sys.exit(0)\n"
            "engine=PPStructureV3()\n"
            "out=engine.predict(input=raw)\n"
            "first=next(iter(out),None) if not isinstance(out,(list,tuple,dict)) else (out[0] if isinstance(out,(list,tuple)) and out else out)\n"
            "j=getattr(first,'json',None)\n"
            "j=j() if callable(j) else j\n"
            "if not isinstance(j,dict) and isinstance(first,dict):\n"
            "    j=first.get('json') or first.get('result') or first.get('res')\n"
            "if not isinstance(j,dict):\n"
            "    print(json.dumps({'ok':False,'stage':'parse_json','err':'invalid json'},ensure_ascii=False)); sys.exit(0)\n"
            "print(json.dumps({'ok':True,'pp_json':j}, ensure_ascii=False))\n"
        )

        try:
            runner_path.write_text(script, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(runner_path), str(image_path)],
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            out = (proc.stdout or "").strip()
            if not out:
                return None, f"stage=fallback_subprocess err=empty stdout {PIN_GUIDE}"
            payload = json.loads(out)
            if not isinstance(payload, dict):
                return None, f"stage=fallback_subprocess err=invalid payload {PIN_GUIDE}"
            if payload.get("ok") and isinstance(payload.get("pp_json"), dict):
                return {"pp_json": payload["pp_json"], "pp_meta": {"fallback": "subprocess"}}, None
            return None, f"stage={payload.get('stage','fallback_subprocess')} err={payload.get('err','unknown')} {PIN_GUIDE}"
        except Exception as e:
            return None, f"stage=fallback_subprocess err={e} {PIN_GUIDE}"
        finally:
            try:
                runner_path.unlink(missing_ok=True)
            except Exception:
                j = None
        if isinstance(j, dict):
            return j
        if isinstance(first, dict):
            for k in ("json", "result", "res"):
                cand = first.get(k)
                if isinstance(cand, dict):
                    return cand
        return None

    def detect(self, image_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not self.enabled or cv2 is None or np is None:
            return None, "PaddleOCR 설치 필요: pip install paddlepaddle paddleocr"
        ok, init_err = self._ensure_engine()
        if not ok:
            return None, f"{init_err} {PIN_GUIDE}"
        try:
            data = np.fromfile(str(image_path), dtype=np.uint8)
            if data.size == 0:
                return None, "stage=detect_load empty image buffer"
            raw = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if raw is None:
                return None, "stage=detect_load imdecode failed"
        except Exception as e:
            return None, f"stage=detect_load err={e}"

        output = None
        try:
            assert self._engine is not None
            output = self._engine.predict(input=raw)
        except Exception as e0:
            try:
                assert self._engine is not None
                output = self._engine.predict(input=str(image_path))
            except Exception as e:
                if self._is_unimplemented_runtime_error(e0) or self._is_unimplemented_runtime_error(e):
                    fb, fb_err = self._run_subprocess_fallback(image_path)
                    if fb is not None:
                        return fb, None
                    return None, f"stage=paddle_runtime_unimplemented err={e} {PIN_GUIDE}; fallback={fb_err}"
                return None, f"stage=detect_predict err={e}"

        first = self._first_output(output)
        j = self._extract_json(first)
        if not isinstance(j, dict):
            meta = {
                "output_type": type(output).__name__,
                "first_type": type(first).__name__ if first is not None else "None",
                "first_keys": self._safe_keys(first),
            }
            return None, f"stage=parse_json invalid json payload meta={json.dumps(meta, ensure_ascii=False)}"

        pp_meta = {
            "output_type": type(output).__name__,
            "first_type": type(first).__name__ if first is not None else "None",
            "first_keys": self._safe_keys(first),
            "json_keys": self._safe_keys(j),
        }
        _ = pp_meta
        return {"pp_json": j}, None

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
        self._layout_keys_logged = False
        self.paddle_client = PaddleStructureClient()
        self.log(
            f"ℹ️ PaddleOCR={self.paddle_client.paddleocr_version} "
            f"Paddle={self.paddle_client.paddle_version}"
        )
        if not self.paddle_client.enabled:
            self.log("PaddleOCR 설치 필요: pip install paddlepaddle paddleocr")
            if self.paddle_client.init_error:
                self.log(f"❌ [Fail] P000 stage=init_engine err={self.paddle_client.init_error[:200]}")
            self.start_btn.configure(state=tk.DISABLED)

    def _set_progress_safe(self, processed: int, total: int) -> None:
        def _apply() -> None:
            self.progress_label_var.set(f"진행률: {processed}/{total}")
            self.progress_var.set((processed / max(1, total)) * 100.0)

        self.root.after(0, _apply)

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
        if not self.paddle_client.enabled:
            self.log("PaddleOCR 설치 필요: pip install paddlepaddle paddleocr")
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

            for pdf_path in self.input_files:
                if self.stop_event.is_set():
                    break

                saved, errs, done_pages = self._process_pdf(
                    pdf_path=pdf_path,
                    output_root=output_root,
                    dpi=dpi,
                    workers=workers,
                    paddle_client=self.paddle_client,
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
            self.log(f"❌ [Fail] P000 stage=run_pipeline err={str(e)[:200]}")
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
        paddle_client: PaddleStructureClient,
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
                fut = executor.submit(self._process_page, t, crops_dir, errors_dir, paddle_client, pdf_stem)
                pending[fut] = t

            while pending and not self.stop_event.is_set():
                done, _ = wait(pending.keys(), timeout=0.2, return_when=FIRST_COMPLETED)
                for fut in done:
                    task = pending.pop(fut)
                    done_pages += 1
                    processed = processed_pages_before + done_pages
                    self._set_progress_safe(processed, total_pages_all)

                    try:
                        saved_count, is_error = fut.result()
                        total_saved += saved_count
                        total_errors += 1 if is_error else 0
                    except Exception as e:
                        total_errors += 1
                        self.log(f"❌ [Fail] P{task.page_number:03d} stage=future_result err={str(e)[:200]}")

                    nxt = next(task_iter, None)
                    if nxt and not self.stop_event.is_set():
                        nf = executor.submit(self._process_page, nxt, crops_dir, errors_dir, paddle_client, pdf_stem)
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
        paddle_client: PaddleStructureClient,
        pdf_stem: str,
    ) -> Tuple[int, bool]:
        if self.stop_event.is_set():
            return 0, False

        img = Image.open(task.page_png_path)
        w, h = img.size

        try:
            data, raw = paddle_client.detect(image_path=task.page_png_path)
            if data is None:
                stage = "detect"
                if raw and "stage=" in raw:
                    stage = raw.split("stage=", 1)[1].split()[0]
                self._write_page_error(
                    errors_dir,
                    task.page_number,
                    task.page_png_path,
                    raw or "parse failed",
                    stage=stage,
                )
                self.log(f"❌ [Fail] P{task.page_number:03d} stage={stage} err={(raw or 'parse fail')}")
                if stage == "paddle_runtime_unimplemented":
                    self.log(f"❌ [Fail] P{task.page_number:03d} stage={stage} err={PIN_GUIDE}")
                return 0, True

            anchors, objects = self._normalize_structure(data, w, h)
            if not anchors:
                trace = data.get("_trace", {}) if isinstance(data, dict) else {}
                self._write_page_error(
                    errors_dir,
                    task.page_number,
                    task.page_png_path,
                    "anchors=0",
                    stage="parse_anchors",
                    extras={
                        "pp_json_keys": trace.get("pp_json_keys", []),
                        "text_candidates": trace.get("text_candidates", []),
                        "pp_meta": trace.get("pp_meta", {}),
                        "pp_json_sample": trace.get("pp_json_sample", {}),
                    },
                )
                self.log(f"❌ [Fail] P{task.page_number:03d} stage=parse_anchors err=anchors=0")
                return 0, True

            crops, dropped, errors = self._build_anchor_slice_regions(anchors, objects, w, h)
            if errors > 0 and not crops:
                self._write_page_error(
                    errors_dir,
                    task.page_number,
                    task.page_png_path,
                    "anchor overlap conflict",
                    stage="slice",
                    extras={"anchors": len(anchors), "objects": len(objects), "errors": errors},
                )
                self.log(f"❌ [Fail] P{task.page_number:03d} stage=slice err=anchor overlap conflict")
                self.log(f"🧾 [AnchorSlice] P{task.page_number:03d} anchors={len(anchors)} saved=0 dropped={dropped} errors={errors}")
                return 0, True

            saved = 0
            for seq, (qid, x1, y1, x2, y2) in enumerate(crops, start=1):
                if self.stop_event.is_set():
                    break
                crop_img = img.crop((x1, y1, x2, y2))
                out_name = f"P{task.page_number:03d}_Q{seq:03d}_N{qid:04d}.png"
                crop_img.save(crops_dir / out_name)
                saved += 1

            if DEBUG_MODE:
                self._save_debug_overlay(crops_dir, task.page_number, task.page_png_path, anchors, objects, crops)

            self.log(
                f"🧾 [AnchorSlice] P{task.page_number:03d} anchors={len(anchors)} "
                f"saved={saved} dropped={dropped} errors={errors}"
            )
            return saved, False
        except Exception as e:
            self._write_page_error(errors_dir, task.page_number, task.page_png_path, str(e), stage="process_page")
            self.log(f"❌ [Fail] P{task.page_number:03d} stage=process_page err={str(e)[:200]}")
            return 0, True
        finally:
            img.close()

    def _write_page_error(
        self,
        errors_dir: Path,
        page_num: int,
        png_src: Path,
        raw_text: str,
        stage: str = "unknown",
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        png_dst = errors_dir / f"P{page_num:03d}.png"
        json_dst = errors_dir / f"P{page_num:03d}.json"
        try:
            png_dst.write_bytes(png_src.read_bytes())
        except Exception:
            pass
        payload = {"page": page_num, "stage": stage, "raw": raw_text[:MAX_DEBUG_JSON_CHARS], "timestamp": time.time()}
        if extras:
            payload.update(extras)
        json_dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _poly_to_bbox(poly: Any) -> Optional[List[int]]:
        if isinstance(poly, dict):
            for key in ("points", "bbox", "xyxy", "box", "polygon", "text_region"):
                if key in poly:
                    return PDFCutterApp._poly_to_bbox(poly.get(key))
            if all(k in poly for k in ("x1", "y1", "x2", "y2")):
                return PDFCutterApp._poly_to_bbox([poly.get("x1"), poly.get("y1"), poly.get("x2"), poly.get("y2")])

        if not isinstance(poly, (list, tuple)):
            return None

        # Case A: flat [x1, y1, x2, y2]
        if len(poly) == 4 and all(isinstance(v, (int, float)) for v in poly):
            x1, y1, x2, y2 = [int(v) for v in poly]
            lo_x, hi_x = min(x1, x2), max(x1, x2)
            lo_y, hi_y = min(y1, y2), max(y1, y2)
            if hi_x <= lo_x or hi_y <= lo_y:
                return None
            return [lo_x, lo_y, hi_x, hi_y]

        # Case B: polygon [[x, y], ...]
        if len(poly) < 4:
            return None
        try:
            xs = [int(p[0]) for p in poly if isinstance(p, (list, tuple)) and len(p) >= 2]
            ys = [int(p[1]) for p in poly if isinstance(p, (list, tuple)) and len(p) >= 2]
        except Exception:
            return None
        if len(xs) < 2 or len(ys) < 2:
            return None
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    @staticmethod
    def _pick_list_by_keys(source: Any, keys: List[str]) -> List[Any]:
        if isinstance(source, list):
            return source
        if isinstance(source, dict):
            for key in keys:
                val = source.get(key)
                if isinstance(val, list):
                    return val
        return []

    @staticmethod
    def _extract_text_bbox_candidates(item: Any) -> List[Tuple[str, Optional[List[int]]]]:
        out: List[Tuple[str, Optional[List[int]]]] = []
        if isinstance(item, dict):
            text = ""
            for k in ("text", "ocrText", "rec_text", "content"):
                if isinstance(item.get(k), str):
                    text = item.get(k).strip()
                    if text:
                        break
            bbox = None
            for bk in ("text_region", "bbox", "xyxy", "points", "polygon"):
                if bk in item:
                    bbox = PDFCutterApp._poly_to_bbox(item.get(bk))
                    if bbox is not None:
                        break
            if text:
                out.append((text, bbox))
            for lk in ("words", "lines", "text_lines", "ocr", "ocr_result"):
                for node in PDFCutterApp._pick_list_by_keys(item.get(lk), [lk]):
                    out.extend(PDFCutterApp._extract_text_bbox_candidates(node))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            bbox = PDFCutterApp._poly_to_bbox(item[0])
            text = ""
            info = item[1]
            if isinstance(info, (list, tuple)) and info:
                text = str(info[0]).strip()
            elif isinstance(info, str):
                text = info.strip()
            if text:
                out.append((text, bbox))
        return out

    def _normalize_structure(self, data: Dict[str, Any], page_w: int, page_h: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        pp_json = data.get("pp_json", {}) if isinstance(data, dict) else {}
        layout = self._pick_list_by_keys(pp_json, ["prunedResult", "res", "result", "layout", "outputs", "regions"])
        if not layout:
            layout = self._pick_list_by_keys(data.get("layout", []) if isinstance(data, dict) else [], ["res", "layout"])
        trace_samples: List[Dict[str, Any]] = []

        if DEBUG_MODE and layout and not self._layout_keys_logged and isinstance(layout[0], dict):
            self.log(f"🔎 [PPStructure] layout[0].keys={sorted(layout[0].keys())}")
            self._layout_keys_logged = True

        strict_pat = re.compile(r"^\s*\d{4}\s*$")
        weak_pat = re.compile(r"^\s*0*\d{1,4}\s*$")
        anchors: List[Dict[str, Any]] = []
        weak_anchor_candidates: List[Dict[str, Any]] = []
        objects: List[Dict[str, Any]] = []

        for block in layout:
            if not isinstance(block, dict):
                continue
            btype = str(block.get("type", block.get("label", block.get("category", "text")))).lower()
            bx = self._poly_to_bbox(block.get("bbox")) or self._poly_to_bbox(block)

            obj_type = "text"
            if "figure" in btype or "image" in btype:
                obj_type = "figure"
            elif "table" in btype:
                obj_type = "table"
            if bx is not None:
                objects.append({"type": obj_type, "bbox": bx})

            if btype not in {"text", "title", "list", "paragraph"}:
                continue
            lines = self._pick_list_by_keys(block, ["res", "words", "lines", "text_lines", "ocr", "ocr_result"])
            candidate_nodes = [block] + lines
            for line in candidate_nodes:
                for txt, line_bbox in self._extract_text_bbox_candidates(line):
                    if line_bbox is None:
                        continue
                    if len(trace_samples) < 20 and TRACE_MODE:
                        trace_samples.append({"text": txt, "bbox": line_bbox})
                    is_strict = bool(strict_pat.match(txt))
                    is_weak = bool(weak_pat.match(txt))
                    if not (is_strict or is_weak):
                        continue
                    qid = int(txt) if txt else 0
                    lx1, ly1, lx2, ly2 = line_bbox
                    bw = max(0, lx2 - lx1)
                    bh = max(0, ly2 - ly1)
                    cx = (line_bbox[0] + line_bbox[2]) / 2
                    col = 0 if cx < (page_w * 0.5) else 1

                    if not (bh <= (0.12 * page_h) and bw <= (0.20 * page_w)):
                        continue
                    if col == 0 and lx1 > (0.35 * page_w):
                        continue
                    if col == 1 and lx1 < (0.50 * page_w):
                        continue

                    cand = {"id": qid, "bbox": line_bbox, "col": col}
                    if is_strict:
                        anchors.append(cand)
                    else:
                        weak_anchor_candidates.append(cand)

        if not anchors and weak_anchor_candidates:
            anchors = weak_anchor_candidates

        if DEBUG_MODE and not anchors and layout:
            sample = layout[0] if isinstance(layout[0], dict) else {"sample": str(layout[0])}
            self.log(f"⚠️ [PPStructure] anchors=0 sample={json.dumps(sample, ensure_ascii=False)[:400]}")
        if isinstance(data, dict):
            pp_json_sample: Dict[str, Any] = {}
            if isinstance(pp_json, dict):
                for k in list(pp_json.keys())[:5]:
                    v = pp_json.get(k)
                    pp_json_sample[str(k)] = str(v)[:200]
            data["_trace"] = {
                "pp_meta": data.get("pp_meta", {}),
                "pp_json_keys": sorted(pp_json.keys()) if isinstance(pp_json, dict) else [],
                "pp_json_sample": pp_json_sample,
                "text_candidates": trace_samples,
            }

        anchors.sort(key=lambda a: (a["col"], a["bbox"][1], a["bbox"][0]))
        return anchors, objects

    def _build_anchor_slice_regions(
        self,
        anchors: List[Dict[str, Any]],
        objects: List[Dict[str, Any]],
        w: int,
        h: int,
    ) -> Tuple[List[Tuple[int, int, int, int, int]], int, int]:
        margin = max(10, int(0.01 * h))
        pad_x = max(30, int(0.015 * w))
        pad_y = max(30, int(0.015 * h))

        by_col = {0: [], 1: []}
        for a in anchors:
            by_col[a["col"]].append(a)

        candidates: List[Dict[str, Any]] = []
        error_count = 0
        for col in (0, 1):
            col_anchors = sorted(by_col[col], key=lambda a: a["bbox"][1])
            x1 = 0 if col == 0 else int(w * 0.5)
            x2 = int(w * 0.5) if col == 0 else w
            for i, anchor in enumerate(col_anchors):
                y_top = anchor["bbox"][1]
                if i + 1 < len(col_anchors):
                    y_bottom = col_anchors[i + 1]["bbox"][1] - margin
                else:
                    y_bottom = h
                if y_bottom <= y_top:
                    error_count += 1
                    continue
                box = [x1, y_top, x2, y_bottom]

                for obj in objects:
                    if obj["type"] not in {"figure", "table"}:
                        continue
                    ox1, oy1, ox2, oy2 = obj["bbox"]
                    cx = (ox1 + ox2) / 2
                    cy = (oy1 + oy2) / 2
                    if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
                        box = [min(box[0], ox1), min(box[1], oy1), max(box[2], ox2), max(box[3], oy2)]

                if col == 0:
                    box[2] = min(box[2], int(w * 0.60))
                else:
                    box[0] = max(box[0], int(w * 0.40))
                if box[2] <= box[0]:
                    error_count += 1
                    continue

                candidates.append({"id": anchor["id"], "bbox": box, "col": col})

        candidates.sort(key=lambda it: (it["col"], it["bbox"][1]))

        for col in (0, 1):
            col_items = [c for c in candidates if c["col"] == col]
            for i in range(len(col_items) - 1):
                cur = col_items[i]["bbox"]
                nxt = col_items[i + 1]["bbox"]
                overlap = cur[3] - nxt[1]
                if overlap <= 0:
                    continue
                if overlap <= int(0.15 * h):
                    nxt[1] = cur[3] + margin
                    if nxt[1] >= nxt[3]:
                        return [], len(anchors), error_count + 1
                else:
                    return [], len(anchors), error_count + 1

        out: List[Tuple[int, int, int, int, int]] = []
        for it in candidates:
            x1, y1, x2, y2 = it["bbox"]
            x1 = _clamp(x1 - pad_x, 0, w - 1)
            y1 = _clamp(y1 - pad_y, 0, h - 1)
            x2 = _clamp(x2 + pad_x, 1, w)
            y2 = _clamp(y2 + pad_y, 1, h)
            w_box = x2 - x1
            h_box = y2 - y1
            area = w_box * h_box
            if w_box > (w * 0.70) and h_box < (h * 0.12):
                continue
            if h_box < max(120, int(0.10 * h)) or area < int(0.01 * w * h):
                continue
            out.append((max(0, min(9999, int(it["id"]))), x1, y1, x2, y2))

        dropped = max(0, len(anchors) - len(out))
        return out, dropped, error_count

    def _save_debug_overlay(
        self,
        crops_dir: Path,
        page_num: int,
        page_png: Path,
        anchors: List[Dict[str, Any]],
        objects: List[Dict[str, Any]],
        crops: List[Tuple[int, int, int, int, int]],
    ) -> None:
        if cv2 is None or np is None:
            return
        try:
            data = np.fromfile(str(page_png), dtype=np.uint8)
            canvas = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            canvas = None
        if canvas is None:
            return
        for a in anchors:
            x1, y1, x2, y2 = a["bbox"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for obj in objects:
            if obj["type"] not in {"figure", "table"}:
                continue
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for _, x1, y1, x2, y2 in crops:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imwrite(str(crops_dir / f"debug_P{page_num:03d}.jpg"), canvas)


def main() -> None:
    root = tk.Tk()
    app = PDFCutterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
