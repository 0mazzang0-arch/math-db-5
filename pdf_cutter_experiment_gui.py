import ast
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk

# [환경설정] PaddleOCR 충돌 방지 및 최적화
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_new_ir"] = "0"
os.environ["PPSTRUCTURE_V3_ISOLATION"] = "1"

GLOBAL_ISOLATION_MODE = os.environ.get("PPSTRUCTURE_V3_ISOLATION", "0").strip().lower() in {"1", "true", "yes", "on"}

try:
    import cv2
except Exception:
    cv2 = None
try:
    import numpy as np
except Exception:
    np = None

if not GLOBAL_ISOLATION_MODE:
    try:
        from paddleocr import PPStructureV3
        PADDLE_OCR_AVAILABLE = True
    except Exception:
        PPStructureV3 = None
        PADDLE_OCR_AVAILABLE = False
else:
    PPStructureV3 = None
    PADDLE_OCR_AVAILABLE = True

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
    ISOLATION_MODE = GLOBAL_ISOLATION_MODE

    def __init__(self) -> None:
        self.enabled = bool((self.ISOLATION_MODE or PADDLE_OCR_AVAILABLE) and cv2 is not None and np is not None)
        self._engine: Optional[Any] = None
        self.paddleocr_version = "unknown"
        self.paddle_version = "unknown"
        self.init_error: Optional[str] = None
        if not self.ISOLATION_MODE:
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
        else:
            self.paddleocr_version = "isolation-subprocess"
            self.paddle_version = "isolation-subprocess"

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
        for attr in ("result", "res", "data"):
            cand = getattr(first, attr, None)
            if isinstance(cand, dict):
                return cand
        return None

    @staticmethod
    def _safe_keys(obj: Any) -> List[str]:
        if isinstance(obj, dict):
            return sorted([str(k) for k in obj.keys()])
        return []

    @staticmethod
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

    @staticmethod
    def _safe_object_dir_keys(obj: Any, limit: int = 80) -> List[str]:
        if obj is None or isinstance(obj, dict):
            return []
        try:
            names = [n for n in dir(obj) if not n.startswith("_")]
            return sorted(names)[:limit]
        except Exception:
            return []

    def _run_isolation_runner(self, image_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        runner_path = Path(__file__).resolve().parent / "v3_isolation_runner.py"
        if not runner_path.exists():
            return None, "stage=isolation_runner err=runner file not found"
        try:
            proc = subprocess.run(
                [sys.executable, str(runner_path), str(image_path), "--profile", os.environ.get("PPSTRUCTURE_V3_PROFILE", "fast")],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=900,
                check=False,
                env=self._isolation_env(),
            )
            lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
            if not lines:
                stderr_tail = (proc.stderr or "").strip()[-2000:]
                return None, f"stage=isolation_runner err=empty stdout stderr={stderr_tail}"
            payload = None
            for ln in reversed(lines):
                try:
                    payload = json.loads(ln)
                    break
                except Exception:
                    continue
            if payload is None:
                stderr_tail = (proc.stderr or "").strip()[-2000:]
                return None, f"stage=isolation_runner err=invalid json stdout stderr={stderr_tail}"
            if not isinstance(payload, dict):
                return None, "stage=isolation_runner err=invalid payload"
            if payload.get("ok") and isinstance(payload.get("pp_json"), dict):
                return {
                    "pp_json": payload["pp_json"],
                    "pp_obj": payload.get("pp_obj", {}),
                    "pp_meta": {"mode": "isolation_subprocess", "fallback": "v3_isolation_runner"},
                }, None
            stderr_tail = (proc.stderr or "").strip()[-2000:]
            return None, f"stage={payload.get('stage','isolation_runner')} err={payload.get('err','unknown')} stderr_tail={stderr_tail}"
        except subprocess.TimeoutExpired as e:
            stderr_tail = ((e.stderr or "") if isinstance(e.stderr, str) else str(e.stderr or "")).strip()[-2000:]
            return None, f"stage=isolation_runner_timeout err=timeout stderr_tail={stderr_tail}"
        except Exception as e:
            return None, f"stage=isolation_runner err={e}"

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
                encoding="utf-8",
                errors="replace",
                timeout=180,
                check=False,
                env=self._isolation_env(),
            )
            lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
            if not lines:
                return None, f"stage=fallback_subprocess err=empty stdout {PIN_GUIDE}"
            payload = None
            for ln in reversed(lines):
                try:
                    payload = json.loads(ln)
                    break
                except Exception:
                    continue
            if payload is None:
                return None, f"stage=fallback_subprocess err=invalid json payload {PIN_GUIDE}"
            if not isinstance(payload, dict):
                return None, f"stage=fallback_subprocess err=invalid payload {PIN_GUIDE}"
            if payload.get("ok") and isinstance(payload.get("pp_json"), dict):
                return {
                    "pp_json": payload["pp_json"],
                    "pp_obj": payload.get("pp_obj", {}),
                    "pp_meta": {"fallback": "subprocess", "mode": "runtime_unimplemented_fallback"},
                }, None
            return None, f"stage={payload.get('stage','fallback_subprocess')} err={payload.get('err','unknown')} {PIN_GUIDE}"
        except Exception as e:
            return None, f"stage=fallback_subprocess err={e} {PIN_GUIDE}"
        finally:
            try:
                runner_path.unlink(missing_ok=True)
            except Exception:
                pass

    def detect(self, image_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not self.enabled or cv2 is None or np is None:
            return None, "PaddleOCR 설치 필요: pip install paddlepaddle paddleocr"
        if self.ISOLATION_MODE:
            isolated, isolated_err = self._run_isolation_runner(image_path)
            if isolated is not None:
                return isolated, None
            return None, isolated_err
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
        pp_obj = self._extract_first_object_fields(first)
        if not isinstance(j, dict):
            meta = {
                "output_type": type(output).__name__,
                "first_type": type(first).__name__ if first is not None else "None",
                "first_keys": self._safe_keys(first),
                "first_dir_keys": self._safe_object_dir_keys(first),
            }
            return None, f"stage=parse_json invalid json payload meta={json.dumps(meta, ensure_ascii=False)}"

        pp_meta = {
            "output_type": type(output).__name__,
            "first_type": type(first).__name__ if first is not None else "None",
            "first_keys": self._safe_keys(first),
            "json_keys": self._safe_keys(j),
        }
        _ = pp_meta
        return {"pp_json": j, "pp_obj": pp_obj, "pp_meta": pp_meta}, None

# =========================================================
# 유틸 함수 모음
# =========================================================
def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def _area(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0: return 0.0
    ua = _area(a) + _area(b) - inter
    return inter / ua if ua > 0 else 0.0

def _contains(big, small, ratio=0.8):
    bx1, by1, bx2, by2 = big
    sx1, sy1, sx2, sy2 = small
    ix1, iy1 = max(bx1, sx1), max(by1, sy1)
    ix2, iy2 = min(bx2, sx2), min(by2, sy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    sa = _area(small)
    return sa > 0 and (inter / sa) >= ratio

def dedup_items(items, iou_th=0.70):
    items = sorted(items, key=lambda it: (it["bbox"][1], it["bbox"][0]))
    kept = []
    for it in items:
        b = it["bbox"]
        drop = False
        for kt in kept:
            kb = kt["bbox"]
            if _iou(b, kb) >= iou_th or _contains(kb, b, 0.8) or _contains(b, kb, 0.8):
                if _area(b) > _area(kb):
                    kt["bbox"] = b
                    kt["id"] = it.get("id", kt.get("id", 0))
                    kt["kind"] = "MC" if (kt.get("kind") == "MC" or it.get("kind") == "MC") else it.get("kind", "SA")
                else:
                    if it.get("kind") == "MC":
                        kt["kind"] = "MC"
                drop = True
                break
        if not drop: kept.append(it)
    return kept

def apply_overlap_cut(items, H):
    items = sorted(items, key=lambda it: it["bbox"][1])
    min_gap = max(10, int(0.01 * H))
    min_question_h = max(180, int(0.18 * H))

    for i in range(len(items) - 1):
        b = items[i]["bbox"]
        nb = items[i + 1]["bbox"]
        y1, y2 = b[1], b[3]
        next_y1 = nb[1]

        if (next_y1 - y1) >= min_question_h:
            cut_y2 = next_y1 - min_gap
            if cut_y2 < y2:
                items[i]["bbox"][3] = cut_y2
        else:
            pass
    return items

def final_garbage_filter(items, W, H):
    base_min_h = max(100, int(0.08 * H))
    min_area = int(0.01 * W * H)
    sa_min_h = max(180, int(0.15 * H))

    out = []
    for it in items:
        kind = it.get("kind", "Q")
        x1, y1, x2, y2 = it["bbox"]
        w_box = x2 - x1
        h_box = y2 - y1
        area = w_box * h_box

        if area < min_area: continue
        if kind == "SA":
            if h_box < sa_min_h: continue
        else:
            if h_box < base_min_h: continue
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
        self.warmup_thread: Optional[threading.Thread] = None
        self.log_queue: "queue.Queue[str]" = queue.Queue()

        self.output_root_var = tk.StringVar(value=str(DEFAULT_OUTPUT_ROOT))
        self.workers_var = tk.IntVar(value=DEFAULT_WORKERS)
        self.dpi_var = tk.IntVar(value=DEFAULT_DPI)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_label_var = tk.StringVar(value="진행률: 0/0")
        self.full_profile_var = tk.BooleanVar(value=False)

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
        ttk.Checkbutton(opt_frame, text="정밀모드(full)", variable=self.full_profile_var).pack(side=tk.LEFT, padx=10)

        ctl_frame = ttk.Frame(top)
        ctl_frame.pack(fill=tk.X, pady=6)

        self.start_btn = ttk.Button(ctl_frame, text="Start", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=4)
        self.stop_btn = ttk.Button(ctl_frame, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)
        self.warmup_btn = ttk.Button(ctl_frame, text="V3 모델 캐시 워밍업", command=self.warmup_v3_cache)
        self.warmup_btn.pack(side=tk.LEFT, padx=4)

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

    def _runner_profile(self) -> str:
        return "full" if bool(self.full_profile_var.get()) else "fast"

    def _isolation_env(self) -> Dict[str, str]:
        os.environ["PPSTRUCTURE_V3_ISOLATION"] = "1"
        env = os.environ.copy()
        env["PPSTRUCTURE_V3_ISOLATION"] = "1"
        return env

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
        os.environ["PPSTRUCTURE_V3_ISOLATION"] = "1"
        os.environ["PPSTRUCTURE_V3_PROFILE"] = self._runner_profile()
        self.log(f"ℹ️ Runner profile={self._runner_profile()} isolation=1")

        self.worker_thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self.worker_thread.start()

    def warmup_v3_cache(self) -> None:
        if self.warmup_thread and self.warmup_thread.is_alive():
            self.log("⚠️ V3 워밍업이 이미 실행 중입니다.")
            return
        self.warmup_btn.configure(state=tk.DISABLED)
        self.warmup_thread = threading.Thread(target=self._run_warmup_subprocess, daemon=True)
        self.warmup_thread.start()

    def _run_warmup_subprocess(self) -> None:
        os.environ["PPSTRUCTURE_V3_ISOLATION"] = "1"
        cmd = [
            sys.executable,
            "-c",
            "from paddleocr import PPStructureV3; print('warmup'); PPStructureV3(); print('ok')",
        ]
        self.log(f"ℹ️ [Warmup] start cmd={' '.join(cmd[:2])} ...")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=self._isolation_env(),
            )

            def _pump(stream, prefix: str) -> None:
                if stream is None:
                    return
                for line in stream:
                    self.log(f"{prefix} {line.rstrip()}")

            t_out = threading.Thread(target=_pump, args=(proc.stdout, "[Warmup][stdout]"), daemon=True)
            t_err = threading.Thread(target=_pump, args=(proc.stderr, "[Warmup][stderr]"), daemon=True)
            t_out.start()
            t_err.start()
            try:
                rc = proc.wait(timeout=900)
            except subprocess.TimeoutExpired:
                proc.kill()
                self.log("❌ [Warmup] timeout(900s)")
                rc = -1
            t_out.join(timeout=2)
            t_err.join(timeout=2)
            if rc == 0:
                self.log("✅ [Warmup] 완료")
            else:
                self.log(f"❌ [Warmup] 실패 rc={rc}")
        except Exception as e:
            self.log(f"❌ [Warmup] stage=run err={e}")
        finally:
            self.root.after(0, lambda: self.warmup_btn.configure(state=tk.NORMAL))

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
            if self.paddle_client.ISOLATION_MODE and workers != 1:
                self.log("⚠ Isolation mode: workers forced to 1")
                workers = 1
                self.root.after(0, lambda: self.workers_var.set(1))
            elif self.paddle_client.ISOLATION_MODE:
                self.log("⚠ Isolation mode: workers forced to 1")
                workers = 1
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

        if paddle_client.ISOLATION_MODE:
            return self._process_pdf_isolation_batch(
                tasks=tasks,
                pages_dir=pages_dir,
                crops_dir=crops_dir,
                errors_dir=errors_dir,
                pdf_stem=pdf_stem,
                dpi=dpi,
                total_pages_all=total_pages_all,
                processed_pages_before=processed_pages_before,
            )

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

    def _retry_single_page_with_region(self, page_png_path: Path) -> Optional[Dict[str, Any]]:
        runner_path = Path(__file__).resolve().parent / "v3_isolation_runner.py"
        if not runner_path.exists():
            return None
        cmd = [
            sys.executable,
            str(runner_path),
            str(page_png_path),
            "--profile",
            "fast",
            "--warmup",
            "0",
            "--force_region_detection",
            "1",
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=240,
                check=False,
                env=self._isolation_env(),
            )
            for line in reversed((proc.stdout or "").splitlines()):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    return payload
            return None
        except Exception:
            return None

    def _process_pdf_isolation_batch(
        self,
        tasks: List[PageTask],
        pages_dir: Path,
        crops_dir: Path,
        errors_dir: Path,
        pdf_stem: str,
        dpi: int,
        total_pages_all: int,
        processed_pages_before: int,
    ) -> Tuple[int, int, int]:
        runner_path = Path(__file__).resolve().parent / "v3_isolation_runner.py"
        if not runner_path.exists():
            self.log("❌ [Fail] P000 stage=isolation_runner err=runner file not found")
            return 0, len(tasks), 0

        task_by_name = {t.page_png_path.name: t for t in tasks}
        total_saved = 0
        total_errors = 0
        done_pages = 0

        profile = self._runner_profile()
        cmd = [sys.executable, str(runner_path), "--pages_dir", str(pages_dir), "--dpi", str(dpi), "--profile", profile]
        self.log(f"ℹ️ [IsolationBatch] start pages={len(tasks)} profile={profile}")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=self._isolation_env(),
        )

        stdout_queue: "queue.Queue[Optional[str]]" = queue.Queue()
        stderr_queue: "queue.Queue[Optional[str]]" = queue.Queue()

        def _pump_stdout() -> None:
            try:
                if proc.stdout is None:
                    return
                for line in proc.stdout:
                    stdout_queue.put(line)
            finally:
                stdout_queue.put(None)

        def _pump_stderr() -> None:
            try:
                if proc.stderr is None:
                    return
                for line in proc.stderr:
                    stderr_queue.put(line)
            finally:
                stderr_queue.put(None)

        t_out = threading.Thread(target=_pump_stdout, daemon=True)
        t_err = threading.Thread(target=_pump_stderr, daemon=True)
        t_out.start()
        t_err.start()

        stdout_done = False
        stderr_done = False
        deadline = time.monotonic() + 1800

        while not stdout_done and not self.stop_event.is_set():
            while True:
                try:
                    line = stderr_queue.get_nowait()
                except queue.Empty:
                    break
                if line is None:
                    stderr_done = True
                    break
                self.log(f"[IsolationRunner] {line.rstrip()}")

            try:
                line = stdout_queue.get(timeout=0.2)
            except queue.Empty:
                if time.monotonic() > deadline:
                    proc.kill()
                    self.log("❌ [Fail] P000 stage=isolation_runner_timeout err=no stdout within 1800s")
                    break
                continue

            if line is None:
                stdout_done = True
                break

            raw_line = line.strip()
            if not raw_line:
                continue

            try:
                payload = json.loads(raw_line)
            except Exception as e:
                self.log(f"⚠️ [IsolationBatch] invalid json line err={e}")
                continue

            page_file = str(payload.get("page_file", ""))
            if payload.get("ok") is False and (not page_file or page_file == "__BATCH__"):
                self.log(
                    f"❌ [IsolationBatch] runner fatal stage={payload.get('stage','unknown')} err={str(payload.get('err','unknown'))[:200]}"
                )
                total_errors += len(tasks)
                try:
                    proc.kill()
                except Exception:
                    pass
                stdout_done = True
                break

            task = task_by_name.get(page_file)
            if task is None:
                if payload.get("ok") is False:
                    self.log(f"⚠️ [IsolationBatch] page mapping miss page_file={page_file} stage={payload.get('stage','unknown')}")
                continue

            done_pages += 1
            processed = processed_pages_before + done_pages
            self._set_progress_safe(processed, total_pages_all)

            if not payload.get("ok"):
                total_errors += 1
                self._write_page_error(
                    errors_dir,
                    task.page_number,
                    task.page_png_path,
                    str(payload.get("err", "runner page failed")),
                    stage=str(payload.get("stage", "runner")),
                    extras={"runner_payload": payload},
                )
                self.log(
                    f"❌ [Fail] P{task.page_number:03d} stage={payload.get('stage','runner')} err={str(payload.get('err','unknown'))[:200]}"
                )
                continue

            img = Image.open(task.page_png_path)
            w, h = img.size
            try:
                data = {
                    "pp_json": payload.get("pp_json", {}),
                    "pp_obj": payload.get("pp_obj", {}),
                    "pp_meta": payload.get("pp_meta", {}),
                }
                anchors, objects = self._normalize_structure(data, w, h)
                if not anchors:
                    if profile == "fast":
                        retry_payload = self._retry_single_page_with_region(task.page_png_path)
                        if retry_payload is not None and retry_payload.get("ok"):
                            retry_data = {
                                "pp_json": retry_payload.get("pp_json", {}),
                                "pp_obj": retry_payload.get("pp_obj", {}),
                                "pp_meta": retry_payload.get("pp_meta", {}),
                            }
                            anchors, objects = self._normalize_structure(retry_data, w, h)
                            if anchors:
                                self.log(f"♻️ [Recovery] P{task.page_number:03d} anchors restored via region_detection=ON")
                    if not anchors:
                        trace = data.get("_trace", {}) if isinstance(data, dict) else {}
                        total_errors += 1
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
                                "trace_stats": trace.get("trace_stats", {}),
                                "pp_obj_keys": trace.get("pp_obj_keys", []),
                            },
                        )
                        self.log(f"❌ [Fail] P{task.page_number:03d} stage=parse_anchors err=anchors=0")
                        continue

                crops, dropped, errors = self._build_anchor_slice_regions(anchors, objects, w, h)
                if errors > 0 and not crops:
                    total_errors += 1
                    self._write_page_error(
                        errors_dir,
                        task.page_number,
                        task.page_png_path,
                        "anchor overlap conflict",
                        stage="slice",
                        extras={"anchors": len(anchors), "objects": len(objects), "errors": errors},
                    )
                    self.log(f"❌ [Fail] P{task.page_number:03d} stage=slice err=anchor overlap conflict")
                    self.log(
                        f"🧾 [AnchorSlice] P{task.page_number:03d} anchors={len(anchors)} saved=0 dropped={dropped} errors={errors}"
                    )
                    continue

                saved = 0
                for seq, (qid, x1, y1, x2, y2) in enumerate(crops, start=1):
                    crop_img = img.crop((x1, y1, x2, y2))
                    out_name = f"P{task.page_number:03d}_Q{seq:03d}_N{qid:04d}.png"
                    crop_img.save(crops_dir / out_name)
                    saved += 1

                if DEBUG_MODE:
                    self._save_debug_overlay(crops_dir, task.page_number, task.page_png_path, anchors, objects, crops)

                total_saved += saved
                self.log(
                    f"🧾 [AnchorSlice] P{task.page_number:03d} anchors={len(anchors)} "
                    f"saved={saved} dropped={dropped} errors={errors}"
                )
            except Exception as e:
                total_errors += 1
                self._write_page_error(errors_dir, task.page_number, task.page_png_path, str(e), stage="process_page")
                self.log(f"❌ [Fail] P{task.page_number:03d} stage=process_page err={str(e)[:200]}")
            finally:
                img.close()

        if self.stop_event.is_set():
            proc.kill()

        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

        while not stderr_done:
            try:
                line = stderr_queue.get_nowait()
            except queue.Empty:
                break
            if line is None:
                stderr_done = True
                break
            self.log(f"[IsolationRunner] {line.rstrip()}")

        if proc.returncode not in (0, None):
            self.log(f"⚠️ [IsolationBatch] runner exit code={proc.returncode}")

        if done_pages == 0 and total_saved == 0 and total_errors == 0:
            self.log("❌ [IsolationBatch] runner produced no usable page payloads")
            total_errors = len(tasks)

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
                extras = None
                if raw and "stderr_tail=" in raw:
                    stderr_tail = raw.split("stderr_tail=", 1)[1][:2000]
                    extras = {"stderr_tail": stderr_tail}
                self._write_page_error(
                    errors_dir,
                    task.page_number,
                    task.page_png_path,
                    raw or "parse failed",
                    stage=stage,
                    extras=extras,
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
                        "trace_stats": trace.get("trace_stats", {}),
                        "pp_obj_keys": trace.get("pp_obj_keys", []),
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
    def _extract_text_bbox_candidates(
        item: Any,
        depth: int = 0,
        max_depth: int = 4,
        visited: Optional[set] = None,
        cap: int = 500,
    ) -> List[Tuple[str, Optional[List[int]]]]:
        out: List[Tuple[str, Optional[List[int]]]] = []
        if depth > max_depth or item is None:
            return out
        if visited is None:
            visited = set()
        try:
            oid = id(item)
            if oid in visited:
                return out
            visited.add(oid)
        except Exception:
            pass

        if isinstance(item, dict):
            text = ""
            for k in ("text", "ocrText", "rec_text", "content", "transcription"):
                if isinstance(item.get(k), str):
                    text = item.get(k).strip()
                    if text:
                        break
            bbox = None
            for bk in ("text_region", "bbox", "xyxy", "points", "polygon", "poly"):
                if bk in item:
                    bbox = PDFCutterApp._poly_to_bbox(item.get(bk))
                    if bbox is not None:
                        break
            if text:
                out.append((text, bbox))
                if len(out) >= cap:
                    return out

            for val in item.values():
                if len(out) >= cap:
                    break
                if not isinstance(val, (dict, list, tuple)):
                    continue
                out.extend(PDFCutterApp._extract_text_bbox_candidates(val, depth + 1, max_depth, visited, cap - len(out)))
        elif isinstance(item, (list, tuple)):
            if len(item) >= 2:
                bbox = PDFCutterApp._poly_to_bbox(item[0])
                text = ""
                info = item[1]
                if isinstance(info, (list, tuple)) and info:
                    text = str(info[0]).strip()
                elif isinstance(info, str):
                    text = info.strip()
                if text:
                    out.append((text, bbox))
                    if len(out) >= cap:
                        return out
            for node in item:
                if len(out) >= cap:
                    break
                if not isinstance(node, (dict, list, tuple)):
                    continue
                out.extend(PDFCutterApp._extract_text_bbox_candidates(node, depth + 1, max_depth, visited, cap - len(out)))
        return out[:cap]

    @staticmethod
    def _obj_get(source: Any, key: str, default: Any = None) -> Any:
        if isinstance(source, dict):
            return source.get(key, default)
        try:
            value = getattr(source, key, default)
        except Exception:
            value = default
        return default if value is None else value

    @staticmethod
    def _ensure_list(source: Any) -> List[Any]:
        if isinstance(source, list):
            return source
        if isinstance(source, tuple):
            return list(source)
        return []

    def _collect_text_candidates_from_source(self, source: Any, trace_samples: List[Dict[str, Any]], cap: int = 40) -> List[Tuple[str, List[int]]]:
        results: List[Tuple[str, List[int]]] = []
        if source is None:
            return results
        for txt, bbox in self._extract_text_bbox_candidates(source):
            if bbox is None:
                continue
            if len(trace_samples) < cap and TRACE_MODE:
                trace_samples.append({"text": txt, "bbox": bbox})
            results.append((str(txt), bbox))
        return results

    def _normalize_structure(self, data: Dict[str, Any], page_w: int, page_h: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        pp_json = data.get("pp_json", {}) if isinstance(data, dict) else {}
        pp_obj = data.get("pp_obj", {}) if isinstance(data, dict) else {}
        if isinstance(pp_json, dict) and isinstance(pp_json.get("res"), str):
            try:
                parsed_res = ast.literal_eval(pp_json.get("res"))
                if isinstance(parsed_res, dict):
                    pp_json["res"] = parsed_res
                else:
                    if isinstance(data, dict):
                        data.setdefault("_parse_errors", []).append("stage=parse_json res string did not evaluate to dict")
            except Exception as e:
                if isinstance(data, dict):
                    data.setdefault("_parse_errors", []).append(f"stage=parse_json res string parse failed err={e}")

        layout = self._pick_list_by_keys(pp_json, ["prunedResult", "res", "result", "layout", "outputs", "regions"])
        if not layout:
            layout = self._pick_list_by_keys(data.get("layout", []) if isinstance(data, dict) else [], ["res", "layout"])

        if DEBUG_MODE and layout and not self._layout_keys_logged and isinstance(layout[0], dict):
            self.log(f"🔎 [PPStructure] layout[0].keys={sorted(layout[0].keys())}")
            self._layout_keys_logged = True

        strict_pat = re.compile(r"^\s*\d{4}\s*$")
        weak_pat = re.compile(r"^\s*0*\d{1,4}\s*$")
        leading_num_pat = re.compile(r"^\s*0*(\d{1,4})([.)]|\s|$)")
        strict_num_pat = re.compile(r"^\s*0*(\d{1,4})\s*([.)]|$)")
        fallback_num_pat = re.compile(r"\b0*(\d{1,4})\b")

        anchors: List[Dict[str, Any]] = []
        objects: List[Dict[str, Any]] = []
        candidate_cap = 500
        trace_cap = 30
        token_keys = ("ocr", "text", "rec", "word", "line")

        trace_stats: Dict[str, int] = {
            "candidate_texts": 0,
            "candidate_with_bbox": 0,
            "regex_pass": 0,
            "passA": 0,
            "passB": 0,
            "passC": 0,
            "anchors": 0,
        }
        trace_samples: List[Dict[str, Any]] = []

        def _collect_sources_from_token_keys(root: Any, depth: int = 0, max_depth: int = 4, visited: Optional[set] = None) -> List[Any]:
            out: List[Any] = []
            if root is None or depth > max_depth:
                return out
            if visited is None:
                visited = set()
            rid = id(root)
            if rid in visited:
                return out
            visited.add(rid)
            if isinstance(root, dict):
                for k, v in root.items():
                    key = str(k).lower()
                    if isinstance(v, (dict, list, tuple)) and any(tok in key for tok in token_keys):
                        out.append(v)
                    if isinstance(v, (dict, list, tuple)):
                        out.extend(_collect_sources_from_token_keys(v, depth + 1, max_depth, visited))
            elif isinstance(root, (list, tuple)):
                for it in root:
                    if isinstance(it, (dict, list, tuple)):
                        out.extend(_collect_sources_from_token_keys(it, depth + 1, max_depth, visited))
            return out

        def _parse_qid(txt: str, allow_fallback: bool = False) -> Optional[int]:
            m = leading_num_pat.match(txt) or strict_num_pat.match(txt)
            if m:
                qid0 = int(m.group(1))
                return qid0 if 0 < qid0 <= 9999 else None
            if not allow_fallback:
                return None
            m2 = fallback_num_pat.search(txt)
            if not m2:
                return None
            qid1 = int(m2.group(1))
            return qid1 if 0 < qid1 <= 9999 else None

        def _pass_geom(bbox: List[int], stage: str) -> bool:
            lx1, ly1, lx2, ly2 = bbox
            bw = max(0, lx2 - lx1)
            bh = max(0, ly2 - ly1)
            cx = (lx1 + lx2) / 2
            col = 0 if cx < (page_w * 0.5) else 1
            if stage == "A":
                if not (bh <= (0.12 * page_h) and bw <= (0.20 * page_w)):
                    return False
                if col == 0 and lx1 > (0.35 * page_w):
                    return False
                if col == 1 and lx1 < (0.50 * page_w):
                    return False
                return True
            if stage == "B":
                if not (bh <= (0.18 * page_h) and bw <= (0.30 * page_w)):
                    return False
                if col == 0 and lx1 > (0.35 * page_w):
                    return False
                if col == 1 and lx1 < (0.50 * page_w):
                    return False
                return True
            if not (bh <= (0.22 * page_h) and bw <= (0.35 * page_w)):
                return False
            if col == 0 and lx1 > (0.40 * page_w):
                return False
            if col == 1 and lx1 < (0.45 * page_w):
                return False
            return True

        ocr_candidate_sources: List[Any] = []
        seen_sources: set = set()

        def _push_source(source: Any) -> None:
            if source is None:
                return
            sid = id(source)
            if sid in seen_sources:
                return
            seen_sources.add(sid)
            ocr_candidate_sources.append(source)

        _push_source(self._obj_get(pp_obj, "overall_ocr_res"))
        for parsing_item in self._ensure_list(self._obj_get(pp_obj, "parsing_res_list")):
            _push_source(self._obj_get(parsing_item, "overall_ocr_res"))
        for obj_key in ("region_det_res", "layout_det_res"):
            _push_source(self._obj_get(pp_obj, obj_key))

        for src in _collect_sources_from_token_keys(pp_obj):
            _push_source(src)
        if isinstance(pp_json, dict):
            for src in _collect_sources_from_token_keys(pp_json):
                _push_source(src)
            if isinstance(pp_json.get("res"), dict):
                for src in _collect_sources_from_token_keys(pp_json.get("res")):
                    _push_source(src)

        records: List[Dict[str, Any]] = []
        rec_seen: set = set()

        for source in ocr_candidate_sources:
            for txt, line_bbox in self._extract_text_bbox_candidates(source, max_depth=4, cap=candidate_cap):
                trace_stats["candidate_texts"] += 1
                if line_bbox is None:
                    continue
                trace_stats["candidate_with_bbox"] += 1
                rec_key = (str(txt), tuple(line_bbox))
                if rec_key in rec_seen:
                    continue
                rec_seen.add(rec_key)
                lx1, ly1, lx2, ly2 = line_bbox
                col = 0 if ((lx1 + lx2) / 2) < (page_w * 0.5) else 1
                records.append({"text": str(txt), "bbox": line_bbox, "col": col, "strict": bool(strict_pat.match(str(txt)) or weak_pat.match(str(txt)))})
                if len(records) >= candidate_cap:
                    break
            if len(records) >= candidate_cap:
                break

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
            for line in [block] + lines:
                for txt, line_bbox in self._extract_text_bbox_candidates(line, max_depth=4, cap=120):
                    trace_stats["candidate_texts"] += 1
                    if line_bbox is None:
                        continue
                    trace_stats["candidate_with_bbox"] += 1
                    rec_key = (str(txt), tuple(line_bbox))
                    if rec_key in rec_seen:
                        continue
                    rec_seen.add(rec_key)
                    lx1, _, lx2, _ = line_bbox
                    col = 0 if ((lx1 + lx2) / 2) < (page_w * 0.5) else 1
                    records.append({"text": str(txt), "bbox": line_bbox, "col": col, "strict": bool(strict_pat.match(str(txt)) or weak_pat.match(str(txt)))})
                    if len(records) >= candidate_cap:
                        break
                if len(records) >= candidate_cap:
                    break
            if len(records) >= candidate_cap:
                break

        regex_records: List[Dict[str, Any]] = []
        strict_regex_count = 0
        for rec in records:
            qid = _parse_qid(rec["text"], allow_fallback=False)
            if qid is None:
                continue
            trace_stats["regex_pass"] += 1
            cand = {"id": qid, "bbox": rec["bbox"], "col": rec["col"], "strict": rec["strict"], "text": rec["text"]}
            regex_records.append(cand)
            if cand["strict"]:
                strict_regex_count += 1

        if strict_regex_count == 0:
            for rec in records:
                if any((r["bbox"] == rec["bbox"] and r["text"] == rec["text"]) for r in regex_records):
                    continue
                qid = _parse_qid(rec["text"], allow_fallback=True)
                if qid is None:
                    continue
                trace_stats["regex_pass"] += 1
                regex_records.append({"id": qid, "bbox": rec["bbox"], "col": rec["col"], "strict": False, "text": rec["text"]})

        def _filter_by_stage(stage: str, source: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            filtered = [c for c in source if _pass_geom(c["bbox"], stage)]
            trace_stats[f"pass{stage}"] = len(filtered)
            if stage == "A" and TRACE_MODE:
                for c in filtered[:trace_cap]:
                    trace_samples.append({"text": c["text"], "bbox": c["bbox"], "id": c["id"]})
            strict_only = [c for c in filtered if c["strict"]]
            return strict_only if strict_only else filtered

        anchors = _filter_by_stage("A", regex_records)
        if not anchors:
            anchors = _filter_by_stage("B", regex_records)
        if not anchors:
            anchors = _filter_by_stage("C", regex_records)

        dedup_anchors: List[Dict[str, Any]] = []
        seen_anchor = set()
        for a in anchors:
            ak = (a["id"], tuple(a["bbox"]), a["col"])
            if ak in seen_anchor:
                continue
            seen_anchor.add(ak)
            dedup_anchors.append({"id": a["id"], "bbox": a["bbox"], "col": a["col"]})
        anchors = dedup_anchors
        trace_stats["anchors"] = len(anchors)

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
                "parse_errors": data.get("_parse_errors", []),
                "trace_stats": trace_stats,
                "pp_obj_keys": sorted(pp_obj.keys())[:50] if isinstance(pp_obj, dict) and not anchors else [],
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
