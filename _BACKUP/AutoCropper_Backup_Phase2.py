import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import os
import sys
import threading
import re
import json
import io
import time

# --- 라이브러리 임포트 및 예외처리 ---
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageTk
import cv2
import numpy as np

# Google GenAI (설치 안되어 있어도 실행되도록 안전장치)
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# =========================================================
# [전역 설정] Tesseract 경로 (사용자 환경 고정)
# =========================================================
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# [설정] 기본 작업 경로 (Config 대체)
DEFAULT_WORK_DIR = os.path.join(os.path.expanduser("~"), "Downloads", "AutoCropper_Work")
if not os.path.exists(DEFAULT_WORK_DIR):
    try: os.makedirs(DEFAULT_WORK_DIR)
    except: pass

# [설정] API 키 (필요시 여기에 입력)
GOOGLE_API_KEY_HARDCODED = "AIzaSyBO9106GmrTWQYTrwzeDbM_d-F1n9gMlGs" 
MODEL_NAME = "gemini-3-flash-preview"
# =========================================================

class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AutoCropper Ultimate V32 (OCR + Cutter Integration)")
        self.geometry("1600x1000") # 컷팅기 사이즈에 맞춤
        
        # 스타일 설정
        style = ttk.Style()
        style.theme_use('clam')
        
        # 탭 컨트롤 생성
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        
        # 탭 1: OCR 변환기 (DirectOCRGUI 로직 이식)
        self.tab_ocr = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_ocr, text="   1. PDF 텍스트 입히기 (OCR)   ")
        self.app_ocr = DirectOCRTab(self.tab_ocr)
        
        # 탭 2: 쎈 컷팅기 (SsenCutterV31 로직 이식)
        self.tab_cutter = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_cutter, text="   2. 문제 자르기 (Cropper)   ")
        self.app_cutter = SsenCutterTab(self.tab_cutter)

# =================================================================================
# [탭 1] Direct OCR 변환기 (기존 DirectOCRGUI 코드 100% 보존)
# =================================================================================
class DirectOCRTab:
    def __init__(self, parent):
        self.frame = parent
        
        # 기본 저장 경로 (다운로드/OCR_Direct_Result)
        self.default_save_dir = os.path.join(os.path.expanduser("~"), "Downloads", "OCR_Direct_Result")
        
        self.is_running = False
        self.setup_ui()

    def setup_ui(self):
        # 1. 입력 선택
        lf_input = tk.LabelFrame(self.frame, text="1. 원본 파일 선택 (PDF)", padx=10, pady=10, font=("bold", 10))
        lf_input.pack(fill="x", padx=10, pady=5)

        self.input_mode = tk.StringVar(value="file")
        
        f_radio = tk.Frame(lf_input)
        f_radio.pack(anchor="w", pady=2)
        tk.Radiobutton(f_radio, text="파일 하나만 (File)", variable=self.input_mode, value="file", command=self.toggle_input).pack(side="left", padx=5)
        tk.Radiobutton(f_radio, text="폴더 전체 (Folder)", variable=self.input_mode, value="folder", command=self.toggle_input).pack(side="left", padx=5)

        f_path = tk.Frame(lf_input)
        f_path.pack(fill="x", pady=5)
        self.ent_input = tk.Entry(f_path, width=50)
        self.ent_input.pack(side="left", fill="x", expand=True)
        self.btn_input = tk.Button(f_path, text="📂 선택", command=self.select_input)
        self.btn_input.pack(side="left", padx=5)

        # 2. 출력 선택
        lf_output = tk.LabelFrame(self.frame, text="2. 저장 폴더 (결과물)", padx=10, pady=10, font=("bold", 10))
        lf_output.pack(fill="x", padx=10, pady=5)
        
        f_out = tk.Frame(lf_output)
        f_out.pack(fill="x")
        self.ent_output = tk.Entry(f_out, width=50)
        self.ent_output.insert(0, self.default_save_dir)
        self.ent_output.pack(side="left", fill="x", expand=True)
        tk.Button(f_out, text="💾 변경", command=self.select_output).pack(side="left", padx=5)

        # 3. 실행 버튼 & 진행바
        self.btn_run = tk.Button(self.frame, text="🚀 OCR 변환 시작 (무조건 성공)", command=self.start_thread, bg="#2196F3", fg="white", font=("bold", 12), height=2)
        self.btn_run.pack(fill="x", padx=10, pady=15)
        
        self.lbl_progress = tk.Label(self.frame, text="대기 중...")
        self.lbl_progress.pack(anchor="w", padx=10)
        
        self.progress = ttk.Progressbar(self.frame, orient="horizontal", length=100, mode="determinate")
        self.progress.pack(fill="x", padx=10, pady=(0, 10))

        # 4. 로그창
        lf_log = tk.LabelFrame(self.frame, text="진행 로그", padx=10, pady=5)
        lf_log.pack(fill="both", expand=True, padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(lf_log, height=10)
        self.log_text.pack(fill="both", expand=True)

    def toggle_input(self):
        self.btn_input.config(text="📂 폴더 선택" if self.input_mode.get() == "folder" else "📄 파일 선택")

    def select_input(self):
        if self.input_mode.get() == "folder":
            p = filedialog.askdirectory()
        else:
            p = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if p:
            self.ent_input.delete(0, tk.END)
            self.ent_input.insert(0, p)

    def select_output(self):
        p = filedialog.askdirectory()
        if p:
            self.ent_output.delete(0, tk.END)
            self.ent_output.insert(0, p)

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def start_thread(self):
        if self.is_running: return
        in_path = self.ent_input.get()
        if not in_path:
            messagebox.showwarning("경고", "파일이나 폴더를 선택해주세요.")
            return
        
        self.is_running = True
        self.btn_run.config(state="disabled", text="작업 중... (응답 없음 아님)")
        threading.Thread(target=self.process, args=(in_path, self.ent_output.get())).start()

    def process(self, in_path, out_path):
        try:
            if not os.path.exists(out_path):
                os.makedirs(out_path)
                self.log(f"📁 결과 폴더 생성: {out_path}")

            # 작업 리스트
            tasks = []
            if self.input_mode.get() == "folder":
                for f in os.listdir(in_path):
                    if f.lower().endswith(".pdf"):
                        tasks.append(os.path.join(in_path, f))
            else:
                tasks.append(in_path)

            if not tasks:
                self.log("❌ 처리할 PDF 파일이 없습니다.")
                return

            total_files = len(tasks)
            self.log(f"🔎 총 {total_files}개 파일 작업을 시작합니다.\n" + "="*40)

            for i, src_file in enumerate(tasks):
                fname = os.path.basename(src_file)
                dst_file = os.path.join(out_path, f"OCR_{fname}")
                
                self.log(f"[{i+1}/{total_files}] 시작: {fname}")
                self.lbl_progress.config(text=f"파일 처리 중 ({i+1}/{total_files}): {fname}")
                
                try:
                    # ==========================================
                    # [핵심 로직] PyMuPDF + Pytesseract 직접 연결
                    # ==========================================
                    doc = fitz.open(src_file)
                    out_doc = fitz.open() # 결과물 PDF
                    
                    total_pages = len(doc)
                    
                    for p_idx, page in enumerate(doc):
                        # 진행률 표시 (파일 단위가 아니라 페이지 단위로 쪼개서 보여줌)
                        self.progress['value'] = ((i * total_pages + p_idx) / (total_files * total_pages)) * 100
                        self.frame.update_idletasks() # UI 갱신 (root 대신 frame)
                        
                        try:
                            # 1. 고화질 이미지 변환 (300 DPI)
                            pix = page.get_pixmap(dpi=300)
                            img_data = pix.tobytes("png")
                            pil_img = Image.open(io.BytesIO(img_data))
                            
                            # 2. OCR 수행 -> PDF 데이터 획득
                            pdf_bytes = pytesseract.image_to_pdf_or_hocr(pil_img, extension='pdf', lang='kor+eng')
                            
                            # 3. 결과 합치기
                            img_pdf = fitz.open("pdf", pdf_bytes)
                            out_doc.insert_pdf(img_pdf)
                            
                        except Exception as e:
                            self.log(f"   ⚠️ {p_idx+1}페이지 OCR 실패(이미지로 대체): {e}")
                            # 실패 시 원본 이미지만이라도 넣어서 페이지 누락 방지
                            out_doc.insert_pdf(fitz.open("pdf", img_data))

                    # 저장
                    out_doc.save(dst_file)
                    self.log(f"   ✅ 완료 ({total_pages}페이지)")
                    
                except Exception as e:
                    self.log(f"   ❌ 파일 처리 실패: {e}")

            self.progress['value'] = 100
            self.lbl_progress.config(text="모든 작업 완료!")
            self.log("="*40 + "\n🎉 모든 작업이 끝났습니다!")
            
            # 폴더 열기
            try: os.startfile(out_path)
            except: pass

        except Exception as e:
            self.log(f"⛔ 치명적 오류: {e}")
        finally:
            self.is_running = False
            self.btn_run.config(state="normal", text="🚀 OCR 변환 시작 (무조건 성공)")

# =================================================================================
# [탭 2] 쎈 컷팅기 (SsenCutterV31FinalIntegrity 코드 100% 보존)
# =================================================================================
class SsenCutterTab:
    def __init__(self, parent):
        self.root = parent # 탭 프레임을 root로 취급
        
        # API Key 처리 (Config 대체)
        self.api_key = GOOGLE_API_KEY_HARDCODED
        
        self.client = None
        self.init_api()
        
        self.doc = None
        self.current_page = 0
        self.zoom_save = 3.0
        self.zoom_display = 1.0 # 초기화
        
        # [CRITICAL] 저장 경로 강제 고정 (작업대) - Config 대체
        self.save_dir = DEFAULT_WORK_DIR
        
        # 폴더가 없으면 만듭니다.
        if not os.path.exists(self.save_dir):
            try: os.makedirs(self.save_dir)
            except: pass
            
        print(f"🔒 [System] 저장 경로가 '{self.save_dir}'로 고정되었습니다.")
        
        self.batch_regions = []
        
        self.unit_map = {1: "Default"}
        self.ref_files_map = {} 
        
        # 상태 변수
        self.mode_var = tk.StringVar(value="Q") 
        self.use_ai_filter = tk.BooleanVar(value=True) 
        self.use_unit_mode = tk.BooleanVar(value=False) 
        
        self.is_running = False
        self.setup_ui()

    def init_api(self):
        if not HAS_GENAI:
            print("⚠️ google-genai 라이브러리가 없습니다. AI 기능을 사용할 수 없습니다.")
            return
        if not self.api_key: 
            return
        try:
            self.client = genai.Client(api_key=self.api_key)
            print(f"✅ AI 연결 성공")
        except: pass

    def setup_ui(self):
        # [왼쪽] 패널
        left = tk.Frame(self.root, width=450, bg="#f0f0f0")
        left.pack(side="left", fill="y"); left.pack_propagate(False)

        # 1. 모드 선택
        lf_mode = tk.LabelFrame(left, text="1. 작업 모드 선택", bg="#e3f2fd", padx=5, pady=5, font=("bold", 10))
        lf_mode.pack(fill="x", padx=5, pady=5)
        
        tk.Radiobutton(lf_mode, text="문제(Q) 자르기", variable=self.mode_var, value="Q", 
                       command=self.update_ui_state, bg="#e3f2fd", font=("bold", 11)).pack(anchor="w")
        tk.Radiobutton(lf_mode, text="해설(A) 자르기 (참조 모드)", variable=self.mode_var, value="A", 
                       command=self.update_ui_state, bg="#e3f2fd", font=("bold", 11)).pack(anchor="w")

        # 2. 파일 관리
        lf_file = tk.LabelFrame(left, text="2. 파일 및 폴더", bg="#f0f0f0", padx=5, pady=5)
        lf_file.pack(fill="x", padx=5, pady=5)
        
        self.btn_open = tk.Button(lf_file, text="📂 작업할 PDF 열기 (OCR 필수)", command=self.open_pdf, bg="#4a90e2", fg="white")
        self.btn_open.pack(fill="x", pady=2)
        self.lbl_file = tk.Label(lf_file, text="파일 없음", bg="#f0f0f0"); self.lbl_file.pack()
        
        # 저장 폴더 변경 (현재 고정됨, 버튼은 존재하되 비활성 유지)
        tk.Button(lf_file, text="저장 폴더 변경 (현재 고정됨)", state="disabled").pack(fill="x", pady=2)
        
        # [해설 모드 전용] 참조 폴더 UI
        self.frm_ref = tk.Frame(lf_file, bg="#f0f0f0")
        self.frm_ref.pack(fill="x", pady=5)
        tk.Label(self.frm_ref, text="참조할 문제(Q) 폴더:", bg="#f0f0f0", fg="blue").pack(anchor="w")
        tk.Button(self.frm_ref, text="📂 참조 폴더 선택", command=self.select_ref_dir).pack(fill="x")
        self.lbl_ref = tk.Label(self.frm_ref, text="선택 안됨", bg="#f0f0f0", fg="gray", wraplength=400)
        self.lbl_ref.pack()

        # 3. 책 정보 & 단원 (문제 모드용)
        self.lf_book = tk.LabelFrame(left, text="3. 책 정보 & 단원 관리", bg="#e8f5e9", padx=5, pady=5)
        self.lf_book.pack(fill="x", padx=5, pady=5)
        
        tk.Label(self.lf_book, text="책 이름:", bg="#e8f5e9").pack(side="left")
        self.ent_bookname = tk.Entry(self.lf_book, width=15); self.ent_bookname.pack(side="left", padx=5)
        self.ent_bookname.insert(0, "쎈수학")
        
        self.chk_unit = tk.Checkbutton(self.lf_book, text="단원 구분 사용", variable=self.use_unit_mode, command=self.toggle_unit_ui, bg="#e8f5e9", font=("bold", 9))
        self.chk_unit.pack(side="left")

        # 단원 리스트 컨테이너
        self.frm_unit_list = tk.Frame(self.lf_book, bg="#e8f5e9") 
        self.frm_unit_list.pack(fill="x", pady=5)
        
        f_u_add = tk.Frame(self.frm_unit_list, bg="#e8f5e9")
        f_u_add.pack(fill="x")
        self.ent_u_p = tk.Entry(f_u_add, width=4); self.ent_u_p.pack(side="left")
        self.ent_u_n = tk.Entry(f_u_add); self.ent_u_n.pack(side="left", fill="x", expand=True)
        tk.Button(f_u_add, text="➕", command=self.add_unit, width=3).pack(side="right")
        
        self.tree = ttk.Treeview(self.frm_unit_list, columns=("p","n"), show="headings", height=4)
        self.tree.heading("p", text="P"); self.tree.column("p", width=40, anchor="center")
        self.tree.heading("n", text="단원명"); self.tree.column("n", width=120)
        self.tree.pack(fill="x")
        
        # [V29 누락 복구] 삭제 버튼
        tk.Button(self.frm_unit_list, text="선택 삭제", command=self.delete_unit).pack(anchor="e")

        # 4. 해설 모드 전용 필터
        self.lf_ans_filter = tk.LabelFrame(left, text="3. 단원 필터 (해설 모드)", bg="#fff3e0", padx=5, pady=5)
        tk.Label(self.lf_ans_filter, text="특정 단원 문제만 자르기:", bg="#fff3e0", fg="red").pack(anchor="w")
        self.combo_filter = ttk.Combobox(self.lf_ans_filter, state="readonly")
        self.combo_filter.pack(fill="x", pady=2)
        tk.Button(self.lf_ans_filter, text="🔄 목록 갱신", command=self.refresh_filter_list).pack(anchor="e")

        # 5. 공통 설정
        lf_common = tk.LabelFrame(left, text="4. 컷팅 & 패턴", bg="#f0f0f0", padx=5, pady=5)
        lf_common.pack(fill="x", padx=5, pady=5)
        
        tk.Label(lf_common, text="번호 패턴 (직접 입력 가능):", bg="#f0f0f0").pack(anchor="w")
        self.combo_pat = ttk.Combobox(lf_common)
        self.combo_pat['values'] = ("0001 (4자리)", "1. (숫자+점)", "1 (숫자)", "(1)", "[1]", "Q1")
        self.combo_pat.set("0001"); self.combo_pat.pack(fill="x", pady=2)
        
        self.chk_ai = tk.Checkbutton(lf_common, text="AI 문맥필터 사용", variable=self.use_ai_filter, bg="#f0f0f0", fg="blue")
        # API 키 없으면 비활성 처리
        if not self.client:
            self.chk_ai.config(state="disabled", text="AI 문맥필터 (API키 필요)")
        self.chk_ai.pack(anchor="w", pady=2)

        f_pad = tk.Frame(lf_common, bg="#f0f0f0"); f_pad.pack(fill="x", pady=2)
        tk.Label(f_pad, text="Top여백:", bg="#f0f0f0").pack(side="left")
        self.spin_top = tk.Spinbox(f_pad, from_=0, to=100, width=4); self.spin_top.insert(0,"10"); self.spin_top.pack(side="left")
        tk.Label(f_pad, text="Btm여백:", bg="#f0f0f0").pack(side="left")
        self.spin_btm = tk.Spinbox(f_pad, from_=0, to=200, width=4); self.spin_btm.insert(0,"30"); self.spin_btm.pack(side="left")

        # 6. 실행 (V29 기능 유지)
        lf_run = tk.LabelFrame(left, text="5. 실행", bg="#f0f0f0", padx=5, pady=5)
        lf_run.pack(fill="x", padx=5, pady=5)
        tk.Button(lf_run, text="영역 초기화", command=self.clear_regions).pack(fill="x")
        f_p = tk.Frame(lf_run, bg="#f0f0f0"); f_p.pack(fill="x", pady=2)
        self.ent_s = tk.Entry(f_p, width=5); self.ent_s.pack(side="left")
        tk.Label(f_p, text="~", bg="#f0f0f0").pack(side="left")
        self.ent_e = tk.Entry(f_p, width=5); self.ent_e.pack(side="left")
        
        tk.Button(lf_run, text="🔍 진단(Scan)", command=lambda: self.run(mode="scan"), bg="#2196F3", fg="white").pack(fill="x", pady=2)
        tk.Button(lf_run, text="🚀 실행(Cut)", command=lambda: self.run(mode="cut"), bg="#d32f2f", fg="white", font=("bold", 12)).pack(fill="x")

        # [오른쪽]
        right = tk.PanedWindow(self.root, orient="vertical")
        right.pack(side="right", fill="both", expand=True)
        self.canvas = tk.Canvas(right, bg="#555", cursor="cross"); right.add(self.canvas, minsize=600)
        self.log_text = scrolledtext.ScrolledText(right, height=10); right.add(self.log_text, minsize=200)

        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)
        # 탭 환경에서는 root 바인딩 충돌 주의 -> 포커스에 따라 처리되거나 전역 바인딩 유지
        self.root.bind("<Left>", lambda e: self.move_page(-1))
        self.root.bind("<Right>", lambda e: self.move_page(1))

        self.update_ui_state()

    # --- [V29 복구] 스마트 패턴 변환기 (범용성 유지) ---
    def get_smart_pattern(self):
        user_input = self.combo_pat.get().strip()
        if " (" in user_input: user_input = user_input.split(" (")[0].strip()
        if "^" in user_input or "\\" in user_input: return user_input # 이미 정규식

        # 숫자만 있는 경우
        if user_input.isdigit():
            if len(user_input) > 1 and user_input.startswith("0"): 
                return r"^\d{" + str(len(user_input)) + r"}$"
            else: return r"^\d+$"
        
        # 특수문자 처리
        if user_input.endswith("."): return r"^\d+\.$"
        if user_input.startswith("(") and user_input.endswith(")"): return r"^\(\d+\)$"
        if user_input.startswith("[") and user_input.endswith("]"): return r"^\[\d+\]$"
        
        # [V29 기능 복구] 문자+숫자 조합 (예: 문1, 예제1, Q1 등 모든 접두어 지원)
        match = re.match(r"([^\d]+)(\d+)", user_input)
        if match: 
            prefix = re.escape(match.group(1))
            return f"^{prefix}\\d+$"
        
        return re.escape(user_input)

    # --- UI 로직 ---
    def update_ui_state(self):
        mode = self.mode_var.get()
        if mode == "Q":
            self.lf_book.pack(fill="x", padx=5, pady=5, after=self.frm_ref.master)
            self.lf_ans_filter.pack_forget()
            self.lbl_ref.config(state="disabled")
        else:
            self.lf_book.pack_forget()
            self.lf_ans_filter.pack(fill="x", padx=5, pady=5, after=self.frm_ref.master)
            self.lbl_ref.config(state="normal")

    def toggle_unit_ui(self):
        if self.use_unit_mode.get(): self.frm_unit_list.pack(fill="x", pady=5)
        else: self.frm_unit_list.pack_forget()

    # --- [V29 복구] 단원 관리 (삭제 기능 포함) ---
    def add_unit(self):
        try:
            self.unit_map[int(self.ent_u_p.get())] = self.ent_u_n.get()
            self.refresh_tree()
            self.ent_u_p.delete(0,"end"); self.ent_u_n.delete(0,"end")
        except: pass

    def delete_unit(self):
        sel = self.tree.selection()
        if sel:
            p = self.tree.item(sel[0])['values'][0]
            if p == 1: return
            del self.unit_map[p]
            self.refresh_tree()

    def refresh_tree(self):
        for i in self.tree.get_children(): self.tree.delete(i)
        for p, n in sorted(self.unit_map.items()): self.tree.insert("", "end", values=(p, n))

    def get_unit_name(self, p):
        u = ""
        for sp in sorted(self.unit_map.keys()):
            if sp <= p: u = self.unit_map[sp]
            else: break
        return u

    # --- 참조 폴더 로직 ---
    def select_ref_dir(self):
        p = filedialog.askdirectory()
        if p:
            self.ref_dir = p
            self.lbl_ref.config(text=p)
            self.refresh_filter_list()

    def refresh_filter_list(self):
        if not hasattr(self, 'ref_dir'): return
        files = os.listdir(self.ref_dir)
        units = set()
        self.ref_files_map = {} 

        for f in files:
            if not f.endswith("_Q.png"): continue
            parts = f.replace("_Q.png", "").split("_")
            # V26+ 파일명 구조: {Book}_{Unit}_{Num}_Q.png
            if len(parts) >= 3:
                u_part = parts[1] # 단원명
                units.add(u_part)
                n_part = parts[-1] # 번호
                if n_part.isdigit():
                    self.ref_files_map[(u_part, n_part)] = f

        self.combo_filter['values'] = ["전체(All)"] + sorted(list(units))
        self.combo_filter.current(0)
        self.log(f"ℹ️ 참조 로드: 문제 {len(self.ref_files_map)}개 / 단원 {len(units)}개")

    # --- 실행 ---
    def run(self, mode):
        if not self.batch_regions: 
            messagebox.showwarning("!", "파란 박스 필요"); return
        self.is_running = True
        threading.Thread(target=self.process, args=(mode,)).start()

    def process(self, mode):
        try:
            s = int(self.ent_s.get()) - 1
            e = int(self.ent_e.get())
            pad_t = int(self.spin_top.get())
            pad_b = int(self.spin_btm.get())
            
            # [V29] 스마트 패턴 적용
            pat = self.get_smart_pattern()
            
            book = self.ent_bookname.get().strip()
            task_mode = self.mode_var.get()
            filter_unit = self.combo_filter.get()
        except: return

        cnt = 0
        if mode == "scan": self.log(f"🔎 진단 패턴: {pat}")

        for p_idx in range(s, e):
            if not self.is_running: break
            
            if task_mode == "Q" and self.use_unit_mode.get():
                curr_unit = self.get_unit_name(p_idx+1)
                log_head = f"P.{p_idx+1} [{curr_unit}]"
            else:
                curr_unit = ""
                log_head = f"P.{p_idx+1}"
            
            self.log(f"\n📄 {log_head} ({mode})...")

            try:
                page = self.doc.load_page(p_idx)
                if mode == "cut":
                    mat = fitz.Matrix(3.0, 3.0)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    if pix.n==3: img_cv = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                    else: img_cv = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)

                for r_i, (rx1, ry1, rx2, ry2) in enumerate(self.batch_regions):
                    rect = fitz.Rect(rx1, ry1, rx2, ry2)
                    words = page.get_text("words", clip=rect, sort=True)
                    
                    cands = []
                    for i, w in enumerate(words):
                        txt = w[4].strip()
                        if re.match(pat, txt):
                            ctx = " ".join([words[j][4] for j in range(i+1, min(i+10, len(words)))])
                            cands.append({'t': txt, 'y': w[1], 'c': ctx})
                            # [V29 복구] 진단 모드 상세 로그
                            if mode == "scan": self.log(f"   후보: {txt} | {ctx[:20]}...")

                    if not cands: continue
                    if mode == "scan": continue

                    # AI 필터
                    valid = []
                    if self.use_ai_filter.get() and self.client and cands:
                        prompt = "Identify REAL problem numbers.\n"
                        for k, c in enumerate(cands): prompt += f"[{k}] {c['t']} | {c['c']}\n"
                        prompt += "JSON [true, false...]"
                        try:
                            resp = self.client.models.generate_content(
                                model=MODEL_NAME, contents=prompt,
                                config=types.GenerateContentConfig(response_mime_type="application/json"))
                            bools = json.loads(resp.text.replace("```json","").replace("```",""))
                            for k, b in enumerate(bools): 
                                if k < len(cands) and b: valid.append(cands[k])
                        except: valid = cands
                    else: valid = cands

                    # 해설 모드 필터링
                    final_valid = []
                    if task_mode == "A":
                        for item in valid:
                            n_clean = re.sub(r"[^\d]", "", item['t'])
                            found = False
                            target_u = ""
                            
                            if filter_unit != "전체(All)":
                                if (filter_unit, n_clean) in self.ref_files_map:
                                    found = True; target_u = filter_unit
                            else:
                                for u_k, n_k in self.ref_files_map.keys():
                                    if n_k == n_clean: found = True; target_u = u_k; break
                            
                            if found:
                                item['target_unit'] = target_u
                                final_valid.append(item)
                    else:
                        final_valid = valid

                    # 컷팅
                    if not final_valid: continue
                    final_valid.sort(key=lambda x: x['y'])
                    scale = 3.0
                    rt, rb = int(ry1*scale), int(ry2*scale)
                    
                    for k, item in enumerate(final_valid):
                        cy = item['y']
                        y1 = max(rt, int(cy*scale - pad_t*scale))
                        if k < len(final_valid)-1:
                            y2 = int(final_valid[k+1]['y']*scale - pad_b*scale)
                        else: y2 = min(rb, int(ry2*scale)) # 끝점 보정
                        
                        if y2 - y1 < 20: continue
                        
                        crop = img_cv[y1:y2, int(rx1*scale):int(rx2*scale)]
                        num = re.sub(r"[^\d]", "", item['t'])
                        
                        if task_mode == "Q":
                            if self.use_unit_mode.get(): fname = f"{book}_{curr_unit}_{num}_Q.png"
                            else: fname = f"{book}_{num}_Q.png"
                        else:
                            u_name = item.get('target_unit', '')
                            if u_name: fname = f"{book}_{u_name}_{num}_A.png"
                            else: fname = f"{book}_{num}_A.png"

                        path = os.path.join(self.save_dir, fname)
                        dup = 1
                        bn = os.path.splitext(fname)[0]
                        while os.path.exists(path):
                            path = os.path.join(self.save_dir, f"{bn}_{dup}.png"); dup+=1
                            
                        ret, buf = cv2.imencode(".png", crop)
                        if ret: 
                            with open(path, "wb") as f: f.write(buf)
                        cnt += 1
                    
                    self.log(f"   👉 [구역{r_i+1}] {len(final_valid)}개 저장")

            except Exception as e: self.log(f"❌ Err: {e}")

        self.is_running = False
        if mode == "cut":
            messagebox.showinfo("완료", f"{cnt}개 완료!")
            os.startfile(self.save_dir)

    def log(self, s): self.log_text.insert(tk.END, s+"\n"); self.log_text.see(tk.END)
    def open_pdf(self):
        f = filedialog.askopenfilename()
        if not f: return
        self.doc = fitz.open(f); self.current_page = 0
        base = os.path.dirname(f); n = os.path.splitext(os.path.basename(f))[0]
        # self.save_dir = os.path.join(base, f"{n}_Result") # [삭제] 기존 로직 무시
        # os.makedirs(self.save_dir, exist_ok=True) # [삭제]
        self.show_page()
        if not self.ent_s.get(): self.ent_s.insert(0,"1"); self.ent_e.insert(0,str(len(self.doc)))
    def select_save_dir(self):
        # [수정] 경로 변경 불가 안내
        messagebox.showinfo("안내", f"저장 경로는 시스템에 의해 '{self.save_dir}'로 고정되어 있습니다.")
        
    def move_page(self, d):
        if self.doc: self.current_page = max(0, min(len(self.doc)-1, self.current_page+d)); self.show_page()
    def show_page(self):
        if not self.doc: return
        p = self.doc.load_page(self.current_page)
        h = self.canvas.winfo_height() or 900
        self.zoom_display = (h*0.95)/p.rect.height
        pix = p.get_pixmap(matrix=fitz.Matrix(self.zoom_display, self.zoom_display))
        self.tk_img = ImageTk.PhotoImage(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        self.canvas.delete("all"); self.canvas.config(scrollregion=(0,0, pix.width, pix.height))
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.redraw()
    def on_down(self, e): 
        self.sx = self.canvas.canvasx(e.x); self.sy = self.canvas.canvasy(e.y)
        self.rid = self.canvas.create_rectangle(self.sx, self.sy, self.sx, self.sy, outline="blue", width=2)
    def on_drag(self, e): self.canvas.coords(self.rid, self.sx, self.sy, self.canvas.canvasx(e.x), self.canvas.canvasy(e.y))
    def on_up(self, e):
        self.canvas.delete(self.rid)
        x1, x2 = sorted([self.sx, self.canvas.canvasx(e.x)])
        y1, y2 = sorted([self.sy, self.canvas.canvasy(e.y)])
        if x2-x1 > 10:
            z = self.zoom_display
            self.batch_regions.append((x1/z, y1/z, x2/z, y2/z))
            self.redraw(); self.log(f"영역 추가 (총 {len(self.batch_regions)})")
    def clear_regions(self): self.batch_regions = []; self.redraw()
    
    # [누락된 함수 복구] 붓(Brush) 역할
    def redraw(self):
        self.canvas.delete("region_rect")
        self.canvas.delete("region_text")
        
        z = self.zoom_display
        
        for i, (rx1, ry1, rx2, ry2) in enumerate(self.batch_regions):
            sx1, sy1, sx2, sy2 = rx1*z, ry1*z, rx2*z, ry2*z
            self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline="red", width=2, tags="region_rect")
            self.canvas.create_text(sx1, sy1, anchor="sw", text=str(i+1), fill="red", font=("bold", 12), tags="region_text")

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()