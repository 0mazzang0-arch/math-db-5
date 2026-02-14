# main.py
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox, simpledialog, Toplevel
import os
import shutil
import threading
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
import webbrowser 
import numpy as np
from datetime import datetime
from PIL import Image, ImageTk, ImageOps  # 이미지 프리뷰용 (pip install pillow 필요)
import difflib # 유사도 정렬용

import config
import notion_api
import gemini_api
import concept_manager
import concept_sync
# ==================================================================================
# [설정] AI 해설 영역을 구분하는 절대적인 기준선입니다. 
# 이 문자열이 발견되면, 이 밑의 내용은 무조건 삭제되고 새로운 해설로 교체됩니다.
# ==================================================================================
AI_SECTION_MARKER = "\n\n\n---\n## 🤖 AI 상세 해설\n"
# ==========================================================
# [Configuration] 전역 상수 및 경로 설정
# ==========================================================
ERROR_DIR = os.path.join(config.DRIVE_WATCH_FOLDER, "_ERROR_FILES")
COMPLETED_DIR = os.path.join(config.DRIVE_WATCH_FOLDER, "_COMPLETED")

def ensure_dirs():
    if not os.path.exists(ERROR_DIR): os.makedirs(ERROR_DIR)
    if not os.path.exists(COMPLETED_DIR): os.makedirs(COMPLETED_DIR)


def backup_main_source_phase2():
    backup_dir = os.path.join(os.path.dirname(__file__), "_BACKUP")
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, "main_Backup_Phase2.py")
    try:
        shutil.copy2(__file__, backup_path)
    except Exception:
        pass

# ==========================================================
# [Main Class] MathBot V27 Control Center (Full Integration)
# ==========================================================
class AutoMathBot:
    def __init__(self, root):
        self.root = root
        self.root.title(f"MathBot V27 (The Control Center - Full Logic)")
        self.root.geometry("1600x950") 
        
        # [Data Containers]
        self.md_files = []        
        self.md_contents = []     
        self.md_numbers = []      
        self.search_corpus = []   
        
        self.concept_map = {}     # 제목 -> Notion Page ID
        self.history_data = [] 
        
        self.current_preview_image = None # 이미지 객체 유지용 (GC 방지)
        self.current_page_id = None       # 현재 선택된 문제의 노션 ID
        self.current_problem_file = None  # 현재 선택된 문제 파일명

        ensure_dirs()
        
        # [UI Construction] 탭 구조 생성
        self.setup_main_tabs()
        
        # [Critical] 데이터 로딩 스레드 시작
        threading.Thread(target=self.load_data, daemon=True).start()

    # ==========================================================
    # [UI Layout] 3-Tab Structure
    # ==========================================================
    def setup_main_tabs(self):
        # 스타일 설정
        style = ttk.Style()
        style.configure("TNotebook.Tab", font=("맑은 고딕", 11, "bold"), padding=[10, 5])
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # --- Tab 1: Dashboard (종합 상황실) ---
        self.tab_dashboard = tk.Frame(self.notebook)
        self.notebook.add(self.tab_dashboard, text="  📊 대시보드 (Monitor)  ")
        self.setup_tab1_dashboard()

        # --- Tab 2: Concept Fortress (실전개념 관리) ---
        self.tab_concepts = tk.Frame(self.notebook)
        self.notebook.add(self.tab_concepts, text="  🛡️ 실전개념 관리 (Concept DB)  ")
        self.setup_tab2_concepts()

        # --- Tab 3: Problem CMS (기출문제 관리) ---
        self.tab_problems = tk.Frame(self.notebook)
        self.notebook.add(self.tab_problems, text="  📝 기출문제 CMS (Problem Manager)  ")
        self.setup_tab3_problems()

    # ----------------------------------------------------------
    # Tab 1: Dashboard Implementation
    # ----------------------------------------------------------
    def setup_tab1_dashboard(self):
        paned = tk.PanedWindow(self.tab_dashboard, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=10, pady=10)
        
        frame_left = tk.Frame(paned)
        frame_right = tk.Frame(paned)
        paned.add(frame_left)
        paned.add(frame_right)

        # Title & Status
        tk.Label(frame_left, text="🚀 MathBot System Log", font=("맑은 고딕", 16, "bold")).pack(pady=10)
        
        self.btn_run = tk.Button(frame_left, text="시스템 초기화 중...", command=self.start_process, 
                                 bg="#cccccc", fg="black", font=("맑은 고딕", 12, "bold"), height=2, state="disabled")
        self.btn_run.pack(fill="x", padx=20, pady=10)
        
        self.lbl_status = tk.Label(frame_left, text="대기 중", font=("맑은 고딕", 11), fg="blue")
        self.lbl_status.pack(pady=5)


        # System Log
        frame_log = tk.LabelFrame(frame_left, text="실시간 시스템 로그")
        frame_log.pack(fill="both", expand=True, padx=5, pady=5)
        self.log_area = scrolledtext.ScrolledText(frame_log, state='disabled', font=("Consolas", 10))
        self.log_area.pack(fill="both", expand=True, padx=5, pady=5)

        # History
        frame_hist = tk.LabelFrame(frame_right, text="작업 히스토리 (더블클릭 이동)")
        frame_hist.pack(fill="both", expand=True, padx=5, pady=5)
        self.list_hist = tk.Listbox(frame_hist, font=("맑은 고딕", 11), bg="#f9f9f9")
        self.list_hist.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        sb_hist = tk.Scrollbar(frame_hist, command=self.list_hist.yview)
        sb_hist.pack(side="right", fill="y")
        self.list_hist.config(yscrollcommand=sb_hist.set)
        self.list_hist.bind("<Double-Button-1>", self.on_history_double_click)

    # ----------------------------------------------------------
    # Tab 2: Concept Fortress Implementation
    # ----------------------------------------------------------
    def setup_tab2_concepts(self):
        paned = tk.PanedWindow(self.tab_concepts, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Left: List
        frame_list = tk.LabelFrame(paned, text="실전개념 DB 목록 (다중선택: Ctrl/Shift)")
        paned.add(frame_list, width=600)
        
        # Treeview Setting
        columns = ("title", "status")
        self.tree_concepts = ttk.Treeview(frame_list, columns=columns, show="headings", selectmode="extended")
        self.tree_concepts.heading("title", text="개념 제목", anchor="w")
        self.tree_concepts.heading("status", text="상태", anchor="center")
        self.tree_concepts.column("title", width=400)
        self.tree_concepts.column("status", width=80, anchor="center")
        
        # Tags for Coloring (V24 Style)
        self.tree_concepts.tag_configure("evenrow", background="white")
        self.tree_concepts.tag_configure("oddrow", background="#f0f0f5")
        self.tree_concepts.tag_configure("suspect", background="#ffebee", foreground="red")
        self.tree_concepts.tag_configure("group_a", background="#e3f2fd")
        self.tree_concepts.tag_configure("group_b", background="#ffffff")
        
        sb_tree = tk.Scrollbar(frame_list, command=self.tree_concepts.yview)
        self.tree_concepts.configure(yscrollcommand=sb_tree.set)
        self.tree_concepts.pack(side="left", fill="both", expand=True)
        sb_tree.pack(side="right", fill="y")
        
        self.tree_concepts.bind("<<TreeviewSelect>>", self.on_concept_select)

        # Toolbar
        frame_toolbar = tk.Frame(frame_list)
        frame_toolbar.pack(side="bottom", fill="x", pady=5)
        
        # Row 1
        frame_row1 = tk.Frame(frame_toolbar)
        frame_row1.pack(fill="x", pady=2)
        tk.Button(frame_row1, text="🔗 병합 (Merge)", command=self.on_merge_btn_click, bg="#FF9800", fg="white").pack(side="left", fill="x", expand=True, padx=2)
        tk.Button(frame_row1, text="🧲 유사정렬 (Sort)", command=self.on_sort_similarity, bg="#2196F3", fg="white").pack(side="left", fill="x", expand=True, padx=2)
        tk.Button(frame_row1, text="🔄 새로고침", command=self.update_concept_list).pack(side="right", padx=2)
        
        # Row 2
        frame_row2 = tk.Frame(frame_toolbar)
        frame_row2.pack(fill="x", pady=2)
        tk.Button(frame_row2, text="🧹 태그제거", command=self.on_remove_tag, bg="#9E9E9E", fg="white").pack(side="left", fill="x", expand=True, padx=2)
        tk.Button(frame_row2, text="🏳️ 무시하기 (Whitelist)", command=self.on_whitelist, bg="#607D8B", fg="white").pack(side="left", fill="x", expand=True, padx=2)

        # Right: Detail Editor
        frame_detail = tk.LabelFrame(paned, text="상세 내용 에디터")
        paned.add(frame_detail)
        
        frame_edit_tools = tk.Frame(frame_detail)
        frame_edit_tools.pack(fill="x", padx=5, pady=5)
        tk.Button(frame_edit_tools, text="💾 수정 저장", command=self.on_save_concept_edit, bg="#4CAF50", fg="white").pack(side="left")
        tk.Button(frame_edit_tools, text="🗑️ 삭제", command=self.on_delete_concept, bg="#f44336", fg="white").pack(side="right")
        
        self.txt_concept_content = scrolledtext.ScrolledText(frame_detail, font=("맑은 고딕", 10))
        self.txt_concept_content.pack(fill="both", expand=True, padx=5, pady=5)

    # ----------------------------------------------------------
    # Tab 3: Problem CMS Implementation (New Feature)
    # ----------------------------------------------------------
    def setup_tab3_problems(self):
        # 3-Panel Layout: List | Editor | Linker
        paned = tk.PanedWindow(self.tab_problems, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=5, pady=5)
        
        # --- Left: Problem List ---
        frame_left = tk.LabelFrame(paned, text="기출문제 목록 (Local Repo)")
        paned.add(frame_left, width=350)
        
        self.entry_prob_search = tk.Entry(frame_left)
        self.entry_prob_search.pack(fill="x", padx=5, pady=5)
        self.entry_prob_search.bind("<Return>", self.filter_problem_list)
        tk.Button(frame_left, text="🔍 검색", command=self.filter_problem_list).pack(fill="x", padx=5)
        
        self.list_problems = tk.Listbox(frame_left, font=("맑은 고딕", 10), bg="#fafafa")
        sb_prob = tk.Scrollbar(frame_left, command=self.list_problems.yview)
        self.list_problems.config(yscrollcommand=sb_prob.set)
        self.list_problems.pack(side="left", fill="both", expand=True)
        sb_prob.pack(side="right", fill="y")
        self.list_problems.bind("<<ListboxSelect>>", self.on_problem_select)

        # --- Center: Editor (Image + Metadata + Text) ---
        frame_center = tk.LabelFrame(paned, text="문제 에디터 (Quick Fix)")
        paned.add(frame_center, width=600)
        
        # 1. Image Preview Area
        self.frame_preview = tk.Frame(frame_center, height=250, bg="black")
        self.frame_preview.pack(fill="x", padx=5, pady=5)
        self.lbl_image_preview = tk.Label(self.frame_preview, text="이미지 미리보기", fg="white", bg="black")
        self.lbl_image_preview.pack(fill="both", expand=True)
        
        # 2. Metadata Editor
        frame_meta = tk.LabelFrame(frame_center, text="메타데이터 수정")
        frame_meta.pack(fill="x", padx=5, pady=5)
        
        tk.Label(frame_meta, text="난이도:").grid(row=0, column=0, padx=2)
        self.combo_diff = ttk.Combobox(frame_meta, values=["최상", "상", "중", "하"], width=5)
        self.combo_diff.grid(row=0, column=1, padx=2)
        
        tk.Label(frame_meta, text="출처:").grid(row=0, column=2, padx=2)
        self.entry_source = tk.Entry(frame_meta, width=15)
        self.entry_source.grid(row=0, column=3, padx=2)
        
        tk.Label(frame_meta, text="연도:").grid(row=0, column=4, padx=2)
        self.entry_year = tk.Entry(frame_meta, width=6)
        self.entry_year.grid(row=0, column=5, padx=2)
        
        tk.Button(frame_meta, text="💾 메타 저장", command=self.save_problem_metadata, bg="#FFC107").grid(row=0, column=6, padx=10)

        # 3. Text Editor
        notebook_editor = ttk.Notebook(frame_center)
        notebook_editor.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.txt_prob_text = scrolledtext.ScrolledText(notebook_editor, height=10)
        notebook_editor.add(self.txt_prob_text, text="문제 본문 (Problem)")
        
        self.txt_sol_text = scrolledtext.ScrolledText(notebook_editor, height=10)
        notebook_editor.add(self.txt_sol_text, text="AI 해설 (Solution)")
        
        tk.Button(frame_center, text="💾 텍스트(본문/해설) 노션 반영", command=self.save_problem_text, bg="#4CAF50", fg="white").pack(fill="x", padx=5, pady=5)

        # --- Right: Linker ---
        frame_right = tk.LabelFrame(paned, text="🔗 실전개념 링커 (The Linker)")
        paned.add(frame_right, width=300)
        
        self.list_linked_concepts = tk.Listbox(frame_right, bg="#e0f7fa")
        self.list_linked_concepts.pack(fill="both", expand=True, padx=5, pady=5)
        
        tk.Button(frame_right, text="➕ 개념 연결 (Connect)", command=self.open_linker_dialog, bg="#2196F3", fg="white").pack(fill="x", padx=5, pady=2)
        tk.Button(frame_right, text="➖ 연결 해제 (Disconnect)", command=self.disconnect_concept, bg="#f44336", fg="white").pack(fill="x", padx=5, pady=2)
        
        tk.Label(frame_right, text="-"*30).pack(pady=5)
        tk.Button(frame_right, text="🗑️ [위험] 문제 완전 삭제", command=self.delete_problem_complete, bg="#000000", fg="red").pack(fill="x", padx=5, pady=20)

    # ==========================================================
    # [Logic] Helper Functions & Shared Logs
    # ==========================================================
    def log(self, msg):
        try:
            self.log_area.config(state='normal')
            
            # [Memory Guard] 로그가 너무 길어지면 앞부분을 잘라내어 메모리 폭발 방지
            # 현재 라인 수가 3000줄을 넘어가면, 가장 오래된 500줄을 삭제함
            if float(self.log_area.index('end-1c')) > 3000:
                self.log_area.delete('1.0', '500.0')
                
            self.log_area.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
            self.log_area.see(tk.END)
            self.log_area.config(state='disabled')
        except: pass
    
    def add_history(self, msg, url):
        self.history_data.append(url)
        self.list_hist.insert(tk.END, msg)
        self.list_hist.see(tk.END)

    def on_history_double_click(self, event):
        selection = self.list_hist.curselection()
        if not selection: return
        idx = selection[0]
        if idx < len(self.history_data):
            webbrowser.open(self.history_data[idx])

    def move_to_dir(self, src_path, dest_dir, filename):
        try:
            if not os.path.exists(dest_dir): os.makedirs(dest_dir)
            shutil.move(src_path, os.path.join(dest_dir, filename))
        except Exception as e:
            self.log(f"⚠ 파일 이동 실패 ({filename}): {e}")

    def git_push_updates(self, repo_path):
        """[V24 Logic] Git Sync Robustness (With Timeout Safety)"""
        try:
            # 타임아웃 60초 설정: 1분 안에 반응 없으면 강제 종료하고 다음 작업으로 넘어감
            # 이렇게 해야 프로그램이 멈추지 않음.
            
            # 1. Pull (Rebase)
            subprocess.run(["git", "pull", "--rebase"], cwd=repo_path, check=False, 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
            
            # 2. Add
            subprocess.run(["git", "add", "."], cwd=repo_path, check=True, 
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
            
            # 3. Commit
            subprocess.run(["git", "commit", "-m", "Auto Upload by MathBot"], cwd=repo_path, check=False, 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
            
            # 4. Push (가장 위험한 구간)
            subprocess.run(["git", "push"], cwd=repo_path, check=True, capture_output=True, text=True, timeout=90)
            
            self.root.after(0, lambda: self.log(f"🚀 [Git] 업로드 완료: {os.path.basename(repo_path)}"))
            
        except subprocess.TimeoutExpired:
            self.root.after(0, lambda: self.log(f"❌ [Git Error] 시간 초과! (Timeout). 네트워크를 확인하세요."))
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else str(e)
            self.root.after(0, lambda: self.log(f"❌ [Git Error] Push 실패: {error_msg}"))
        except Exception as e:
            self.root.after(0, lambda: self.log(f"❌ [Git Error] 알 수 없는 오류: {e}"))

    def normalize_text(self, text):
        split_patterns = [r'##\s*풀이', r'##\s*정답', r'##\s*해설', r'\*\*풀이\*\*', r'Sol\)', r'Solution', r'정답']
        for pat in split_patterns:
            parts = re.split(pat, text, flags=re.IGNORECASE)
            if len(parts) > 1: text = parts[0]; break
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        garbage = ['$$', '$', '{', '}', ' ', '*', '#', '-', '[', ']', '(', ')', '`', '|', '&', '\\', 'hline', 'clines']
        for g in garbage: text = text.replace(g, '')
        return text

    def extract_numbers(self, text):
        return set(re.findall(r'\d+(?:\.\d+)?', text))

    def load_data(self):
        self.log("📡 시스템 시동 중... (V27 Full Mode)")
        
        # 1. Notion Sync
        try:
            self.log("   (1/3) Notion DB 동기화...")
            total = notion_api.sync_db_to_memory(lambda x: None)
            self.log(f"   ✅ Notion 데이터 확보: {total}개")
        except Exception as e:
            self.log(f"   ❌ Notion Sync 실패: {e}")
            
        # 2. Local MD Files
        self.log("   (2/3) 로컬 문제 데이터 스캔...")
        self.md_files = []
        if os.path.exists(config.MD_DIR_PATH):
            files = [f for f in os.listdir(config.MD_DIR_PATH) if f.lower().endswith(".md")]
            for f in files:
                try:
                    with open(os.path.join(config.MD_DIR_PATH, f), "r", encoding="utf-8") as file:
                        content = file.read()
                        self.md_files.append(f)
                        self.md_contents.append(content)
                        self.search_corpus.append(self.normalize_text(content))
                        self.md_numbers.append(self.extract_numbers(content))
                except: pass
            self.refresh_problem_list()
        
        # Build Vectorizer
        self.log(f"   ⚙️ 검색 엔진 빌드 중 ({len(self.md_files)}개)...")
        if self.search_corpus:
            self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))
            self.tfidf_matrix = self.vectorizer.fit_transform(self.search_corpus)
        else:
            self.vectorizer = None

        # 3. Concept Map
        self.log("   (3/3) 실전개념 DB 로드...")
        try:
            self.concept_map = concept_sync.get_existing_map() or {}
            self.update_concept_list()
        except: pass
        
        self.log("✅ 모든 준비 완료. 대시보드가 활성화되었습니다.")
        self.btn_run.config(state="normal", bg="#4CAF50", fg="white", text="▶ 자동화 시작")
        self.lbl_status.config(text="준비 완료", fg="green")
        
        # [플래그 초기화]
        self.is_running = False 

    # ▼▼▼▼▼ [수정됨: 시작/중지 토글 로직 적용] ▼▼▼▼▼
    def start_process(self):
        if not self.is_running:
            # 시작 로직
            self.is_running = True
            self.btn_run.config(text="⏹ 자동화 중지 (클릭)", bg="#f44336") # 빨간색으로 변경
            self.lbl_status.config(text="🔥 자동화 루프 가동 중... (24h)", fg="blue")
            threading.Thread(target=self.process_logic, daemon=True).start()
        else:
            # 중지 로직
            self.is_running = False
            self.btn_run.config(state="disabled", text="중지 요청 중...")
            self.log("🛑 사용자 요청에 의해 루프를 정지합니다. (현재 작업 완료 후 종료)")
    # ▲▲▲▲▲ [여기까지 교체] ▲▲▲▲▲
    
    # ==========================================================
    # [Logic] Tab 2: Concept Manager Handlers
    # ==========================================================
    def update_concept_list(self):
        for item in self.tree_concepts.get_children():
            self.tree_concepts.delete(item)
        concepts = concept_manager.load_concepts()
        sorted_concepts = sorted(concepts, key=lambda x: x['title'])
        for i, c in enumerate(sorted_concepts):
            title = c['title']
            tags = ()
            if "(중복의심)" in title:
                tags = ("suspect",)
                stat = "⚠️ 확인필요"
            else:
                tags = ("evenrow",) if i % 2 == 0 else ("oddrow",)
                stat = "정상"
            self.tree_concepts.insert("", "end", values=(title, stat), tags=tags)

    def on_concept_select(self, event):
        sel = self.tree_concepts.selection()
        if not sel: return
        if len(sel) > 1:
            self.txt_concept_content.delete("1.0", tk.END)
            self.txt_concept_content.insert(tk.END, f"✅ {len(sel)}개 항목 선택됨.\n[병합] 또는 [삭제]를 수행하세요.")
            return
        vals = self.tree_concepts.item(sel[0])['values']
        if not vals: return
        title = vals[0]
        concepts = concept_manager.load_concepts()
        target = next((i for i in concepts if i['title'] == title), None)
        self.txt_concept_content.delete("1.0", tk.END)
        if target: self.txt_concept_content.insert(tk.END, target.get('content', ''))

    def get_selected_concepts(self):
        return [self.tree_concepts.item(i)['values'][0] for i in self.tree_concepts.selection()]

    def on_sort_similarity(self):
        """[V24 Logic] 유사도 정렬 + 색상 교차 완벽 복구"""
        self.log("🧲 유사도 분석 및 정렬 중... (시간이 걸릴 수 있음)")
        self.root.update() 
        
        # 정렬된 리스트 받아옴
        sorted_list = concept_manager.get_similarity_clusters()
        
        # Treeview 비우기
        for item in self.tree_concepts.get_children():
            self.tree_concepts.delete(item)
            
        # 그룹별 색상 칠하기 로직 (V24 그대로 적용)
        current_bg_tag = "group_a" # 시작 색상
        prev_title = ""
        
        for title in sorted_list:
            status_text = "정상"
            row_tag = ()
            
            # 1. 중복의심 태그 우선 적용
            if "(중복의심)" in title:
                row_tag = ("suspect",)
                status_text = "⚠️ 확인필요"
            else:
                # 2. 그룹 색상 로직
                if prev_title:
                    norm_curr = concept_manager.normalize_fingerprint(title)
                    norm_prev = concept_manager.normalize_fingerprint(prev_title)
                    sim = difflib.SequenceMatcher(None, norm_curr, norm_prev).ratio()
                    
                    # 유사도가 뚝 떨어지면(0.4 미만) 다른 그룹으로 간주 -> 배경색 변경
                    if sim < 0.4:
                        current_bg_tag = "group_b" if current_bg_tag == "group_a" else "group_a"
                
                row_tag = (current_bg_tag,)
                
            self.tree_concepts.insert("", "end", values=(title, status_text), tags=row_tag)
            prev_title = title # 다음 비교를 위해 저장
        
        self.log(f"✅ 정렬 완료 ({len(sorted_list)}개)")
        messagebox.showinfo("정렬 완료", "유사한 항목끼리 배경색을 묶어서 정렬했습니다.\n목록을 확인하세요.")

    def on_merge_btn_click(self):
        titles = self.get_selected_concepts()
        if len(titles) < 2: return messagebox.showwarning("경고", "2개 이상 선택하세요.")
        
        clean = re.sub(r'^\(중복의심\)\s*\[\d+%\]\s*', '', titles[0])
        master = simpledialog.askstring("병합", f"대표 제목(Master) 입력:\n후보: {titles[0]}", initialvalue=clean)
        if not master: return

        if master not in titles:
            if not messagebox.askyesno("확인", f"'{master}'는 목록에 없습니다. '{titles[0]}'을 이 이름으로 변경하며 병합합니까?"): return
            if master != titles[0]:
                self.log(f"ℹ️ 시스템상 '{titles[0]}' 기준으로 병합됩니다. 완료 후 이름을 수정하세요.")
                master = titles[0]

        slaves = [t for t in titles if t != master]
        if concept_manager.merge_concepts_manual(master, slaves):
            self.log("📡 노션 동기화 (병합)...")
            for s in slaves:
                k = s.replace(" ", "")
                if k in self.concept_map:
                    concept_sync.delete_concept_page(self.concept_map[k])
                    del self.concept_map[k]
            
            concepts = concept_manager.load_concepts()
            m_item = next((i for i in concepts if i['title'] == master), None)
            if m_item:
                k = master.replace(" ", "")
                if k in self.concept_map:
                    concept_sync.update_concept_page(self.concept_map[k], master, m_item.get('content', ""))
            
            self.update_concept_list()
            self.log("✅ 병합 완료")

    def on_remove_tag(self):
        titles = self.get_selected_concepts()
        cnt = 0
        for t in titles:
            if "(중복의심)" not in t: continue
            res = concept_manager.remove_suspect_tag(t)
            if res == True:
                old_k = t.replace(" ", "")
                if old_k in self.concept_map:
                    concept_sync.delete_concept_page(self.concept_map[old_k])
                    del self.concept_map[old_k]
                cnt += 1
        if cnt: self.update_concept_list(); self.log(f"🧹 {cnt}개 태그 제거")

    def on_whitelist(self):
        titles = self.get_selected_concepts()
        if len(titles) < 2: return
        cnt = 0
        for i in range(len(titles)):
            for j in range(i+1, len(titles)):
                concept_manager.add_to_whitelist(titles[i], titles[j])
                cnt += 1
        self.log(f"🏳️ {cnt}쌍 화이트리스트 등록")

    def on_save_concept_edit(self):
        titles = self.get_selected_concepts()
        if len(titles) != 1: return
        title = titles[0]
        content = self.txt_concept_content.get("1.0", tk.END).strip()
        if messagebox.askyesno("저장", f"'{title}' 수정 저장?"):
            if concept_manager.manual_update_concept(title, content):
                k = title.replace(" ", "")
                if k in self.concept_map:
                    concept_sync.update_concept_page(self.concept_map[k], title, content)
                self.log(f"💾 수정 완료: {title}")

    def on_delete_concept(self):
        titles = self.get_selected_concepts()
        if not titles: return
        if messagebox.askyesno("삭제", f"{len(titles)}개 영구 삭제?"):
            for t in titles:
                k = t.replace(" ", "")
                if k in self.concept_map:
                    concept_sync.delete_concept_page(self.concept_map[k])
                    del self.concept_map[k]
                concept_manager.delete_concept(t)
            self.update_concept_list()
            self.txt_concept_content.delete("1.0", tk.END)
            self.log("🗑️ 삭제 완료")

    # ==========================================================
    # [Logic] Tab 3: Problem CMS Handlers
    # ==========================================================
    def refresh_problem_list(self):
        self.list_problems.delete(0, tk.END)
        for f in sorted(self.md_files):
            self.list_problems.insert(tk.END, f)
            
    def filter_problem_list(self, event=None):
        query = self.entry_prob_search.get().lower().strip()
        self.list_problems.delete(0, tk.END)
        for f in self.md_files:
            if query in f.lower():
                self.list_problems.insert(tk.END, f)

    def find_local_image_path(self, filename_base):
        exts = ['.jpg', '.png', '.jpeg']
        for repo in config.LOCAL_REPO_PATHS:
            for ext in exts:
                path = os.path.join(repo, filename_base + ext)
                if os.path.exists(path): return path
        for ext in exts:
            path = os.path.join(COMPLETED_DIR, filename_base + ext)
            if os.path.exists(path): return path
        return None

    def on_problem_select(self, event):
        sel = self.list_problems.curselection()
        if not sel: return
        filename = self.list_problems.get(sel[0])
        self.current_problem_file = filename
        
        # 1. Load MD Content
        md_path = os.path.join(config.MD_DIR_PATH, filename)
        content = ""
        try:
            with open(md_path, "r", encoding="utf-8") as f: content = f.read()
        except: content = "파일을 읽을 수 없습니다."
        
        prob_text = ""
        sol_text = ""
        if "## 문제" in content:
            parts = content.split("## 문제")
            if len(parts) > 1:
                sub = parts[1].split("## 해설")
                prob_text = sub[0].strip()
                if len(sub) > 1: sol_text = sub[1].strip()
        
        self.txt_prob_text.delete("1.0", tk.END)
        self.txt_prob_text.insert(tk.END, prob_text)
        self.txt_sol_text.delete("1.0", tk.END)
        self.txt_sol_text.insert(tk.END, sol_text)
        
        # 2. Load Image Preview
        base_name = os.path.splitext(filename)[0]
        img_path = self.find_local_image_path(base_name)
        
        if img_path:
            try:
                pil_img = Image.open(img_path)
                pil_img.thumbnail((500, 230))
                self.current_preview_image = ImageTk.PhotoImage(pil_img)
                self.lbl_image_preview.config(image=self.current_preview_image, text="")
            except:
                self.lbl_image_preview.config(image="", text="이미지 로드 실패")
        else:
            self.lbl_image_preview.config(image="", text="이미지 없음")

        # 3. Find Notion Page
        page_id, _ = notion_api.find_page_id(filename)
        self.current_page_id = page_id
        
        if page_id:
            self.lbl_status.config(text=f"Connected: {page_id}", fg="green")
            self.list_linked_concepts.delete(0, tk.END)
            self.list_linked_concepts.insert(tk.END, "-> '개념 연결' 버튼을 눌러 추가하세요.")
        else:
            self.lbl_status.config(text="Notion Not Found", fg="red")

    def save_problem_metadata(self):
        if not hasattr(self, 'current_page_id') or not self.current_page_id:
            return messagebox.showerror("오류", "노션 페이지와 연결되지 않았습니다.")
        messagebox.showinfo("알림", "현재 API 구조상 텍스트 속성 외 업데이트는 제한적입니다.")

    def save_problem_text(self):
        if not hasattr(self, 'current_page_id') or not self.current_page_id: return
        messagebox.showinfo("알림", "안전한 수정을 위해 '속성(Property)' 업데이트만 지원합니다.")

    def open_linker_dialog(self):
        if not hasattr(self, 'current_page_id') or not self.current_page_id:
            return messagebox.showerror("오류", "문제를 먼저 선택하세요.")
            
        top = Toplevel(self.root)
        top.title("실전개념 연결 (The Linker)")
        top.geometry("400x500")
        
        lbl = tk.Label(top, text="연결할 개념을 검색하세요:")
        lbl.pack(pady=5)
        
        entry = tk.Entry(top)
        entry.pack(fill="x", padx=10)
        
        lst = tk.Listbox(top, selectmode=tk.MULTIPLE)
        lst.pack(fill="both", expand=True, padx=10, pady=5)
        
        all_concepts = sorted(self.concept_map.keys())
        for c in all_concepts: lst.insert(tk.END, c)
        
        def filter_list(e=None):
            q = entry.get().lower()
            lst.delete(0, tk.END)
            for c in all_concepts:
                if q in c.lower(): lst.insert(tk.END, c)
        entry.bind("<KeyRelease>", filter_list)
        
        def do_link():
            sel = lst.curselection()
            if not sel: return
            c_ids = []
            c_titles = []
            for i in sel:
                t = lst.get(i)
                c_titles.append(t)
                c_ids.append(self.concept_map[t])
            res, msg = notion_api.update_page_properties(self.current_page_id, {}, concept_ids=c_ids)
            if res:
                messagebox.showinfo("성공", f"{len(c_titles)}개 개념이 연결되었습니다.")
                top.destroy()
            else:
                messagebox.showerror("실패", msg)
                
        tk.Button(top, text="🔗 선택한 개념 연결하기", command=do_link, bg="#2196F3", fg="white").pack(fill="x", padx=10, pady=10)

    def disconnect_concept(self):
        messagebox.showinfo("알림", "노션 API 제한으로 연결 해제는 노션에서 직접 하시는 게 안전합니다.")

    def delete_problem_complete(self):
        if not hasattr(self, 'current_problem_file') or not self.current_problem_file: return
        fname = self.current_problem_file
        
        if messagebox.askyesno("경고", f"정말 '{fname}'을 완전히 삭제합니까?\n(로컬 파일 + 노션 페이지 + Git에서 모두 사라집니다)"):
            if self.current_page_id:
                notion_api.delete_concept_page(self.current_page_id) 
                self.log(f"🗑️ 노션 페이지 삭제 완료: {self.current_page_id}")
            
            md_path = os.path.join(config.MD_DIR_PATH, fname)
            if os.path.exists(md_path): os.remove(md_path)
            
            base = os.path.splitext(fname)[0]
            img_path = self.find_local_image_path(base)
            if img_path and os.path.exists(img_path): os.remove(img_path)
            
            self.refresh_problem_list()
            self.txt_prob_text.delete("1.0", tk.END)
            self.lbl_image_preview.config(image="", text="삭제됨")
            self.log(f"🗑️ 완전 삭제 완료: {fname}")

    # ==========================================================
    # [Logic] AI Judge & Process Loop (V24 Full Restoration)
    # ==========================================================
    def call_ai_judge(self, ocr_text, candidates):
        try:
            candidate_text = ""
            for i, cand in enumerate(candidates):
                candidate_text += f"\n[Candidate {i+1}]: {cand[0]}\nContext: {cand[1][:300]}...\n"

            prompt = f"""
            Role: Mathematical Document Matcher.
            Task: Compare the [OCR Text] with the [Candidates] and identify the exact match.
            STRICTLY CHECK METADATA: Year, Grade, Subject, Month, Authority.
            
            [OCR Text]
            {ocr_text}

            {candidate_text}

            Instructions:
            1. Compare the mathematical structure, numbers, and key terms.
            2. Ignore minor OCR errors.
            3. If Candidate 1 matches, reply "1".
            4. If Candidate 2 matches, reply "2".
            5. If Candidate 3 matches, reply "3".
            6. If NONE match, reply "0".
            
            OUTPUT ONLY THE NUMBER.
            """
            
            response = gemini_api.execute_with_key_rotation(
                gemini_api.analysis_model, 
                [prompt],
                generation_config={"temperature": 0.0, "max_output_tokens": 300},
                request_options=gemini_api.REQUEST_OPTIONS
            )
            
            result = response.text.strip()
            self.log(f"⚖️ [AI Judge] 판결: {result}")
            
            if "1" in result: return 0
            if "2" in result: return 1
            if "3" in result: return 2
            return -1
        except Exception as e:
            self.log(f"⚠️ 심판관 오류: {e}")
            return -1

    # ==================================================================================
    # [New Engine] 이미지 합체 & 레거시 처리기 (Robustness & Integrity)
    # ==================================================================================
    def merge_images_vertical(self, path1, path2, output_path):
        """
        [Over-engineering] 두 이미지를 세로로 이어 붙입니다.
        폭이 다를 경우 큰 쪽에 맞춰 리사이징하여 정렬을 맞춥니다. (절대 실패 방지)
        """
        try:
            img1 = Image.open(path1)
            img2 = Image.open(path2)
            
            # 폭 맞추기 (Width Matching) - 큰 쪽에 무조건 맞춤
            w1, h1 = img1.size
            w2, h2 = img2.size
            target_w = max(w1, w2)
            
            # img1 리사이징 (필요시)
            if w1 < target_w:
                ratio = target_w / w1
                new_h1 = int(h1 * ratio)
                img1 = img1.resize((target_w, new_h1), Image.Resampling.LANCZOS)
                h1 = new_h1
            
            # img2 리사이징 (필요시)
            if w2 < target_w:
                ratio = target_w / w2
                new_h2 = int(h2 * ratio)
                img2 = img2.resize((target_w, new_h2), Image.Resampling.LANCZOS)
                h2 = new_h2
                
            # 캔버스 생성 (흰색 배경)
            merged_img = Image.new('RGB', (target_w, h1 + h2), (255, 255, 255))
            merged_img.paste(img1, (0, 0))
            merged_img.paste(img2, (0, h1))
            
            merged_img.save(output_path, quality=100)
            self.log(f"🧩 [Merge] 이미지 합체 성공: {os.path.basename(output_path)}")
            return True
        except Exception as e:
            self.log(f"❌ [Merge Fail] 이미지 병합 실패 (개별 처리로 전환합니다): {e}")
            return False

    def process_deep_file_legacy(self, path, img):
        """
        [Core Logic] 기존 Track A의 심층 분석 로직을 100% 원본 그대로 보존한 실행기입니다.
        일반 파일, 병합된 파일, 타임아웃된 파일 모두 이 함수를 통과합니다.
        """
        try:
            # [Debouncing] 파일 전송 안정화 대기 (1초) - 함수 진입 시점에도 한 번 더 체크 (과잉 방어)
            try:
                size_init = os.path.getsize(path)
                time.sleep(1)
                if size_init != os.path.getsize(path): return # 전송 중이면 조용히 리턴 (다음 루프에서 처리)
            except: return

            # 점수 파일명 오인 방지 정규식 (기존 로직 유지)
            if re.search(r'_[1-9]\.[a-zA-Z]+$', img):
                self.move_to_dir(path, ERROR_DIR, img)
                return

            self.root.after(0, lambda f=img: self.log(f"\n🧠 [Track A] 심층 분석 시작: {f}"))
            
            try:
                # ------------------------------------------------------------------
                # 1. OCR & Hybrid Search Engine (기존 로직 완벽 보존)
                # ------------------------------------------------------------------
                search_text = gemini_api.get_pure_ocr_text(path)
                
                final_score = 0.0
                best_file = None
                is_new_problem = False
                
                if self.vectorizer and search_text:
                    query_norm = self.normalize_text(search_text)
                    vec = self.vectorizer.transform([query_norm])
                    sims = cosine_similarity(vec, self.tfidf_matrix).flatten()
                    
                    top_indices = sims.argsort()[-3:][::-1]
                    best_idx = top_indices[0]
                    base_score = sims[best_idx]
                    
                    ocr_nums = self.extract_numbers(search_text)
                    md_nums = self.md_numbers[best_idx]
                    
                    num_bonus = 0.0
                    if ocr_nums and md_nums:
                        intersection = ocr_nums.intersection(md_nums)
                        recall = len(intersection) / len(ocr_nums)
                        if recall >= 0.8: num_bonus = 0.3
                        elif recall >= 0.5: num_bonus = 0.15
                    
                    final_score = base_score + num_bonus
                    best_file = self.md_files[best_idx]
                    
                    self.log(f"📊 점수 분석: Base({base_score:.2f}) + NumBonus({num_bonus:.2f}) = {final_score:.2f}")

                    match_decision = "NEW"
                    if final_score >= 0.8: match_decision = "MATCH"
                    elif final_score >= 0.4:
                        self.log(f"⚖️ 점수 애매함 ({final_score:.2f}). AI 심판관 소환!")
                        judge_candidates = []
                        for idx in top_indices:
                            judge_candidates.append((self.md_files[idx], self.md_contents[idx], sims[idx]))
                        winner_idx = self.call_ai_judge(search_text, judge_candidates)
                        if winner_idx != -1:
                            best_file = judge_candidates[winner_idx][0]
                            final_score = 0.99
                            match_decision = "MATCH"
                            self.log(f"🎉 AI 심판관이 매칭 확정: {best_file}")
                        else:
                            match_decision = "NEW"
                            self.log("⚖️ AI 심판관 판결: 일치하는 문제 없음 -> 신규 생성")
                    else: match_decision = "NEW"

                    if match_decision == "MATCH":
                        self.root.after(0, lambda f=best_file, s=final_score: self.log(f"🔍 [매칭] 기존 문제 업데이트: {f} (Final: {s:.2f})"))
                        is_new_problem = False
                    else:
                        self.root.after(0, lambda s=final_score: self.log(f"🆕 [신규] 생성 모드 (Final: {s:.2f})"))
                        is_new_problem = True
                else:
                    is_new_problem = True

                # ------------------------------------------------------------------
                # 2. 분석 (Forensic Mode)
                # ------------------------------------------------------------------
                self.root.after(0, lambda: self.log("🧠 상세 분석 중 (Tag Mode)..."))
                json_data = gemini_api.analyze_image_structure(path)
                
                if not json_data: 
                    self.root.after(0, lambda: self.log("❌ 분석 데이터 추출 실패. ERROR 이동."))
                    self.move_to_dir(path, ERROR_DIR, img)
                    return

                try:
                    deep_root_basename = os.path.basename(os.path.normpath(config.DEEP_WATCH_DIR)).strip()
                    parent_folder = os.path.basename(os.path.dirname(path)).strip()
                    folder_tag = "" if parent_folder == deep_root_basename else parent_folder

                    if "db_columns" not in json_data or not isinstance(json_data["db_columns"], dict):
                        json_data["db_columns"] = {}

                    if "tags" not in json_data["db_columns"] or not isinstance(json_data["db_columns"]["tags"], list):
                        json_data["db_columns"]["tags"] = []

                    if folder_tag and folder_tag not in json_data["db_columns"]["tags"]:
                        json_data["db_columns"]["tags"].append(folder_tag)
                        self.log(f"🏷️ [Auto Tag] 폴더명 태그 추가: {folder_tag}")
                except Exception as e:
                    self.log(f"⚠️ [Tag Error] 폴더 태그 추가 실패 (무시하고 진행): {e}")

# ------------------------------------------------------------------
# ------------------------------------------------------------------
                # 3. 개념 ID (Concept ID)
                # ------------------------------------------------------------------
                detected_concept_ids = []
                pcs = json_data.get("body_content", {}).get("practical_concepts", [])
                
                # [진단용 덫] 도대체 뭐가 들어오고 있는지 확인
                self.root.after(0, lambda p=pcs: self.log(f"🧪 pcs type={type(p)}, sample0={type(p[0]) if isinstance(p, list) and p else None}"))
                
                for c in pcs:
                    if not isinstance(c, dict):
                        self.root.after(0, lambda err=c: self.log(f"⚠️ [Loop Error] practical_concepts 요소 불량: {type(err)} -> {err}"))
                        continue
                    
                    self.process_single_concept(c) 
                    title_key = c.get('title', '').replace(" ", "")
                    if title_key in self.concept_map:
                        detected_concept_ids.append(self.concept_map[title_key])

                # ------------------------------------------------------------------
                # 4. GitHub & Body Content Packaging

                # ------------------------------------------------------------------
                # 4. GitHub & Body Content Packaging
                # ------------------------------------------------------------------
                repo_idx = 4 
                target_repo_path = config.LOCAL_REPO_PATHS[repo_idx]
                target_repo_name = config.REPO_NAMES[repo_idx]
                
                if is_new_problem: src_name = os.path.splitext(img)[0]
                else: src_name = os.path.splitext(best_file)[0]
                    
                _, ext = os.path.splitext(img)
                safe_name = f"{src_name}{ext}".replace(" ", "_").replace("[", "").replace("]", "").replace("(", "").replace(")", "")
                github_url = f"https://raw.githubusercontent.com/{config.GITHUB_USERNAME}/{target_repo_name}/main/{safe_name}"
                
                if "body_content" in json_data:
                    json_data["body_content"]["image_url"] = github_url
                    if is_new_problem:
                        json_data["body_content"]["problem_text"] = search_text if search_text else "OCR 텍스트 없음"
                    else:
                        if "problem_text" in json_data["body_content"]: del json_data["body_content"]["problem_text"]
                    if not json_data["body_content"].get("verbatim_handwriting"):
                        json_data["body_content"]["verbatim_handwriting"] = search_text or "OCR 텍스트 없음"

                # ------------------------------------------------------------------
                # 5. Notion Page Creation / Update (속성 및 본문 업데이트)
                # ------------------------------------------------------------------
                page_id = None
                if is_new_problem:
                    new_title = os.path.splitext(img)[0]
                    page_id, msg = notion_api.create_new_problem_page(new_title, json_data.get("db_columns", {}), detected_concept_ids)
                    
                    if page_id:
                        notion_api.append_children(page_id, json_data.get("body_content", {}))
                        self.root.after(0, lambda t=new_title: self.log(f"✨ [생성] {t}"))
                        
                        # [MD 파일 생성] - 기존의 안전장치(Overwrite vs New) 로직 100% 유지
                        md_filename = f"{new_title}.md"
                        md_path = os.path.join(config.MD_DIR_PATH, md_filename)
                        ai_response_text = json_data.get("body_content", {}).get("ai_solution", "해설 없음")

                        try:
                            current_content = ""
                            if os.path.exists(md_path):
                                with open(md_path, 'r', encoding='utf-8') as f_read: current_content = f_read.read()

                            final_new_content = ""
                            # [Critical Logic] AI_SECTION_MARKER 기준으로 덮어쓰기 로직
                            if AI_SECTION_MARKER in current_content:
                                clean_problem_part = current_content.split(AI_SECTION_MARKER)[0]
                                final_new_content = clean_problem_part.rstrip() + AI_SECTION_MARKER + "\n" + ai_response_text
                                self.root.after(0, lambda: self.log(f"♻️ [갱신] 기존 문제 수정본은 유지하고, AI 해설만 교체했습니다."))
                            elif current_content.strip() != "":
                                final_new_content = current_content.rstrip() + AI_SECTION_MARKER + "\n" + ai_response_text
                                self.root.after(0, lambda: self.log(f"⚠️ [구조변경] 구버전 파일에 안전 구분선을 추가했습니다."))
                            else:
                                problem_base = "# " + str(new_title) + "\n\n## 문제\n" + str(search_text) + "\n"
                                final_new_content = problem_base.rstrip() + AI_SECTION_MARKER + "\n" + ai_response_text
                                self.root.after(0, lambda: self.log("📝 [신규] 새 MD 파일을 생성했습니다."))

                            with open(md_path, "w", encoding="utf-8") as f_write: f_write.write(final_new_content)
                        except Exception as e_md:
                            self.root.after(0, lambda: self.log(f"❌ MD 작성 중 오류: {e_md}"))
                else:
                    page_id, err = notion_api.find_page_id(best_file)
                    if page_id:
                        notion_api.update_page_properties(page_id, json_data.get("db_columns", {}), concept_ids=detected_concept_ids)
                        notion_api.append_children(page_id, json_data.get("body_content", {}))
                        self.root.after(0, lambda: self.log(f"✅ Notion 업데이트 완료"))
                    else:
                        self.root.after(0, lambda: self.log(f"❌ 노션 페이지 매칭 거부: {err}"))
                        self.move_to_dir(path, ERROR_DIR, img)
                        return
                
                # ------------------------------------------------------------------
                # 6. 마무리 (파일 이동 및 Git Push)
                # ------------------------------------------------------------------
                if page_id:
                    page_url = f"https://www.notion.so/{page_id.replace('-', '')}"
                    self.root.after(0, lambda t=src_name, u=page_url: self.add_history(f"✅ {t}", u))
                    final_local_path = os.path.join(target_repo_path, safe_name)
                    try: 
                        shutil.move(path, final_local_path)
                        self.git_push_updates(target_repo_path)
                    except Exception as e: self.log(f"⚠ 이동/업로드 실패: {e}")
                else:
                    self.move_to_dir(path, ERROR_DIR, img)

            except Exception as e_inner:
                self.root.after(0, lambda: self.log(f"💣 개별 파일 처리 중 오류: {e_inner}"))
                self.move_to_dir(path, ERROR_DIR, img)

        except Exception as e_outer:
            self.log(f"🔥 치명적 오류 발생 (Deep Legacy): {e_outer}")
            self.move_to_dir(path, ERROR_DIR, img)

    # ▼▼▼▼▼ [여기서부터 복사해서 기존 process_logic 함수를 완전히 덮어쓰세요] ▼▼▼▼▼
    # ▼▼▼▼▼ [여기서부터 복사해서 기존 process_logic 함수를 완전히 덮어쓰세요] ▼▼▼▼▼
    # ▼▼▼▼▼ [수정: 함수 분리 없이 모든 로직을 내부에 때려 넣은 무결성 버전] ▼▼▼▼▼
    # ▼▼▼▼▼ [Final Integrity Ver: Merge + Append + Full Logic Inlined] ▼▼▼▼▼
    def process_logic(self):
        """
        [MathBot V28 Hybrid Engine: The Complete Integration (Monolithic)]
        
        [Logic Flow]
        1. Track A (Deep Analysis):
           - Watch '[1]_오답분석_Deep'
           - Merge System: Wait 60s for '_1' + '_2' pair. Merge if found.
           - Mode System: Detect '_add'/'_plus' for Append Mode.
           - Execution: Full Forensic Analysis (No Abstraction).
        2. Track B (Fast Collection):
           - Watch '[2]_자료수집_Fast'
           - Simple OCR + CategoryBrain.
        3. Concept Track:
           - Watch '실전개념'
           - Flexible Extraction.
           
        [Zero-Compromise Principle]
        - No functions used for core logic to ensure visibility.
        - Logic is duplicated for 'Normal Processing' and 'Timeout Processing'.
        - MD generation splits strictly into Append vs Overwrite paths.
        """
        import json # 로컬 임포트 (누락 방지)
        
        # [설정] 노션 캐시 갱신 주기 (초 단위) - 30분마다 갱신
        CACHE_REFRESH_INTERVAL = 1800 
        last_cache_refresh_time = time.time()
        
        # [설정] 병합 대기 시간 (60초)
        MERGE_WAIT_TIMEOUT = 60 
        
        # [Memory] 대기실: { "파일명_base": {"path": "경로", "timestamp": 시간} }
        pending_queue = {} 

        self.root.after(0, lambda: self.log("🚀 [System] 하이브리드 엔진 가동 (Merge: ON, Append/Overwrite: ON)"))
        self.root.after(0, lambda: self.log(f"👁️ [Track A] 감시 중..."))
        self.root.after(0, lambda: self.log(f"👁️ [Track B] 감시 중..."))

        try:
            while self.is_running:
                processed_any = False
                # ------------------------------------------------------------------
                # [안전장치 1] 감시 폴더 존재 확인 (루트 폴더)
                # ------------------------------------------------------------------
                if not os.path.exists(config.WATCH_ROOT_DIR):
                    self.root.after(0, lambda: self.log(f"❌ 감시 루트 폴더가 사라졌습니다: {config.WATCH_ROOT_DIR}"))
                    time.sleep(5)
                    continue

                # ------------------------------------------------------------------
                # [안전장치 2] 노션 캐시 주기적 갱신
                # ------------------------------------------------------------------
                current_time = time.time()
                if current_time - last_cache_refresh_time > CACHE_REFRESH_INTERVAL:
                    self.root.after(0, lambda: self.log("🔄 [System] 노션 데이터베이스 캐시 정기 갱신 중..."))
                    try:
                        notion_api.sync_db_to_memory(lambda x: None)
                        last_cache_refresh_time = current_time
                        self.root.after(0, lambda: self.log("✅ [System] 캐시 갱신 완료."))
                    except Exception as e_sync:
                        self.root.after(0, lambda: self.log(f"⚠️ [System] 캐시 갱신 실패 (무시): {e_sync}"))

                # ==================================================================================
                # [Track A] 오답 분석 모드 (Deep Analysis)
                # ==================================================================================
                if os.path.exists(config.DEEP_WATCH_DIR):
                    # ✅ [재귀 탐색] 하위 폴더까지 이미지 찾기
                    deep_image_paths = []
                    for root, dirs, files in os.walk(config.DEEP_WATCH_DIR):
                        for f in files:
                            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                                deep_image_paths.append(os.path.join(root, f))

                    if deep_image_paths: processed_any = True

                    # 1. 파일 스캔 루프
                    for path in deep_image_paths:
                        if not self.is_running: break

                        img = os.path.basename(path)
                        root_dir = os.path.dirname(path)

                        # ✅ [태그] Deep 폴더 바로 아래 하위폴더명을 태그로 사용
                        # 예: ...\[1]_오답분석_Deep\뉴런\파일.jpg  -> folder_tag="뉴런"
                        # Deep 바로 아래면 태그 없음("")
                        if os.path.normpath(root_dir) == os.path.normpath(config.DEEP_WATCH_DIR):
                            folder_tag = ""
                        else:
                            folder_tag = os.path.basename(root_dir)

                        # [Debouncing] 파일 전송 안정화 대기
                        try:
                            s1 = os.path.getsize(path); time.sleep(0.5)
                            if s1 != os.path.getsize(path): continue
                        except: continue

                        name_base, ext = os.path.splitext(img)


                        # --- [Merge Logic: 대기열 관리] ---------------------------------
                        
                        # CASE 1: _1.jpg 발견 (대기)
                        if name_base.endswith("_1"):
                            core_name = name_base[:-2] # "_1" 제거
                            if core_name in pending_queue: continue # 이미 대기 중

                            pending_queue[core_name] = {
                                "path": path, "filename": img, "timestamp": time.time()
                            }
                            self.root.after(0, lambda n=img: self.log(f"⏳ [Wait] {n} 대기 중... (60초)"))
                            continue # 처리 보류

                        # CASE 2: _2.jpg 발견 (병합 시도)
                        elif name_base.endswith("_2"):
                            core_name = name_base[:-2]
                            if core_name in pending_queue:
                                info = pending_queue.pop(core_name) # 대기열에서 꺼냄
                                partner_path = info["path"]
                                merged_filename = f"{core_name}_merged{ext}"
                                merged_path = os.path.join(config.DEEP_WATCH_DIR, merged_filename)
                                
                                # self.merge_images_vertical 메서드 호출 (상단에 정의됨)
                                if self.merge_images_vertical(partner_path, path, merged_path):
                                    # 병합 성공 시 원본 백업 후 Continue (병합된 파일은 다음 루프에서 감지됨)
                                    backup_dir = os.path.join(config.DRIVE_WATCH_FOLDER, "_MERGED_ORIGINALS")
                                    if not os.path.exists(backup_dir): os.makedirs(backup_dir)
                                    try:
                                        shutil.move(partner_path, os.path.join(backup_dir, info["filename"]))
                                        shutil.move(path, os.path.join(backup_dir, img))
                                        self.log(f"🧹 원본 파일(_1, _2) 백업 완료")
                                    except: pass
                                    continue 
                                else:
                                    pass # 실패 시 _2만이라도 처리 진행
                            else:
                                pass # 짝꿍 없으면 _2만 처리 진행

                        # --- [Core Analysis Logic: 심층 분석 본체 (Inline)] ---------------
                        # CASE 3: 일반 파일 / 병합된 파일 / 짝 없는 _2
                        
                        # [NEW] Mode Detection (Append vs Overwrite)
                        # 파일명에 _add, _plus, _append, _added 등이 있으면 이어쓰기 모드
                        is_append_mode = False
                        if any(trigger in img.lower() for trigger in ["_add", "_plus", "_append", "_added", "추가"]):
                            is_append_mode = True
                            self.root.after(0, lambda: self.log(f"📎 [Mode] 이어 쓰기(Append) 모드 감지: {img}"))
                        else:
                            self.root.after(0, lambda: self.log(f"📝 [Mode] 덮어 쓰기(Overwrite) 모드: {img}"))

                        # 점수 파일명 오인 방지
                        if re.search(r'_[1-9]\.[a-zA-Z]+$', img):
                            self.move_to_dir(path, ERROR_DIR, img)
                            continue

                        self.root.after(0, lambda f=img: self.log(f"\n🧠 [Track A] 심층 분석 시작: {f}"))
                        
                        try:
                            # 1. OCR & Hybrid Search Engine
                            search_text = gemini_api.get_pure_ocr_text(path)
                            final_score = 0.0
                            best_file = None
                            is_new_problem = False
                            
                            if self.vectorizer and search_text:
                                query_norm = self.normalize_text(search_text)
                                vec = self.vectorizer.transform([query_norm])
                                sims = cosine_similarity(vec, self.tfidf_matrix).flatten()
                                top_indices = sims.argsort()[-3:][::-1]
                                best_idx = top_indices[0]
                                base_score = sims[best_idx]
                                
                                ocr_nums = self.extract_numbers(search_text)
                                md_nums = self.md_numbers[best_idx]
                                num_bonus = 0.0
                                if ocr_nums and md_nums:
                                    intersection = ocr_nums.intersection(md_nums)
                                    recall = len(intersection) / len(ocr_nums)
                                    if recall >= 0.8: num_bonus = 0.3
                                    elif recall >= 0.5: num_bonus = 0.15
                                
                                final_score = base_score + num_bonus
                                best_file = self.md_files[best_idx]
                                self.log(f"📊 점수: Base({base_score:.2f}) + Bonus({num_bonus:.2f}) = {final_score:.2f}")

                                match_decision = "NEW"
                                if final_score >= 0.8: match_decision = "MATCH"
                                elif final_score >= 0.4:
                                    self.log(f"⚖️ 점수 애매함 ({final_score:.2f}). AI 심판관 소환!")
                                    judge_candidates = []
                                    for idx in top_indices:
                                        judge_candidates.append((self.md_files[idx], self.md_contents[idx], sims[idx]))
                                    winner_idx = self.call_ai_judge(search_text, judge_candidates)
                                    if winner_idx != -1:
                                        best_file = judge_candidates[winner_idx][0]
                                        final_score = 0.99
                                        match_decision = "MATCH"
                                        self.log(f"🎉 AI 심판관 매칭 확정: {best_file}")
                                    else:
                                        match_decision = "NEW"
                                        self.log("⚖️ AI 심판관 판결: 신규 생성")
                                else: match_decision = "NEW"

                                if match_decision == "MATCH":
                                    self.root.after(0, lambda f=best_file: self.log(f"🔍 [매칭] 기존 문제 업데이트: {f}"))
                                    is_new_problem = False
                                else:
                                    self.root.after(0, lambda: self.log(f"🆕 [신규] 생성 모드"))
                                    is_new_problem = True
                            else:
                                is_new_problem = True

                            # 2. 분석 (Forensic Mode)
                            self.root.after(0, lambda: self.log("🧠 상세 분석 중 (Tag Mode)..."))
                            json_data = gemini_api.analyze_image_structure(path)
                            if not json_data: 
                                self.root.after(0, lambda: self.log("❌ 분석 데이터 추출 실패. ERROR 이동."))
                                self.move_to_dir(path, ERROR_DIR, img)
                                continue
# [Robust Logic] 부모 폴더명을 추출하여 태그로 사용합니다.
                            try:
                                # 👇 [디버그용 추가] 이 줄을 복사해서 붙여넣으세요 👇
                                self.root.after(0, lambda p=path: self.log(f"🧭 [Tag Debug] path={p} | parent={os.path.basename(os.path.dirname(p))} | deep_root={os.path.basename(config.DEEP_WATCH_DIR)}"))
                                
                                parent_folder_path = os.path.dirname(path)
                                parent_folder_name = os.path.basename(parent_folder_path).strip()
                                
                                # 감시 루트 폴더 이름과 다르고, 유효한 문자열일 경우에만 태그로 인정
                                if parent_folder_name and parent_folder_name != os.path.basename(config.DEEP_WATCH_DIR):
                                    # json_data 내부에 db_columns 구조가 없으면 강제로 생성
                                    if "db_columns" not in json_data:
                                        json_data["db_columns"] = {}
                                    if "tags" not in json_data["db_columns"] or not isinstance(json_data["db_columns"]["tags"], list):
                                        json_data["db_columns"]["tags"] = []
                                    
                                    # 중복 방지 후 태그 추가
                                    if parent_folder_name not in json_data["db_columns"]["tags"]:
                                        json_data["db_columns"]["tags"].append(parent_folder_name)
                                        self.root.after(0, lambda tn=parent_folder_name: self.log(f"🏷️ [Auto Tag] 폴더명 태그 추가: {tn}"))
                            except Exception as e_tag:
                                self.root.after(0, lambda err=e_tag: self.log(f"⚠️ [Tag Error] 폴더 태그 추가 실패 (무시하고 진행): {err}"))
# ▲▲▲ [여기까지 붙여넣기] ▲▲▲
                            # 3. 개념 ID 추출
                            detected_concept_ids = []
                            pcs = json_data.get("body_content", {}).get("practical_concepts", [])
                            
                            # [진단용 덫] 도대체 뭐가 들어오고 있는지 확인
                            self.root.after(0, lambda p=pcs: self.log(f"🧪 pcs type={type(p)}, sample0={type(p[0]) if isinstance(p, list) and p else None}"))
                            
                            for c in pcs:
                                if not isinstance(c, dict):
                                    self.root.after(0, lambda err=c: self.log(f"⚠️ [Loop Error] practical_concepts 요소 불량: {type(err)} -> {err}"))
                                    continue
                                
                                self.process_single_concept(c) 
                                title_key = c.get('title', '').replace(" ", "")
                                if title_key in self.concept_map:
                                    detected_concept_ids.append(self.concept_map[title_key])

                            # 4. GitHub & URL
                            repo_idx = 4 
                            target_repo_path = config.LOCAL_REPO_PATHS[repo_idx]
                            target_repo_name = config.REPO_NAMES[repo_idx]
                            src_name = os.path.splitext(img)[0] if is_new_problem else os.path.splitext(best_file)[0]
                            safe_name = f"{src_name}{ext}".replace(" ", "_").replace("[", "").replace("]", "")
                            github_url = f"https://raw.githubusercontent.com/{config.GITHUB_USERNAME}/{target_repo_name}/main/{safe_name}"
                            
                            if "body_content" in json_data:
                                json_data["body_content"]["image_url"] = github_url
                                if is_new_problem: json_data["body_content"]["problem_text"] = search_text or "OCR 텍스트 없음"
                                else:
                                    if "problem_text" in json_data["body_content"]: del json_data["body_content"]["problem_text"]
                                if not json_data["body_content"].get("verbatim_handwriting"):
                                    json_data["body_content"]["verbatim_handwriting"] = search_text or "OCR 텍스트 없음"

                            # 5. Notion & MD Write (Mode Applied Here)
                            page_id = None
                            
                            # (A) 신규 문제인 경우 -> 무조건 생성
                            if is_new_problem:
                                new_title = os.path.splitext(img)[0]
                                page_id, msg = notion_api.create_new_problem_page(new_title, json_data.get("db_columns", {}), detected_concept_ids)
                                if page_id:
                                    notion_api.append_children(page_id, json_data.get("body_content", {}))
                                    self.root.after(0, lambda t=new_title: self.log(f"✨ [생성] {t}"))
                                    
                                    # MD 파일 생성
                                    md_filename = f"{new_title}.md"
                                    md_path = os.path.join(config.MD_DIR_PATH, md_filename)
                                    ai_sol = json_data.get("body_content", {}).get("ai_solution", "해설 없음")
                                    
                                    # [MD Logic: Append vs Overwrite]
                                    try:
                                        final_new = ""
                                        # 이어 쓰기 모드 + 파일 존재
                                        if is_append_mode and os.path.exists(md_path):
                                            with open(md_path, 'r', encoding='utf-8') as f_read: current_content = f_read.read()
                                            # 내용 하단에 구분선 넣고 추가
                                            final_new = current_content + "\n\n---\n## 🧩 [추가 해설 / 강사 풀이]\n" + ai_sol
                                            self.root.after(0, lambda: self.log("📎 [Append] 기존 파일에 내용을 추가했습니다."))
                                        
                                        # 덮어 쓰기 모드 (또는 신규 파일)
                                        else:
                                            current_content = ""
                                            if os.path.exists(md_path):
                                                with open(md_path, 'r', encoding='utf-8') as f_read: current_content = f_read.read()
                                            
                                            # 기존 파일의 안전 마커 확인
                                            if AI_SECTION_MARKER in current_content:
                                                final_new = current_content.split(AI_SECTION_MARKER)[0].rstrip() + AI_SECTION_MARKER + "\n" + ai_sol
                                                self.root.after(0, lambda: self.log(f"♻️ [갱신] 기존 문제 유지, AI 해설만 교체"))
                                            elif current_content.strip() != "":
                                                final_new = current_content.rstrip() + AI_SECTION_MARKER + "\n" + ai_sol
                                                self.root.after(0, lambda: self.log(f"⚠️ [구조변경] 안전 구분선 추가"))
                                            else:
                                                final_new = "# " + str(new_title) + "\n\n## 문제\n" + str(search_text) + "\n" + AI_SECTION_MARKER + "\n" + ai_sol
                                                self.root.after(0, lambda: self.log("📝 [신규] MD 파일 생성"))
                                                
                                        with open(md_path, "w", encoding="utf-8") as f_write: f_write.write(final_new)
                                    except Exception as e_md: self.log(f"❌ MD 작성 오류: {e_md}")
                            
                            # (B) 기존 문제 매칭된 경우 -> Notion 업데이트 & MD 모드 적용
                            else:
                                page_id, err = notion_api.find_page_id(best_file)
                                if page_id:
                                    notion_api.update_page_properties(page_id, json_data.get("db_columns", {}), concept_ids=detected_concept_ids)
                                    # Notion은 기본적으로 Append 방식
                                    notion_api.append_children(page_id, json_data.get("body_content", {}))
                                    self.root.after(0, lambda: self.log(f"✅ Notion 업데이트 완료"))
                                else:
                                    self.root.after(0, lambda: self.log(f"❌ 매칭 실패: {err}"))
                                    self.move_to_dir(path, ERROR_DIR, img)
                                    continue

                            # 6. 마무리 (이동 & Git)
                            if page_id:
                                page_url = f"https://www.notion.so/{page_id.replace('-', '')}"
                                self.root.after(0, lambda t=src_name, u=page_url: self.add_history(f"✅ {t}", u))
                                final_local_path = os.path.join(target_repo_path, safe_name)
                                try:
                                    shutil.move(path, final_local_path)
                                    self.git_push_updates(target_repo_path)
                                except Exception as e: self.log(f"⚠ 이동/업로드 실패: {e}")
                            else:
                                self.move_to_dir(path, ERROR_DIR, img)

                        except Exception as e_inner:
                            self.root.after(0, lambda: self.log(f"💣 처리 중 오류: {e_inner}"))
                            self.move_to_dir(path, ERROR_DIR, img)

                    # --------------------------------------------------------------
                    # [Timeout Check] 타임아웃된 파일 강제 처리 (로직 100% 복제 + Mode 적용)
                    # --------------------------------------------------------------
                    expired_keys = []
                    for c_name, info in pending_queue.items():
                        if time.time() - info["timestamp"] > MERGE_WAIT_TIMEOUT:
                            expired_keys.append(c_name)
                    
                    for key in expired_keys:
                        info = pending_queue.pop(key)
                        self.log(f"⏰ [Timeout] {info['filename']} 대기 시간 초과! 독자 처리 시작.")
                        
                        # [DUPLICATED LOGIC START] - 타임아웃 파일 처리를 위해 로직 반복 (No Abstraction)
                        # 위와 동일한 로직이 타임아웃된 단일 파일(_1)에도 적용됩니다.
                        path = info["path"]
                        img = info["filename"]
                        
                        # [NEW] Mode Check (Duplicated for Timeout)
                        is_append_mode = False
                        if any(trigger in img.lower() for trigger in ["_add", "_plus", "_append", "_added", "추가"]):
                            is_append_mode = True
                            self.root.after(0, lambda: self.log(f"📎 [Mode] 이어 쓰기(Append) - Timeout: {img}"))
                        else:
                            self.root.after(0, lambda: self.log(f"📝 [Mode] 덮어 쓰기(Overwrite) - Timeout: {img}"))

                        try:
                            # 1. OCR & Search
                            search_text = gemini_api.get_pure_ocr_text(path)
                            final_score = 0.0
                            best_file = None
                            is_new_problem = False
                            
                            if self.vectorizer and search_text:
                                query_norm = self.normalize_text(search_text)
                                vec = self.vectorizer.transform([query_norm])
                                sims = cosine_similarity(vec, self.tfidf_matrix).flatten()
                                top_indices = sims.argsort()[-3:][::-1]
                                best_idx = top_indices[0]
                                base_score = sims[best_idx]
                                ocr_nums = self.extract_numbers(search_text)
                                md_nums = self.md_numbers[best_idx]
                                num_bonus = 0.0
                                if ocr_nums and md_nums:
                                    intersection = ocr_nums.intersection(md_nums)
                                    recall = len(intersection) / len(ocr_nums)
                                    if recall >= 0.8: num_bonus = 0.3
                                    elif recall >= 0.5: num_bonus = 0.15
                                final_score = base_score + num_bonus
                                best_file = self.md_files[best_idx]
                                match_decision = "NEW"
                                if final_score >= 0.8: match_decision = "MATCH"
                                elif final_score >= 0.4:
                                    judge_candidates = []
                                    for idx in top_indices: judge_candidates.append((self.md_files[idx], self.md_contents[idx], sims[idx]))
                                    winner_idx = self.call_ai_judge(search_text, judge_candidates)
                                    if winner_idx != -1:
                                        best_file = judge_candidates[winner_idx][0]
                                        final_score = 0.99
                                        match_decision = "MATCH"
                                    else: match_decision = "NEW"
                                else: match_decision = "NEW"

                                if match_decision == "MATCH": is_new_problem = False
                                else: is_new_problem = True
                            else: is_new_problem = True

                            # 2. Analysis
                            json_data = gemini_api.analyze_image_structure(path)
                            if not json_data:
                                self.move_to_dir(path, ERROR_DIR, img)
                                continue
# [Robust Logic] 타임아웃된 파일 태그 처리
                            try:
                                # 👇 [디버그용 추가 2] 여기도 똑같이 붙여넣으세요 👇
                                self.root.after(0, lambda p=path: self.log(f"🧭 [Tag Debug-Timeout] path={p} | parent={os.path.basename(os.path.dirname(p))} | deep_root={os.path.basename(config.DEEP_WATCH_DIR)}"))
                                
                                parent_folder_path_to = os.path.dirname(path)
                                parent_folder_name_to = os.path.basename(parent_folder_path_to).strip()
                                
                                if parent_folder_name_to and parent_folder_name_to != os.path.basename(config.DEEP_WATCH_DIR):
                                    if "db_columns" not in json_data:
                                        json_data["db_columns"] = {}
                                    if "tags" not in json_data["db_columns"] or not isinstance(json_data["db_columns"]["tags"], list):
                                        json_data["db_columns"]["tags"] = []
                                    
                                    if parent_folder_name_to not in json_data["db_columns"]["tags"]:
                                        json_data["db_columns"]["tags"].append(parent_folder_name_to)
                                        self.root.after(0, lambda tn=parent_folder_name_to: self.log(f"🏷️ [Auto Tag-Timeout] 폴더명 태그 추가: {tn}"))
                            except Exception as e_tag_to:
                                self.root.after(0, lambda err=e_tag_to: self.log(f"⚠️ [Tag Error-Timeout] 폴더 태그 추가 실패 (무시하고 진행): {err}"))
# ▲▲▲ [여기까지 붙여넣기] ▲▲▲
                            # 3. Concept ID
                            detected_concept_ids = []
                            pcs = json_data.get("body_content", {}).get("practical_concepts", [])
                            
                            # [진단용 덫] 도대체 뭐가 들어오고 있는지 확인
                            self.root.after(0, lambda p=pcs: self.log(f"🧪 pcs type={type(p)}, sample0={type(p[0]) if isinstance(p, list) and p else None}"))
                            
                            for c in pcs:
                                if not isinstance(c, dict):
                                    self.root.after(0, lambda err=c: self.log(f"⚠️ [Loop Error] practical_concepts 요소 불량: {type(err)} -> {err}"))
                                    continue
                                
                                self.process_single_concept(c) 
                                title_key = c.get('title', '').replace(" ", "")
                                if title_key in self.concept_map:
                                    detected_concept_ids.append(self.concept_map[title_key])

                            # 4. GitHub
                            repo_idx = 4 
                            target_repo_path = config.LOCAL_REPO_PATHS[repo_idx]
                            target_repo_name = config.REPO_NAMES[repo_idx]
                            _, ext = os.path.splitext(img)
                            src_name = os.path.splitext(img)[0] if is_new_problem else os.path.splitext(best_file)[0]
                            safe_name = f"{src_name}{ext}".replace(" ", "_").replace("[", "").replace("]", "")
                            github_url = f"https://raw.githubusercontent.com/{config.GITHUB_USERNAME}/{target_repo_name}/main/{safe_name}"
                            if "body_content" in json_data:
                                json_data["body_content"]["image_url"] = github_url
                                if is_new_problem: json_data["body_content"]["problem_text"] = search_text or "OCR 텍스트 없음"
                                else: 
                                    if "problem_text" in json_data["body_content"]: del json_data["body_content"]["problem_text"]
                                if not json_data["body_content"].get("verbatim_handwriting"):
                                    json_data["body_content"]["verbatim_handwriting"] = search_text or "OCR 텍스트 없음"

                            # 5. Notion & MD (Timeout Branch - Mode Logic Duplicated)
                            page_id = None
                            if is_new_problem:
                                new_title = os.path.splitext(img)[0]
                                page_id, msg = notion_api.create_new_problem_page(new_title, json_data.get("db_columns", {}), detected_concept_ids)
                                if page_id:
                                    notion_api.append_children(page_id, json_data.get("body_content", {}))
                                    self.root.after(0, lambda t=new_title: self.log(f"✨ [Timeout] 생성: {t}"))
                                    
                                    # MD 생성 로직
                                    md_filename = f"{new_title}.md"
                                    md_path = os.path.join(config.MD_DIR_PATH, md_filename)
                                    ai_sol = json_data.get("body_content", {}).get("ai_solution", "해설 없음")
                                    try:
                                        final_new = ""
                                        # Append Logic
                                        if is_append_mode and os.path.exists(md_path):
                                            with open(md_path, 'r', encoding='utf-8') as f_read: current_content = f_read.read()
                                            final_new = current_content + "\n\n---\n## 🧩 [추가 해설 / 강사 풀이]\n" + ai_sol
                                            self.root.after(0, lambda: self.log("📎 [Append] 내용 추가됨 (Timeout)"))
                                        # Overwrite Logic
                                        else:
                                            current_content = ""
                                            if os.path.exists(md_path):
                                                with open(md_path, 'r', encoding='utf-8') as f_read: current_content = f_read.read()
                                            if AI_SECTION_MARKER in current_content:
                                                final_new = current_content.split(AI_SECTION_MARKER)[0].rstrip() + AI_SECTION_MARKER + "\n" + ai_sol
                                            elif current_content.strip() != "":
                                                final_new = current_content.rstrip() + AI_SECTION_MARKER + "\n" + ai_sol
                                            else:
                                                final_new = "# " + str(new_title) + "\n\n## 문제\n" + str(search_text) + "\n" + AI_SECTION_MARKER + "\n" + ai_sol
                                        
                                        with open(md_path, "w", encoding="utf-8") as f_write: f_write.write(final_new)
                                    except: pass
                            else:
                                page_id, err = notion_api.find_page_id(best_file)
                                if page_id:
                                    notion_api.update_page_properties(page_id, json_data.get("db_columns", {}), concept_ids=detected_concept_ids)
                                    notion_api.append_children(page_id, json_data.get("body_content", {}))
                                    self.root.after(0, lambda: self.log(f"✅ [Timeout] 업데이트 완료"))
                                else:
                                    self.move_to_dir(path, ERROR_DIR, img)
                                    continue

                            # 6. Finish
                            if page_id:
                                final_local_path = os.path.join(target_repo_path, safe_name)
                                try:
                                    shutil.move(path, final_local_path)
                                    self.git_push_updates(target_repo_path)
                                except: pass
                            else:
                                self.move_to_dir(path, ERROR_DIR, img)

                        except Exception as e_inner_to:
                            self.log(f"💣 Timeout 파일 처리 중 오류: {e_inner_to}")
                            self.move_to_dir(path, ERROR_DIR, img)
                        # [DUPLICATED LOGIC END]

                # ==================================================================================
                # [Track B] 자료 수집 모드 (기존 로직 100% 유지)
                # ==================================================================================
                if os.path.exists(config.FAST_WATCH_DIR):
                    for root, dirs, files in os.walk(config.FAST_WATCH_DIR):
                        if files: processed_any = True
                        for file in files:
                            if not self.is_running: break
                            
                            if "_Q." in file and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                q_path = os.path.join(root, file)
                                try:
                                    sz1 = os.path.getsize(q_path); time.sleep(0.5)
                                    if sz1 != os.path.getsize(q_path): continue
                                except: continue

                                a_filename = file.replace("_Q.", "_A.")
                                a_path = os.path.join(root, a_filename)
                                has_answer = False
                                if os.path.exists(a_path):
                                    try:
                                        sz_a1 = os.path.getsize(a_path); time.sleep(0.5)
                                        if sz_a1 != os.path.getsize(a_path): continue 
                                        has_answer = True
                                    except: pass
                                
# [이 줄과 라인을 맞추세요]
                                self.root.after(0, lambda f=file: self.log(f"⚡ [Track B] 수집 시작: {f}"))

# ▼▼▼ [여기서부터 붙여넣기] 시작 부분(current_folder_name)이 윗줄 self.root.after와 줄이 딱 맞아야 합니다 ▼▼▼
                                # [Robust Logic] 폴더명을 분석하여 작동 모드를 결정합니다.
                                current_folder_name = os.path.basename(root)
                                is_tag_update_mode = current_folder_name.endswith("_태그추가")
                                
                                if is_tag_update_mode:
                                    # ----------------------------------------------------------------------
                                    # [MODE 1: 태그 업데이트 전용 모드]
                                    # ----------------------------------------------------------------------
                                    tag_to_add = current_folder_name.replace("_태그추가", "").strip()
                                    self.root.after(0, lambda t=tag_to_add: self.log(f"🔄 [Update Mode] 태그 추가 전용 모드 진입. 대상 태그: '{t}'"))

                                    try:
                                        q_text_search = gemini_api.get_pure_ocr_text(q_path)
                                        if not q_text_search or not self.vectorizer:
                                            raise Exception("OCR 실패 또는 검색 엔진 미준비로 검색 불가")

                                        # [검색 엔진 가동]
                                        query_norm = self.normalize_text(q_text_search)
                                        vec = self.vectorizer.transform([query_norm])
                                        sims = cosine_similarity(vec, self.tfidf_matrix).flatten()
                                        top_indices = sims.argsort()[-3:][::-1]
                                        
                                        best_match_file = None
                                        match_found = False

                                        best_idx_candidate = top_indices[0]
                                        base_score = sims[best_idx_candidate]

                                        # 번호 일치 보너스
                                        ocr_nums = self.extract_numbers(q_text_search)
                                        md_nums = self.md_numbers[best_idx_candidate]
                                        num_bonus = 0.0
                                        if ocr_nums and md_nums:
                                            intersection = ocr_nums.intersection(md_nums)
                                            recall = len(intersection) / len(ocr_nums)
                                            if recall >= 0.8: num_bonus = 0.3
                                            elif recall >= 0.5: num_bonus = 0.15
                                        
                                        final_score_candidate = base_score + num_bonus

                                        # 매칭 판정
                                        if final_score_candidate >= 0.8:
                                            best_match_file = self.md_files[best_idx_candidate]
                                            match_found = True
                                            self.log(f"🎯 [검색 성공] 고득점 매칭: {best_match_file} (Score: {final_score_candidate:.2f})")
                                        elif final_score_candidate >= 0.4:
                                            self.log(f"⚖️ [검색 애매] AI 심판관 소환 (Score: {final_score_candidate:.2f})")
                                            judge_candidates = []
                                            for idx in top_indices:
                                                judge_candidates.append((self.md_files[idx], self.md_contents[idx], sims[idx]))
                                            winner_idx = self.call_ai_judge(q_text_search, judge_candidates)
                                            if winner_idx != -1:
                                                best_match_file = judge_candidates[winner_idx][0]
                                                match_found = True
                                                self.log(f"🎉 [AI 심판관] 매칭 확정: {best_match_file}")
                                            else:
                                                self.log("⚖️ [AI 심판관] 불일치 판정.")

                                        # 결과 처리
                                        if match_found and best_match_file and tag_to_add:
                                            page_id_target, err_target = notion_api.find_page_id(best_match_file)
                                            if page_id_target:
                                                update_payload = {"tags": [tag_to_add]}
                                                notion_api.update_page_properties(page_id_target, update_payload)
                                                self.log(f"✅ [태그 업데이트] {best_match_file} -> '{tag_to_add}'")

                                                relative_path = os.path.relpath(root, config.FAST_WATCH_DIR)
                                                target_dir = os.path.join(COMPLETED_DIR, "[2]_자료수집_Fast_Updated", relative_path)
                                                if not os.path.exists(target_dir): os.makedirs(target_dir)
                                                shutil.move(q_path, os.path.join(target_dir, file))
                                                if has_answer: shutil.move(a_path, os.path.join(target_dir, a_filename))
                                                self.log(f"📦 완료 폴더(_Updated)로 이동됨.")
                                            else:
                                                self.log(f"❌ [Notion 404] 페이지 못 찾음: {err_target}")
                                        else:
                                            self.log(f"🚫 [업데이트 스킵] DB 매칭 실패. (파일 유지)")

                                    except Exception as e_update:
                                        self.log(f"💣 [Update Error] {e_update}")
                                        self.move_to_dir(q_path, ERROR_DIR, file)
                                        if has_answer: self.move_to_dir(a_path, ERROR_DIR, a_filename)

                                else:
                                    # ----------------------------------------------------------------------
                                    # [MODE 2: 신규 생성 모드]
                                    # ----------------------------------------------------------------------
                                    try:
                                        q_text = gemini_api.get_pure_ocr_text(q_path) or "OCR 실패"
                                        a_text = ""
                                        if has_answer: a_text = gemini_api.get_pure_ocr_text(a_path) or "OCR 실패"

                                        folder_name = current_folder_name 
                                        if folder_name == os.path.basename(config.FAST_WATCH_DIR): folder_name = "미분류"
                                        
                                        import category_manager
                                        suggested_tags = category_manager.get_suggested_tags(folder_name, q_text)
                                        final_tags = list(dict.fromkeys([folder_name, "기출문제"] + suggested_tags))
                                        # [Robust Logic 1] A단계(단순계산) 태그 자동 부착
                                        is_basic = False
                                        try:
                                            if gemini_api.check_is_basic_drill(q_text):
                                                final_tags.append("#단순계산")
                                                is_basic = True
                                                self.log(f"🏷️ [Auto Tag] 단순 계산 문제 감지 -> '#단순계산'")
                                        except: pass

                                        # [Robust Logic 2] 시대인재급 난이도 태그 자동 부착 (단순계산 아닐 때만)
                                        if not is_basic:
                                            try:
                                                diff_tag = gemini_api.analyze_difficulty_level(q_text)
                                                final_tags.append(f"#{diff_tag}")
                                                self.log(f"🏷️ [Auto Tag] 난이도 판독 -> '#{diff_tag}'")
                                            except: pass

                                        # [Robust Logic 3] 정답 추출기 (Regex Hunter)
                                        extracted_answer = ""
                                        if has_answer and a_text:
                                            try:
                                                # 전략 1: 명시적 키워드 검색
                                                match = re.search(r'(?:정답|답)\s*[:\-\.]?\s*(\d+|[①-⑤]|[a-zA-Z]+)', a_text)
                                                if match: extracted_answer = match.group(1)
                                                else:
                                                    # 전략 2: 텍스트가 매우 짧으면 전체를 정답으로 간주
                                                    clean_a = a_text.strip()
                                                    if len(clean_a) < 10 and any(c.isdigit() for c in clean_a): extracted_answer = clean_a
                                                    else:
                                                        # 전략 3: 마지막 줄에서 숫자 찾기
                                                        lines = clean_a.split('\n')
                                                        last_line = lines[-1].strip()
                                                        num_match = re.search(r'(\d+)', last_line)
                                                        if num_match: extracted_answer = num_match.group(1)
                                            except: pass
                                        q_name_base = os.path.splitext(file)[0].replace("_Q", "")
                                        db_data = {
                                            "main_category": folder_name, "tags": final_tags,
                                            "necessity": "", "key_idea": "", "special_point": "", "source": q_name_base,
                                            "correct_answer": extracted_answer # [NEW] 정답 필드 추가
                                        }
                                        
                                        page_id, msg = notion_api.create_new_problem_page(q_name_base, db_data)
                                        
                                        if page_id:
                                            body_content = {
                                                "problem_text": q_text,
                                                "ai_solution": f"## 해설\n{a_text}" if a_text else "해설 없음",
                                                "verbatim_handwriting": "Track B 자동 수집 모드", "image_url": ""
                                            }
                                            notion_api.append_children(page_id, body_content)
                                            
                                            current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                            md_content = f"""---
type: collection
id: {q_name_base}
category: {folder_name}
tags: {json.dumps(final_tags, ensure_ascii=False)}
date: {current_time_str}
---

# {q_name_base}

## 문제
![problem]({file})

{q_text}

## 해설
![solution]({a_filename})

{a_text}
"""
                                            md_save_path = os.path.join(config.MD_DIR_PATH, f"{q_name_base}.md")
                                            with open(md_save_path, "w", encoding="utf-8") as f: f.write(md_content)
                                                
                                            relative_path = os.path.relpath(root, config.FAST_WATCH_DIR)
                                            target_dir = os.path.join(COMPLETED_DIR, "[2]_자료수집_Fast", relative_path)
                                            if not os.path.exists(target_dir): os.makedirs(target_dir)

                                            shutil.move(q_path, os.path.join(target_dir, file))
                                            if has_answer: shutil.move(a_path, os.path.join(target_dir, a_filename))
                                                
                                            page_url = f"https://www.notion.so/{page_id.replace('-', '')}"
                                            self.root.after(0, lambda t=q_name_base: self.log(f"✅ [Track B] 신규 생성 완료: {t}"))
                                            self.add_history(f"📦 [수집] {q_name_base}", page_url)
                                            time.sleep(3)
                                        else:
                                            self.root.after(0, lambda s=msg: self.log(f"❌ [Notion Fail] {s}"))
                                            self.move_to_dir(q_path, ERROR_DIR, file)
                                            if has_answer: self.move_to_dir(a_path, ERROR_DIR, a_filename)

                                    except Exception as e_b:
                                        self.root.after(0, lambda err=e_b: self.log(f"💣 [Track B Error] {err}"))
                                        self.move_to_dir(q_path, ERROR_DIR, file)
                                        if has_answer: self.move_to_dir(a_path, ERROR_DIR, a_filename)
# ▲▲▲ [여기까지 붙여넣기] ▲▲▲
                
                # ------------------------------------------------------------------
                # [Main Track 2] 실전개념 이미지 처리 (Concept Track - 기존 로직 100% 유지)
                # ------------------------------------------------------------------
                if not self.is_running: break

                files_concept = [f for f in os.listdir(config.CONCEPT_WATCH_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                
                if files_concept:
                    processed_any = True
                    for img in files_concept:
                        if not self.is_running: break
                        path = os.path.join(config.CONCEPT_WATCH_FOLDER, img)
                        repo_idx = 4
                        target_repo_path = config.LOCAL_REPO_PATHS[repo_idx]
                        target_repo_name = config.REPO_NAMES[repo_idx]
                        _, ext = os.path.splitext(img)
                        try:
                            result_json = gemini_api.extract_concepts_flexible(path)
                            if result_json and "concepts" in result_json:
                                for c in result_json["concepts"]:
                                    title = c.get('title', '제목없음')
                                    safe_title = "".join([x for x in title if x.isalnum() or x in (' ', '_', '-')]).strip()
                                    safe_name = f"[개념]_{safe_title}{ext}".replace(" ", "_")
                                    github_url = f"https://raw.githubusercontent.com/{config.GITHUB_USERNAME}/{target_repo_name}/main/{safe_name}"
                                    self.process_single_concept(c, github_url)
                                
                                final_local_path = os.path.join(target_repo_path, safe_name)
                                try:
                                    shutil.move(path, final_local_path)
                                    self.root.after(0, lambda f=safe_name: self.log(f"✅ [개념] {f} 완료"))
                                    self.root.after(0, self.update_concept_list)
                                except: pass
                            else:
                                self.move_to_dir(path, ERROR_DIR, img)
                        except Exception as e:
                            self.move_to_dir(path, ERROR_DIR, img)
                
                if processed_any:
                    time.sleep(0.1) # 파일 감지 시 즉시 재스캔 (CPU 폭주 방지 최소 쿨타임)
                else:
                    time.sleep(2) # 유휴 시 CPU 휴식

        except Exception as e:
            error_msg = f"스레드 치명적 충돌:\n{e}"
            print(error_msg)
            self.root.after(0, lambda: messagebox.showerror("시스템 오류", error_msg))
        
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.btn_run.config(state="normal", text="▶ 자동화 시작", bg="#4CAF50"))
            self.root.after(0, lambda: self.lbl_status.config(text="대기 중 (루프 정지됨)", fg="black"))
    # ▲▲▲▲▲ [여기까지 교체] ▲▲▲▲▲
    # ▲▲▲▲▲ [여기까지 교체] ▲▲▲▲▲
    # ▲▲▲▲▲ [여기까지 복사] ▲▲▲▲▲
    # ▲▲▲▲▲ [여기까지 복사] ▲▲▲▲▲
    # ▲▲▲▲▲ [여기까지 교체] ▲▲▲▲▲

    def process_single_concept(self, concept_data, image_url=None):
        if not isinstance(concept_data, dict):
            self.root.after(0, lambda d=concept_data: self.log(f"⚠️ [Type Error] process_single_concept 입력이 dict 아님: {type(d)} -> {d}"))
            return None
        # 1. 로컬 저장 (내부 장부 기록 - 리스트 표시용)
        concept_manager.save_concept(concept_data)
        
        # 2. Notion 동기화 (누락된 '보고 체계' 복구)
        # 로컬에 저장된 내용을 Notion 실전개념 DB로 즉시 전송합니다.
        try:
            title = concept_data.get('title', '제목없음')
            content = concept_data.get('content', '')
            
            # concept_sync.py에 있는 생성 함수 호출 (이미지 URL 포함)
            # 이제 MathBot이 혼자만 알지 않고 Notion에 보고합니다.
            if hasattr(concept_sync, 'create_concept_page'):
                page_id = concept_sync.create_concept_page(concept_data, image_url)
                
                if page_id:
                    # 전송 성공 시 매핑 테이블(제목->ID) 갱신
                    self.concept_map[title.replace(" ", "")] = page_id
                    self.root.after(0, lambda: self.log(f"📡 [Sync] 노션 업로드 성공: {title}"))
                else:
                    self.root.after(0, lambda: self.log(f"⚠️ [Sync] 노션 업로드 실패 (ID 반환 없음): {title}"))
            else:
                # 혹시 함수 이름이 다를 경우를 대비한 예외 처리
                self.root.after(0, lambda: self.log("⚠️ [System] concept_sync.create_concept_page 함수를 찾을 수 없습니다."))
                
        except Exception as e:
            self.root.after(0, lambda err=e: self.log(f"❌ [Sync Error] 실전개념 동기화 중 오류: {err}"))

if __name__ == "__main__":
    backup_main_source_phase2()
    root = tk.Tk()
    app = AutoMathBot(root)
    root.mainloop()
