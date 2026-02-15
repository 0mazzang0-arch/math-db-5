import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import sys
import os
import datetime

# 클립보드 라이브러리 체크
try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False

class MathBotCommanderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔥 MathBot Commander: AI Control Console (V29 Ultimate) 🔥")
        self.root.geometry("1100x850")

        # ------------------------------
        # 사용자 옵션 (UX/효율)
        # ------------------------------
        self.compression_mode = tk.BooleanVar(value=True)   # 토큰 세이브 모드
        self.korean_comments = tk.BooleanVar(value=True)    # 한글 주석 우선
        self.silent_success_popup = tk.BooleanVar(value=True)  # 성공 팝업 끄기(연속 작업용)
        
        # 스타일 설정
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Pretendard', 10, 'bold'), padding=5)
        style.configure('Header.TLabel', font=('Pretendard', 16, 'bold'), foreground='#2c3e50')
        style.configure('SubHeader.TLabel', font=('Pretendard', 12, 'bold'), foreground='#34495e')
        style.configure('Info.TLabel', font=('Pretendard', 10), foreground='#7f8c8d')

        # ==============================================================================
        # 0. [절대 고정] 시스템 정체성 (Base Context) - AI의 정신 개조용
        # ==============================================================================
        self.base_context = """
[System Protocol: MATHBOT_COMMANDER_V1]
- Role: Lead Python Architect & Math Education Specialist
- User: Project Manager (Strict about performance & completeness)
- Project: MathBot V29 (Automated Math Problem DB System)
- Stack: Python 3.10, ChromaDB, SQLite, Notion API, OpenCV, Watchdog

[Critical Rules - VIOLATION FORBIDDEN]
1. **NO LAZINESS:** Never use placeholders like `# ... existing code ...` or `pass`. Write FULL code.
2. **NO SUMMARY:** Do not summarize logic. Explain "Why" and "How" in code comments.
3. **DIFF ONLY:** When fixing bugs, provide `Unified Diff` or specific function replacements.
4. **PERFORMANCE:** Prioritize execution speed and error handling (Self-Healing).
5. **LANGUAGE:** Korean (Explain).
   - Code Comments: Korean first (필수), English optional (병기 가능).
--------------------------------------------------
[Dev Compression Mode: ACTIVE]
"""
        # V29 인수인계서 전문
        self.handover_spec = """
[Document: Technical Specification for V29]
1. Goal: File-based -> SQLite DB Migration & Self-Healing Agent.
2. Tasks:
   - Replace `concept_book.json` with SQLite `concepts` table.
   - Implement IPC (Named Pipes) for AutoCropper <-> MathBot communication.
   - Add `ErrorHandler` for Notion API 400/502/Timeout (Auto-retry).
   - Integrate RAG (ChromaDB) for 'Similar Problem' search.
3. Constraints:
   - Maintain `tkinter` GUI structure.
   - All logs must be saved to `logs/` directory.
"""
        
        self.create_widgets()
        self.bind_hotkeys()

    def create_widgets(self):
        # --- 레이아웃 프레임 ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. 헤더
        header_label = ttk.Label(main_frame, text="MathBot Commander: AI Control Console", style='Header.TLabel')
        header_label.pack(pady=(0, 10))

        # 2. 버튼 영역 (좌측 메뉴) vs 입력/출력 영역 (우측)
        content_paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        content_paned.pack(fill=tk.BOTH, expand=True)

        # [좌측] 버튼 패널
        btn_frame = ttk.Labelframe(content_paned, text=" 🕹️ 명령 선택 ", padding="10")
        content_paned.add(btn_frame, weight=1)

        # 옵션 (효율/토큰 절약)
        ttk.Label(btn_frame, text="[옵션]", style='SubHeader.TLabel').pack(anchor='w', pady=(5, 5))
        ttk.Checkbutton(btn_frame, text="🗜️ Compression Mode (토큰 절약)", variable=self.compression_mode).pack(anchor='w', pady=2)
        ttk.Checkbutton(btn_frame, text="🇰🇷 한글 주석 우선", variable=self.korean_comments).pack(anchor='w', pady=2)
        ttk.Checkbutton(btn_frame, text="🔕 성공 팝업 끄기", variable=self.silent_success_popup).pack(anchor='w', pady=2)
        ttk.Separator(btn_frame, orient='horizontal').pack(fill='x', pady=10)

        # 기본 작업 버튼
        ttk.Label(btn_frame, text="[기본 작업]", style='SubHeader.TLabel').pack(anchor='w', pady=(5, 5))
        ttk.Button(btn_frame, text="1. 🛠️ 핀셋 수정 (Pincer Edit)", command=lambda: self.process_command('1')).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="2. 📜 전체 코드 작성 (Full Code)", command=lambda: self.process_command('2')).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="3. 🏗️ 아키텍처 설계 (Blueprint)", command=lambda: self.process_command('3')).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="4. 🧹 코드 리팩토링 (Optimization)", command=lambda: self.process_command('4')).pack(fill=tk.X, pady=2)

        # 고급 작업 버튼
        ttk.Label(btn_frame, text="\n[고급 작업]", style='SubHeader.TLabel').pack(anchor='w', pady=(10, 5))
        ttk.Button(btn_frame, text="5. 🚀 V29 인수인계서 발송", command=lambda: self.process_command('5')).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="6. 🧠 로직 시뮬레이션", command=lambda: self.process_command('6')).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="7. 📂 파일 구조 동기화", command=lambda: self.process_command('7')).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="8. ↩️ 롤백 요청 (Rollback)", command=lambda: self.process_command('8')).pack(fill=tk.X, pady=2)

        # 특수 모드 버튼
        ttk.Label(btn_frame, text="\n[특수 모드]", style='SubHeader.TLabel').pack(anchor='w', pady=(10, 5))
        ttk.Button(btn_frame, text="9. 🖼️ 이미지 분석 (OCR JSON)", command=lambda: self.process_command('9')).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="0. 🔄 대화 리셋 요약", command=lambda: self.process_command('0')).pack(fill=tk.X, pady=2)
        
        # 종료 및 상태
        ttk.Separator(btn_frame, orient='horizontal').pack(fill='x', pady=15)
        self.status_label = ttk.Label(btn_frame, text="대기 중...", style='Info.TLabel', wraplength=200)
        self.status_label.pack(side='bottom', fill='x', pady=5)


        # [우측] 입력 및 결과 패널
        right_frame = ttk.Frame(content_paned)
        content_paned.add(right_frame, weight=3)

        # 입력창
        ttk.Label(right_frame, text="STEP 1. 상세 내용 입력 (에러 로그, 목표, 코드 등)", style='SubHeader.TLabel').pack(anchor='w', pady=(0, 5))
        self.input_text = scrolledtext.ScrolledText(right_frame, height=10, font=('Consolas', 10))
        self.input_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 안내 문구
        ttk.Label(right_frame, text="💡 내용을 입력하고 왼쪽의 버튼을 누르면, 아래에 프롬프트가 생성되고 클립보드에 복사됩니다.", style='Info.TLabel').pack(anchor='w')
        self.meter_label = ttk.Label(right_frame, text="길이: 0 chars | 추정 토큰: 0", style='Info.TLabel')
        self.meter_label.pack(anchor='w', pady=(2, 0))

        # 출력창
        ttk.Label(right_frame, text="STEP 2. 생성된 프롬프트 (자동 복사됨)", style='SubHeader.TLabel').pack(anchor='w', pady=(10, 5))
        self.output_text = scrolledtext.ScrolledText(right_frame, height=15, font=('Consolas', 10), bg='#f0f0f0')
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def get_user_input(self):
        return self.input_text.get("1.0", tk.END).strip()

    def set_output(self, text):
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", text)

        # 길이/토큰 미터 (대략)
        chars = len(text)
        est_tokens = max(1, chars // 4)  # 아주 러프한 추정치(한/영 섞이면 오차 있음)
        if hasattr(self, "meter_label"):
            self.meter_label.config(text=f"길이: {chars} chars | 추정 토큰: {est_tokens}")

        # 로그 저장
        self.save_prompt_log(text)

        # 클립보드 복사
        if CLIPBOARD_AVAILABLE:
            pyperclip.copy(text)
            self.status_label.config(text="✅ 클립보드 복사 완료!\n(Ctrl+V 하세요)", foreground="green")
            if not self.silent_success_popup.get():
                messagebox.showinfo("성공", "프롬프트가 생성되고 클립보드에 복사되었습니다.")
        else:
            self.status_label.config(text="⚠️ 클립보드 모듈 없음\n직접 복사하세요.", foreground="red")


    def build_header(self) -> str:
        """
        옵션에 따라 base_context 위에/아래로 압축 규칙을 가변 적용
        """
        header = self.base_context.strip()
        extra_rules = []
        if self.compression_mode.get():
            extra_rules.append("[Compression Mode]\n- 불필요한 설명 금지\n- 결론/행동/산출물 우선\n- 20줄 이내(요청 없으면)\n- 버그 수정은 diff만\n")
        if self.korean_comments.get():
            extra_rules.append("[Korean Comment Rule]\n- 코드 주석은 한글이 기본. (필요 시 영어 병기)\n")
        if extra_rules:
            header = header + "\n\n" + "\n".join(extra_rules)
        return header + "\n\n"

    def save_prompt_log(self, prompt_text: str) -> None:
        """
        생성된 프롬프트를 logs/에 자동 저장 (나중에 품질 개선/재현에 도움)
        """
        try:
            os.makedirs("logs", exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join("logs", f"prompt_{ts}.txt")
            user_content = self.get_user_input()
            with open(path, "w", encoding="utf-8") as f:
                f.write("### USER_INPUT ###\n")
                f.write(user_content + "\n\n")
                f.write("### GENERATED_PROMPT ###\n")
                f.write(prompt_text + "\n")
        except Exception:
            # 로그 저장 실패는 UX를 깨지 않기 위해 조용히 무시
            pass

    def bind_hotkeys(self):
        """
        Ctrl+숫자 단축키로 모드 실행 (연속 작업 효율↑)
        """
        for key in ["1","2","3","4","5","6","7","8","9","0"]:
            self.root.bind(f"<Control-Key-{key}>", lambda e, k=key: self.process_command(k))


    def process_command(self, mode):
        content = self.get_user_input()
        prompt = ""
        header = self.build_header()

        # ------------------------------------------------------------------
        # 1. 핀셋 수정
        # ------------------------------------------------------------------
        if mode == '1':
            if not content:
                messagebox.showwarning("입력 필요", "에러 로그나 수정할 내용을 입력창에 넣으세요.")
                return
            prompt = header + f"""
[Task: Precision Code Fix]
🚨 Error/Issue:
{content}

👉 Action Required:
1. Identify the **Root Cause** in 1 line.
2. Provide **Unified Diff** or **Function Replacement** ONLY.
3. **DO NOT** output the whole file unless requested. Focus on the broken part.
4. If imports are missing, specify them clearly.
"""

        # ------------------------------------------------------------------
        # 2. 전체 코드 작성
        # ------------------------------------------------------------------
        elif mode == '2':
            if not content:
                content = "Target File Not Specified (Please input target filename)"
            
            prompt = header + f"""
[Task: Full Code Generation]
📂 Target: {content}

👉 STRICT Constraints:
1. **NO PLACEHOLDERS.** (e.g., `# ...`, `pass`, `TODO`) are strictly FORBIDDEN.
2. Generate the **Complete, Working Code** from line 1 to the end.
3. Include extensive comments explaining 'Why' this logic is used.
4. If the code is over 500 lines, split it into Part 1 and Part 2.
"""

        # ------------------------------------------------------------------
        # 3. 아키텍처 설계
        # ------------------------------------------------------------------
        elif mode == '3':
            if not content:
                content = "Goal Not Specified"
                
            prompt = header + f"""
[Task: Architecture Design]
💡 Goal: {content}

👉 Output Requirements:
1. **Class Diagram (Text)**: Show relationships between classes.
2. **Data Flow**: Explain how data moves (Input -> Process -> DB).
3. **Bottleneck Analysis**: Predict where it might fail (Speed, Memory, API limits).
4. **Step-by-Step Implementation Plan**: Phase 1, Phase 2, Phase 3.
"""

        # ------------------------------------------------------------------
        # 4. 리팩토링
        # ------------------------------------------------------------------
        elif mode == '4':
            if not content:
                messagebox.showwarning("입력 필요", "리팩토링할 코드를 입력창에 붙여넣으세요.")
                return
                
            prompt = header + f"""
[Task: High-Performance Refactoring]
Current Code:
{content}

👉 Goals:
1. **Optimize Speed:** Reduce complexity (Big O).
2. **Enhance Readability:** Use meaningful variable names.
3. **Robustness:** Add error handling (try-except) where missing.
4. Provide the **Full Optimized Code**.
"""

        # ------------------------------------------------------------------
        # 5. V29 인수인계서
        # ------------------------------------------------------------------
        elif mode == '5':
            # 입력 내용 무시하고 인수인계서 발송
            prompt = header + self.handover_spec + """
\n👉 Action: Read the above specification carefully. Acknowledge your role as Lead Engineer and wait for the first command.
"""

        # ------------------------------------------------------------------
        # 6. 로직 시뮬레이션
        # ------------------------------------------------------------------
        elif mode == '6':
            if not content:
                content = "General Logic Check"
                
            prompt = header + f"""
[Task: Logic Simulation / Thought Experiment]
🧪 Scenario: {content}

👉 Requirement:
1. Do not write code yet.
2. **Simulate** how the current system would react step-by-step.
3. Identify logical flaws or crash points.
4. Propose a solution to handle this edge case.
"""

        # ------------------------------------------------------------------
        # 7. 파일 구조 동기화
        # ------------------------------------------------------------------
        elif mode == '7':
            if not content:
                messagebox.showwarning("입력 필요", "'tree' 명령어나 파일 목록을 입력창에 넣으세요.")
                return
                
            prompt = header + f"""
[Task: Project Structure Sync]
Current File Tree:
{content}

👉 Requirement:
1. Memorize this structure for relative imports.
2. Point out if any critical file (e.g., config.py, logs/) is missing based on V29 specs.
"""

        # ------------------------------------------------------------------
        # 8. 롤백 요청
        # ------------------------------------------------------------------
        elif mode == '8':
            if not content:
                content = "Code not working as expected"
                
            prompt = header + f"""
[Task: EMERGENCY ROLLBACK]
🚨 Reason: {content}

👉 Action:
1. Discard the previous code generation.
2. Revert to the stable version logic.
3. Explain why the previous code failed and how the stable version avoids it.
"""

        # ------------------------------------------------------------------
        # 9. 이미지 분석 (OCR JSON)
        # ------------------------------------------------------------------
        elif mode == '9':
            prompt = header + """
[Task: Image Analysis for MathBot]
👉 Input: (Attached Image)
👉 Output Format: JSON ONLY

{
  "unit": "Subject/Unit Name",
  "difficulty": 1-5,
  "question_type": "Multiple Choice / Short Answer",
  "keywords": ["tag1", "tag2"],
  "content_ocr": "Latex String",
  "solution_hint": "One sentence strategy"
}
"""

        # ------------------------------------------------------------------
        # 0. 대화 리셋 요약
        # ------------------------------------------------------------------
        elif mode == '0':
            prompt = """
[System Command: Summarize Context]
👉 The chat context is getting too long.

Please summarize the current session:
1. **Completed Logic:** What files are finished?
2. **Current Task:** What were we working on?
3. **Pending Errors:** Any unfixed bugs?
4. **Next Step:** What is the very next command I should give?

Output in Korean. 20 lines max.
"""

        # 결과 출력 및 복사
        self.set_output(prompt)

if __name__ == "__main__":
    root = tk.Tk()
    app = MathBotCommanderGUI(root)
    root.mainloop()