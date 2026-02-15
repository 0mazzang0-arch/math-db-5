import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import os
import datetime

# =========================================================
#  범용 AI 질문 콘솔 + MathBot 전용 패널 (기능 통합 버전)
#  - 한 프로그램 안에 두 프로그램 기능을 모두 포함
#  - 버튼 탭(Notebook)으로 "범용" / "MathBot(V29)" 전환
#  - 옵션(토큰 절약/한글 주석/팝업 끄기), 로그 저장, 토큰 미터, 단축키 포함
# =========================================================

# 클립보드 라이브러리 체크
try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False


class CombinedPromptConsoleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 범용 AI 질문 콘솔 + 🔥 MathBot V29 패널 (통합)")
        self.root.geometry("1200x900")

        # ------------------------------
        # 사용자 옵션 (UX/효율)
        # ------------------------------
        self.compression_mode = tk.BooleanVar(value=True)     # 토큰 절약 모드
        self.korean_comments = tk.BooleanVar(value=True)      # 한글 주석 우선
        self.silent_success_popup = tk.BooleanVar(value=True) # 성공 팝업 끄기

        # ------------------------------
        # 스타일 설정
        # ------------------------------
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Pretendard', 10, 'bold'), padding=5)
        style.configure('Header.TLabel', font=('Pretendard', 16, 'bold'), foreground='#2c3e50')
        style.configure('SubHeader.TLabel', font=('Pretendard', 12, 'bold'), foreground='#34495e')
        style.configure('Info.TLabel', font=('Pretendard', 10), foreground='#7f8c8d')

        # ------------------------------
        # Base Context들 (두 프로그램 기능을 모두 보존)
        # ------------------------------

        # (A) MathBot 전용 Base Context
        self.mathbot_base_context = """
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
""".strip()

        # (B) 범용 Base Context
        self.universal_base_context = """
[System Protocol: UNIVERSAL_PROMPT_CONSOLE_V1]
- Role: Senior AI Work Assistant (Coding/Debugging/Explaining/Review)
- User: 비전공자도 이해 가능하게 설명해줘야 함 (수학 강사)
- Output: 요청한 형식(예: diff / JSON / 리스트)을 반드시 지킬 것

[Critical Rules - VIOLATION FORBIDDEN]
1. **NO LAZINESS:** `# ...` / `TODO` / `pass` 같은 빈칸 금지 (요청 시 예외)
2. **SCOPE CONTROL:** 사용자가 지정한 범위(파일/함수/부분) 밖으로 확장하지 말 것
3. **DIFF FIRST:** 기존 코드 수정이면 Unified Diff 또는 함수 교체만(전체 재작성은 요청 시만)
4. **SAFETY:** 실행/삭제/외부 호출 같은 위험 행동은 하기 전 주의점과 백업을 안내
5. **LANGUAGE:** 설명은 한국어. 코드 주석은 한글 우선(필요 시 영어 병기)
--------------------------------------------------
[Dev Compression Mode: OPTIONAL]
""".strip()

        # MathBot 인수인계서 (전용 기능 유지)
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
""".strip()

        self.create_widgets()
        self.bind_hotkeys()

    # ------------------------------
    # UI
    # ------------------------------
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        header_label = ttk.Label(
            main_frame,
            text="범용 AI 질문 콘솔 + MathBot V29 패널 (통합)",
            style='Header.TLabel'
        )
        header_label.pack(pady=(0, 10))

        content_paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        content_paned.pack(fill=tk.BOTH, expand=True)

        # 왼쪽: 탭 + 버튼
        left_frame = ttk.Frame(content_paned)
        content_paned.add(left_frame, weight=1)

        # 옵션 박스 (공통)
        opt_box = ttk.Labelframe(left_frame, text="⚙️ 옵션(공통)", padding="10")
        opt_box.pack(fill=tk.X, padx=5, pady=(0, 10))
        ttk.Checkbutton(opt_box, text="🗜️ Compression Mode (토큰 절약)", variable=self.compression_mode).pack(anchor='w', pady=2)
        ttk.Checkbutton(opt_box, text="🇰🇷 한글 주석 우선", variable=self.korean_comments).pack(anchor='w', pady=2)
        ttk.Checkbutton(opt_box, text="🔕 성공 팝업 끄기", variable=self.silent_success_popup).pack(anchor='w', pady=2)

        # 탭(Notebook): 범용 / MathBot
        nb = ttk.Notebook(left_frame)
        nb.pack(fill=tk.BOTH, expand=True)

        # -------- 범용 탭 --------
        tab_univ = ttk.Frame(nb)
        nb.add(tab_univ, text="🧠 범용")

        univ_box = ttk.Labelframe(tab_univ, text="🕹️ 범용 작업", padding="10")
        univ_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Label(univ_box, text="[자주 쓰는 작업]", style='SubHeader.TLabel').pack(anchor='w', pady=(0, 5))
        ttk.Button(univ_box, text="1. 🛠️ 오류 고치기 (에러/이상동작)", command=lambda: self.process_command('U1')).pack(fill=tk.X, pady=2)
        ttk.Button(univ_box, text="2. 🧱 새 프로그램 만들기 (처음부터)", command=lambda: self.process_command('U2')).pack(fill=tk.X, pady=2)
        ttk.Button(univ_box, text="3. ➕ 기능 추가하기 (기존 코드에 덧붙이기)", command=lambda: self.process_command('U3')).pack(fill=tk.X, pady=2)
        ttk.Button(univ_box, text="4. 📚 개념/코드 이해하기 (쉬운 설명)", command=lambda: self.process_command('U4')).pack(fill=tk.X, pady=2)
        ttk.Button(univ_box, text="5. ✅ 논리 점검하기 (시뮬레이션/엣지케이스)", command=lambda: self.process_command('U5')).pack(fill=tk.X, pady=2)
        ttk.Button(univ_box, text="0. 🔄 대화/작업 리셋 요약", command=lambda: self.process_command('U0')).pack(fill=tk.X, pady=2)

        ttk.Label(univ_box, text="\n[단축키]", style='SubHeader.TLabel').pack(anchor='w', pady=(10, 5))
        ttk.Label(univ_box, text="- Alt+0~5: 범용 실행\n- Ctrl+0~9: MathBot 실행", style='Info.TLabel', wraplength=240).pack(anchor='w')

        # -------- MathBot 탭 --------
        tab_mb = ttk.Frame(nb)
        nb.add(tab_mb, text="🔥 MathBot(V29)")

        mb_box = ttk.Labelframe(tab_mb, text="🕹️ MathBot 전용", padding="10")
        mb_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Label(mb_box, text="[기본 작업]", style='SubHeader.TLabel').pack(anchor='w', pady=(0, 5))
        ttk.Button(mb_box, text="1. 🛠️ 핀셋 수정 (Pincer Edit)", command=lambda: self.process_command('M1')).pack(fill=tk.X, pady=2)
        ttk.Button(mb_box, text="2. 📜 전체 코드 작성 (Full Code)", command=lambda: self.process_command('M2')).pack(fill=tk.X, pady=2)
        ttk.Button(mb_box, text="3. 🏗️ 아키텍처 설계 (Blueprint)", command=lambda: self.process_command('M3')).pack(fill=tk.X, pady=2)
        ttk.Button(mb_box, text="4. 🧹 코드 리팩토링 (Optimization)", command=lambda: self.process_command('M4')).pack(fill=tk.X, pady=2)

        ttk.Label(mb_box, text="\n[고급 작업]", style='SubHeader.TLabel').pack(anchor='w', pady=(10, 5))
        ttk.Button(mb_box, text="5. 🚀 V29 인수인계서 발송", command=lambda: self.process_command('M5')).pack(fill=tk.X, pady=2)
        ttk.Button(mb_box, text="6. 🧠 로직 시뮬레이션", command=lambda: self.process_command('M6')).pack(fill=tk.X, pady=2)
        ttk.Button(mb_box, text="7. 📂 파일 구조 동기화", command=lambda: self.process_command('M7')).pack(fill=tk.X, pady=2)
        ttk.Button(mb_box, text="8. ↩️ 롤백 요청 (Rollback)", command=lambda: self.process_command('M8')).pack(fill=tk.X, pady=2)

        ttk.Label(mb_box, text="\n[특수 모드]", style='SubHeader.TLabel').pack(anchor='w', pady=(10, 5))
        ttk.Button(mb_box, text="9. 🖼️ 이미지 분석 (OCR JSON)", command=lambda: self.process_command('M9')).pack(fill=tk.X, pady=2)
        ttk.Button(mb_box, text="0. 🔄 대화 리셋 요약", command=lambda: self.process_command('M0')).pack(fill=tk.X, pady=2)

        # 상태 라벨 (공통)
        self.status_label = ttk.Label(left_frame, text="대기 중...", style='Info.TLabel', wraplength=260)
        self.status_label.pack(side='bottom', fill='x', padx=5, pady=5)

        # 오른쪽: 입력/출력
        right_frame = ttk.Frame(content_paned)
        content_paned.add(right_frame, weight=3)

        ttk.Label(right_frame, text="STEP 1. 상세 내용 입력 (에러 로그, 목표, 코드 등)", style='SubHeader.TLabel').pack(anchor='w', pady=(0, 5))
        self.input_text = scrolledtext.ScrolledText(right_frame, height=12, font=('Consolas', 10))
        self.input_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        ttk.Label(right_frame, text="💡 내용을 입력하고 왼쪽 버튼(또는 단축키)을 누르면 프롬프트가 생성되고 클립보드에 복사됩니다.", style='Info.TLabel').pack(anchor='w')
        self.meter_label = ttk.Label(right_frame, text="길이: 0 chars | 추정 토큰: 0", style='Info.TLabel')
        self.meter_label.pack(anchor='w', pady=(2, 0))

        ttk.Label(right_frame, text="STEP 2. 생성된 프롬프트 (자동 복사됨)", style='SubHeader.TLabel').pack(anchor='w', pady=(10, 5))
        self.output_text = scrolledtext.ScrolledText(right_frame, height=18, font=('Consolas', 10), bg='#f0f0f0')
        self.output_text.pack(fill=tk.BOTH, expand=True)

    # ------------------------------
    # Common helpers
    # ------------------------------
    def get_user_input(self) -> str:
        return self.input_text.get("1.0", tk.END).strip()

    def set_output(self, text: str, tag: str = "") -> None:
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", text)

        chars = len(text)
        est_tokens = max(1, chars // 4)  # 매우 러프한 추정
        self.meter_label.config(text=f"길이: {chars} chars | 추정 토큰: {est_tokens}")

        self.save_prompt_log(text, tag=tag)

        if CLIPBOARD_AVAILABLE:
            pyperclip.copy(text)
            self.status_label.config(text="✅ 클립보드 복사 완료! (Ctrl+V)", foreground="green")
            if not self.silent_success_popup.get():
                messagebox.showinfo("성공", "프롬프트가 생성되고 클립보드에 복사되었습니다.")
        else:
            self.status_label.config(text="⚠️ pyperclip 미설치: 직접 복사하세요.", foreground="red")

    def build_header(self, base_context: str) -> str:
        """옵션에 따라 헤더(규칙)를 붙여서 프롬프트를 안정/절약 모드로 만듦."""
        extra_rules = []
        if self.compression_mode.get():
            extra_rules.append(
                "[Compression Mode]\n"
                "- 불필요한 설명 금지\n"
                "- 결론/행동/산출물 우선\n"
                "- 20줄 이내(요청 없으면)\n"
                "- 기존 코드 수정은 diff/함수교체 우선\n"
            )
        if self.korean_comments.get():
            extra_rules.append(
                "[Korean Comment Rule]\n"
                "- 코드 주석은 한글이 기본. (필요 시 영어 병기)\n"
            )
        header = base_context.strip() + "\n\n" + "\n".join(extra_rules) if extra_rules else base_context.strip()
        return header.strip() + "\n\n"

    def save_prompt_log(self, prompt_text: str, tag: str = "") -> None:
        """생성 프롬프트를 logs/에 저장(재현/품질개선용 데이터)."""
        try:
            os.makedirs("logs", exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_tag = re_safe_filename(tag) if tag else "prompt"
            path = os.path.join("logs", f"{safe_tag}_{ts}.txt")
            user_content = self.get_user_input()
            with open(path, "w", encoding="utf-8") as f:
                f.write("### USER_INPUT ###\n")
                f.write(user_content + "\n\n")
                f.write("### GENERATED_PROMPT ###\n")
                f.write(prompt_text + "\n")
        except Exception:
            # 로그 저장 실패가 UX를 깨지 않도록 조용히 무시
            pass

    def bind_hotkeys(self):
        """
        단축키:
        - Ctrl+0~9 : MathBot 실행(기존 패널 방식 유지)
        - Alt+0~5  : 범용 실행
        """
        # MathBot
        for key in ["0","1","2","3","4","5","6","7","8","9"]:
            self.root.bind(f"<Control-Key-{key}>", lambda e, k=key: self.process_command(f"M{k}"))

        # Universal
        for key in ["0","1","2","3","4","5"]:
            self.root.bind(f"<Alt-Key-{key}>", lambda e, k=key: self.process_command(f"U{k}"))

    # ------------------------------
    # Command dispatcher
    # ------------------------------
    def process_command(self, mode: str) -> None:
        content = self.get_user_input()
        prompt = ""

        # 범용 / MathBot 구분
        if mode.startswith("U"):
            header = self.build_header(self.universal_base_context)
            prompt = self._build_universal_prompt(mode, content, header)
        elif mode.startswith("M"):
            header = self.build_header(self.mathbot_base_context)
            prompt = self._build_mathbot_prompt(mode, content, header)
        else:
            messagebox.showwarning("오류", f"알 수 없는 모드: {mode}")
            return

        self.set_output(prompt, tag=mode)

    # ------------------------------
    # Universal prompts
    # ------------------------------
    def _build_universal_prompt(self, mode: str, content: str, header: str) -> str:
        if mode == 'U1':
            if not content:
                messagebox.showwarning("입력 필요", "에러 로그/현상/관련 코드를 입력해 주세요.")
                return ""
            return header + f"""
[작업: 오류 고치기(디버깅) / 최소 수정]
현상/에러:
{content}

요구:
1) 원인 후보 3개(우선순위)
2) 가장 유력 원인 1개 + 근거 3줄
3) 수정은 Unified Diff 또는 함수 교체만
4) (가능하면) 재현 방법/확인 방법 2개
""".strip()

        if mode == 'U2':
            if not content:
                content = "만들고 싶은 프로그램/기능을 한 문장으로 적어줘 (예: '엑셀 파일에서 점수 집계하는 GUI')"
            return header + f"""
[작업: 새 프로그램 만들기(처음부터)]
목표:
{content}

요구:
1) 필요한 기능 목록(체크리스트)
2) 파일/폴더 구조 제안
3) 실행 가능한 전체 코드(요청 범위 내)
4) 설치/실행 방법(짧게)

주의:
- 너무 거대하면 Part 1/2로 나눠서 출력
""".strip()

        if mode == 'U3':
            if not content:
                messagebox.showwarning("입력 필요", "기존 코드와 추가하고 싶은 기능을 적어주세요.")
                return ""
            return header + f"""
[작업: 기능 추가(기존 코드에 덧붙이기)]
현재 코드/상태 + 추가 기능 요구:
{content}

요구:
1) 추가할 위치(함수/클래스/파일) 제안
2) 변경은 diff 또는 '추가할 함수/클래스'만
3) 기존 동작이 깨질 수 있는 포인트 3개
""".strip()

        if mode == 'U4':
            if not content:
                messagebox.showwarning("입력 필요", "이해가 안 되는 코드/설명/용어를 붙여넣어 주세요.")
                return ""
            return header + f"""
[작업: 개념/코드 이해하기(쉬운 설명)]
아래 내용이 이해가 안 돼:
{content}

요구(쉬운 말로):
1) 핵심 개념 3개만 뽑아서 2줄씩 설명
2) 이 코드에서 그 개념이 '어디에 쓰였는지' 한 줄
3) 내가 흔히 헷갈릴 포인트 3개
""".strip()

        if mode == 'U5':
            if not content:
                messagebox.showwarning("입력 필요", "점검할 로직/코드/설명을 입력해 주세요.")
                return ""
            return header + f"""
[작업: 논리 점검(시뮬레이션/엣지케이스)]
점검 대상:
{content}

요구:
1) 입력 3가지로 손으로 따라가듯 시뮬레이션
   - 정상 1, 엣지 1, 실패 1
2) 실패 가능 지점 3개
3) 안전장치(예외처리/검증) 최소 3개 제안
""".strip()

        if mode == 'U0':
            return """
[System Command: 대화/작업 리셋 요약]
- 지금 대화/작업이 길어졌어.
- 아래를 한국어로 20줄 이내로 요약해줘.

1) 지금까지 한 일(완료)
2) 현재 막힌 지점/미해결
3) 다음에 내가 해야 할 '딱 1가지'
4) 필요한 입력(내가 추가로 줘야 하는 정보)
""".strip()

        return header + f"[알 수 없는 범용 모드: {mode}]"

    # ------------------------------
    # MathBot prompts (기존 기능 보존)
    # ------------------------------
    def _build_mathbot_prompt(self, mode: str, content: str, header: str) -> str:
        # 0: 리셋 요약
        if mode == 'M0':
            return """
[System Command: Summarize Context]
👉 The chat context is getting too long.

Please summarize the current session:
1. **Completed Logic:** What files are finished?
2. **Current Task:** What were we working on?
3. **Pending Errors:** Any unfixed bugs?
4. **Next Step:** What is the very next command I should give?

Output in Korean. 20 lines max.
""".strip()

        # 1: 핀셋 수정
        if mode == 'M1':
            if not content:
                messagebox.showwarning("입력 필요", "에러 로그나 수정할 내용을 입력창에 넣으세요.")
                return ""
            return header + f"""
[Task: Precision Code Fix]
🚨 Error/Issue:
{content}

👉 Action Required:
1. Identify the **Root Cause** in 1 line.
2. Provide **Unified Diff** or **Function Replacement** ONLY.
3. **DO NOT** output the whole file unless requested. Focus on the broken part.
4. If imports are missing, specify them clearly.
""".strip()

        # 2: 전체 코드 작성
        if mode == 'M2':
            if not content:
                content = "Target File Not Specified (Please input target filename)"
            return header + f"""
[Task: Full Code Generation]
📂 Target: {content}

👉 STRICT Constraints:
1. **NO PLACEHOLDERS.** (e.g., `# ...`, `pass`, `TODO`) are strictly FORBIDDEN.
2. Generate the **Complete, Working Code** from line 1 to the end.
3. Include extensive comments explaining 'Why' this logic is used.
4. If the code is over 500 lines, split it into Part 1 and Part 2.
""".strip()

        # 3: 아키텍처
        if mode == 'M3':
            if not content:
                content = "Goal Not Specified"
            return header + f"""
[Task: Architecture Design]
💡 Goal: {content}

👉 Output Requirements:
1. **Class Diagram (Text)**: Show relationships between classes.
2. **Data Flow**: Explain how data moves (Input -> Process -> DB).
3. **Bottleneck Analysis**: Predict where it might fail (Speed, Memory, API limits).
4. **Step-by-Step Implementation Plan**: Phase 1, Phase 2, Phase 3.
""".strip()

        # 4: 리팩토링
        if mode == 'M4':
            if not content:
                messagebox.showwarning("입력 필요", "리팩토링할 코드를 입력창에 붙여넣으세요.")
                return ""
            return header + f"""
[Task: High-Performance Refactoring]
Current Code:
{content}

👉 Goals:
1. **Optimize Speed:** Reduce complexity (Big O).
2. **Enhance Readability:** Use meaningful variable names.
3. **Robustness:** Add error handling (try-except) where missing.
4. Provide the **Full Optimized Code**.
""".strip()

        # 5: 인수인계서
        if mode == 'M5':
            return header + "\n" + self.handover_spec + "\n\n👉 Action: Read the above specification carefully. Acknowledge your role as Lead Engineer and wait for the first command."

        # 6: 로직 시뮬레이션
        if mode == 'M6':
            if not content:
                content = "General Logic Check"
            return header + f"""
[Task: Logic Simulation / Thought Experiment]
🧪 Scenario: {content}

👉 Requirement:
1. Do not write code yet.
2. **Simulate** how the current system would react step-by-step.
3. Identify logical flaws or crash points.
4. Propose a solution to handle this edge case.
""".strip()

        # 7: 파일 구조 동기화
        if mode == 'M7':
            if not content:
                messagebox.showwarning("입력 필요", "'tree' 명령어나 파일 목록을 입력창에 넣으세요.")
                return ""
            return header + f"""
[Task: Project Structure Sync]
Current File Tree:
{content}

👉 Requirement:
1. Memorize this structure for relative imports.
2. Point out if any critical file (e.g., config.py, logs/) is missing based on V29 specs.
""".strip()

        # 8: 롤백
        if mode == 'M8':
            if not content:
                content = "Code not working as expected"
            return header + f"""
[Task: EMERGENCY ROLLBACK]
🚨 Reason: {content}

👉 Action:
1. Discard the previous code generation.
2. Revert to the stable version logic.
3. Explain why the previous code failed and how the stable version avoids it.
""".strip()

        # 9: 이미지 분석
        if mode == 'M9':
            return header + """
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
""".strip()

        return header + f"[알 수 없는 MathBot 모드: {mode}]"


def re_safe_filename(s: str) -> str:
    # 파일명에 위험한 문자 제거
    return "".join(ch for ch in s if ch.isalnum() or ch in ("-", "_"))[:40] or "prompt"


if __name__ == "__main__":
    root = tk.Tk()
    app = CombinedPromptConsoleGUI(root)
    root.mainloop()
