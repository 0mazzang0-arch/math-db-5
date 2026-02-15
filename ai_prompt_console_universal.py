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

class UniversalPromptConsoleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 범용 AI 질문 콘솔 (Universal Prompt Console) v1")
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
"""

        # V29 인수인계서 전문
        self.handover_spec = ""  # (범용 콘솔에서는 사용하지 않음)
        
        self.create_widgets()
        self.bind_hotkeys()

    def create_widgets(self):
        # --- 레이아웃 프레임 ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. 헤더
        header_label = ttk.Label(main_frame, text="범용 AI 질문 콘솔 (Universal Prompt Console)", style='Header.TLabel')
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

        
        # ------------------------------------------------------------------------------
        # 작업 버튼 (범용)
        # ------------------------------------------------------------------------------
        ttk.Label(btn_frame, text="[자주 쓰는 작업]", style='SubHeader.TLabel').pack(anchor='w', pady=(5, 5))
        ttk.Button(btn_frame, text="1. 🛠️ 오류 고치기 (에러/이상동작)", command=lambda: self.process_command('1')).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="2. 🧱 새 프로그램 만들기 (처음부터)", command=lambda: self.process_command('2')).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="3. ➕ 기능 추가하기 (기존 코드에 덧붙이기)", command=lambda: self.process_command('3')).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="4. 📚 개념/코드 이해하기 (쉬운 설명)", command=lambda: self.process_command('4')).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="5. ✅ 논리 점검하기 (시뮬레이션/엣지케이스)", command=lambda: self.process_command('5')).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="0. 🔄 대화/작업 리셋 요약 (짧게 정리)", command=lambda: self.process_command('0')).pack(fill=tk.X, pady=2)

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
        for key in ["1","2","3","4","5","0"]:
            self.root.bind(f"<Control-Key-{key}>", lambda e, k=key: self.process_command(k))


    def process_command(self, mode):
        content = self.get_user_input()
        prompt = ""
        header = self.build_header()

        # ------------------------------------------------------------------
        # 1. 오류 고치기 (디버깅)
        # ------------------------------------------------------------------
        if mode == '1':
            if not content:
                messagebox.showwarning("입력 필요", "에러 로그 / 문제 상황 / 관련 코드를 입력창에 넣으세요.")
                return

            prompt = header + f"""
[작업: 오류 고치기 (디버깅)]
입력(에러/상황/코드):
{content}

요구:
1) 원인 후보 3개(우선순위)
2) 가장 유력한 원인 1개 + 근거(3줄)
3) 수정안은 **Unified Diff** 또는 **수정된 함수만** (전체 파일 금지)
4) 실행/환경/의존성(버전) 확인 질문이 필요하면 **최소 질문 2개만**
"""

        # ------------------------------------------------------------------
        # 2. 새 프로그램 만들기 (처음부터)
        # ------------------------------------------------------------------
        elif mode == '2':
            if not content:
                messagebox.showwarning("입력 필요", "만들고 싶은 프로그램을 말로 적어주세요. (예: '수학문제 OCR 후 폴더정리 GUI')")
                return

            prompt = header + f"""
[작업: 새 프로그램 만들기]
목표:
{content}

요구:
1) 먼저 요구사항을 5줄로 재정리해줘 (내 말이 모호하면 가정 3개만 적고 진행)
2) 바로 실행 가능한 **완성 코드**를 작성해줘
3) 코드 주석은 한글 우선(이유/흐름 중심)
4) 설치가 필요한 패키지는 맨 위에 `pip install ...`로 정리
5) 400줄이 넘으면 Part 1/2로 나눠줘
"""

        # ------------------------------------------------------------------
        # 3. 기능 추가하기 (기존 코드에 덧붙이기)
        # ------------------------------------------------------------------
        elif mode == '3':
            if not content:
                messagebox.showwarning("입력 필요", "기존 코드 + 추가하고 싶은 기능을 같이 입력하세요.")
                return

            prompt = header + f"""
[작업: 기능 추가하기]
입력(기존 코드 + 원하는 기능):
{content}

제약:
- 기존 동작을 최대한 유지
- 전체 재작성 금지(요청 시만)

출력:
1) 변경 포인트 요약(5줄)
2) **Unified Diff** 우선
   - diff가 어렵다면 '수정된 함수 전체' 방식으로
3) 추가한 기능에 대해 간단한 테스트 방법 3개
"""

        # ------------------------------------------------------------------
        # 4. 개념/코드 이해하기 (쉬운 설명)
        # ------------------------------------------------------------------
        elif mode == '4':
            if not content:
                messagebox.showwarning("입력 필요", "AI가 만든 코드/설명/에러 메시지를 붙여넣으세요.")
                return

            prompt = header + f"""
[작업: 개념/코드 이해하기 (쉬운 설명)]
입력:
{content}

출력 형식:
1) 전체 요약 5줄 (초등교사처럼 쉬운 말)
2) 핵심 개념 3개:
   - 개념 이름:
   - 2줄 설명:
   - 이 코드에서 어디에 쓰였는지(한 줄):
3) 이 코드/설명이 하는 일을 '입력 → 처리 → 출력' 흐름으로 5줄
4) 내가 다음에 무엇을 물어보면 좋은지 질문 3개
"""

        # ------------------------------------------------------------------
        # 5. 논리 점검 (시뮬레이션/엣지케이스)
        # ------------------------------------------------------------------
        elif mode == '5':
            if not content:
                messagebox.showwarning("입력 필요", "검토할 로직/코드/기획을 붙여넣으세요.")
                return

            prompt = header + f"""
[작업: 논리 점검 (시뮬레이션/엣지케이스)]
입력:
{content}

요구:
1) 정상 케이스 1개 / 엣지 케이스 1개 / 실패 케이스 1개를 만들어서
   '손으로 실행하듯' 단계별로 따라가줘
2) 실패 가능 지점 5개(우선순위)
3) 빠르게 안정화하려면 추가해야 할 방어코드/검증 5개
4) 테스트 체크리스트 7개
"""

        # ------------------------------------------------------------------
        # 0. 대화/작업 리셋 요약 (짧게 정리)
        # ------------------------------------------------------------------
        elif mode == '0':
            prompt = """
[System Command: Summarize Context]
👉 대화/작업 맥락이 길어졌습니다. 아래 형식으로 짧게 정리해줘.

출력(한국어, 20줄 이내):
- 지금까지 한 일(완료): 3~6개
- 현재 문제/막힌 점: 1~3개
- 다음 액션(내가 할 것): 3개
- AI에게 다음에 시킬 정확한 질문 1개(복붙용)
"""
        else:
            messagebox.showwarning("알 수 없음", f"지원하지 않는 모드: {mode}")
            return

        # 결과 출력 및 복사

        self.set_output(prompt)

if __name__ == "__main__":
    root = tk.Tk()
    app = UniversalPromptConsoleGUI(root)
    root.mainloop()