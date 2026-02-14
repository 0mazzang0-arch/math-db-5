# gemini_api.py
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, InternalServerError, ServiceUnavailable
import json
import re
import warnings
import time
import base64
import ast
from PIL import Image
from openai import OpenAI

from config import (
    GOOGLE_API_KEYS, OPENAI_API_KEY, MODEL_NAME_OCR, MODEL_NAME_ANALYSIS, OPENAI_MODEL_NAME,
    INSIGHT_SYSTEM_PROMPT, CONCEPT_SYSTEM_PROMPT, OCR_SYSTEM_PROMPT
)

warnings.filterwarnings("ignore")

# [비용 절감] GPT 사용 여부 스위치 (현재 False 권장)
USE_GPT_FALLBACK = False 

# ==========================================================
# [EXTREME PROMPT V30] THE "EVERYTHING" PROMPT
# ==========================================================
# 기존의 모든 태그(LIST 포함)를 살리고, Teacher's Decoding 포맷을 적용합니다.
# AI Annotation만 Teacher's Decoding 내부로 통합됩니다.
# ==========================================================
TAGGED_SYSTEM_PROMPT = r"""
# Role Definition
You are a "Forensic Mathematical Logic Auditor" (디지털 포렌식 수학 논리 감사관) and a "Top-Tier Mathematical Strategist".
Your duty is to extract content from handwritten math solutions with **Zero Tolerance for Omission**, and then independently generate universal mathematical strategies.

**CORE DIRECTIVE (THE PRIME DIRECTIVE):**
1. **NO SUMMARIZATION:** You are FORBIDDEN from summarizing. You must transcribe every detail.
2. **VARIABLE MAPPING (NEW):** You must first define symbols as variables (Step 1), then use them in the narrative (Step 2).
3. **HYBRID SEPARATION OF CONCERNS (CRITICAL):**
   * The user's handwriting (arrows `->`, notes, symbols) represents the **"Teacher's View"**.
   * The `ACTION_PROTOCOL` must act as a **Hybrid**. You MUST look at the Teacher's arrows/logic from SECTION A, use them as your foundation (Anchor), and then expand upon them to propose universal mathematical rules (AI's Proposal).

   ### [CRITICAL ADDITION] ANSWER EXTRACTION
* **Target:** You MUST identify the final answer of the problem.
* **Format:** Extract ONLY the final value (e.g., "3", "5", "42", "3\sqrt{2}", "④").
* **Location:** Look at the end of the solution or inside the `[[GOAL]]` section.
* **Processing (DB):** Put this extracted value into the 'correct_answer' column.
---

# [PART 1] DETAILED EXTRACTION PROTOCOLS

## STEP 1: SYMBOL DEFINITION (Mapped from Teacher's View)
**Goal:** Create a `[[SYMBOL_TABLE]]` that functions as BOTH (1) a dictionary AND (2) a strategic decoder.
**Instruction (ABSOLUTE 4-COLUMN RULE):**
- You MUST output rows in this EXACT 4-column format:
  **`[Symbol] | [Type] | [Verbatim Content] | [Strategic Commentary]`**
- Column meanings:
  1) **Symbol:** 이미지에서 보이는 라벨(예: ①, (가), (핵), (특), (구), Sol1, Sol2, ⚡ 등)
  2) **Type:** 반드시 아래 5개 중 하나로만 선택  
     **Condition / Goal / Key / Trap / Strategy**
  3) **Verbatim Content (NO SUMMARIZATION):** 기호 옆에 적힌 원문을 **그대로** 옮긴다. (의역/요약 금지)
  4) **Strategic Commentary (전략 코멘터리):** 단순 설명이 아니라 **왜 이게 중요한지/다음 행동/함정**을 1~2문장으로 찌른다.
     - 권장 형식: **트리거(신호) → 즉시 행동(도구/치환) → 체크(함정/검증)**
     - **추측 금지:** 원문이 불명확하거나 보이지 않으면 invent 하지 말고 `Unknown`으로 표기.

### 🔍 TARGET TRIGGERS (Do NOT omit)
1) **CONDITIONS (조건) -> Type="Condition"**
   - Look for: `①`, `②`, `③`, `④`, `⑤`, `(가)`, `(나)`, `(다)`, `⚡`, 그리고 “조건”처럼 조건을 명시하는 메모.
   - Verbatim Content: 해당 기호 옆의 조건/식/문장을 그대로.

2) **GOAL (구하는 목표) -> Type="Goal"**
   - Look for: `㊈`, `(구)`, `🎯`, 또는 “구하시오/찾아라/값”처럼 목표를 지정하는 표기.
   - Verbatim Content: 목표 문장을 그대로.

3) **KEY IDEA (핵심) -> Type="Key"**
   - Look for: `㊄`, `(핵)`, `🔑` 또는 핵심 도구를 강조한 표기.
   - Verbatim Content: 핵심 도구/정리/발상을 그대로.

4) **SPECIAL POINT / TRAP (특이점/함정) -> Type="Trap"**
   - Look for: `㊕`, `(특)`, `❗` 또는 함정/예외를 강조한 표기.
   - Verbatim Content: 예외 조건/주의점 메모를 그대로.

5) **STRATEGY / SOLUTION SWITCH (풀이 전략/모드) -> Type="Strategy"**
   - Look for: `Sol1`, `Sol2`, `전략`, `방법`, “정공법/여사건/케이스분류” 같은 풀이 모드 라벨.
   - Verbatim Content: 해당 라벨 옆의 설명을 그대로.

### ✅ NEGATIVE CONSTRAINT (누락 방지 규칙)
- 위 트리거 목록에 없더라도, **기호가 “라벨/번호/풀이 분기 표시”로 기능**한다면 반드시 추출하라.
- 단, 의미를 억지로 채우지 말고, 원문이 불명확하면 **Verbatim Content에 `Unknown`**으로 남겨라.

---

---

## STEP 2: LOGIC NARRATIVE (Evidence-Based Substitution)
**Goal:** Create a `[[LOGIC_NARRATIVE]]` as an evidence-backed proof flow.
**Instruction:** Use symbols from STEP 1, and write every step in "Evidence → Conclusion/Action" form (NO hallucination, NO missing links).

# 👇 [여기서부터 덮어씌우세요 (화살표 조건 삭제 -> 대괄호 절대 규칙 적용)] 👇
### 1. NECESSITY (필연성) -> [[LOGIC_NARRATIVE]]
* **Trigger:** Text enclosed in square brackets `[...]` or marked with `(필)`.
* **Strict Rule:** The user has declared that **ANY text inside `[...]` is "Necessity"**.
    * If you see `[Because of this...]`, treat it immediately as the logical reason.
    * Arrows (`->`) are optional. The bracket `[...]` is the absolute identifier.
* **Substitution Rule:** You MUST use the format **`Symbol(Definition)`**.
    * Example: "**①(Condition)** leads to **[Necessity](Using Formula X)**."

### 2. ACTION PROTOCOL (AI가 제안하는 필연성 & 행동강령) -> [[ACTION_PROTOCOL]]
* **Target:** HYBRID ANCHOR & EXPAND STRATEGY.
* **Instruction (ABSOLUTE FORMAT RULE):**
  - You MUST output Action Protocol as a list of **atomic rules**.
  - **EACH rule MUST contain exactly these 3 labeled lines** (do not omit):
    1) **트리거(Trigger):** The exact signal phrase/pattern from the problem or teacher's notes (e.g., "순서가 정해진", "~사이에", "적어도/최소", "[...]" necessity bracket, etc.).
    2) **행동(Action):** The immediate mathematical transformation/tool to apply (e.g., "자리선택 → 조합", "칸막이 변수설정 → Stars & Bars", "여사건으로 전환", etc.).
    3) **체크(Check):** The most common trap/exception/validation step that prevents wrong counting (e.g., "양끝 포함 여부", "변수 치환 y≥k → y'=y-k", "불가능 구간 컷", etc.).
* **Instruction (QUALITY / NO-LOSS GUARANTEE):**
  - Do NOT shorten content. **Do NOT reduce the number of ideas.**
  - If the teacher used arrows / necessity flow, you MUST anchor rules to that flow, THEN expand to universal reusable rules.
  - Output **at least 3 rules**. If more are needed, output more. Never output fewer than 3.
  - Write in Korean, and keep each rule crisp but complete (Trigger→Action→Check must all be meaningful).

### 3. STRATEGY (전략 로드맵) -> [[LOGIC_NARRATIVE]]
* **Target:** Macro-level Step-by-Step Workflow.
* **Instruction:** Provide a clear, numbered 1-2-3-4 roadmap. Translate any circled numbers into their actual mathematical meanings.

---

## STEP 3: INDEPENDENT MODULES (The Safety Net)
**Instruction:** Extract these sections exactly as is (No structural change).

### 1. PRACTICAL CONCEPTS -> [[PRACTICAL_CONCEPTS]]
* **Trigger:** `㉦` or `(실)`. Format: `Title: ... || Content: ...`

### 2. BASIC CONCEPTS -> [[BASIC_CONCEPTS]]
* **Trigger:** `㊂` or `(기)`. Basic definitions used.

### 3. FIGURE ANALYSIS -> [[FIGURE_ANALYSIS]]
* **Target:** Description of graphs or geometric figures.

### 4. VERBATIM -> [[VERBATIM]]
* **Target:** ALL handwriting. Strict LaTeX. No Korean inside `$`. Every pixel must be translated to LaTeX.

### 5. SUPPLEMENTARY LISTS (Safety Net)
* **Instruction:** If multiple Key Ideas or Special Points exist that didn't fit the Symbol Table, YOU MUST LIST THEM in their own independent tags (`[[KEY_IDEAS_LIST]]`, `[[SPECIAL_POINTS_LIST]]`).
* **Constraint:** Do NOT summarize them into the Database Columns. Keep them raw and detailed.

# 👇 [TAGGED_SYSTEM_PROMPT의 맨 아래 부분을 이것으로 덮어씌우세요] 👇
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# [PART 2] OUTPUT FORMAT (STRICT TAG SYSTEM)

**Generate output strictly in KOREAN.**

# 🚨 **CRITICAL OUTPUT RULES (DO NOT IGNORE):**
1. **NO OMISSION:** You MUST output **ALL** the tags listed below. Do not skip any section.
2. **EMPTY HANDLING:** If you have no content for a section, write `Unknown` or `None` inside the tags. **NEVER omit the tags themselves.**
3. **MANDATORY TAGS:** Specifically, `[[VERBATIM_START]]`, `[[AI_SOLUTION_START]]`, and `[[STRATEGY_START]]` are **REQUIRED** for the system to work. If they are missing, the system crashes.

[[STRATEGY_START]]
(Step-by-step Roadmap: 1. ... 2. ...)
[[STRATEGY_END]]

[[SYMBOL_TABLE_START]]
(Format: Symbol | Meaning | AI Comment)
Example:
① | a,b,c는 음이 아닌 정수 | 변수 범위 제한 확인 필수
(핵) | 중복조합(H)의 활용 | 서로 다른 n개에서 중복 허용 r개 선택
[[SYMBOL_TABLE_END]]

[[LOGIC_NARRATIVE_START]]
(ABSOLUTE FORMAT: Evidence ▶ Conclusion. NOT a wall of text.)
- Write as bullet points. EACH bullet MUST be a complete "근거 → 결론/행동" unit.
- Strict template per bullet:
  * **[키워드/단계]** (Evidence: 원문에서 보이는 조건/표식/식/메모를 짧게 지목 또는 인용) → **(Conclusion/Action: 수학적 결론 또는 다음 행동/도구)**
- Rules (NO FUNCTIONAL SUMMARY / NO HALLUCINATION):
  1) Evidence 없는 결론 금지(추측 금지).
  2) 결론은 반드시 “무엇을 할지(행동/도구/치환/케이스 분류)”로 연결되어야 함.
  3) **NO FUNCTIONAL SUMMARY:** 케이스 분기, 변수 치환, 포함/배제, 중간 계산 결과를 **절대 생략하지 말 것**. (학생이 그대로 따라 적을 수 있을 정도로)
  5) **MINIMUM STEPS:** 최소 8개의 bullet을 출력하라. 부족하면 케이스/치환/계산/결론을 쪼개서 bullet 수를 늘려라.
  4) 원문이 불명확하면 Evidence는 `Unknown`으로 표기하고, 임의로 채우지 말 것.
- Example:
  * **[상황 파악]** “(가) 흰색 카드는 작은 수부터 크기순” (Evidence) → **순서열거가 아니라 ‘자리선택(조합)’으로 환원** (Conclusion)
  * **[변수 세팅]** “(나) 검은 카드 사이 흰색 카드 ≥ 2장” (Evidence) → **y≥2 치환 후 Stars & Bars 적용** (Action)
[[LOGIC_NARRATIVE_END]]


# -------------------------------------------------------
# [LEGACY TAGS FOR PARSER COMPATIBILITY - DO NOT OMIT]
# IMPORTANT: Even if content is empty, you MUST output the tags and put "Unknown" inside.
# -------------------------------------------------------

**MANDATORY FILL:** If any relevant evidence exists in VERBATIM / SYMBOL_TABLE / LOGIC_NARRATIVE, you MUST copy it into the corresponding legacy tag. Do NOT leave it Unknown when evidence exists.


[[NECESSITY_START]]
(필연성: 반드시 채워라. 아래 규칙으로 **SYMBOL_TABLE/Teacher's Decoding에서 직접 가져와라**. 없으면 Unknown)
(RULE-N: 다음 중 하나라도 있으면 반드시 채운다)
- **대괄호[...] 안 문장**은 1순위로 Necessity에 복사
- **(필)** 표시가 있는 문장은 2순위로 Necessity에 복사
- **화살표(->)로 연결된 원인→결과 문장**은 3순위로 Necessity에 복사
- 위가 하나도 없더라도, Teacher's Decoding에서 **"type이 Condition이고 코멘트가 '따라서/그러므로/필연'류"**면 Necessity로 복사
(Format 강제: 반드시 `Symbol | Content | AI_Interpretation` 여러 줄로 작성)
[[NECESSITY_END]]

[[KEY_IDEA_START]]
(RULE-K1: Teacher's Decoding에서 symbol이 (핵) / ㊄ / 🔑 인 행이 **하나라도** 있으면, 그 행(들)을 그대로 Key Idea에 **반드시 복사**하라.)
(RULE-K2: (핵) 표식이 없어도, Teacher's Decoding의 메모(OCR) 또는 Logic Narrative/Verbatim에 '중복조합/여사건/케이스분류/칸막이/Stars and Bars/포함-배제' 같은 **도구명**이 나타나면 그 줄을 Key Idea에 **반드시 복사**하라.)
(OUTPUT MINIMUM: Key Idea는 최소 1줄 이상 출력하라. 근거가 전혀 없으면 `Unknown` 1줄을 출력하라. 태그를 비우지 말 것.)
(Format 강제: `Symbol | Content | AI_Interpretation` 각 줄)
[[KEY_IDEA_END]]

[[SPECIAL_POINT_START]]
(RULE-S1: Teacher's Decoding에서 symbol이 (특) / ㊕ / ❗ 인 행이 **하나라도** 있으면, 그 행(들)을 그대로 Special Point에 **반드시 복사**하라.)
(RULE-S2: (특) 표식이 없어도, Teacher's Decoding의 메모(OCR) 또는 Logic Narrative/Verbatim에 '함정/주의/겹침/중복/배제 누락/케이스 누락/등호 포함 여부/0 포함 여부' 같은 **경고 메모**가 나타나면 그 줄을 Special Point에 **반드시 복사**하라.)
(OUTPUT MINIMUM: Special Point는 최소 1줄 이상 출력하라. 근거가 전혀 없으면 `Unknown` 1줄을 출력하라. 태그를 비우지 말 것.)
(Format 강제: `Symbol | Content | AI_Interpretation` 각 줄)
[[SPECIAL_POINT_END]]

[[GOAL_START]]
(구하는 목표: (구)/㊈/🎯 또는 “구하시오” 문장을 적어라. 없으면 Unknown)
(Format 권장: Symbol | Content | AI_Interpretation)
[[GOAL_END]]

[[CONDITIONS_START]]
(조건: ①②③… 또는 (가)(나)(다) 등의 조건 문장을 적어라. 없으면 Unknown)
(Format 권장: Symbol | Content | AI_Interpretation)
[[CONDITIONS_END]]


[[ACTION_PROTOCOL_START]]
(Format: MUST be a numbered list of rules. EACH rule MUST have 3 labeled lines.)
1) 트리거(Trigger): ...
   행동(Action): ...
   체크(Check): ...
2) 트리거(Trigger): ...
   행동(Action): ...
   체크(Check): ...
(Write at least 3 rules. Do NOT omit any of the 3 lines per rule.)
[[ACTION_PROTOCOL_END]]

[[PRACTICAL_CONCEPTS_START]]
(Format: Title: ... || Content: ...)
[[PRACTICAL_CONCEPTS_END]]

[[BASIC_CONCEPTS_START]]
(Basic definitions)
[[BASIC_CONCEPTS_END]]

[[FIGURE_ANALYSIS_START]]
(Graph description)
[[FIGURE_ANALYSIS_END]]

[[CORRECT_ANSWER_START]]
(Extracted final answer only, e.g., 3, 5, 149)
[[CORRECT_ANSWER_END]]

[[VERBATIM_START]]
(Pixel-perfect transcription)
[[VERBATIM_END]]

[[AI_SOLUTION_START]]
(Standard solution)
[[AI_SOLUTION_END]]

[[KEY_IDEAS_LIST_START]]
(Supplementary list for safety: List ALL extra key ideas here)
[[KEY_IDEAS_LIST_END]]

[[SPECIAL_POINTS_LIST_START]]
(Supplementary list for safety: List ALL extra special points here)
[[SPECIAL_POINTS_LIST_END]]

[[DB_COLUMNS_START]]
ABSOLUTE RULE: DB_COLUMNS는 **요약이 아니라 복사**다. 아래 레거시 태그의 내용을 그대로 복붙하라. (Do NOT paraphrase. Do NOT shorten.)
- necessity := [[NECESSITY_START]] 내부 내용 그대로 (태그 안이 Unknown이면 Unknown 그대로)
- key_idea := [[KEY_IDEA_START]] 내부 내용 그대로 (Unknown이면 Unknown 그대로)
- special_point := [[SPECIAL_POINT_START]] 내부 내용 그대로 (Unknown이면 Unknown 그대로)

necessity: (여기에 necessity를 위 규칙대로 그대로 복사)
key_idea: (여기에 key_idea를 위 규칙대로 그대로 복사)
special_point: (여기에 special_point를 위 규칙대로 그대로 복사)
correct_answer: (정답이 있으면 정답, 없으면 Unknown)
[[DB_COLUMNS_END]]

[[DEEP_INSIGHT_START]]
(Leave empty)
[[DEEP_INSIGHT_END]]
"""

gpt_client = None
if OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"):
    try: gpt_client = OpenAI(api_key=OPENAI_API_KEY)
    except: pass

CURRENT_KEY_INDEX = 0

def initialize_api():
    global CURRENT_KEY_INDEX
    if not GOOGLE_API_KEYS: return
    genai.configure(api_key=GOOGLE_API_KEYS[CURRENT_KEY_INDEX])

def rotate_api_key():
    global CURRENT_KEY_INDEX
    if len(GOOGLE_API_KEYS) <= 1: return False
    prev_index = CURRENT_KEY_INDEX
    CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(GOOGLE_API_KEYS)
    genai.configure(api_key=GOOGLE_API_KEYS[CURRENT_KEY_INDEX])
    print(f"\n🔄 [Quota] API Key 교체 완료 (Key {prev_index + 1} -> Key {CURRENT_KEY_INDEX + 1})")
    return True

initialize_api()
REQUEST_OPTIONS = {"timeout": 600}

# [모델 설정]
analysis_model = genai.GenerativeModel(model_name=MODEL_NAME_ANALYSIS, system_instruction=TAGGED_SYSTEM_PROMPT)
search_model = genai.GenerativeModel(model_name=MODEL_NAME_OCR, system_instruction=OCR_SYSTEM_PROMPT, generation_config={"temperature": 0.0})
insight_model = genai.GenerativeModel(model_name=MODEL_NAME_ANALYSIS, system_instruction=INSIGHT_SYSTEM_PROMPT)
concept_model = genai.GenerativeModel(model_name=MODEL_NAME_ANALYSIS, system_instruction=CONCEPT_SYSTEM_PROMPT)

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f: return base64.b64encode(f.read()).decode('utf-8')

# ==========================================================
# [JSON Helper Functions] - [복구 완료] 삭제되었던 함수들 100% 복구
# ==========================================================
def clean_json_text(text):
    if not text: return ""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```', '', text)
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx : end_idx+1]
    return text.strip()

def repair_json_content(text):
    if not text: return "{}"
    text = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', text)
    def replace_newlines_in_string(match):
        content = match.group(1).replace('\n', '\\n').replace('\r', '')
        return f'"{content}"'
    try:
        text = re.sub(r'"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', replace_newlines_in_string, text, flags=re.DOTALL)
    except: pass
    return text

def try_advanced_parsing(text):
    try: return json.loads(text)
    except:
        try: return json.loads(repair_json_content(text))
        except:
            try:
                py_text = text.replace("true", "True").replace("false", "False").replace("null", "None")
                return ast.literal_eval(py_text)
            except: return None

def parse_broken_json(text):
    # [복구 완료] 혹시 모를 JSON 포맷 에러 시 강제 추출용 함수
    print("⚠ [Warning] JSON 파싱 실패. 정규표현식으로 강제 추출을 시도합니다.")
    fallback_data = {
        "db_columns": {"necessity": "", "key_idea": "", "special_point": ""},
        "body_content": {}
    }
    patterns = {
        "necessity": r'"necessity"\s*:\s*"([^"]+)"',
        "key_idea": r'"key_idea"\s*:\s*"([^"]+)"',
        "goal": r'"goal"\s*:\s*"([^"]+)"',
        "ai_solution": r'"ai_solution"\s*:\s*"([^"]+)"'
    }
    for key, pat in patterns.items():
        match = re.search(pat, text)
        if match:
            val = match.group(1)
            if key in ["necessity", "key_idea"]:
                fallback_data["db_columns"][key] = val
            else:
                fallback_data["body_content"][key] = val
    return fallback_data

# ==========================================================
# [The Monster Parser V30] Manual & Verbose Extraction (Restored & Enhanced)
# ==========================================================
def parse_tagged_response(text):
    print("🚜 [Parser V30] 데이터 추출 시작 (Manual & Verbose Mode)...")
    
    data = {
        "db_columns": {"necessity": "", "key_idea": "", "special_point": "", "correct_answer": ""},
        "body_content": {
            "symbol_table": [],      # [V35] 기호 정의 테이블
            "logic_narrative": [],   # [V35] 논리 서술 리스트
            
            "key_ideas_list": [],    # [Safety] 추가 핵심 리스트
            "special_points_list": [], # [Safety] 추가 특이점 리스트
            
            "practical_concepts": [], "basic_concepts": [],     
            "figure_analysis": "",    "verbatim_handwriting": "", 
            "ai_solution": "",        "instructor_solution": "",
            "conditions": [], "goal": "" # Legacy 호환용
        }
    }

    # [Verbose Extraction Helper] - [복구 완료] 로그 및 부분 매칭 기능
    def extract_section(start_tag, end_tag, debug_name=None):
        # 1. 태그 정규화
        base_start = start_tag.replace("[", "").replace("]", "")
        base_end = end_tag.replace("[", "").replace("]", "")
        
        # 2. [Core] 정석 매칭 (괄호, 공백, 특수문자 무시하고 태그 찾기)
        pattern = r'[\#\*\s\[\]]*' + base_start + r'[\#\*\s\[\]]*(.*?)[\#\*\s\[\]]*' + base_end + r'[\#\*\s\[\]]*'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match: return match.group(1).strip()
        
        # 3. [Safety Net 1] Fallback (종료 태그를 AI가 빼먹었을 때)
        # 시작 태그부터... 다음 'START' 태그가 나오기 전까지 몽땅 긁어옴
        pattern_loose = r'[\#\*\s\[\]]*' + base_start + r'[\#\*\s\[\]]*(.*)'
        match_loose = re.search(pattern_loose, text, re.DOTALL | re.IGNORECASE)
        if match_loose:
            content = match_loose.group(1).strip()
            # 다음 섹션의 시작 태그가 보이면 거기서 자른다.
            next_tag_match = re.search(r'[\#\*\s\[\]]*[A-Z_]+_START[\#\*\s\[\]]*', content, re.IGNORECASE)
            if next_tag_match: return content[:next_tag_match.start()].strip()
            return content
            
        # 4. [Safety Net 2] Last Resort (AI 해설 전용)
        # 태그가 완전히 깨졌을 때 한글 키워드 'AI 해설'로 찾기
        if "AI_SOLUTION" in start_tag:
            alt_match = re.search(r'#+\s*AI\s*(정석\s*)?해설(.*?)(?=#+|$)', text, re.DOTALL | re.IGNORECASE)
            if alt_match: return alt_match.group(2).strip()
            
        return ""

# [Helper] 리스트 파싱 (불릿 포인트 제거 + 기호 보존 Fix)
    # [Helper] 리스트 파싱 (함수명 변경 및 로직 확정)
    def parse_list(raw_text):
        if not raw_text: return []
        lines = raw_text.split('\n')
        cleaned = []
        for line in lines:
            # [핀셋 수정] ①, (가) 보존 (불릿과 1. 2. 같은 번호만 제거)
            line = re.sub(r'^\s*([\*\-]\s*|\d+\.\s*)', '', line).strip()
            if line: cleaned.append(line)
        return cleaned

    # [복구 완료] 중복 제거 및 병합 로직 (리스트 합치기용)
    def merge_and_deduplicate(single_text, list_text):
        items = []
        # 1. Single text 처리
        if single_text: items.append(single_text)
        # 2. List text 처리
        if list_text:
            lines = clean_list(list_text)
            for line in lines:
                is_duplicate = False
                for existing in items:
                    if line in existing or existing in line: # 포함 관계 확인
                        is_duplicate = True
                        break
                if not is_duplicate:
                    items.append(line)
        return "\n".join(items)

    # [V30 New Logic] 파이프(|) 구분 Teacher's Decoding 파서
    def parse_teacher_decoding(raw_text, item_type):
        items = []
        if not raw_text: return items
        lines = raw_text.split('\n')
        for line in lines:
            if not line.strip(): continue
            parts = [p.strip() for p in line.split('|')]
            
            symbol = parts[0] if len(parts) > 0 else "Note"
            content = parts[1] if len(parts) > 1 else parts[0]
            ai_comment = parts[2] if len(parts) > 2 else ""
            
            if len(parts) == 1: 
                symbol = "Note"
                content = line
            
            items.append({
                "type": item_type,
                "symbol": symbol,
                "content": content,
                "ai_comment": ai_comment
            })
        return items

# 1. [STEP 1] Symbol Table 파싱
    raw_symbols = extract_section("SYMBOL_TABLE_START", "SYMBOL_TABLE_END")
    symbol_list = []

    if raw_symbols:
        for line in raw_symbols.splitlines():
            line = line.strip()
            if not line or line.startswith("Example") or line.startswith("("):
                continue
            if "|" not in line:
                # 포맷 깨진 줄도 보존 (누락 방지)
                symbol_list.append({
                    "symbol": "Unknown",
                    "type": "Trap",
                    "content": line,
                    "ai_comment": "(SYMBOL_TABLE 포맷 오류: '|' 없음)"
                })
                continue

            parts = [p.strip() for p in line.split("|")]

            # 4열 기대: Symbol | Type | Verbatim Content | Strategic Commentary
            sym = parts[0] if len(parts) > 0 else ""
            dtype = parts[1] if len(parts) > 1 else ""
            content = parts[2] if len(parts) > 2 else ""
            comment = parts[3] if len(parts) > 3 else ""

            # 누락 방지: 최소한이라도 채움
            if not dtype:
                dtype = "Condition"
            if not content:
                content = "Unknown"

            symbol_list.append({
                "symbol": sym,
                "type": dtype,
                "content": content,
                "ai_comment": comment
            })

    # ✅ 핵심: 이제부터 Notion은 teacher_decoding만 보면 4열이 항상 맞는다
    data["body_content"]["teacher_decoding"] = symbol_list

    # (선택) 구버전 호환을 위해 symbol_table도 남기고 싶으면 아래처럼 저장
    data["body_content"]["symbol_table"] = [
        {"symbol": x["symbol"], "meaning": x["content"], "comment": x["ai_comment"], "type": x["type"]}
        for x in symbol_list
    ]

    # [신규 복구] 전략 로드맵 & 행동 강령 추출
    data["body_content"]["strategy_overview"] = extract_section("STRATEGY_START", "STRATEGY_END")
    data["body_content"]["action_protocol"] = extract_section("ACTION_PROTOCOL_START", "ACTION_PROTOCOL_END")

    # 2. [STEP 2] Logic Narrative 파싱
    raw_logic = extract_section("LOGIC_NARRATIVE_START", "LOGIC_NARRATIVE_END")
    data["body_content"]["logic_narrative"] = parse_list(raw_logic)

        # -------------------------------------------------------
    # [V30+ Critical Fix] Legacy Tags -> DB Columns (NO SUMMARY, COPY ONLY)
    # -------------------------------------------------------
    legacy_necessity = extract_section("NECESSITY_START", "NECESSITY_END", "LegacyNecessity")
    legacy_key_idea = extract_section("KEY_IDEA_START", "KEY_IDEA_END", "LegacyKeyIdea")
    legacy_special = extract_section("SPECIAL_POINT_START", "SPECIAL_POINT_END", "LegacySpecial")

    def _clean_legacy_block(s: str) -> str:
        if not s:
            return ""
        s = s.strip()
        # 태그 안에 안내 문구만 있고 실제 내용이 없는 경우를 방지
        # (괄호로 시작하는 안내 라인들 제거는 "요약"이 아니라 "프롬프트 안내문 제거"임
        lines = []
        for line in s.splitlines():
            t = line.strip()
            if not t:
                continue
            # 프롬프트 안내문 패턴 제거 (필요 최소)
            if t.startswith("(") and t.endswith(")"):
                continue
            if t.lower().startswith("rule-"):
                continue
            lines.append(t)
        return "\n".join(lines).strip()

    legacy_necessity = _clean_legacy_block(legacy_necessity)
    legacy_key_idea = _clean_legacy_block(legacy_key_idea)
    legacy_special = _clean_legacy_block(legacy_special)

    data["body_content"]["legacy_necessity_raw"] = legacy_necessity
    data["body_content"]["legacy_key_idea_raw"] = legacy_key_idea
    data["body_content"]["legacy_special_point_raw"] = legacy_special

    # Unknown 처리 표준화 (빈칸 방지)
    def _normalize_unknown(s: str) -> str:
        if not s or not s.strip():
            return "Unknown"
        ss = s.strip()
        if ss.lower() == "unknown":
            return "Unknown"
        return ss

    def _to_db_index_string(items, max_len: int = 180) -> str:
        pieces = []
        for item in items:
            t = (item or "").strip()
            if not t:
                continue
            pieces.append(t)

        one_line = " / ".join(pieces).strip()
        one_line = _normalize_unknown(one_line)
        if one_line == "Unknown":
            return "Unknown"
        if len(one_line) > max_len:
            return one_line[:max_len - 3].rstrip() + "..."
        return one_line

    strict_key_contents = []
    strict_trap_contents = []
    strict_necessity_contents = []
    for row in symbol_list:
        dtype = (row.get("type") or "").strip().lower()
        content = (row.get("content") or "")
        ai_comment = (row.get("ai_comment") or "")

        if dtype == "key":
            if content and content.strip():
                strict_key_contents.append(content)
            if ai_comment and ai_comment.strip():
                strict_key_contents.append(ai_comment)
        if dtype == "trap":
            if content and content.strip():
                strict_trap_contents.append(content)
            if ai_comment and ai_comment.strip():
                strict_trap_contents.append(ai_comment)

        for source_text in (content, ai_comment):
            if not source_text:
                continue
            for bracket_text in re.findall(r"\[[^\[\]]+\]", source_text):
                strict_necessity_contents.append(bracket_text)

    # ✅ DB 컬럼은 teacher_decoding 증거 기반 Strict 규칙으로만 저장
    data["db_columns"]["necessity"] = _to_db_index_string(strict_necessity_contents)
    data["db_columns"]["key_idea"] = _to_db_index_string(strict_key_contents)
    data["db_columns"]["special_point"] = _to_db_index_string(strict_trap_contents)


    # 3. [Safety Nets] 독립 리스트 파싱
    data["body_content"]["key_ideas_list"] = parse_list(extract_section("KEY_IDEAS_LIST_START", "KEY_IDEAS_LIST_END", "KeyList"))

    data["body_content"]["special_points_list"] = parse_list(extract_section("SPECIAL_POINTS_LIST_START", "SPECIAL_POINTS_LIST_END", "SpecList"))
    # 4. [Independent Modules] 실전개념, 기본개념, 그래프, 정답, 원문
    pc_raw = extract_section("PRACTICAL_CONCEPTS_START", "PRACTICAL_CONCEPTS_END", "PracConcept")
    pc_list = []
    if pc_raw:
        for line in pc_raw.split('\n'):
            parts = re.split(r'\|\|', line)
            if len(parts) >= 2:
                pc_list.append({"title": parts[0].replace("Title:", "").strip(), "content": parts[1].replace("Content:", "").strip()})
    data["body_content"]["practical_concepts"] = pc_list

    data["body_content"]["basic_concepts"] = parse_list(extract_section("BASIC_CONCEPTS_START", "BASIC_CONCEPTS_END", "BasicConcept"))
    data["body_content"]["figure_analysis"] = extract_section("FIGURE_ANALYSIS_START", "FIGURE_ANALYSIS_END", "Figure")
    data["db_columns"]["correct_answer"] = extract_section("CORRECT_ANSWER_START", "CORRECT_ANSWER_END", "Answer")
    data["body_content"]["verbatim_handwriting"] = extract_section("VERBATIM_START", "VERBATIM_END", "Verbatim")
    data["body_content"]["ai_solution"] = extract_section("AI_SOLUTION_START", "AI_SOLUTION_END", "AISolution")
    data["body_content"]["instructor_solution"] = extract_section("DEEP_INSIGHT_START", "DEEP_INSIGHT_END", "Insight")

    return data

def execute_with_key_rotation(model, content, **kwargs):
    if "safety_settings" not in kwargs:
        kwargs["safety_settings"] = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

    max_attempts = len(GOOGLE_API_KEYS) + 1 
    for attempt in range(max_attempts):
        try:
            return model.generate_content(content, **kwargs)
        except (ResourceExhausted, InternalServerError, ServiceUnavailable) as e:
            print(f"⚠ Gemini API Error. 키 교체 시도... ({e})")
            if rotate_api_key(): time.sleep(2); continue 
            else: raise e
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print(f"⚠ 429 Rate Limit. 키 교체 시도...")
                if rotate_api_key(): time.sleep(2); continue
            raise e 
    raise Exception("All Gemini Keys Exhausted")

def analyze_image_structure(image_path):
    import concept_manager
    raw_concepts = concept_manager.load_concepts()
    concept_list_text = "\n".join([f"- {c['title']}: {c.get('content','...')[:50]}" for c in raw_concepts[:50]])
    
    print("\n--- [Stage 1] 구조 분석 (V30 Forensic Juggernaut Mode) 시작 ---")
    
    user_prompt = f"""
    Analyze the image strictly according to the 'Forensic Auditor' protocol.
    Fill all the tags in KOREAN.
    
    [Reference Concept DB Context]
    {concept_list_text}
    """
    
    try:
        response = execute_with_key_rotation(
            analysis_model, 
            [user_prompt, Image.open(image_path)],
            generation_config={"max_output_tokens": 8192, "temperature": 0.1},
            request_options=REQUEST_OPTIONS
        )
        result_data = parse_tagged_response(response.text)
        
        # [검증 로직 보존]
        body = result_data["body_content"]
        if not body["verbatim_handwriting"]:
            if body["strategy_overview"] or body["ai_solution"]:
                print("⚠️ [Warn] 손글씨(Verbatim)는 없지만, 해설/전략이 있어 통과합니다.")
            else:
                print("❌ [Error] 분석 데이터 전무 (Verbatim/Strategy/Solution 모두 없음).")
                raise Exception("Critical Extraction Failed (Empty Data)")
            
        print("✅ [Plan A] 태그 파싱 성공")

    except Exception as e:
        print(f"⚠ [Plan A 실패] {e}. [Plan B] GPT 용병 투입 여부 확인...")
        return None

    print("\n--- [Stage 2] 심층 추론 시작 ---")
    concept_db_text = json.dumps(concept_manager.load_concepts(), ensure_ascii=False, indent=2)
    base_sol = result_data.get("body_content", {}).get("ai_solution", "")
    
    # [Insight 생성 로직 보존]
    # (주의) 파서가 더미 텍스트를 잡아와서 if문이 False가 되는 치명적 버그를 원천 차단함.
    # 더미 텍스트가 있든 없든, 무조건 Stage 2 독립 에이전트를 가동하여 덮어씌움!
    deep_insight = generate_deep_insight(image_path, base_sol, concept_db_text) 
    result_data["body_content"]["instructor_solution"] = deep_insight
    
    print("✅ 분석 완료.")
    return result_data

def generate_deep_insight(image_path, base_solution_text, concept_db_text):
    final_prompt = INSIGHT_SYSTEM_PROMPT.replace("[USER_CONCEPT_DB]", concept_db_text)
    user_prompt = f"[Standard Solution]:\n{base_solution_text}\n\nBased on the image and solution, provide the 1-Tier Insight."
    try:
        dynamic_insight_model = genai.GenerativeModel(model_name=MODEL_NAME_ANALYSIS, system_instruction=final_prompt)
        response = execute_with_key_rotation(
            dynamic_insight_model, [user_prompt, Image.open(image_path)], request_options=REQUEST_OPTIONS
        )
        return response.text.strip()
    except: return "심층 분석 생성 실패"

def call_gpt4o_fallback(system_instr, user_instr, image_path):
    return None

def get_pure_ocr_text(image_path):
    try:
        response = execute_with_key_rotation(
            search_model, ["Execute OCR.", Image.open(image_path)], request_options=REQUEST_OPTIONS
        )
        return response.text.replace("```latex", "").replace("```", "").strip()
    except: return None

def extract_concepts_flexible(image_path):
    try:
        response = execute_with_key_rotation(
            concept_model, ["Extract concepts strictly in JSON.", Image.open(image_path)],
            generation_config={"response_mime_type": "application/json"}, request_options=REQUEST_OPTIONS
        )
        cleaned = clean_json_text(response.text)
        return json.loads(cleaned)
    except: return None
    
# [NEW] A단계(단순계산) 판독기 - Track B 전용
def check_is_basic_drill(text):
    if not text or len(text) < 5: return False
    try:
        # Flash 모델을 사용하여 빠르게 판단 (True/False)
        prompt = f"""
        Role: Math Problem Classifier.
        Task: Analyze the text and determine if it is a "Simple Calculation Drill" (A-step,단순 연산,기초 문제).
        Text: {text[:500]}
        Output: Return ONLY 'TRUE' if it is a simple drill, 'FALSE' otherwise.
        """
        resp = execute_with_key_rotation(search_model, [prompt], request_options=REQUEST_OPTIONS)
        return "TRUE" in resp.text.strip().upper()
    except: return False

# [NEW] 시대인재급 난이도 판독기 (Level 1~4 Classifier)
def analyze_difficulty_level(text):
    if not text: return "기본개념"
    try:
        # Flash 모델에게 '문제의 관상(Heuristics)'을 보고 판단하라고 지시
        prompt = f"""
        Role: Math Problem Difficulty Classifier.
        Task: Classify the difficulty of the given math problem text into one of 4 levels.
        
        [Criteria]
        1. LEVEL_1 (Basic Concept): Short text (1-3 lines), asks for simple calculation or basic definition.
        2. LEVEL_2 (Entry Semi-Killer): Standard 4-point problem. Has 1-2 conditions. Typical textbook style.
        3. LEVEL_3 (Deep Semi-Killer): Hard 4-point. Keywords: "Defined function g(x)", "Differentiability", "Select all correct (ㄱ,ㄴ,ㄷ)", "Fill in the blank". Requires logical deduction.
        4. LEVEL_4 (Killer): Very long text, complex conditions, new function definitions, finding Max/Min in complex situations. 

        Input Text:
        {text[:800]}

        Output: ONLY return one word: "LEVEL_1", "LEVEL_2", "LEVEL_3", or "LEVEL_4".
        """
        resp = execute_with_key_rotation(search_model, [prompt], request_options=REQUEST_OPTIONS)
        result = resp.text.strip().upper()
        
        if "LEVEL_4" in result: return "킬러"
        if "LEVEL_3" in result: return "준킬러_심화"
        if "LEVEL_2" in result: return "준킬러_진입"
        return "기본개념"
    except: return "기본개념"
