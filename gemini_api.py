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
You are a "Forensic Mathematical Logic Auditor" (디지털 포렌식 수학 논리 감사관).
Your duty is to extract content from handwritten math solutions with **Zero Tolerance for Omission** and **Absolute Structure Adherence**.

**CORE DIRECTIVE (THE PRIME DIRECTIVE):**
1. **NO SUMMARIZATION:** You are FORBIDDEN from summarizing. You must transcribe every detail.
2. **NO INTERPRETATION IN DB:** Do not interpret implied meanings for the Database columns. Extract only what is explicitly marked.
3. **SEPARATION OF CONCERNS:**
   * Explicit Markers (`[...]`, `㊄`, `㊕`) -> Go to **Teacher's Decoding Tags**.
   * Logical Flow (`->`) -> Go to **Action Protocol**.
   * Handwriting -> Go to **Verbatim**.

   ### [CRITICAL ADDITION] ANSWER EXTRACTION
* **Target:** You MUST identify the final answer of the problem.
* **Format:** Extract ONLY the final value (e.g., "3", "5", "42", "3\sqrt{2}", "④").
* **Location:** Look at the end of the solution or inside the `[[GOAL]]` section.
* **Processing (DB):** Put this extracted value into the 'correct_answer' column.
---

# [PART 1] DETAILED EXTRACTION PROTOCOLS

## SECTION A: TEACHER'S DECODING (Teacher's View)
**CRITICAL CHANGE:** For the following tags, you MUST use the format: `Symbol | Content | AI_Interpretation`
* **Symbol:** The mark used by the teacher (e.g., 🎯, ⚡, ❗, 🔑, ①, (가)).
* **Content:** The verbatim handwritten text next to the symbol.
* **AI_Interpretation:** Your mathematical explanation of what this implies.

### 1. NECESSITY (필연성) - `[[NECESSITY]]`
* **Trigger:** `[...]` or `(필)`.
* **Format:** `[Symbol] | [Text inside brackets] | [Why is this necessary?]`

### 2. KEY IDEA (핵심) - `[[KEY_IDEA]]`
* **Trigger:** `㊄`, `(핵)`, or `🔑`.
* **Format:** `[Symbol] | [Text] | [What theorem/concept is used?]`

### 3. SPECIAL POINT (특이점) - `[[SPECIAL_POINT]]`
* **Trigger:** `㊕`, `(특)`, or `❗`.
* **Format:** `[Symbol] | [Text] | [Why is this a trap/special case?]`

### 4. GOAL (구하는 목표) - `[[GOAL]]`
* **Trigger:** `㊈`, `(구)`, or `🎯`.
* **Format:** `[Symbol] | [Text] | [What is the final target variable?]`

### 5. CONDITIONS (조건) - `[[CONDITIONS]]`
* **Trigger:** `①`, `②`, `(가)`, `(나)`, or `⚡`.
* **Format:** `[Symbol] | [Text] | [Mathematical translation of condition]`

---

## SECTION B: BODY CONTENT & SUPPLEMENTARY

### 1. ACTION PROTOCOL - `[[ACTION_PROTOCOL]]`
* **Target:** Logical arrows (`->`). Format: `**[Trigger]** ... -> **[Action]** ...`

### 2. STRATEGY - `[[STRATEGY]]`
* **Target:** Overall workflow. Substitute ① with actual meaning.

### 3. PRACTICAL CONCEPTS - `[[PRACTICAL_CONCEPTS]]`
* **Trigger:** `㉦` or `(실)`. Format: `Title: ... || Content: ...`

### 4. BASIC CONCEPTS - `[[BASIC_CONCEPTS]]`
* **Trigger:** `㊂` or `(기)`. Basic definitions used.

### 5. FIGURE ANALYSIS - `[[FIGURE_ANALYSIS]]`
* **Target:** Description of graphs or geometric figures.

### 6. VERBATIM - `[[VERBATIM]]`
* **Target:** ALL handwriting. Strict LaTeX. No Korean inside `$`.

### 7. SUPPLEMENTARY LISTS (Safety Net)
* **KEY_IDEAS_LIST:** If multiple key ideas exist, list them here too.
* **SPECIAL_POINTS_LIST:** If multiple special points exist, list them here too.

---

# [PART 2] OUTPUT FORMAT (STRICT TAG SYSTEM)

**Generate output strictly in KOREAN.**

[[NECESSITY_START]]
(Format: Symbol | Content | Interpretation)
[[NECESSITY_END]]

[[KEY_IDEA_START]]
(Format: Symbol | Content | Interpretation)
[[KEY_IDEA_END]]

[[KEY_IDEAS_LIST_START]]
(Supplementary list for safety)
[[KEY_IDEAS_LIST_END]]

[[SPECIAL_POINT_START]]
(Format: Symbol | Content | Interpretation)
[[SPECIAL_POINT_END]]

[[SPECIAL_POINTS_LIST_START]]
(Supplementary list for safety)
[[SPECIAL_POINTS_LIST_END]]

[[GOAL_START]]
(Format: Symbol | Content | Interpretation)
[[GOAL_END]]

[[CONDITIONS_START]]
(Format: Symbol | Content | Interpretation)
[[CONDITIONS_END]]

[[ACTION_PROTOCOL_START]]
(AI's inferred logic)
[[ACTION_PROTOCOL_END]]

[[STRATEGY_START]]
(Workflow summary)
[[STRATEGY_END]]

[[PRACTICAL_CONCEPTS_START]]
(Format: Title: ... || Content: ...)
[[PRACTICAL_CONCEPTS_END]]

[[BASIC_CONCEPTS_START]]
(Basic definitions)
[[BASIC_CONCEPTS_END]]

[[FIGURE_ANALYSIS_START]]
(Graph description)
[[FIGURE_ANALYSIS_END]]

[[VERBATIM_START]]
(Pixel-perfect transcription)
[[VERBATIM_END]]

[[AI_SOLUTION_START]]
(Standard solution)
[[AI_SOLUTION_END]]

[[DEEP_INSIGHT_START]]
(1-Tier Instructor's Insight)
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
        "db_columns": {"necessity": "", "key_idea": "", "special_point": ""},
        "body_content": {
            "teacher_decoding": [], # [신규] 선생님의 시선 통합
            "conditions": [], "special_points": [], "key_ideas": [], # [복구] Legacy 리스트 필드
            "basic_concepts": [], "practical_concepts": [], 
            "figure_analysis": "", "verbatim_handwriting": "", 
            "ai_solution": "", "instructor_solution": "",
            "strategy_overview": "", "action_protocol": ""
        }
    }

    # [Verbose Extraction Helper] - [복구 완료] 로그 및 부분 매칭 기능
    def extract_section(start_tag, end_tag, debug_name):
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        match = re.search(pattern, text, re.DOTALL)
        if match: return match.group(1).strip()
        
        # Fallback: 루즈 매칭
        pattern_loose = re.escape(start_tag) + r"(.*)"
        match_loose = re.search(pattern_loose, text, re.DOTALL)
        if match_loose:
            content = match_loose.group(1).strip()
            next_tag_match = re.search(r'\[\[.*?_START\]\]', content)
            if next_tag_match: return content[:next_tag_match.start()].strip()
            return content
        return ""

    def clean_list(raw_text):
        if not raw_text: return []
        lines = raw_text.split('\n')
        cleaned = []
        for line in lines:
            line = re.sub(r'^[\s\*\-\d\.]+', '', line).strip() 
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

    # --- 1. Extraction & Integration (Dual Processing) ---
    print("  >> 선생님의 시선(Decoding) 추출 및 레거시 데이터 병합 중...")
    decoding_list = []
    
    # 1. Necessity
    raw_nec = extract_section("[[NECESSITY_START]]", "[[NECESSITY_END]]", "Necessity")
    decoding_list.extend(parse_teacher_decoding(raw_nec, "필연성"))
    data["db_columns"]["necessity"] = raw_nec.replace("|", " ").replace("\n", " ")

    # 2. Key Idea (Merge Single + List Tag) [복구된 로직]
    k1 = extract_section("[[KEY_IDEA_START]]", "[[KEY_IDEA_END]]", "KeyIdea(Single)")
    k2 = extract_section("[[KEY_IDEAS_LIST_START]]", "[[KEY_IDEAS_LIST_END]]", "KeyIdea(List)")
    raw_key_merged = merge_and_deduplicate(k1, k2)
    
    # 통합된 텍스트를 Decoding List에도 넣고, DB 컬럼에도 넣음 (이중 저장)
    decoding_list.extend(parse_teacher_decoding(raw_key_merged, "핵심 아이디어"))
    data["db_columns"]["key_idea"] = raw_key_merged.replace("|", " ").replace("\n", " ")
    data["body_content"]["key_ideas"] = clean_list(k2) # [Legacy List 보존]

    # 3. Special Point (Merge Single + List Tag) [복구된 로직]
    s1 = extract_section("[[SPECIAL_POINT_START]]", "[[SPECIAL_POINT_END]]", "SpecialPoint(Single)")
    s2 = extract_section("[[SPECIAL_POINTS_LIST_START]]", "[[SPECIAL_POINTS_LIST_END]]", "SpecialPoint(List)")
    raw_sp_merged = merge_and_deduplicate(s1, s2)
    
    decoding_list.extend(parse_teacher_decoding(raw_sp_merged, "특이점"))
    data["db_columns"]["special_point"] = raw_sp_merged.replace("|", " ").replace("\n", " ")
    data["body_content"]["special_points"] = clean_list(s2) # [Legacy List 보존]

    # 4. Goal
    raw_goal = extract_section("[[GOAL_START]]", "[[GOAL_END]]", "Goal")
    decoding_list.extend(parse_teacher_decoding(raw_goal, "구하는 목표"))
    data["body_content"]["goal"] = raw_goal

    # 5. Conditions
    raw_cond = extract_section("[[CONDITIONS_START]]", "[[CONDITIONS_END]]", "Conditions")
    decoding_list.extend(parse_teacher_decoding(raw_cond, "조건"))
    data["body_content"]["conditions"] = clean_list(raw_cond) # [Legacy List 보존]
    
    # 최종 통합 리스트 저장
    data["body_content"]["teacher_decoding"] = decoding_list

    # --- 2. Body Content Extraction (Restored All Fields) ---
    print("  >> 본문 콘텐츠 및 그래프/기본개념 추출 중...")
    data["body_content"]["verbatim_handwriting"] = extract_section("[[VERBATIM_START]]", "[[VERBATIM_END]]", "Verbatim")
    data["body_content"]["ai_solution"] = extract_section("[[AI_SOLUTION_START]]", "[[AI_SOLUTION_END]]", "AI Solution")
    data["body_content"]["instructor_solution"] = extract_section("[[DEEP_INSIGHT_START]]", "[[DEEP_INSIGHT_END]]", "Insight")
    
    data["body_content"]["strategy_overview"] = extract_section("[[STRATEGY_START]]", "[[STRATEGY_END]]", "Strategy")
    data["body_content"]["action_protocol"] = extract_section("[[ACTION_PROTOCOL_START]]", "[[ACTION_PROTOCOL_END]]", "ActionProtocol")
    
    # [복구] Figure Analysis & Basic Concepts
    data["body_content"]["figure_analysis"] = extract_section("[[FIGURE_ANALYSIS_START]]", "[[FIGURE_ANALYSIS_END]]", "Figure")
    data["body_content"]["basic_concepts"] = clean_list(extract_section("[[BASIC_CONCEPTS_START]]", "[[BASIC_CONCEPTS_END]]", "BasicConcepts"))

    # 실전개념 정제
    pc_raw = extract_section("[[PRACTICAL_CONCEPTS_START]]", "[[PRACTICAL_CONCEPTS_END]]", "PracticalConcepts")
    pc_list = []
    if pc_raw:
        lines = pc_raw.split('\n')
        for line in lines:
            parts = re.split(r'\|\|', line)
            if len(parts) >= 2:
                title_part = parts[0].replace("Title:", "").strip()
                content_part = parts[1].replace("Content:", "").strip()
                if title_part:
                    pc_list.append({"title": title_part, "content": content_part})
    data["body_content"]["practical_concepts"] = pc_list

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
    if not result_data["body_content"]["instructor_solution"]:
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