# notion_api.py
import requests
import json
import re
import os
import time
import difflib
import unicodedata
from datetime import datetime
from config import NOTION_API_KEY, NOTION_DATABASE_ID, MD_DIR_PATH

# ==========================================================
# [Configuration] 헤더 및 상수 설정 (절대 타협 없음)
# ==========================================================
HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json"
}

CACHE_FILE_PATH = os.path.join(MD_DIR_PATH, "notion_db_cache.json")

# [Global State] 메모리 캐시 및 검색 최적화 맵
NOTION_CACHE = [] 
IS_CACHE_READY = False
FAST_LOOKUP_MAP = {} # { "정규화된제목": "page_id" }
GHOST_MAP = {}       # { "정규화된제목(확통제거)": "page_id" } - 확통 과목 매칭용

# ==========================================================
# [Core Logic 1] 통신 안전장치 (Robust Request System)
# ==========================================================
def robust_request(method, url, payload=None, retries=5):
    """
    [기존 기능 보존] 네트워크 불안정, API 속도 제한(429), 서버 오류(5xx)를 
    5회까지 재시도하며 방어하는 통신 함수입니다.
    """
    last_error = None
    for attempt in range(retries):
        try:
            if method == "POST":
                res = requests.post(url, headers=HEADERS, json=payload, timeout=20)
            elif method == "PATCH":
                res = requests.patch(url, headers=HEADERS, json=payload, timeout=20)
            else:
                res = requests.get(url, headers=HEADERS, timeout=20)
            
            # 200 OK: 성공 시 즉시 반환
            if res.status_code == 200:
                return res
            
            # 429 Too Many Requests: 지수 백오프(Exponential Backoff)로 대기
            if res.status_code == 429:
                wait_time = 2 ** attempt
                print(f"⚠️ [Notion 429] 속도 제한 감지! {wait_time}초 대기 후 재시도...")
                time.sleep(wait_time)
                continue
                
            # 5xx Server Error: 노션 서버 문제, 잠시 대기 후 재시도
            if 500 <= res.status_code < 600:
                print(f"⚠️ [Notion {res.status_code}] 서버 내부 오류. 재시도 {attempt+1}/{retries}")
                time.sleep(1)
                continue
                
            # 409 Conflict / 502 Bad Gateway / 503 Service Unavailable / 504 Gateway Timeout
            if res.status_code in [409, 502, 503, 504]:
                print(f"⚠️ [Notion {res.status_code}] 일시적 오류. 재시도 {attempt+1}/{retries}")
                time.sleep(1)
                continue
                
            # 그 외 4xx 에러는 재시도해도 소용없으므로 에러 메시지 저장 후 반환
            last_error = res.text
            return res 
            
        except Exception as e:
            last_error = str(e)
            print(f"💥 통신 예외 발생 (시도 {attempt+1}/{retries}): {e}")
            time.sleep(1)
            
    print(f"❌ [Final Fail] 5회 재시도 모두 실패. Last Error: {last_error}")
    return None

# ==========================================================
# [Core Logic 2] 정규화 및 지문 추출 (Forensic Text Analysis)
# ==========================================================
def normalize_aggressive(text):
    """
    [기존 기능 보존] HML V229 로직 이식.
    제목의 본질(알맹이)만 남기고 껍데기를 벗겨내어 매칭률을 극대화합니다.
    """
    if not text: return ""
    
    # 1. 유니코드 정규화 (NFC)
    text = unicodedata.normalize('NFC', text)
    
    # 2. 불필요한 수식어 제거 (공통범위 등)
    text = text.replace("공통범위", "").replace("공통", "")
    
    # 3. 점수 표기([3.00점]) 처리: 점수 뒤에 붙은 문과/이과/예체능 제거
    score_match = re.search(r'(\[\d+\.\d+점\])', text)
    if score_match:
        split_idx = score_match.start()
        front = text[:split_idx]
        back = text[split_idx:]
        back = back.replace("문과", "").replace("이과", "").replace("예체능", "")
        text = front + back
    else:
        text = re.sub(r'(문과|이과|예체능)\s*$', '', text)

    # 4. 잡다한 괄호 및 이미지 태그 제거
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    for _ in range(3): text = re.sub(r'(_img\d*|_\d+)\s*$', '', text)
    
    # 5. 최종 필터: 숫자, 영어, 한글만 남김 (특수문자/공백 제거)
    text = re.sub(r'[^0-9a-zA-Z가-힣]', '', text).lower()
    return text.strip()

def extract_fingerprint(text):
    """
    [기존 기능 보존] 문제의 6가지 신원 정보(지문)를 추출합니다.
    이 정보는 캐싱 시스템에서 문제의 동일성을 판단하는 보조 지표로 사용됩니다.
    """
    fingerprint = {
        "year": None,
        "month": None,
        "number": None,
        "subject": set(),
        "authority": set(),
        "grade": set()
    }
    
    # 1. 연도 (Year)
    year_match = re.search(r'(20\d{2})', text)
    if year_match: fingerprint["year"] = int(year_match.group(1))

    # 2. 월 (Month)
    if "수능" in text or "대학수학능력시험" in text: fingerprint["month"] = 11
    else:
        month_match = re.search(r'(\d{1,2})월', text)
        if month_match: fingerprint["month"] = int(month_match.group(1))

    # 3. 문제 번호 (Number)
    clean_text_for_num = re.sub(r'\d+(\.\d+)?점', '', text)
    nums = re.findall(r'\d+', clean_text_for_num)
    if nums:
        # 뒤에서부터 찾되 30번 이하인 숫자를 문제 번호로 간주
        for n in reversed(nums):
            val = int(n)
            if 1 <= val <= 30:
                fingerprint["number"] = val
                break
    
    # 4. 과목 (Subject)
    if "가형" in text or "이과" in text: fingerprint["subject"].add("가형")
    if "나형" in text or "문과" in text: fingerprint["subject"].add("나형")
    if "미적" in text: fingerprint["subject"].add("미적")
    if "기하" in text: fingerprint["subject"].add("기하")
    if "확통" in text or "확률" in text: fingerprint["subject"].add("확통")
    if "공통" in text: fingerprint["subject"].add("공통")

    # 5. 출제 기관 (Authority)
    if "사관" in text: fingerprint["authority"].add("사관")
    if "경찰" in text: fingerprint["authority"].add("경찰")
    if "교육청" in text or "학평" in text or "학력" in text: fingerprint["authority"].add("교육청")
    if "평가원" in text or "모의" in text: fingerprint["authority"].add("평가원")
    if "수능" in text or "대학수학능력" in text: fingerprint["authority"].add("수능")

    # 6. 학년 (Grade)
    if "고1" in text: fingerprint["grade"].add("고1")
    if "고2" in text: fingerprint["grade"].add("고2")
    if "고3" in text: fingerprint["grade"].add("고3")

    return fingerprint

# ==========================================================
# [Caching System] 로컬 파일 기반 증분 동기화 (Sync & Cache)
# ==========================================================
class SetEncoder(json.JSONEncoder):
    """[기존 기능 보존] JSON 저장 시 Set 자료형을 List로 변환"""
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

def load_local_cache():
    """[기존 기능 보존] 로컬 JSON 캐시 파일 로드"""
    if not os.path.exists(CACHE_FILE_PATH): return None
    try:
        with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except: return None

def save_local_cache(data):
    """[기존 기능 보존] 로컬 JSON 캐시 파일 저장"""
    try:
        with open(CACHE_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=SetEncoder, ensure_ascii=False, indent=2)
    except Exception as e: print(f"⚠ 캐시 저장 실패: {e}")

def sync_db_to_memory(log_func=print):
    """
    [기존 기능 보존] Notion DB의 모든 데이터를 긁어와 메모리에 올립니다.
    로컬 파일이 있으면 그걸 먼저 로드하고, 변경된 부분만 API로 가져오는 '증분 업데이트'를 수행합니다.
    """
    global NOTION_CACHE, IS_CACHE_READY, FAST_LOOKUP_MAP, GHOST_MAP
    
    local_data = load_local_cache()
    existing_map = {} 
    last_synced_time = None
    
    # 1. 로컬 캐시 로드 및 맵핑 구축
    if local_data:
        # log_func(f"📂 [System] 로컬 캐시 로드 ({len(local_data)}개).")
        for item in local_data:
            existing_map[item["id"]] = item
            
            # Fast Lookup Map 구축 (정규화된 제목 -> PageID)
            raw_title = item.get("title", "")
            norm_key = normalize_aggressive(raw_title)
            FAST_LOOKUP_MAP[norm_key] = item["id"]
            
            # Ghost Map 구축 (확통 과목 특화 매칭)
            if "확률과 통계" in raw_title or "확률과통계" in raw_title:
                stripped_src = raw_title.replace("확률과 통계", "").replace("확률과통계", "")
                stripped_key = normalize_aggressive(stripped_src)
                GHOST_MAP[stripped_key] = item["id"]
            
            # 마지막 수정 시간 추적
            item_time = item.get("last_edited_time", "")
            if item_time:
                if not last_synced_time or item_time > last_synced_time:
                    last_synced_time = item_time
    else:
        print("✨ [System] 로컬 캐시 없음. 전체 다운로드 시작...")

    # 2. Notion API 호출 (증분 업데이트)
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    payload = {"page_size": 100}
    
    # 마지막 동기화 시간 이후에 변경된 데이터만 가져오도록 필터 설정
    if last_synced_time:
        payload["filter"] = {"timestamp": "last_edited_time", "last_edited_time": {"after": last_synced_time}}
    
    has_more = True
    next_cursor = None
    fetched_count = 0
    
    while has_more:
        if next_cursor: payload["start_cursor"] = next_cursor
        try:
            res = requests.post(url, headers=HEADERS, json=payload)
            if res.status_code != 200: 
                time.sleep(2)
                continue
            
            data = res.json()
            results = data.get("results", [])
            
            for page in results:
                try:
                    page_id = page["id"]
                    edited_time = page["last_edited_time"]
                    props = page["properties"]
                    
# ----------------------------------------------------------------------------------------------------
                    # [수술 부위] 공백 제목 무시 및 출처 컬럼 승격 로직 (Invisible Wall 방어)
                    # ----------------------------------------------------------------------------------------------------
                    title_obj = props.get("문제&풀이", {})
                    t_list = title_obj.get("title", []) or title_obj.get("rich_text", [])
                    
                    # 핵심: .strip()을 추가하여 공백만 있는 좀비 제목을 빈 문자열("")로 처리
                    raw_title = t_list[0].get("plain_text", "").strip() if t_list else ""

                    # 제목이 텅 비었다면(공백 포함) '출처' 컬럼을 제목으로 승격
                    if not raw_title:
                        src_obj = props.get("출처", {})
                        s_list = src_obj.get("rich_text", []) or src_obj.get("title", [])
                        if s_list: raw_title = s_list[0].get("plain_text", "").strip()

                    if raw_title:
                    # ----------------------------------------------------------------------------------------------------
                        norm_key = normalize_aggressive(raw_title)
                        FAST_LOOKUP_MAP[norm_key] = page_id
                        
                        if "확률과 통계" in raw_title or "확률과통계" in raw_title:
                            stripped_src = raw_title.replace("확률과 통계", "").replace("확률과통계", "")
                            stripped_key = normalize_aggressive(stripped_src)
                            GHOST_MAP[stripped_key] = page_id

                        # 데이터 객체 생성 및 저장
                        new_item = {
                            "id": page_id, "title": raw_title, "last_edited_time": edited_time,
                            "norm": norm_key, "fp": extract_fingerprint(raw_title)
                        }
                        existing_map[page_id] = new_item
                except: continue
                
            fetched_count += len(results)
            has_more = data.get("has_more", False)
            next_cursor = data.get("next_cursor")
        except: time.sleep(2)

    # 최종 캐시 리스트 생성
    NOTION_CACHE = list(existing_map.values())
    IS_CACHE_READY = True
    
    # 변경사항이 있거나 로컬 데이터가 없었으면 파일로 저장
    if fetched_count > 0 or not local_data:
        # print(f"💾 [System] 캐시 파일 업데이트... (총 {len(NOTION_CACHE)}개)")
        save_local_cache(NOTION_CACHE)
    
    return len(NOTION_CACHE)

def find_page_id(filename, debug=False):
    """
    [기존 기능 보존] 제목으로 Notion Page ID를 찾습니다.
    1. Exact Match (정규화 후 비교)
    2. Ghost Match (확통 과목 특수 처리)
    """
    global FAST_LOOKUP_MAP, GHOST_MAP
    
    # 캐시가 준비되지 않았으면 로드 시도
    if not FAST_LOOKUP_MAP and not NOTION_CACHE: 
        sync_db_to_memory()
        if not FAST_LOOKUP_MAP: return None, "DB_CACHE_EMPTY"

    name_body = os.path.splitext(filename)[0]
    target_norm = normalize_aggressive(name_body)
    
    # 1. Direct Match (1:1 Map Lookup)
    if target_norm in FAST_LOOKUP_MAP:
        if debug: print(f"🚀 [HML Match] 100% 일치: {filename}")
        return FAST_LOOKUP_MAP[target_norm], None
        
    # 2. Forced Match (Ghost Map for Prob/Stat)
    # 조건: 2021년 이후 + 고3 + 23~30번 문제인 경우
    year_match = re.search(r'(\d{4})년', name_body)
    year = int(year_match.group(1)) if year_match else 0
    
    clean_name = re.sub(r'\[.*?\]', '', name_body)
    nums = re.findall(r'(\d+)', clean_name)
    q_num = 0
    if nums:
        for n in reversed(nums):
            if int(n) < 100: 
                q_num = int(n)
                break
                
    is_high3 = "고3" in name_body
    
    if year >= 2021 and is_high3 and 23 <= q_num <= 30:
        if target_norm in GHOST_MAP:
            if debug: print(f"👻 [Ghost Match] 확통 강제 매칭: {filename}")
            return GHOST_MAP[target_norm], None

    return None, "NO_MATCH"

# ==========================================================
# [Helpers] LaTeX & Block Rendering (한글 수식 복구 기능 포함)
# ==========================================================
def make_rich_text_list(content):
    if not content: return []
    content = str(content)
    content = content.replace("\\\\", "\\")
    # [수정] Notion이 못 읽는 LaTeX 문서 태그 제거 및 변환 (청소 작업)
    content = re.sub(r'\\begin\{itemize\}', '', content)
    content = re.sub(r'\\end\{itemize\}', '', content)
    content = re.sub(r'\\begin\{enumerate\}', '', content)
    content = re.sub(r'\\end\{enumerate\}', '', content)
    content = re.sub(r'\\item\s*', '\n• ', content)        
    content = re.sub(r'\\textbf\{([^}]+)\}', r'\1', content) 
    content = re.sub(r'\\underline\{([^}]+)\}', r'\1', content) 
    content = re.sub(r'\\textcircled\{([^}]+)\}', r'(\1)', content) 
    content = re.sub(r'\\quad', '  ', content)              
    
    # 1. 이중 백슬래시 과다 이스케이프 정리
    content = content.replace("\\\\\\\\", "\\\\")
    
    # 정규식으로 수식 블록 분리
    pattern = r'(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\$[\s\S]+?\$|\\\([\s\S]+?\\\))'
    tokens = re.split(pattern, content)
    rich_text = []
    
    # [Helper] 한글 래핑 함수 (수식 내 한글 깨짐 방지)
    def wrap_korean(match):
        return f"\\text{{{match.group(0)}}}"

    for token in tokens:
        if not token: continue
        token_strip = token.strip()
        
        if not token_strip:
            rich_text.append({"type": "text", "text": {"content": token}})
            continue

        is_equation = False
        expr = ""
        
        # 수식 태그 감지 및 껍데기 벗기기
        if token_strip.startswith("$$") and token_strip.endswith("$$"):
            expr = token_strip[2:-2].strip(); is_equation = True
        elif token_strip.startswith("\\[") and token_strip.endswith("\\]"):
            expr = token_strip[2:-2].strip(); is_equation = True
        elif token_strip.startswith("$") and token_strip.endswith("$"):
            expr = token_strip[1:-1].strip(); is_equation = True
        elif token_strip.startswith("\\(") and token_strip.endswith("\\)"):
            expr = token_strip[2:-2].strip(); is_equation = True
            
        if is_equation:
            # 빈 수식 방어 (Notion 400 Error 방지)
            if not expr or expr.strip() == "":
                rich_text.append({"type": "text", "text": {"content": " "}})
            else:
                # 한글 처리 로직
                if re.search(r'[가-힣]+', expr) and not "\\text" in expr:
                    expr = re.sub(r'([가-힣]+)', wrap_korean, expr)
                
                # 재검사
                if not expr.strip():
                     rich_text.append({"type": "text", "text": {"content": " "}})
                else:
                    rich_text.append({"type": "equation", "equation": {"expression": expr}})
        else:
            # 일반 텍스트 2000자 제한 처리
            if len(token) > 1900:
                chunks = [token[i:i+1900] for i in range(0, len(token), 1900)]
                for c in chunks:
                    rich_text.append({"type": "text", "text": {"content": c}})
            else:
                rich_text.append({"type": "text", "text": {"content": token}})
            
    return rich_text

def create_block(type, content, color="default", icon=None):
    """[기존 기능 보존] 블록 생성 헬퍼 함수"""
    if not content: return []
    
    # 블록 수식 ($$...$$) 단독 처리
    if content.strip().startswith("$$") and content.strip().endswith("$$"):
        expr = content.strip().replace("$$", "").strip()
        if expr: return [{"object": "block", "type": "equation", "equation": {"expression": expr}}]
        
    full_rich_text = make_rich_text_list(content)
    if not full_rich_text: return []
    
    # 2000자 제한 방지 (청크 분할)
    chunks = [full_rich_text[i:i + 100] for i in range(0, len(full_rich_text), 100)]
    blocks = []
    for chunk in chunks:
        block = {"object": "block", "type": type, type: {"rich_text": chunk}}
        if color != "default" and type != "paragraph": block[type]["color"] = color
        if type == "callout" and icon: block[type]["icon"] = {"emoji": icon}
        blocks.append(block)
    return blocks

# ==========================================================
# [Core Logic 3] 페이지 생성 및 속성 업데이트 (Properties)
# ==========================================================
def create_new_problem_page(title, db_data, concept_ids=None):
    """
    [기존 기능 보존] Notion DB에 새 페이지를 생성합니다.
    모든 메타데이터(난이도, 등급, 유형, 출처 등)를 매핑합니다.
    """
    url = "https://api.notion.com/v1/pages"
    
    props = {
        "이름": {"title": [{"text": {"content": title}}]}
    }
    
    # 메타데이터 매핑
    if db_data.get("main_category"):
        props["대분류"] = {"select": {"name": str(db_data["main_category"])}}
    if db_data.get("sub_category"):
        props["중분류"] = {"select": {"name": str(db_data["sub_category"])}}
    if db_data.get("difficulty"):
        props["난이도"] = {"select": {"name": str(db_data["difficulty"])}}
    if db_data.get("grade"):
        props["등급"] = {"select": {"name": str(db_data["grade"])}}
    if db_data.get("type"):
        props["유형"] = {"select": {"name": str(db_data["type"])}}

    # 출처 처리 (Safe Logic: 텍스트로 입력하여 오류 방지)
    if db_data.get("source"):
        src_val = str(db_data["source"])
        props["출처"] = {"rich_text": [{"text": {"content": src_val}}]}
    
    # 필연성/핵심/특이점 (Legacy Support for DB Filtering)
    if db_data.get("necessity"):
        props["필연성"] = {"rich_text": [{"text": {"content": str(db_data["necessity"])[:2000]}}]}
    if db_data.get("key_idea"):
        props["핵심 아이디어"] = {"rich_text": [{"text": {"content": str(db_data["key_idea"])[:2000]}}]}
    if db_data.get("special_point"):
        props["특이점"] = {"rich_text": [{"text": {"content": str(db_data["special_point"])[:2000]}}]}
    # [NEW] 정답 (Correct Answer) - 안전장치 적용 (Over-engineering)
    if db_data.get("correct_answer"):
        # 정답이 너무 길면(해설이 딸려오면) 100자로 자르는 방어 로직 적용
        ans_val = str(db_data["correct_answer"]).strip()
        if len(ans_val) > 100: ans_val = ans_val[:100]
        props["정답"] = {"rich_text": [{"text": {"content": ans_val}}]}
    # 태그 처리 (Multi-select)
    if db_data.get("tags"):
        tag_list = []
        raw_tags = db_data["tags"]
        if isinstance(raw_tags, str): raw_tags = [t.strip() for t in raw_tags.split(',')]
        for t in raw_tags: tag_list.append({"name": str(t)})
        if tag_list: props["태그"] = {"multi_select": tag_list}

    # 개념 연결 (Relation)
    if concept_ids and isinstance(concept_ids, list):
        relation_list = [{"id": cid} for cid in concept_ids]
        props["실전개념"] = {"relation": relation_list}

    payload = {"parent": {"database_id": NOTION_DATABASE_ID}, "properties": props}
    
    res = robust_request("POST", url, payload)
    if res and res.status_code == 200:
        page_data = res.json()
        page_id = page_data["id"]
        
        # 캐시 갱신 (즉시 검색 가능하도록)
        norm_key = normalize_aggressive(title)
        FAST_LOOKUP_MAP[norm_key] = page_id
        return page_id, "Success"
    else:
        err_msg = res.text if res else "No Response"
        return None, f"Create Failed: {err_msg}"

def update_page_properties(page_id, db_data, concept_ids=None):
    """[기존 기능 보존] 기존 페이지 속성 업데이트"""
    nec = db_data.get("necessity") or ""
    key = db_data.get("key_idea") or ""
    spe = db_data.get("special_point") or ""
    
    properties = {
        "필연성": {"rich_text": [{"type": "text", "text": {"content": str(nec)}}]},
        "핵심 아이디어": {"rich_text": [{"type": "text", "text": {"content": str(key)}}]},
        "특이점": {"rich_text": [{"type": "text", "text": {"content": str(spe)}}]},
    }
    
    if concept_ids and isinstance(concept_ids, list):
        relation_list = [{"id": cid} for cid in concept_ids]
        properties["실전개념"] = {"relation": relation_list}
        
    payload = {"properties": properties}
    url = f"https://api.notion.com/v1/pages/{page_id}"
    
    res = robust_request("PATCH", url, payload)
    if res and res.status_code == 200: return True, "성공"
    return False, res.text if res else "Update Failed"
def make_heading_2(text, color="default"):
    return {"object": "block", "type": "heading_2", "heading_2": {"rich_text": make_rich_text_list(text), "color": color}}

def make_text_block(text):
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": make_rich_text_list(text)}}

# ----------------------------------------------------------------------------------------------------
# [수술 부위 2] 400 Error (Limit 100) 방어용 Callout 생성기 (리스트 반환)
# Insight 내용이 길어 수식 조각이 100개를 넘으면, 여러 개의 Callout 블록으로 쪼개서 반환합니다.
# ----------------------------------------------------------------------------------------------------
def make_callout(text, icon="💡"):
    full_rich_text = make_rich_text_list(text)
    if not full_rich_text: return []

    # 안전하게 90개씩 끊어서 블록 분할 (Notion 제한: 100개)
    chunk_size = 90
    chunks = [full_rich_text[i:i + chunk_size] for i in range(0, len(full_rich_text), chunk_size)]
    
    blocks = []
    for chunk in chunks:
        blocks.append({
            "object": "block", 
            "type": "callout", 
            "callout": {"rich_text": chunk, "icon": {"emoji": icon}}
        })
    return blocks
# ----------------------------------------------------------------------------------------------------
def make_quote_block(text):
    """
    텍스트를 인용구(Quote) 블록으로 변환합니다. 
    내부의 모든 LaTeX 수식이 완벽하게 노션 수식 객체로 렌더링되도록 강제합니다.
    """
    if not text or text.strip() == "":
        return {
            "object": "block",
            "type": "quote",
            "quote": {"rich_text": [{"type": "text", "text": {"content": " "}}]}
        }
        
    # [핵심 수술 부위] 텍스트를 생으로 넣지 않고 반드시 수식 변환기를 거치게 함
    rendered_rich_text = make_rich_text_list(text)
    
    # 만약 변환기가 실패해서 빈 배열이 오면 최소한의 공백이라도 넣어 에러 방지
    if not rendered_rich_text:
        rendered_rich_text = [{"type": "text", "text": {"content": " "}}]
        
    return {
        "object": "block",
        "type": "quote",
        "quote": {"rich_text": rendered_rich_text}
    }
# ==========================================================
# [Core Logic 4] 본문 내용 생성 (The Body Builder) - V30
# ==========================================================
# ==========================================
# 1. 원본 주석을 100% 살린 표 생성기
# ==========================================
def make_teacher_decoding_table(decoding_list):
    """
    [신규 기능] '선생님의 시선' 데이터를 Notion 표(Table)로 변환합니다.
    구조: [기호 | 구분 | 내용 | AI 해석]
    """
    if not decoding_list: return None
    
    table_block = {
        "object": "block",
        "type": "table",
        "table": {
            "table_width": 4,
            "has_column_header": True,
            "has_row_header": False,
            "children": [
                # 헤더 행
                {
                    "type": "table_row",
                    "table_row": {
                        "cells": [
                            [{"text": {"content": "기호"}}],
                            [{"text": {"content": "구분"}}],
                            [{"text": {"content": "선생님의 메모 (OCR)"}}],
                            [{"text": {"content": "💡 AI의 해석 (Interpretation)"}}]
                        ]
                    }
                }
            ]
        }
    }
    
    # 데이터 행 추가
    for item in decoding_list:
        symbol = item.get("symbol", "")
        dtype = item.get("type", "")
        content = item.get("content", "")
        comment = item.get("ai_comment", "")
        
        row = {
            "type": "table_row",
            "table_row": {
                "cells": [
                    make_rich_text_list(symbol) or [{"type": "text", "text": {"content": " "}}],
                    make_rich_text_list(dtype) or [{"type": "text", "text": {"content": " "}}],
                    make_rich_text_list(content) or [{"type": "text", "text": {"content": " "}}],
                    make_rich_text_list(comment) or [{"type": "text", "text": {"content": " "}}]
                ]
            }
        }
        table_block["table"]["children"].append(row)
        
    return table_block

def append_children(page_id, body_content):
    """
    [핵심] 페이지 본문에 블록들을 순서대로 쌓아 올립니다.
    선생님 요청 순서:
    1. 📸 원본 문제 (Image)
    2. 🧠 선생님의 시선 (Teacher's Decoding) -> Table
    3. 🤖 행동 강령 (Action Protocol) -> Callout
    4. ✍️ 손글씨 풀이 (Verbatim) -> Quote
    5. 🎓 AI 정석 해설 (Standard Solution) -> Text
    6. 📚 실전 개념 (My Dictionary) -> Toggle/Callout
    7. 🏆 Insight -> Callout
    """
    
    # [Pre-flight Check] 블록 데이터 소독 (빈 수식 제거 등)
    def sanitize_blocks_recursive(blocks):
        clean_blocks = []
        for block in blocks:
            # 1. Rich Text 검사
            for type_key in ["paragraph", "heading_1", "heading_2", "heading_3", "callout", "quote", "bulleted_list_item", "numbered_list_item"]:
                if type_key in block and "rich_text" in block[type_key]:
                    new_rich_text = []
                    for rt in block[type_key]["rich_text"]:
                        if rt.get("type") == "equation":
                            expr = rt.get("equation", {}).get("expression", "")
                            if not expr or not str(expr).strip():
                                new_rich_text.append({"type": "text", "text": {"content": " "}})
                            else: new_rich_text.append(rt)
                        elif rt.get("type") == "text":
                            content = rt.get("text", {}).get("content", "")
                            if content: new_rich_text.append(rt)
                        else: new_rich_text.append(rt)
                    
                    if not new_rich_text:
                        new_rich_text = [{"type": "text", "text": {"content": " "}}]
                    block[type_key]["rich_text"] = new_rich_text

            # 2. Block Equation 검사
            if block.get("type") == "equation":
                expr = block.get("equation", {}).get("expression", "")
                if not expr or not str(expr).strip(): continue

            clean_blocks.append(block)
        return clean_blocks

    all_blocks = []
    
    # -------------------------------------------------------
    # 1. 📸 원본 이미지 (Image)
    # -------------------------------------------------------
    img_url = body_content.get("image_url")
    if img_url and img_url.startswith("http"):
        all_blocks.append(make_heading_2("📸 원본 문제 & 필기"))
        all_blocks.append({
            "object": "block",
            "type": "image",
            "image": {
                "type": "external",
                "external": {"url": img_url}
            }
        })
    
    # -------------------------------------------------------
    # 2. 🧠 선생님의 시선 (Teacher's Decoding) [신규/통합]
    # -------------------------------------------------------
    decoding_list = body_content.get("teacher_decoding", [])
    if decoding_list:
        all_blocks.append(make_heading_2("🧠 선생님의 시선 (Teacher's Decoding)", "blue_background"))
        table = make_teacher_decoding_table(decoding_list)
        if table: all_blocks.append(table)
        all_blocks.append(make_text_block(" ")) # 공백

    # -------------------------------------------------------
    # 3. 🤖 행동 강령 & 전략 (Action Protocol & Algorithm)
    # -------------------------------------------------------
    strategy = body_content.get("strategy_overview", "")
    protocol = body_content.get("action_protocol", "")
    
    if strategy or protocol:
        all_blocks.append(make_heading_2("🤖 AI가 제안하는 필연성 & 행동강령"))
        if strategy:
            all_blocks.append(make_text_block(f"🗺️ 전략 로드맵:\n{strategy}"))
        if protocol:
            all_blocks.append(make_text_block(f"⚡ AI가 제안하는 필연성 & 행동강령:\n{protocol}"))
        all_blocks.append(make_text_block(" "))

    # -------------------------------------------------------
    # 4. ✍️ 선생님의 손필기 풀이 (Verbatim)
    # -------------------------------------------------------
    verbatim = body_content.get("verbatim_handwriting", "")
    if verbatim:
        all_blocks.append(make_heading_2("✍️ 선생님의 손필기 풀이 (Verbatim)"))
        all_blocks.append(make_quote_block(verbatim))
        all_blocks.append(make_text_block(" "))

    # -------------------------------------------------------
    # 5. 🎓 AI 정석 해설 (Standard Solution)
    # -------------------------------------------------------
    ai_sol = body_content.get("ai_solution", "")
    if ai_sol:
        all_blocks.append(make_heading_2("🎓 AI 정석 해설 (Standard Solution)"))
        chunks = [ai_sol[i:i+2000] for i in range(0, len(ai_sol), 2000)]
        for chunk in chunks:
            all_blocks.append(make_text_block(chunk))
        all_blocks.append(make_text_block(" "))

    # -------------------------------------------------------
    # 6. 📚 실전 개념 (My Dictionary)
    # -------------------------------------------------------
    concepts = body_content.get("practical_concepts", [])
    if concepts:
        all_blocks.append(make_heading_2("📚 실전 개념 (My Dictionary)"))
        for c in concepts:
            title = c.get("title", "개념")
            content = c.get("content", "")
            all_blocks.append({
                "object": "block",
                "type": "toggle",
                "toggle": {
                    "rich_text": [{"type": "text", "text": {"content": f"📌 {title}"}, "annotations": {"bold": True}}],
                    "children": [
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [{"type": "text", "text": {"content": content[:2000]}}]
                            }
                        }
                    ]
                }
            })
        all_blocks.append(make_text_block(" "))

    # -------------------------------------------------------
    # 7. 🏆 1타 강사의 Insight (마무리)
    # -------------------------------------------------------
    insight = body_content.get("instructor_solution", "")
    if insight:
        all_blocks.append(make_heading_2("🏆 1타 강사의 Insight", "yellow_background"))
        all_blocks.extend(make_callout(insight, "🔥"))

    # =======================================================
    # [Final Step] 블록 전송 (Batch Upload)
    # =======================================================
    # 전송 전 최종 소독
    final_children = sanitize_blocks_recursive([c for c in all_blocks if c])
    
    batch_size = 90
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    
    is_all_success = True
    
    for i in range(0, len(final_children), batch_size):
        batch = final_children[i:i + batch_size]
        payload = {"children": batch}
        
        success_chunk = False
        last_error = ""
        
        # [복구 완료] Retry Logic with Unarchive Handling
        for attempt in range(3):
            try:
                res = requests.patch(url, headers=HEADERS, json=payload)
                
                if res.status_code == 200:
                    success_chunk = True
                    break 
                
                # [Error Handling] Archived Error -> 페이지 복구 시도
                elif res.status_code == 400 and "archived" in res.text.lower():
                    print(f"💀 [Notion] 페이지가 삭제됨(Archived) 감지. 강제 복구(Unarchive) 시도 중...")
                    restore_url = f"https://api.notion.com/v1/pages/{page_id}"
                    restore_payload = {"archived": False}
                    restore_res = requests.patch(restore_url, headers=HEADERS, json=restore_payload)
                    
                    if restore_res.status_code == 200:
                        print(f"🧟 [Notion] 페이지 복구 성공! 블록 전송 재시도...")
                        time.sleep(1)
                        continue 
                    else:
                        print(f"⚰️ [Notion] 페이지 복구 실패: {restore_res.text}")
                
                # 그 외 에러
                else:
                    last_error = res.text
                    print(f"⚠️ [Append Fail] {res.status_code}: {res.text[:150]}...")
                    time.sleep(1)
                    
            except Exception as e:
                last_error = str(e)
                print(f"⚠️ [Append Error] {e}")
                time.sleep(1)
        
        if not success_chunk:
            print(f"❌ [Critical] 블록 전송 실패. Reason: {last_error}")
            is_all_success = False
            break 
            
    if is_all_success:
        return True, "성공"
    else:
        return False, f"실패: {last_error}"