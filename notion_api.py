# notion_api.py
import requests
import json
import re
import os
import time
import copy
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
        if next_cursor:
            payload["start_cursor"] = next_cursor
        elif "start_cursor" in payload:
            payload.pop("start_cursor", None)
        try:
            res = robust_request("POST", url, payload)
            if not res or res.status_code != 200:
                err_text = "No Response"
                if res is not None:
                    err_text = f"{res.status_code}: {(res.text or '')[:120]}"
                log_func(f"⚠ [Notion Sync] query 실패, 안전 종료: {err_text}")
                break
            
            data = res.json()
            results = data.get("results", [])
            
            for page in results:
                try:
                    if page.get("archived", False):
                        continue
                    page_id = page["id"]
                    edited_time = page["last_edited_time"]
                    props = page["properties"]
                    
# ----------------------------------------------------------------------------------------------------
                    # [수술 부위] 공백 제목 무시 및 출처 컬럼 승격 로직 (Invisible Wall 방어)
                    # ----------------------------------------------------------------------------------------------------
                    # 1순위: 이름(title/rich_text)
                    name_obj = props.get("이름", {})
                    name_list = name_obj.get("title", []) or name_obj.get("rich_text", [])
                    raw_title = name_list[0].get("plain_text", "").strip() if name_list else ""

                    # 2순위: 문제&풀이(title/rich_text)
                    if not raw_title:
                        title_obj = props.get("문제&풀이", {})
                        t_list = title_obj.get("title", []) or title_obj.get("rich_text", [])
                        raw_title = t_list[0].get("plain_text", "").strip() if t_list else ""

                    # 3순위: 출처(rich_text/title)
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
        except Exception as e:
            log_func(f"⚠ [Notion Sync] 예외, 안전 종료: {str(e)[:120]}")
            break

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
        "문제&풀이": {"title": [{"text": {"content": title}}]}
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
    properties = {}

    def _get_existing_page_tags(target_page_id):
        """기존 페이지의 '태그' multi_select 이름 목록 조회"""
        get_url = f"https://api.notion.com/v1/pages/{target_page_id}"
        res = robust_request("GET", get_url)
        if not res or res.status_code != 200:
            return None
        try:
            props = res.json().get("properties", {})
            tag_items = props.get("태그", {}).get("multi_select", [])
            return [item.get("name", "") for item in tag_items if item.get("name")]
        except Exception:
            return None

    def _to_rich_text_prop(value):
        return {"rich_text": [{"type": "text", "text": {"content": str(value if value is not None else "")}}]}

    if "necessity" in db_data:
        properties["필연성"] = _to_rich_text_prop(db_data.get("necessity"))
    if "key_idea" in db_data:
        properties["핵심 아이디어"] = _to_rich_text_prop(db_data.get("key_idea"))
    if "special_point" in db_data:
        properties["특이점"] = _to_rich_text_prop(db_data.get("special_point"))

    if "tags" in db_data:
        existing_tags = _get_existing_page_tags(page_id)
        if existing_tags is None:
            return False, "TAG_GET_FAILED"

        raw_tags = db_data.get("tags")
        if isinstance(raw_tags, str):
            new_tags = [t.strip() for t in re.split(r"[,/\n]", raw_tags)]
        elif isinstance(raw_tags, list):
            new_tags = [str(t).strip() for t in raw_tags]
        else:
            new_tags = []

        merged_tags = []
        seen_keys = set()

        for t in (existing_tags or []):
            normalized_key = re.sub(r"\s+", "", str(t)).lower()
            if normalized_key and normalized_key not in seen_keys:
                seen_keys.add(normalized_key)
                merged_tags.append(str(t).strip())

        for t in new_tags:
            cleaned = str(t).strip()
            normalized_key = re.sub(r"\s+", "", cleaned).lower()
            if normalized_key and normalized_key not in seen_keys:
                seen_keys.add(normalized_key)
                merged_tags.append(cleaned)

        # 병합 후 비어 있으면 안전하게 스킵(태그 클리어는 별도 기능)
        if merged_tags:
            properties["태그"] = {"multi_select": [{"name": tag} for tag in merged_tags]}
    
    if concept_ids is not None and isinstance(concept_ids, list):
        relation_list = [{"id": cid} for cid in concept_ids]
        properties["실전개념"] = {"relation": relation_list}

    if not properties:
        return True, "NO_CHANGES"

    payload = {"properties": properties}
    url = f"https://api.notion.com/v1/pages/{page_id}"
    
    res = robust_request("PATCH", url, payload)
    if res and res.status_code == 200: return True, "성공"
    return False, res.text if res else "Update Failed"


def archive_page(page_id: str):
    """페이지를 archived 처리하고 로컬 캐시/맵에서도 제거합니다."""
    global NOTION_CACHE, FAST_LOOKUP_MAP, GHOST_MAP

    url = f"https://api.notion.com/v1/pages/{page_id}"
    res = robust_request("PATCH", url, {"archived": True})
    if not res or res.status_code != 200:
        err = res.text if res else "No Response"
        return False, f"ERR: {err}"

    NOTION_CACHE = [item for item in NOTION_CACHE if item.get("id") != page_id]

    for key in [k for k, v in FAST_LOOKUP_MAP.items() if v == page_id]:
        del FAST_LOOKUP_MAP[key]
    for key in [k for k, v in GHOST_MAP.items() if v == page_id]:
        del GHOST_MAP[key]

    try:
        save_local_cache(NOTION_CACHE)
    except Exception:
        pass

    return True, "OK"
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
# ----------------------------------------------------------------------------------
# [V31 New Renderer] 선생님의 시선: Step 1 & Step 2 분리 렌더링
# ----------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------
# [V35 Renderer] 누락되었던 렌더링 헬퍼 함수 복구
# ----------------------------------------------------------------------------------
def make_symbol_table(symbol_list):
    """ [Step 1] 기호 정의 테이블 생성 (3열: 기호 | 의미 | AI 주석) """
    if not symbol_list: return None
    
    table_rows = []
    # Header (3열)
    table_rows.append({
        "type": "table_row",
        "table_row": {
            "cells": [
                [{"text": {"content": "기호 (Symbol)", "link": None}, "annotations": {"bold": True, "color": "blue"}}],
                [{"text": {"content": "의미 (Definition)", "link": None}, "annotations": {"bold": True, "color": "blue"}}],
                [{"text": {"content": "AI 주석 (Comment)", "link": None}, "annotations": {"bold": True, "color": "gray"}}]
            ]
        }
    })
    # Body (3열 데이터 매핑)
    for item in symbol_list:
        sym = item.get("symbol", "")
        mean = item.get("meaning", "")
        comment = item.get("comment", "")
        table_rows.append({
            "type": "table_row",
            "table_row": {
                "cells": [
                    make_rich_text_list(sym),
                    make_rich_text_list(mean),
                    make_rich_text_list(comment)
                ]
            }
        })

    return {"object": "block", "type": "table", "table": {"table_width": 3, "has_column_header": True, "children": table_rows}}

def make_logic_narrative_blocks(narrative_list):
    """ [Step 2] 논리 서술 블록 생성 """
    if not narrative_list: return []
    blocks = []
    for line in narrative_list:
        icon = "👉"
        if "[상황" in line: icon = "🧐"
        elif "[핵심" in line or "(핵)" in line: icon = "🔑"
        elif "[특이" in line or "(특)" in line: icon = "⚠️"
        elif "[필연" in line or "[행동" in line or "따라서" in line: icon = "🚀"
        
        blocks.append({
            "object": "block", "type": "callout",
            "callout": {"rich_text": make_rich_text_list(line), "icon": {"emoji": icon}, "color": "gray_background"}
        })
    return blocks
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


def _split_text_chunks(text, max_len=2000):
    text = "" if text is None else str(text)
    if len(text) <= max_len:
        return [text]
    return [text[i:i + max_len] for i in range(0, len(text), max_len)]


def sanitize_blocks_recursive(blocks):
    """
    전송 직전 Notion 제한(문자열 2000자, 빈 수식 등)을 만족하도록 블록을 재귀 정리합니다.
    표/행 구조는 절대 강등하지 않고 셀 내부 rich_text만 정리합니다.
    """
    if not isinstance(blocks, list):
        return []

    rich_text_owner_keys = [
        "paragraph", "heading_1", "heading_2", "heading_3", "callout", "quote",
        "bulleted_list_item", "numbered_list_item", "toggle", "to_do"
    ]

    def sanitize_rich_text_list(rich_text_list):
        sanitized = []
        for rt in (rich_text_list or []):
            if not isinstance(rt, dict):
                continue

            rt_type = rt.get("type")
            if rt_type == "equation":
                expr = str(rt.get("equation", {}).get("expression", ""))
                if not expr.strip():
                    sanitized.append({"type": "text", "text": {"content": " "}})
                    continue
                expr_chunks = _split_text_chunks(expr, 2000)
                if len(expr_chunks) > 1:
                    print(f"🩹 [Notion Recover] action=split chunk=0 size={len(expr)}")
                for expr_chunk in expr_chunks:
                    new_rt = copy.deepcopy(rt)
                    new_rt.setdefault("equation", {})["expression"] = expr_chunk
                    sanitized.append(new_rt)

            elif rt_type == "text":
                text_obj = rt.get("text", {})
                content = str(text_obj.get("content", ""))
                if not content:
                    continue
                content_chunks = _split_text_chunks(content, 2000)
                if len(content_chunks) > 1:
                    print(f"🩹 [Notion Recover] action=split chunk=0 size={len(content)}")
                for content_chunk in content_chunks:
                    new_rt = copy.deepcopy(rt)
                    new_rt.setdefault("text", {})["content"] = content_chunk
                    sanitized.append(new_rt)

            else:
                sanitized.append(rt)

        return sanitized or [{"type": "text", "text": {"content": " "}}]

    clean_blocks = []
    for block in blocks:
        if not isinstance(block, dict):
            print(f"⚠️ [Notion Block TypeError] block is {type(block)} -> {str(block)[:80]}")
            continue

        block = copy.deepcopy(block)
        block_type = block.get("type")

        if block_type == "equation":
            expr = str(block.get("equation", {}).get("expression", ""))
            if not expr.strip():
                continue
            if len(expr) > 2000:
                print(f"🩹 [Notion Recover] action=truncate chunk=0 size={len(expr)}")
            block.setdefault("equation", {})["expression"] = expr[:2000]

        for owner_key in rich_text_owner_keys:
            if owner_key in block and isinstance(block[owner_key], dict) and "rich_text" in block[owner_key]:
                block[owner_key]["rich_text"] = sanitize_rich_text_list(block[owner_key].get("rich_text", []))

        if block_type == "table_row" and isinstance(block.get("table_row"), dict):
            cells = block["table_row"].get("cells", [])
            normalized_cells = []
            for cell in cells:
                normalized_cells.append(sanitize_rich_text_list(cell if isinstance(cell, list) else []))
            block["table_row"]["cells"] = normalized_cells

        for child_container_key in [block_type] + rich_text_owner_keys + ["table"]:
            container = block.get(child_container_key)
            if isinstance(container, dict) and "children" in container:
                container["children"] = sanitize_blocks_recursive(container.get("children", []))

        clean_blocks.append(block)

    return clean_blocks


def _patch_with_retry(url, payload, chunk_no, batch_size):
    res = robust_request("PATCH", url, payload, retries=5)
    if res and res.status_code == 200:
        return res

    if res and (res.status_code == 429 or res.status_code in [500, 502, 503, 504]):
        print(f"🩹 [Notion Recover] action=retry chunk={chunk_no} size={batch_size}")
        time.sleep(1)
        retry_res = robust_request("PATCH", url, payload, retries=5)
        return retry_res if retry_res else res

    return res


def _send_children_in_chunks(page_id, children_blocks, chunk_size):
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"

    for i in range(0, len(children_blocks), chunk_size):
        batch = children_blocks[i:i + chunk_size]
        payload = {"children": batch}
        chunk_no = i // chunk_size + 1

        res = _patch_with_retry(url, payload, chunk_no, len(batch))

        if res and res.status_code == 200:
            continue

        if not res:
            print("❌ [Notion Fail] status=NO_RESPONSE err=No Response")
            return False, "No Response", i

        if res.status_code == 400 and "archived" in res.text.lower():
            restore_url = f"https://api.notion.com/v1/pages/{page_id}"
            restore_res = robust_request("PATCH", restore_url, {"archived": False})
            if restore_res and restore_res.status_code == 200:
                print(f"🩹 [Notion Recover] action=unarchive chunk={chunk_no} size={len(batch)}")
                retry_res = _patch_with_retry(url, payload, chunk_no, len(batch))
                if retry_res and retry_res.status_code == 200:
                    continue
                retry_text = (retry_res.text if retry_res else "No Response")
                retry_code = (retry_res.status_code if retry_res else "NO_RESPONSE")
                print(f"❌ [Notion Fail] status={retry_code} err={str(retry_text)[:300]}")
                return False, (retry_text or "unknown error"), i

            restore_text = (restore_res.text if restore_res else "No Response")
            restore_code = (restore_res.status_code if restore_res else "NO_RESPONSE")
            print(f"❌ [Notion Fail] status={restore_code} err={str(restore_text)[:300]}")
            return False, (restore_text or "unknown error"), i

        print(f"❌ [Notion Fail] status={res.status_code} err={res.text[:300]}")
        return False, (res.text or "unknown error"), i

    return True, "성공", len(children_blocks)


def safe_append_children(page_id, children_blocks):
    """
    Notion append 안정 래퍼:
    - 100개 제한 회피(90 -> 50 -> 30 chunk)
    - rich_text/equation 2000자 제한 정규화
    - archived 자동 복구
    - 최대 3회 시도(중복 업로드 방지)
    """
    if isinstance(children_blocks, dict):
        children_blocks = build_children_blocks(children_blocks)

    remaining_children = sanitize_blocks_recursive(children_blocks or [])
    attempts = [90, 50, 30]
    last_error = ""

    for attempt_idx, chunk_size in enumerate(attempts, start=1):
        if not remaining_children:
            return True, "성공"

        ok, msg, next_start = _send_children_in_chunks(page_id, remaining_children, chunk_size=chunk_size)
        if ok:
            return True, "성공"

        last_error = msg
        remaining_children = remaining_children[next_start:]
        print(
            f"🩹 [Notion Recover] action=retry chunk={attempt_idx} size={len(remaining_children)}"
        )
        if attempt_idx < len(attempts):
            time.sleep(1)

    return False, f"실패: {str(last_error)[:300]}"


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
    
    final_children = build_children_blocks(body_content)
    return safe_append_children(page_id, final_children)


def _build_children_blocks_impl(body_content):
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
    # [0.5] 🗺️ 전체 전략 (Strategy Map) - [신규 배치: 최상단]
    # -------------------------------------------------------
    strategy = body_content.get("strategy_overview", "")
    if strategy:
        all_blocks.append(make_heading_2("🗺️ 전체 전략 (Strategy Map)"))
        all_blocks.append(make_callout(strategy, "🧭"))
        all_blocks.append(make_text_block(" "))

    # -------------------------------------------------------
    # -------------------------------------------------------
        # -------------------------------------------------------
    # 2. 🧠 선생님의 시선 (Teacher's Decoding) [UNIFIED: Always 4-Column]
    #    ✅ 출력 통일 원칙:
    #    - 입력이 symbol_table 이든 teacher_decoding 이든 상관없이
    #      항상 make_teacher_decoding_table(4열)로 렌더링한다.
    #    - 3열 make_symbol_table 경로는 더 이상 사용하지 않는다. (혼재 방지)
    # -------------------------------------------------------
    symbol_data = body_content.get("symbol_table", [])
    decoding_list = body_content.get("teacher_decoding", [])
    logic_data = body_content.get("logic_narrative", [])

    # [A] teacher_decoding이 없고 symbol_table만 있는 경우 -> 4열 teacher_decoding 형태로 변환
    # symbol_table item keys: symbol / meaning / comment
    # teacher_decoding item keys: symbol / type / content / ai_comment
    if (not decoding_list) and symbol_data:
        decoding_list = []
        for it in symbol_data:
            # 방어: dict 아니면 무시
            if not isinstance(it, dict):
                continue
            decoding_list.append({
                "symbol": it.get("symbol", ""),
                "type": it.get("type", "") or "Condition",          # type이 없으면 기본 Condition
                "content": it.get("meaning", ""),                   # meaning -> content
                "ai_comment": it.get("comment", ""),                # comment -> ai_comment
            })

    # [B] teacher_decoding이 있는데, item 키명이 옛날/혼합일 수 있음 -> 키 정규화
    if decoding_list:
        normalized = []
        # GPT 제안: 복구용 타입 집합 정의
        TYPE_SET = {"Condition", "Goal", "Key", "Trap", "Strategy", "Example"}
        
        for it in decoding_list:
            if not isinstance(it, dict):
                continue

            # [GPT 핀셋 복구] content가 타입처럼 생기고 type이 전부 Condition으로 고정된 경우 -> 스왑 복구
            current_type = it.get("type", "")
            current_content = it.get("content", "")
            
            # 만약 타입이 'Condition'인데 내용물이 'Goal' 같은 거라면? -> 칸 밀림 현상임!
            if (current_type == "Condition") and (current_content in TYPE_SET):
                it["type"] = current_content  # 내용을 타입으로 격상
                it["content"] = "Unknown"     # 내용은 비어있으니 Unknown 처리
            
            normalized.append({
                "symbol": it.get("symbol", ""),
                "type": it.get("type", ""),
                "content": it.get("content", "") or it.get("meaning", ""),  # 혹시 content 대신 meaning이면 흡수
                "ai_comment": it.get("ai_comment", "") or it.get("comment", ""),
            })
        decoding_list = normalized

    if decoding_list or logic_data:
        all_blocks.append(make_heading_2("🧠 선생님의 시선 (Teacher's Decoding)", "blue_background"))

        # 2-1. 기호 정의 테이블 (항상 4열로 통일)
        if decoding_list:
            all_blocks.append(make_text_block("📌 기호 정의 (Symbol Map)"))
            table = make_teacher_decoding_table(decoding_list)
            if table:
                all_blocks.append(table)
            all_blocks.append(make_text_block(" "))

        # 2-2. 논리 서술 (이야기)
        if logic_data:
            all_blocks.append(make_text_block("📝 논리적 풀이 흐름 (Logic Narrative)"))
            all_blocks.extend(make_logic_narrative_blocks(logic_data))
            all_blocks.append(make_text_block(" "))


    # -------------------------------------------------------
    # [2.5] ⚡ 행동 강령 (Action Protocol) - [신규 배치: 서술 직후]
    # -------------------------------------------------------
    protocol = body_content.get("action_protocol", "")
    if protocol:
        all_blocks.append(make_heading_2("⚡ 행동 강령 (Action Protocol)"))
        all_blocks.append(make_callout(protocol, "🚀"))
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
    
    # ✅ [응급처치 & 무적 방어막] all_blocks 안에 list가 섞여 있으면 평탄화
    flattened = []
    for b in all_blocks:
        if not b:
            continue
        if isinstance(b, list):
            flattened.extend(b)
        else:
            flattened.append(b)

    return flattened


def build_children_blocks(body_content):
    return _build_children_blocks_impl(body_content or {})
