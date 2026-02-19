# concept_manager.py
import json
import os
import shutil
import re
import difflib
import logging
import threading
import unicodedata
from datetime import datetime

from config import MD_DIR_PATH
import database_manager


FILE_LOCK = threading.RLock()
# ==========================================================
# [Configuration] 경로 및 상수 (절대 타협 없음)
# ==========================================================
DB_PATH = os.path.join(MD_DIR_PATH, "mathbot.sqlite3")
BACKUP_DIR = os.path.join(MD_DIR_PATH, "concept_history")
SOURCE_BACKUP_DIR = os.path.join(os.path.dirname(__file__), "_BACKUP")
WHITELIST_PATH = os.path.join(MD_DIR_PATH, "concept_whitelist.json")

# [유사도 임계값 - 엄격]
SIMILARITY_THRESHOLD_HIGH = 0.85  # 이 이상이면 무조건 병합 (Append)
SIMILARITY_THRESHOLD_WARN = 0.40  # 이 이상이면 (중복의심) 태그 부착

# [Global State] find_page_id 지원용 캐시
CONCEPT_CACHE = []
FAST_LOOKUP_MAP = {}  # { "정규화된제목": "concept_id" }
GHOST_MAP = {}        # { "정규화된제목(확통제거)": "concept_id" }

# 로깅 설정
logging.basicConfig(
    filename='concept_manager.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)


def log_debug(msg): logging.debug(msg)


def log_info(msg):
    print(f"ℹ️ [Manager] {msg}")
    logging.info(msg)
    try:
        database_manager.write_log("INFO", msg)
    except Exception:
        pass


def log_warn(msg):
    print(f"⚠️ [Manager] {msg}")
    logging.warning(msg)
    try:
        database_manager.write_log("WARN", msg)
    except Exception:
        pass


def log_error(msg):
    print(f"❌ [Manager] {msg}")
    logging.error(msg)
    try:
        database_manager.write_log("ERROR", msg)
    except Exception:
        pass


# ==========================================================
# [Defensive Change] 소스 파일 자동 백업
# ==========================================================
def backup_current_source_file():
    """덮어쓰기 이전 보존 관례를 유지하기 위한 소스 백업 루틴."""
    try:
        os.makedirs(SOURCE_BACKUP_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = os.path.join(SOURCE_BACKUP_DIR, f"concept_manager.py.runtime_backup_{ts}")
        shutil.copy2(__file__, dst)
    except Exception:
        pass


# ==========================================================
# [Core Logic 0] Whitelist (면죄부 시스템)
# ==========================================================
def load_whitelist():
    """사용자가 '중복 아님'으로 지정한 쌍을 로드"""
    if not os.path.exists(WHITELIST_PATH):
        return []
    try:
        with open(WHITELIST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def add_to_whitelist(title_a, title_b):
    """(A, B)는 서로 다른 개념임을 영구 기록"""
    data = load_whitelist()
    pair = sorted([title_a, title_b])
    if pair not in data:
        data.append(pair)
        try:
            os.makedirs(os.path.dirname(WHITELIST_PATH), exist_ok=True)
            with open(WHITELIST_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            log_info(f"🏳️ [Whitelist] '{title_a}' vs '{title_b}' 무시 목록 등록.")
        except Exception as e:
            log_error(f"Whitelist 저장 실패: {e}")


def is_whitelisted(title_a, title_b):
    """이 두 개가 화이트리스트에 있는지 확인"""
    data = load_whitelist()
    pair = sorted([title_a, title_b])
    return pair in data


# ==========================================================
# [Core Logic 1] 변태적 정규화 (Fingerprint)
# ==========================================================
def normalize_fingerprint(text):
    """
    텍스트의 영혼만 추출.
    1. 소문자화 + 양옆 공백 제거
    2. (중복의심) [xx%] 태그 제거 (순수 제목만 비교 위해)
    3. 노이즈 단어 제거 (수학, 개념 등)
    4. 특수문자 전멸시킴
    """
    if not text:
        return ""

    text = re.sub(r'^\(중복의심\)\s*\[\d+%\]\s*', '', str(text))
    text = text.lower().strip()
    noise_words = ["실전개념", "기본개념", "수학개념", "공식정리", "개념정리", "수학", "개념", "공식", "정리"]
    for word in noise_words:
        text = text.replace(word, "")
    text = re.sub(r'[^a-z0-9가-힣]', '', text)
    return text


def calculate_similarity(s1, s2):
    if not s1 or not s2:
        return 0.0
    norm1 = normalize_fingerprint(s1)
    norm2 = normalize_fingerprint(s2)
    if not norm1 or not norm2:
        return 0.0
    return difflib.SequenceMatcher(None, norm1, norm2).ratio()


# ==========================================================
# [Ghost Map 로직] 파일명 매칭 정규화
# ==========================================================
def normalize_aggressive(text):
    if not text:
        return ""

    text = unicodedata.normalize('NFC', str(text))
    text = text.replace("공통범위", "").replace("공통", "")

    score_match = re.search(r'(\[\d+\.\d+점\])', text)
    if score_match:
        split_idx = score_match.start()
        front = text[:split_idx]
        back = text[split_idx:]
        back = back.replace("문과", "").replace("이과", "").replace("예체능", "")
        text = front + back
    else:
        text = re.sub(r'(문과|이과|예체능)\s*$', '', text)

    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    for _ in range(3):
        text = re.sub(r'(_img\d*|_\d+)\s*$', '', text)

    text = re.sub(r'[^0-9a-zA-Z가-힣]', '', text).lower()
    return text.strip()


# ==========================================================
# [SQLite I/O] 안전 제일 (Safety First)
# ==========================================================
def ensure_backup_dir():
    if not os.path.exists(BACKUP_DIR):
        try:
            os.makedirs(BACKUP_DIR)
        except Exception:
            pass


def create_snapshot():
    if not os.path.exists(DB_PATH):
        return
    ensure_backup_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"mathbot_backup_{timestamp}.sqlite3"
    try:
        shutil.copy2(DB_PATH, os.path.join(BACKUP_DIR, backup_filename))
    except Exception:
        pass


def _row_to_legacy_item(row):
    return {
        "id": row.get("id"),
        "title": row.get("title", ""),
        "content": row.get("content", ""),
        "fingerprint": row.get("fingerprint", ""),
        "notion_page_id": row.get("notion_page_id"),
        "created_at": row.get("created_at"),
        "last_updated": row.get("last_updated"),
    }


def load_concepts():
    """
    [Thread-Safe] SQLite DB 로드.
    main.py 호환을 위해 기존 JSON 리스트 구조로 반환.
    """
    with FILE_LOCK:
        try:
            database_manager.init_db()
            rows = database_manager.fetch_all_concepts()
            return [_row_to_legacy_item(r) for r in rows]
        except Exception as e:
            log_error(f"DB 로드 실패: {e}")
            return []


def save_all_concepts(data):
    """Atomic Replace: 쓰다가 죽어도 트랜잭션 롤백"""
    with FILE_LOCK:
        try:
            normalized = []
            for item in data:
                title = item.get("title", "")
                normalized.append({
                    "title": title,
                    "content": item.get("content", ""),
                    "fingerprint": item.get("fingerprint") or normalize_fingerprint(title),
                    "notion_page_id": item.get("notion_page_id"),
                    "created_at": item.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "last_updated": item.get("last_updated"),
                })
            database_manager.replace_all_concepts(normalized)
            return True
        except Exception as e:
            log_error(f"CRITICAL: DB 저장 실패 {e}")
            return False


# ==========================================================
# [Main Logic] 개념 저장 (The Fortress Gatekeeper)
# ==========================================================
def save_concept(new_concept):
    """
    [알고리즘: Fortress V2 - Tag & Append]
    1. Fingerprint 완전 일치 -> 무조건 병합
    2. 유사도 > 85% -> 무조건 병합 (Smart Format)
    3. 유사도 > 40% -> (중복의심) 태그 붙여서 생성 (단, 화이트리스트 있으면 패스)
    4. 나머지 -> 신규 생성
    """
    if not new_concept or "title" not in new_concept:
        return "INVALID"

    raw_title = new_concept['title'].strip()
    raw_content = new_concept.get('content', "").strip()

    if len(raw_content) < 10:
        log_info(f"내용 부실로 저장 거부: {raw_title}")
        return "REJECTED_WEAK"

    create_snapshot()
    data = load_concepts()

    best_match_idx = -1
    highest_sim = 0.0
    match_type = "NONE"

    target_fingerprint = normalize_fingerprint(raw_title)

    for idx, item in enumerate(data):
        existing_title = item.get('title', "")

        if is_whitelisted(raw_title, existing_title):
            continue

        existing_fingerprint = normalize_fingerprint(existing_title)

        if target_fingerprint == existing_fingerprint:
            best_match_idx = idx
            highest_sim = 1.0
            match_type = "EXACT"
            break

        sim = calculate_similarity(raw_title, existing_title)
        if sim > highest_sim:
            highest_sim = sim
            best_match_idx = idx

    if match_type == "EXACT" or highest_sim >= SIMILARITY_THRESHOLD_HIGH:
        target_item = data[best_match_idx]
        target_title = target_item.get('title')
        old_content = target_item.get('content', "")

        if normalize_fingerprint(raw_content) in normalize_fingerprint(old_content):
            log_info(f"🛡️ [Skip] '{raw_title}' 내용은 이미 '{target_title}'에 있음.")
            return "SKIPPED_DUPLICATE"

        today = datetime.now().strftime("%Y-%m-%d")
        append_header = f"\n\n\n--- 📅 [추가: {today}] (유사도 {int(highest_sim*100)}%) ---\n"

        target_item['content'] = old_content + append_header + raw_content
        target_item['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        target_item['fingerprint'] = normalize_fingerprint(target_item.get('title', ''))

        save_all_concepts(data)
        log_info(f"🔗 [Merged] '{raw_title}' -> '{target_title}' 병합 완료.")
        sync_db_to_memory(lambda _: None)
        return "MERGED"

    elif highest_sim >= SIMILARITY_THRESHOLD_WARN:
        sim_percent = int(highest_sim * 100)
        origin_title = data[best_match_idx]['title'] if best_match_idx >= 0 else "Unknown"

        tagged_title = f"(중복의심) [{sim_percent}%] {raw_title}"

        log_warn(f"⚠️ [Suspect] '{raw_title}' vs '{origin_title}' ({sim_percent}%). 태그 부착 저장.")

        new_concept['title'] = tagged_title
        new_concept['created_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_concept['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        warn_msg = f"> ⚠️ **시스템 경고:** 이 개념은 '{origin_title}'과 {sim_percent}% 유사합니다.\n\n"
        new_concept['content'] = warn_msg + raw_content
        new_concept['fingerprint'] = normalize_fingerprint(tagged_title)

        data.append(new_concept)
        save_all_concepts(data)
        sync_db_to_memory(lambda _: None)
        return "TAGGED_NEW"

    new_concept['created_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_concept['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_concept['fingerprint'] = normalize_fingerprint(raw_title)
    data.append(new_concept)
    save_all_concepts(data)
    log_info(f"✨ [New] '{raw_title}' 신규 등록.")
    sync_db_to_memory(lambda _: None)
    return "NEW"


# ==========================================================
# [Notion ID Matching Support] sync/find
# ==========================================================
def sync_db_to_memory(log_func=print):
    """
    SQLite concepts 데이터를 메모리에 로드하여 FAST_LOOKUP_MAP/GHOST_MAP 재구축.
    """
    global CONCEPT_CACHE, FAST_LOOKUP_MAP, GHOST_MAP
    data = load_concepts()

    CONCEPT_CACHE = data
    FAST_LOOKUP_MAP = {}
    GHOST_MAP = {}

    for item in data:
        raw_title = item.get("title", "")
        concept_id = str(item.get("notion_page_id") or item.get("id") or "")
        if not raw_title or not concept_id:
            continue

        norm_key = normalize_aggressive(raw_title)
        if norm_key:
            FAST_LOOKUP_MAP[norm_key] = concept_id

        if "확률과 통계" in raw_title or "확률과통계" in raw_title:
            stripped_src = raw_title.replace("확률과 통계", "").replace("확률과통계", "")
            stripped_key = normalize_aggressive(stripped_src)
            if stripped_key:
                GHOST_MAP[stripped_key] = concept_id

    try:
        log_func(f"✅ [ConceptDB] 메모리 동기화 완료 ({len(CONCEPT_CACHE)}개)")
    except Exception:
        pass
    return len(CONCEPT_CACHE)


def find_page_id(filename, debug=False):
    """
    파일명을 정규화하여 concept/notion page id를 찾습니다.
    1. Direct Match
    2. Ghost Match (확통 강제 매칭)
    """
    global FAST_LOOKUP_MAP, CONCEPT_CACHE, GHOST_MAP

    if not FAST_LOOKUP_MAP and not CONCEPT_CACHE:
        sync_db_to_memory(lambda _: None)
        if not FAST_LOOKUP_MAP:
            return None, "DB_CACHE_EMPTY"

    name_body = os.path.splitext(filename)[0]
    target_norm = normalize_aggressive(name_body)

    if target_norm in FAST_LOOKUP_MAP:
        if debug:
            print(f"🚀 [HML Match] 100% 일치: {filename}")
        return FAST_LOOKUP_MAP[target_norm], None

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
            if debug:
                print(f"👻 [Ghost Match] 확통 강제 매칭: {filename}")
            return GHOST_MAP[target_norm], None

    return None, "NO_MATCH"


# ==========================================================
# [Helper Tools] UI 연동용 도구들
# ==========================================================
def delete_concept(target_title):
    create_snapshot()
    data = load_concepts()
    new_data = [d for d in data if d['title'] != target_title]
    if len(data) != len(new_data):
        save_all_concepts(new_data)
        sync_db_to_memory(lambda _: None)
        return True
    return False


def manual_update_concept(target_title, new_content):
    create_snapshot()
    data = load_concepts()
    for item in data:
        if item['title'] == target_title:
            item['content'] = new_content
            item['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            item['fingerprint'] = normalize_fingerprint(item.get('title', ''))
            save_all_concepts(data)
            sync_db_to_memory(lambda _: None)
            return True
    return False


def merge_concepts_manual(master_title, slave_titles):
    """사용자가 UI에서 선택한 것들 강제 병합"""
    create_snapshot()
    data = load_concepts()

    master_item = next((d for d in data if d['title'] == master_title), None)
    if not master_item:
        return False

    slaves = [d for d in data if d['title'] in slave_titles]

    today = datetime.now().strftime("%Y-%m-%d")
    merged_content = master_item.get('content', "")

    for slave in slaves:
        s_title = slave.get('title')
        s_content = slave.get('content', "")
        header = f"\n\n--- 🔗 [병합됨: {s_title} | {today}] ---\n"
        merged_content += header + s_content

    master_item['content'] = merged_content
    master_item['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    master_item['fingerprint'] = normalize_fingerprint(master_item.get('title', ''))

    slave_set = set(slave_titles)
    new_data = [d for d in data if d['title'] not in slave_set]

    save_all_concepts(new_data)
    sync_db_to_memory(lambda _: None)
    return True


def remove_suspect_tag(target_title):
    """(중복의심) 태그 제거 (청소 기능)"""
    clean_title = re.sub(r'^\(중복의심\)\s*\[\d+%\]\s*', '', target_title)

    if clean_title == target_title:
        return False

    create_snapshot()
    data = load_concepts()

    if any(d['title'] == clean_title for d in data):
        log_warn(f"태그 제거 불가: '{clean_title}'이 이미 존재함. 병합 기능을 사용하세요.")
        return "EXISTS"

    for item in data:
        if item['title'] == target_title:
            item['title'] = clean_title
            item['fingerprint'] = normalize_fingerprint(clean_title)
            content = item.get('content', "")
            content = re.sub(r'> ⚠️ \*\*시스템 경고:\*\*.*?\n\n', '', content, flags=re.DOTALL)
            item['content'] = content
            save_all_concepts(data)
            sync_db_to_memory(lambda _: None)
            return True

    return False


def get_similarity_clusters():
    """
    [UI 정렬용] 전체 개념을 N*N 비교하여 유사한 것끼리 묶은 리스트 반환
    (성능 무시, 결과 지향)
    """
    data = load_concepts()
    if not data:
        return []

    items = []
    for d in data:
        items.append({
            'title': d['title'],
            'norm': normalize_fingerprint(d['title']),
            'visited': False
        })

    clusters = []

    for i in range(len(items)):
        if items[i]['visited']:
            continue

        current_cluster = [items[i]['title']]
        items[i]['visited'] = True
        base_norm = items[i]['norm']

        for j in range(i + 1, len(items)):
            if items[j]['visited']:
                continue

            sim = difflib.SequenceMatcher(None, base_norm, items[j]['norm']).ratio()
            if sim >= 0.4:
                current_cluster.append(items[j]['title'])
                items[j]['visited'] = True

        clusters.append(current_cluster)

    sorted_titles = []
    for cl in clusters:
        cl.sort()
        sorted_titles.extend(cl)

    return sorted_titles


# 초기화 루틴
try:
    backup_current_source_file()
    database_manager.init_db()
    sync_db_to_memory(lambda _: None)
except Exception:
    pass
