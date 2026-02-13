# concept_manager.py
import json
import os
import shutil
import re
import difflib
import time
import logging
import threading
from datetime import datetime
from config import MD_DIR_PATH


FILE_LOCK = threading.RLock()
# ==========================================================
# [Configuration] 경로 및 상수 (절대 타협 없음)
# ==========================================================
DB_PATH = os.path.join(MD_DIR_PATH, "concept_book.json")
BACKUP_DIR = os.path.join(MD_DIR_PATH, "concept_history")
TEMP_DB_PATH = os.path.join(MD_DIR_PATH, "concept_book.tmp")
WHITELIST_PATH = os.path.join(MD_DIR_PATH, "concept_whitelist.json")

# [유사도 임계값 - 엄격]
SIMILARITY_THRESHOLD_HIGH = 0.85  # 이 이상이면 무조건 병합 (Append)
SIMILARITY_THRESHOLD_WARN = 0.40  # 이 이상이면 (중복의심) 태그 부착

# 로깅 설정 (변태적으로 상세하게)
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
def log_warn(msg): 
    print(f"⚠️ [Manager] {msg}")
    logging.warning(msg)
def log_error(msg): 
    print(f"❌ [Manager] {msg}")
    logging.error(msg)

# ==========================================================
# [Core Logic 0] Whitelist (면죄부 시스템)
# ==========================================================
def load_whitelist():
    """사용자가 '중복 아님'으로 지정한 쌍을 로드"""
    if not os.path.exists(WHITELIST_PATH): return []
    try:
        with open(WHITELIST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except: return []

def add_to_whitelist(title_a, title_b):
    """(A, B)는 서로 다른 개념임을 영구 기록"""
    data = load_whitelist()
    # 순서 무관하게 저장 (항상 정렬해서 저장)
    pair = sorted([title_a, title_b])
    if pair not in data:
        data.append(pair)
        try:
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
    if not text: return ""
    
    # 1. 태그 제거 (기존에 붙은 태그 무시하고 알맹이만 비교)
    text = re.sub(r'^\(중복의심\)\s*\[\d+%\]\s*', '', text)
    
    # 2. 기본 정제
    text = text.lower().strip()
    
    # 3. 노이즈 단어 제거 (긴 것부터)
    noise_words = ["실전개념", "기본개념", "수학개념", "공식정리", "개념정리", "수학", "개념", "공식", "정리"]
    for word in noise_words:
        text = text.replace(word, "")
    
    # 4. 특수문자 제거 (한글, 영문, 숫자 외 제거)
    text = re.sub(r'[^a-z0-9가-힣]', '', text)
    
    return text

# ==========================================================
# [Core Logic 2] 유사도 계산 (Sim Radar)
# ==========================================================
def calculate_similarity(s1, s2):
    if not s1 or not s2: return 0.0
    norm1 = normalize_fingerprint(s1)
    norm2 = normalize_fingerprint(s2)
    if not norm1 or not norm2: return 0.0
    return difflib.SequenceMatcher(None, norm1, norm2).ratio()

# ==========================================================
# [File I/O] 안전 제일 (Safety First)
# ==========================================================
def ensure_backup_dir():
    if not os.path.exists(BACKUP_DIR):
        try: os.makedirs(BACKUP_DIR)
        except: pass

def create_snapshot():
    if not os.path.exists(DB_PATH): return
    ensure_backup_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"concept_book_backup_{timestamp}.json"
    try: shutil.copy2(DB_PATH, os.path.join(BACKUP_DIR, backup_filename))
    except: pass

def load_concepts():
    """
    [Thread-Safe] JSON DB 로드.
    파일을 읽는 도중에 다른 스레드가 쓰지 못하도록 락을 겁니다.
    """
    with FILE_LOCK:
        if not os.path.exists(DB_PATH): return []
        try:
            with open(DB_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except: return []

def save_all_concepts(data):
    """Atomic Save: 쓰다가 죽어도 DB는 깨지지 않는다."""
    with FILE_LOCK:
        try:
            with open(TEMP_DB_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            if os.path.exists(DB_PATH): os.remove(DB_PATH)
            os.rename(TEMP_DB_PATH, DB_PATH)
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
    if not new_concept or "title" not in new_concept: return

    raw_title = new_concept['title'].strip()
    raw_content = new_concept.get('content', "").strip()
    
    # [방어] 내용 부실(10자 미만) 차단
    if len(raw_content) < 10:
        log_info(f"내용 부실로 저장 거부: {raw_title}")
        return

    create_snapshot()
    data = load_concepts()
    
    # -------------------------------------------------------
    # 1. 전수 조사 (Full Scan)
    # -------------------------------------------------------
    best_match_idx = -1
    highest_sim = 0.0
    match_type = "NONE" # EXACT, HIGH, MID, NONE
    
    target_fingerprint = normalize_fingerprint(raw_title)

    for idx, item in enumerate(data):
        existing_title = item.get('title', "")
        
        # 0. 화이트리스트 확인 (면죄부)
        if is_whitelisted(raw_title, existing_title):
            continue # 이 녀석과는 비교하지 않는다
            
        existing_fingerprint = normalize_fingerprint(existing_title)
        
        # 1-1. 완전 일치 (Priority 1)
        if target_fingerprint == existing_fingerprint:
            best_match_idx = idx
            highest_sim = 1.0
            match_type = "EXACT"
            break 
        
        # 1-2. 유사도 계산
        sim = calculate_similarity(raw_title, existing_title)
        if sim > highest_sim:
            highest_sim = sim
            best_match_idx = idx

    # -------------------------------------------------------
    # 2. 판정 및 실행 (Decision)
    # -------------------------------------------------------
    
    # [CASE A] 병합 (Append) - 완전 일치 or 85% 이상
    if match_type == "EXACT" or highest_sim >= SIMILARITY_THRESHOLD_HIGH:
        target_item = data[best_match_idx]
        target_title = target_item.get('title')
        old_content = target_item.get('content', "")
        
        # 내용 중복 체크 (단순 포함)
        if normalize_fingerprint(raw_content) in normalize_fingerprint(old_content):
            log_info(f"🛡️ [Skip] '{raw_title}' 내용은 이미 '{target_title}'에 있음.")
            return

        # [Smart Formatting]
        today = datetime.now().strftime("%Y-%m-%d")
        # 이미지 URL이나 출처가 있으면 좋겠지만, 현재는 날짜로 구분
        append_header = f"\n\n\n--- 📅 [추가: {today}] (유사도 {int(highest_sim*100)}%) ---\n"
        
        target_item['content'] = old_content + append_header + raw_content
        target_item['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        save_all_concepts(data)
        log_info(f"🔗 [Merged] '{raw_title}' -> '{target_title}' 병합 완료.")
        return

    # [CASE B] 중복 의심 (Tagging) - 40% ~ 84%
    elif highest_sim >= SIMILARITY_THRESHOLD_WARN:
        sim_percent = int(highest_sim * 100)
        origin_title = data[best_match_idx]['title']
        
        # 태그 부착: "(중복의심) [82%] 원래제목"
        tagged_title = f"(중복의심) [{sim_percent}%] {raw_title}"
        
        log_warn(f"⚠️ [Suspect] '{raw_title}' vs '{origin_title}' ({sim_percent}%). 태그 부착 저장.")
        
        new_concept['title'] = tagged_title
        new_concept['created_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_concept['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 내용 상단에도 경고 문구 삽입
        warn_msg = f"> ⚠️ **시스템 경고:** 이 개념은 '{origin_title}'과 {sim_percent}% 유사합니다.\n\n"
        new_concept['content'] = warn_msg + raw_content
        
        data.append(new_concept)
        save_all_concepts(data)
        return

    # [CASE C] 신규 생성 (New)
    new_concept['created_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_concept['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data.append(new_concept)
    save_all_concepts(data)
    log_info(f"✨ [New] '{raw_title}' 신규 등록.")

# ==========================================================
# [Helper Tools] UI 연동용 도구들
# ==========================================================
def delete_concept(target_title):
    create_snapshot()
    data = load_concepts()
    new_data = [d for d in data if d['title'] != target_title]
    if len(data) != len(new_data):
        save_all_concepts(new_data)
        return True
    return False

def manual_update_concept(target_title, new_content):
    create_snapshot()
    data = load_concepts()
    for item in data:
        if item['title'] == target_title:
            item['content'] = new_content
            item['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_all_concepts(data)
            return True
    return False

def merge_concepts_manual(master_title, slave_titles):
    """사용자가 UI에서 선택한 것들 강제 병합"""
    create_snapshot()
    data = load_concepts()
    
    master_item = next((d for d in data if d['title'] == master_title), None)
    if not master_item: return False
    
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
    
    # Slave 삭제 (제목에서 태그 떼고 비교하는 로직 등 불필요, 정확한 제목으로 삭제)
    slave_set = set(slave_titles)
    new_data = [d for d in data if d['title'] not in slave_set]
    
    save_all_concepts(new_data)
    return True

def remove_suspect_tag(target_title):
    """(중복의심) 태그 제거 (청소 기능)"""
    # 정규식으로 태그 부분만 날림
    clean_title = re.sub(r'^\(중복의심\)\s*\[\d+%\]\s*', '', target_title)
    
    if clean_title == target_title: return False # 태그 없음
    
    create_snapshot()
    data = load_concepts()
    
    # 혹시 태그 뗀 이름이 이미 존재하면? -> 병합해야 함 (복잡도 증가)
    # 여기서는 "이미 존재하면 실패 처리"하고 사용자에게 "병합하세요"라고 하는 게 안전함
    if any(d['title'] == clean_title for d in data):
        log_warn(f"태그 제거 불가: '{clean_title}'이 이미 존재함. 병합 기능을 사용하세요.")
        return "EXISTS" # 특수 리턴
    
    for item in data:
        if item['title'] == target_title:
            item['title'] = clean_title
            # 내용 상단의 경고 문구도 제거 시도
            content = item.get('content', "")
            content = re.sub(r'> ⚠️ \*\*시스템 경고:\*\*.*?\n\n', '', content, flags=re.DOTALL)
            item['content'] = content
            save_all_concepts(data)
            return True
            
    return False

def get_similarity_clusters():
    """
    [UI 정렬용] 전체 개념을 N*N 비교하여 유사한 것끼리 묶은 리스트 반환
    (성능 무시, 결과 지향)
    """
    data = load_concepts()
    if not data: return []
    
    # 1. (제목, 정규화된제목) 리스트 생성
    items = []
    for d in data:
        items.append({
            'title': d['title'],
            'norm': normalize_fingerprint(d['title']),
            'visited': False
        })
        
    clusters = []
    
    # 2. 클러스터링 (Greedy)
    for i in range(len(items)):
        if items[i]['visited']: continue
        
        # 새로운 클러스터 시작
        current_cluster = [items[i]['title']]
        items[i]['visited'] = True
        base_norm = items[i]['norm']
        
        for j in range(i+1, len(items)):
            if items[j]['visited']: continue
            
            # 유사도 비교 (기준: 0.4 이상이면 같은 그룹으로 간주)
            sim = difflib.SequenceMatcher(None, base_norm, items[j]['norm']).ratio()
            if sim >= 0.4:
                current_cluster.append(items[j]['title'])
                items[j]['visited'] = True
        
        clusters.append(current_cluster)
        
    # 3. 플랫 리스트로 변환 (클러스터 간 구분은 UI에서 처리하든 그냥 나열하든)
    # 여기서는 유사한 것끼리 인접하게 배치된 단일 리스트 반환
    sorted_titles = []
    for cl in clusters:
        # 클러스터 내부는 가나다순 정렬
        cl.sort()
        sorted_titles.extend(cl)
        
    return sorted_titles