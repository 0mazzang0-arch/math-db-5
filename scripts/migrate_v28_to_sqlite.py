import json
import os
import shutil
import sys
from datetime import datetime

# [추가됨] 부모 폴더(거실)를 볼 수 있게 시야를 넓혀주는 코드
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MD_DIR_PATH
import database_manager


JSON_PATH = os.path.join(MD_DIR_PATH, "concept_book.json")
SQLITE_PATH = os.path.join(MD_DIR_PATH, "mathbot.sqlite3")
BACKUP_DIR = os.path.join(MD_DIR_PATH, "concept_history")


def normalize_fingerprint(text):
    import re
    if not text:
        return ""
    text = re.sub(r'^\(중복의심\)\s*\[\d+%\]\s*', '', str(text))
    text = text.lower().strip()
    noise_words = ["실전개념", "기본개념", "수학개념", "공식정리", "개념정리", "수학", "개념", "공식", "정리"]
    for word in noise_words:
        text = text.replace(word, "")
    text = re.sub(r'[^a-z0-9가-힣]', '', text)
    return text


def backup_json_file():
    if not os.path.exists(JSON_PATH):
        return None
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"concept_book_backup_{ts}.json")
    shutil.copy2(JSON_PATH, backup_path)
    return backup_path


def load_legacy_json():
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"JSON 파일이 없습니다: {JSON_PATH}")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("concept_book.json 형식 오류: 리스트가 아닙니다")
    return data


def migrate():
    print("=== V28 JSON -> SQLite 마이그레이션 시작 ===")
    print(f"JSON: {JSON_PATH}")
    print(f"SQLite: {SQLITE_PATH}")

    backup_path = backup_json_file()
    if backup_path:
        print(f"[1/4] JSON 백업 완료: {backup_path}")

    legacy = load_legacy_json()
    print(f"[2/4] JSON 로드 완료: {len(legacy)}건")

    payload = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for item in legacy:
        title = item.get("title", "")
        payload.append({
            "title": title,
            "content": item.get("content", ""),
            "fingerprint": item.get("fingerprint") or normalize_fingerprint(title),
            "notion_page_id": item.get("notion_page_id"),
            "created_at": item.get("created_at") or now,
            "last_updated": item.get("last_updated") or item.get("created_at") or now,
        })

    database_manager.init_db()
    database_manager.replace_all_concepts(payload)
    print(f"[3/4] SQLite 저장 완료: {len(payload)}건")

    rows = database_manager.fetch_all_concepts()
    print(f"[4/4] 검증 완료: DB row={len(rows)}")

    if len(rows) != len(payload):
        raise RuntimeError("마이그레이션 건수 불일치")

    print("=== 완료 ===")


if __name__ == "__main__":
    migrate()
