# concept_sync.py
import json
import os
import requests
import time
import logging
from config import NOTION_API_KEY, NOTION_CONCEPT_DB_ID, MD_DIR_PATH

# 로깅 설정 (Sync 전용)
logging.basicConfig(filename='concept_sync.log', level=logging.INFO, format='%(asctime)s %(message)s')

HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json"
}

JSON_PATH = os.path.join(MD_DIR_PATH, "concept_book.json")

def robust_request(method, url, payload=None, retries=5):
    """
    [지침] 3번이 아니라 5번 재시도. 끈질기게 붙는다.
    """
    for attempt in range(retries):
        try:
            if method == "POST": res = requests.post(url, headers=HEADERS, json=payload, timeout=10)
            elif method == "PATCH": res = requests.patch(url, headers=HEADERS, json=payload, timeout=10)
            else: res = requests.get(url, headers=HEADERS, timeout=10)
            
            if res.status_code == 200: return res
            
            # 400번대 에러(Client Error)는 재시도해도 소용없음 -> 바로 리턴
            if 400 <= res.status_code < 500:
                logging.error(f"❌ [Client Error] {res.status_code}: {res.text}")
                return None
                
            logging.warning(f"⚠️ [Server Error] {res.status_code}. Retrying ({attempt+1}/{retries})...")
            time.sleep(2 * (attempt + 1)) # 지수 백오프
            
        except Exception as e:
            logging.error(f"💣 [Network Exception] {e}. Retrying...")
            time.sleep(2)
    return None

def get_existing_map():
    print("📡 노션 DB 스캔 중...", end="")
    concept_map = {} 
    
    url = f"https://api.notion.com/v1/databases/{NOTION_CONCEPT_DB_ID}/query"
    has_more = True
    next_cursor = None

    while has_more:
        payload = {"page_size": 100}
        if next_cursor: payload["start_cursor"] = next_cursor

        res = robust_request("POST", url, payload)
        if not res: 
            print("❌ 지도 확보 실패")
            return None 
        
        data = res.json()
        for page in data["results"]:
            try:
                page_id = page["id"]
                props = page["properties"]
                # 제목 추출 (방어적으로)
                title_list = props.get("개념명", {}).get("title", [])
                if title_list:
                    title_text = title_list[0]["plain_text"]
                    # 여기서도 공백 제거 버전으로 매핑 (Manager와 동일 로직 적용은 아님, 단순 ID 조회용)
                    normalized_key = title_text.replace(" ", "")
                    concept_map[normalized_key] = page_id
            except: continue
        
        has_more = data.get("has_more", False)
        next_cursor = data.get("next_cursor")
        print(".", end="")
    
    print(f"\n✅ 지도 완료: {len(concept_map)}개")
    return concept_map

def update_concept_page(page_id, title, content, image_url=None):
    """
    [강화된 업데이트]
    내용이 2000자를 넘어가면 노션 API가 에러를 뱉음.
    따라서 '내용' 속성(Property)에는 앞부분 2000자만 넣고,
    전체 내용은 페이지 본문(Children)에 블록으로 쏴야 함.
    하지만 사용자 요구상 '속성' 업데이트가 우선이므로 2000자 컷팅 방어를 확실히 함.
    """
    url = f"https://api.notion.com/v1/pages/{page_id}"
    
    # 노션 RichText 한계: 2000자
    safe_content = content[:2000] if content else ""
    
    payload = {
        "properties": {
            "개념명": {"title": [{"text": {"content": title}}]},
            "내용": {"rich_text": [{"text": {"content": safe_content}}]}
        }
    }
    
    res = robust_request("PATCH", url, payload)
    
    # [추가] 만약 내용이 바뀌어서 본문에도 업데이트가 필요하다면?
    # 일단 요구사항은 '중복 방지'이므로 속성 업데이트에 집중.
    return True if res else False

def create_concept_page(concept, image_url=None):
    url = "https://api.notion.com/v1/pages"
    title = concept.get("title", "제목없음")
    content = concept.get("content", "")
    
    safe_content = content[:2000]

    children = [
        {
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": [{"text": {"content": safe_content}}],
                "icon": {"emoji": "💡"}
            }
        }
    ]
    
    if image_url:
        children.append({
            "object": "block",
            "type": "image",
            "image": {"type": "external", "external": {"url": image_url}}
        })

    payload = {
        "parent": {"database_id": NOTION_CONCEPT_DB_ID},
        "properties": {
            "개념명": {"title": [{"text": {"content": title}}]},
            "내용": {"rich_text": [{"text": {"content": safe_content}}]}
        },
        "children": children
    }

    res = robust_request("POST", url, payload)
    if res:
        try: return res.json()["id"]
        except: return True
    return False

def append_image_to_page(page_id, image_url):
    if not image_url: return False
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    payload = {
        "children": [
            {
                "object": "block",
                "type": "image",
                "image": {"type": "external", "external": {"url": image_url}}
            }
        ]
    }
    res = robust_request("PATCH", url, payload)
    return True if res else False

def delete_concept_page(page_id):
    url = f"https://api.notion.com/v1/pages/{page_id}"
    payload = {"archived": True}
    res = robust_request("PATCH", url, payload)
    return True if res else False