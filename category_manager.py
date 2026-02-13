# category_manager.py
import os
import json
import re
from config import CATEGORY_FILE_PATH, MD_DIR_PATH

# ==================================================================================
# [System Principle] Complexity is Irrelevant. Result is King.
# 이 모듈은 18,000개의 텍스트 미로를 '계층형 JSON 지도'로 변환하고,
# 폴더명(대분류)이라는 나침반을 이용해 Level 3~4 수준의 최적 태그를 찾아냅니다.
# ==================================================================================

CACHE_FILE_PATH = os.path.join(MD_DIR_PATH, "category_map.json")

class CategoryBrain:
    def __init__(self):
        self.category_tree = {} # { "수학1": { "지수": { ... } } }
        self.is_ready = False
        self.initialize_brain()

    def initialize_brain(self):
        """
        [지도 구축 프로토콜]
        1. 캐시된 지도(JSON)가 있으면 0.1초 만에 로드.
        2. 없으면 텍스트 파일(Raw Data)을 파싱하여 지도를 새로 제작(Build).
        """
        if os.path.exists(CACHE_FILE_PATH):
            try:
                with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
                    self.category_tree = json.load(f)
                self.is_ready = True
                print(f"🧠 [Brain] 고속 지도(Cache) 로드 완료.")
                return
            except Exception as e:
                print(f"⚠️ [Brain] 캐시 손상. 재구축합니다. ({e})")

        # 캐시가 없거나 손상되었으면 원본 텍스트 파싱
        self.build_map_from_txt()

    def build_map_from_txt(self):
        """
        [Stack Machine Parser]
        들여쓰기(공백, 탭)나 기호를 분석하여 부모-자식 관계를 추적하는 강력한 파서.
        어떤 더러운 포맷의 텍스트가 와도 논리적 계층구조(Tree)로 변환해냅니다.
        """
        if not os.path.exists(CATEGORY_FILE_PATH):
            print(f"❌ [Brain] 분류 파일 없음: {CATEGORY_FILE_PATH}")
            return

        print("🏗️ [Brain] 텍스트 파일 분석 및 지도 구축 중... (최초 1회 실행)")
        tree = {}
        path_stack = [] # 현재 위치를 추적하는 스택 [(level, name), ...]

        try:
            with open(CATEGORY_FILE_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                raw_line = line.rstrip()
                if not raw_line: continue

                # 1. 들여쓰기 레벨 계산 (탭=4공백 치환)
                clean_line = raw_line.replace('\t', '    ')
                indent_level = (len(clean_line) - len(clean_line.lstrip())) // 2 # 2칸을 1레벨로 간주
                
                # 2. 내용 정제 (특수문자 제거, 순수 텍스트만)
                # 대괄호, 번호 등 제거하고 핵심 키워드만 남김
                content = re.sub(r'^[0-9\.\-\(\)\[\]]+', '', clean_line.strip()).strip()
                content = re.sub(r'[\[\]]', '', content) # 혹시 남은 대괄호 제거
                
                if not content: continue

                # 3. 스택 조정 (현재 레벨보다 깊은 애들은 팝)
                while path_stack and path_stack[-1][0] >= indent_level:
                    path_stack.pop()
                
                # 4. 트리 구성
                current_node = tree
                for _, p_name in path_stack:
                    if p_name not in current_node:
                        current_node[p_name] = {}
                    current_node = current_node[p_name]
                
                # 현재 항목 등록
                if content not in current_node:
                    current_node[content] = {}
                
                # 스택에 푸시
                path_stack.append((indent_level, content))

            # 캐시 저장
            with open(CACHE_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(tree, f, ensure_ascii=False, indent=2)
            
            self.category_tree = tree
            self.is_ready = True
            print("✅ [Brain] 지도 구축 완료 & 캐시 저장.")

        except Exception as e:
            print(f"💣 [Brain] 지도 구축 실패: {e}")

    def search_best_path(self, folder_guide, ocr_text):
        """
        [핵심 알고리즘]
        1. Folder Guide(대분류)로 탐색 범위를 좁힘 (가지치기).
        2. OCR Text에 있는 단어가 트리 노드에 있는지 전수 조사 (Recursive Search).
        3. 가장 깊고 정확한 매칭을 찾되, Level 3~4 수준으로 부모를 리턴.
        """
        if not self.is_ready: return []
        
        # 1. 대분류 진입 (폴더명과 유사한 최상위 키 찾기)
        target_root = None
        
        # 폴더명 정규화 (예: "[수학1]" -> "수학1")
        clean_folder = re.sub(r'[\[\]_\d]', '', folder_guide).strip()
        
        for root_key in self.category_tree.keys():
            # 폴더명이 트리의 대분류에 포함되거나, 대분류가 폴더명에 포함되면 진입
            if clean_folder in root_key or root_key in clean_folder:
                target_root = self.category_tree[root_key]
                break
        
        # 대분류를 못 찾으면 전체 트리에서 검색 (Fallback)
        search_scope = target_root if target_root else self.category_tree
        
        # 2. 재귀적 키워드 매칭
        # 모든 노드를 순회하며 OCR 텍스트에 등장하는 키워드를 찾음
        candidates = [] # (depth, path_list)
        
        def traverse(node, current_path, level):
            node_name = current_path[-1] if current_path else ""
            
            # 검색: 현재 노드 이름이 OCR 텍스트에 있는가?
            # (단, 2글자 이상이어야 함. '수', '식' 같은 1글자는 노이즈)
            if len(node_name) >= 2 and node_name in ocr_text:
                candidates.append((level, list(current_path)))
            
            # 자식 노드 순회
            for child_name, child_node in node.items():
                traverse(child_node, current_path + [child_name], level + 1)

        # 탐색 시작
        if isinstance(search_scope, dict):
            for r_key, r_node in search_scope.items():
                traverse(r_node, [r_key], 1)
        
        if not candidates:
            return []

        # 3. 최적 후보 선정
        # 전략: 가장 깊은(구체적인) 매칭을 찾은 뒤, 역으로 Level 3~4 부모를 리턴
        # 정렬 기준: Depth(깊이) 내림차순 -> 길이 내림차순
        candidates.sort(key=lambda x: (x[0], len(x[1][-1])), reverse=True)
        
        best_match = candidates[0] # (level, path_list)
        best_path = best_match[1]
        
        # 4. Level 3~4 조정 (User Requirement)
        # 경로가 [대분류, 중분류, 소분류, 세분류, ...] 일 때
        # 인덱스 2(Level 3) 또는 3(Level 4)까지 잘라서 리턴
        
        cut_index = min(len(best_path), 4) # 최대 Level 4까지만
        final_tags = best_path[:cut_index]
        
        return final_tags

# 전역 인스턴스 (Singleton)
brain = CategoryBrain()

def get_suggested_tags(folder_name, ocr_text):
    """
    외부(main.py)에서 호출하는 유일한 인터페이스.
    입력: "[수학1]", "지수함수의 그래프가..."
    출력: ["수학1", "지수함수", "지수함수의 활용"]
    """
    try:
        # 1. 브레인 가동하여 경로 탐색
        found_tags = brain.search_best_path(folder_name, ocr_text)
        
        # 2. 태그가 없으면 기본값(폴더명)이라도 리턴
        if not found_tags:
            # 대괄호 제거된 폴더명
            clean_folder = re.sub(r'[\[\]]', '', folder_name).strip()
            return [clean_folder]
            
        return found_tags
        
    except Exception as e:
        print(f"💣 [Tagging Error] {e}")
        return [folder_name] # 에러나면 본전(폴더명)치기