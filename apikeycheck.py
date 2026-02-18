# apikeycheck.py
import os
from google import genai
from config import GOOGLE_API_KEYS

# 키 로드 실패 시 안전장치
if not GOOGLE_API_KEYS:
    print("🚨 오류: .env 파일에서 GOOGLE_API_KEYS를 찾을 수 없습니다.")
    exit()

# 첫 번째 키로 클라이언트 생성
client = genai.Client(api_key=GOOGLE_API_KEYS[0])

print(f"🔍 현재 내 API 키({GOOGLE_API_KEYS[0][:5]}***)로 사용 가능한 모델 검색 중...\n")

try:
    # ⚠️ 수정됨: client.models.list_models() -> client.models.list()
    # 최신 SDK에서는 메서드 이름이 'list'로 짧아졌습니다.
    for m in client.models.list():
        # 모델 이름에서 'models/' 접두사 제거 (예: models/gemini-pro -> gemini-pro)
        name = m.name.split("/")[-1]
        
        # 'gemini'가 포함된 모델만 필터링해서 보기 좋게 출력
        if "gemini" in name:
            if "thinking" in name or "deep" in name:
                print(f"🌟 [대박 발견] {name} (추론 모델)")
            elif "exp" in name:
                print(f"🧪 [실험 버전] {name}")
            else:
                print(f"   - {name}")

except Exception as e:
    print(f"❌ 에러 발생: {e}")