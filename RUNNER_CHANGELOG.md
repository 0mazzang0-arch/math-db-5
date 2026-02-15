# PPStructureV3 Speed/Robustness Update

## 핵심 변경점

- `v3_isolation_runner.py`
  - `--profile {fast,full}` 추가 (기본 `fast`)
    - `fast`는 layout+ocr 중심으로 무거운 모듈(table/formula/chart/region/orientation/unwarp/seal)을 기본 OFF.
    - `full`은 기존 전체 모듈 ON.
  - `--force_region_detection {-1,0,1}` 추가
    - 기본 `-1`(profile 기본값 사용), `1`이면 region_detection 강제 ON.
  - `fast`에서 `PP-StructureV3-fast.yaml`이 존재하면 우선 로드 시도, 실패 시 flag 기반 init으로 자동 폴백.
  - predict 시에도 동일한 `use_xxx` 플래그 override를 재전달(버전별 init 플래그 무시 대비).
  - 타이밍 메타 강화: `t_init_ms`, `t_predict_ms`, `t_page_total_ms`, `t_emit_ms`.
  - stderr 요약 로그 추가: `P003 predict=xxxxms total=xxxxms`.

- `PP-StructureV3-fast.yaml`
  - 빠른 실행용 미니 설정 템플릿 추가.
  - 환경에 따라 YAML schema가 다를 수 있어, 러너는 YAML 실패 시 자동 폴백하도록 구성.

- `pdf_cutter_experiment_gui.py`
  - 옵션에 `정밀모드(full)` 체크박스 추가 (기본 OFF=fast).
  - GUI 시작 시 `PPSTRUCTURE_V3_ISOLATION=1`을 자동 강제 세팅(수동 PowerShell 입력 불필요).
  - 러너 subprocess 호출마다 `env=os.environ.copy()` 기반으로 isolation env를 명시 전달.
  - 러너 배치 호출 시 `--profile` 전달.
  - 시작 시 현재 profile을 환경변수(`PPSTRUCTURE_V3_PROFILE`)로 반영하여 단일 러너 호출도 동일 profile 사용.
  - fast에서 `anchors=0`인 페이지는 해당 페이지만 자동 재시도:
    - `--profile fast --force_region_detection 1 --warmup 0`
    - 복구되면 진행, 실패하면 기존 에러 격리 로직 유지.

## 실행 예시

```bash
# fast batch (기본)
python v3_isolation_runner.py --pages_dir C:/temp/pages --dpi 250 --profile fast --warmup 1

# full batch
python v3_isolation_runner.py --pages_dir C:/temp/pages --dpi 250 --profile full --warmup 1

# single + region_detection 강제 ON (fallback 테스트용)
python v3_isolation_runner.py C:/temp/pages/P003.png --profile fast --force_region_detection 1 --warmup 0
```
