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

- 안정화 보강
  - runner init은 `PPStructureV3()` 단순 호출로 고정(`show_log` 전달 제거).
  - 배치 init 실패 시 `{"ok": false, "page_file": "__BATCH__", ...}` 1줄을 stdout으로 즉시 emit.
  - GUI는 `page_file=="__BATCH__"` 또는 빈 page_file의 `ok=false`를 runner fatal로 간주해 전체 페이지를 실패 처리.
  - GUI에서 결과가 전혀 없는 경우(`saved=0,error=0,done=0`)도 전체 실패로 보정.

- FAST/JSON 보정
  - `fast` 기본에서 `use_region_detection=True`로 동작(anchors 안정성 우선), `table/formula/chart`만 OFF.
  - stdout JSONL emit은 `ensure_ascii=True`로 고정.
  - `pp_json["res"]`가 문자열이면 `json.loads`/`ast.literal_eval`로 파싱 시도, 실패 시 `pp_json`를 비워 안전 JSON만 출력.

- 디버그/검증 강화
  - GUI 옵션으로 runner 원본 출력 저장 지원(`runner_out.jsonl`, `runner_err.log`).
  - fast 모드에서 stderr에 `Chart2Table`/`FormulaNet` 로딩 문자열이 감지되면 경고 1회 출력.
  - `scripts/validate_jsonl.py` 추가: JSONL 파싱 실패 라인 번호 + 앞뒤 1줄 컨텍스트 출력.
  - stderr 요약 로그에 `t_init_ms`, `t_predict_ms`, `t_emit_ms` 분리 출력.

- YAML 기반 fast/full 로딩
  - `v3_isolation_runner.py`는 profile별로 `configs/PP-StructureV3_fast.yaml` / `configs/PP-StructureV3_full.yaml`를 `paddlex_config`로 로드 시도.
  - `scripts/export_ppv3_yaml.py`로 full YAML export 가능 (`configs/PP-StructureV3_full.yaml`).
  - batch init 실패 시 stdout JSONL 1줄(`page_file=__BATCH__`) 출력 후 종료코드 1 반환.

- payload 크기 제어
  - runner CLI에 `--payload {min,full}` 추가(기본 `min`).
  - `min`은 `ok/page_file/timing/anchors/objects/pp_meta`만 stdout에 포함(대용량 `pp_json`/`pp_obj` 미포함).
  - GUI 배치 호출은 기본 `--payload min`, 단일 isolation 호출은 호환 위해 `--payload full`.
  - GUI의 runner 로그 저장 체크박스 기본값을 OFF로 변경.

- 자동 YAML 생성
  - `scripts/export_ppv3_yaml.py`: `PPStructureV3_full.yaml` 자동 export.
  - `scripts/make_fast_yaml.py`: full YAML을 기반으로 fast YAML 자동 생성(구조 유지, 존재 키만 수정).
  - fast 생성 시 `use_doc_preprocessor`는 유지(true), `use_region_detection`은 강제 true.
