# DEVELOPMENT

## 문서 원칙

이 레포의 개발 문서는 `docs/` 아래만 둔다.

- 임시 문서를 루트나 다른 경로에 만들지 않는다.
- 변경 사항을 누적 로그처럼 덧붙이지 않는다.
- 구조 변경이 있으면 `docs/PROJECT.md`를 최신 상태로 갱신한다.

가장 중요한 문서는:
- `docs/PROJECT.md`

작업 전에 모듈 경계와 호출 흐름을 확인하고, 변경 후에는 이 문서가 코드와 일치하도록 유지한다.

## 현재 개발 패턴

### 1. 모델 수정

모델 아키텍처 연구는 아래 레이어에서 진행한다.

- `src/nested_learning/model.py`
- `src/nested_learning/hope/block.py`
- `src/nested_learning/titan/*`
- `src/nested_learning/cms.py`
- `src/nested_learning/backbones.py`
- `src/nested_learning/fast_state.py`
- `src/nested_learning/optim/*`

권장 방식:
- 외부 공개 인터페이스는 유지
- 내부 구현은 자유롭게 교체
- 변경 후 관련 테스트 파일 케이스를 업데이트

### 2. 외부 사용 표면 수정

외부 사용 표면은 아래 네 축만 기준으로 본다.

- model build: `factory.py`
- train: `trainer.py`
- infer: `inference.py`
- checkpoint: `checkpoint.py`

이 레이어의 함수 시그니처를 바꿀 때는:
- `api.py`
- `__init__.py`
- `cli.py`
- `docs/PROJECT.md`

를 함께 갱신한다.

### 3. train.py의 위치

루트 `train.py`는 thin wrapper다.

용도:
- `python train.py --config-name ...` 형태의 직접 실행

비용:
- 중복 엔트리포인트처럼 보일 수 있음

원칙:
- 실제 학습 로직은 넣지 않는다.
- `trainer.run_training_loop()` 호출만 담당하게 유지한다.

## 설정 관리

현재 유지되는 config는 소수다.

- `configs/pilot.yaml`
- `configs/pilot_smoke.yaml`
- `configs/hope/pilot.yaml`
- `configs/hope/pilot_attention.yaml`
- `configs/hope/pilot_selfmod.yaml`
- `configs/hope/pilot_transformer.yaml`

규칙:
- 실험성 config를 대량으로 늘리지 않는다.
- 새로운 variant가 필요하면 기존 config를 기반으로 최소 개수만 추가한다.
- 사용되지 않는 field가 늘어나면 코드 또는 config를 정리해 불일치를 줄인다.

## 테스트 관리

테스트는 모두 `tests/` 아래 모듈 단위로 유지한다.

원칙:
- 새 파일을 무분별하게 만들지 않는다.
- 기존 테스트 파일에 케이스를 추가하는 쪽을 우선한다.
- 내부 구현이 아니라 인터페이스와 동작 기준으로 검증한다.

권장 매핑:
- model/block 변경: `tests/test_model.py`, `tests/test_hope_block.py`, `tests/test_variants.py`
- CMS 변경: `tests/test_cms*.py`
- selfmod 변경: `tests/test_self_modifying_titans.py`, `tests/test_selfmod_*.py`
- optimizer 변경: `tests/test_optim.py`
- teach signal 변경: `tests/test_teach_signal.py`

## 작업 체크리스트

### 아키텍처 변경 전

1. `docs/PROJECT.md`에서 관련 모듈 책임과 의존 관계를 확인한다.
2. 변경이 공개 인터페이스인지 내부 구현인지 먼저 구분한다.

### 아키텍처 변경 후

1. `docs/PROJECT.md` 업데이트
2. 관련 테스트 갱신
3. `README.md`의 공개 사용법이 바뀌었으면 함께 수정

## 공개 인터페이스 요약

Python:

- `build_model`
- `build_model_from_cfg`
- `build_optimizer`
- `build_dataloader`
- `train_step`
- `run_training_loop`
- `next_token_loss`
- `generate`
- `save_checkpoint`
- `load_checkpoint`

CLI:

- `nl smoke`
- `nl train`
- `nl infer`

이 집합이 현재 외부 계약이다. 내부 리팩터링 시 우선적으로 보존한다.
