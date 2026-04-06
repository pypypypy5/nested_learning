# PROJECT

## 목적

이 레포는 Google Nested Learning / HOPE 계열 메커니즘을 차용한 policy 모델 연구용 코어 레포다.

현재 목표는 다음 두 가지다.

- 모델 아키텍처와 내부 메커니즘을 빠르게 수정하며 연구할 수 있을 것
- 외부에서는 최소한의 표면으로 `build -> train -> save/load -> infer`만 수행할 수 있을 것

이 문서는 현재 코드베이스의 단일 소스 오브 트루스다. 모듈 경계, 입출력, 의존 관계, 호출 흐름은 여기 기준으로 파악한다.

## 최상위 구조

- `src/nested_learning/`
  모델 코어와 외부 공개 API가 있는 패키지 본체
- `configs/`
  로컬 연구 실행용 Hydra config
- `tests/`
  모듈 인터페이스 기준 테스트
- `docs/`
  최신 개발 문서
- `train.py`
  Hydra 기반 로컬 학습 진입 래퍼

## 공개 인터페이스

### Python API

패키지 최상위 import로 바로 접근 가능하다.

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

구현 위치:
- `nested_learning.__init__`
- `nested_learning.api`

### CLI

`nl` 엔트리포인트는 세 가지 명령만 제공한다.

- `nl smoke`
- `nl train`
- `nl infer`

구현 위치:
- `src/nested_learning/cli.py`

### 루트 스크립트

- `train.py`

역할:
- Hydra config를 읽는다.
- device를 resolve한다.
- `nested_learning.trainer.run_training_loop`를 호출한다.

주의:
- 실제 학습 로직은 여기 없고 thin wrapper다.
- 유지 이유는 `python train.py --config-name ...` 형태의 직접 실행 편의성 때문이다.

## 모듈 목록

### 1. 외부 API 레이어

#### `src/nested_learning/__init__.py`

역할:
- 패키지 버전 노출
- 공개 API 재노출

입력:
- 없음

출력:
- import 가능한 공개 심볼 집합

side effect:
- 없음

#### `src/nested_learning/api.py`

역할:
- 외부에서 쓸 함수형 API 집합을 한곳에 모은다.

입력:
- 내부 모듈 함수 import

출력:
- 일관된 공개 API

side effect:
- 없음

### 2. 실행/오케스트레이션 레이어

#### `src/nested_learning/cli.py`

역할:
- 최소 CLI 표면 제공

명령:
- `smoke`: config로 모델 생성 후 랜덤 토큰 forward
- `train`: config 기반 로컬 train loop 실행
- `infer`: config + optional checkpoint 기반 generation

입력:
- CLI 인자
- Hydra config

출력:
- JSON 형태 stdout

side effect:
- checkpoint 로드 가능
- 학습 시 checkpoint 저장 가능

#### `src/nested_learning/config_utils.py`

역할:
- Hydra config directory resolve
- config compose

입력:
- config name
- overrides
- optional config dir

출력:
- resolved `DictConfig`

side effect:
- Hydra global state 초기화

#### `src/nested_learning/factory.py`

역할:
- config/dataclass로부터 모델, optimizer, dataloader를 생성
- grouped Hydra config unwrap

핵심 함수:
- `unwrap_config`
- `build_model_from_cfg`
- `build_model`
- `build_optimizer`
- `build_dataloader`

입력:
- `DictConfig`, `dict`, dataclass config

출력:
- `torch.nn.Module`
- `torch.optim.Optimizer`
- `DataLoader`

side effect:
- 없음

의존:
- `data.py`
- `model.py`
- `titan/model.py`
- `optim/m3.py`
- `levels.py`

#### `src/nested_learning/trainer.py`

역할:
- 현재 레포의 로컬 학습 규칙 정의
- next-token loss 계산
- teach signal 계산
- 단일 train step 및 간단한 train loop 실행

핵심 함수:
- `compute_teach_signal`
- `next_token_loss`
- `train_step`
- `run_training_loop`

입력:
- model
- optimizer
- token batch
- config

출력:
- step metrics dict

side effect:
- model parameter update
- 내부 memory update
- optional checkpoint 저장

의존:
- `factory.py`
- `checkpoint.py`

주의:
- 현재 `train_step`은 next-token CE 기반 학습만 구현한다.
- `cfg.train` 아래 많은 legacy field가 남아 있어도 실제로는 일부만 소비한다.

현재 실사용 train config field:
- `train.steps`
- `train.log_interval`
- `train.device`
- `train.save_path` (있을 때만 사용)

#### `src/nested_learning/inference.py`

역할:
- autoregressive generation 제공

핵심 함수:
- `generate`

입력:
- model
- prompt token tensor
- generation hyperparameters

출력:
- generated token tensor

side effect:
- 없음

주의:
- `torch.inference_mode()` 사용
- tokenizer/text layer는 현재 레포 범위 밖이다. 입력은 token ids 기준이다.

#### `src/nested_learning/checkpoint.py`

역할:
- model/optimizer state 저장 및 로드

핵심 함수:
- `save_checkpoint`
- `load_checkpoint`

입력:
- model
- optional optimizer
- path

출력:
- loaded payload dict

side effect:
- 파일시스템 read/write

### 3. 모델 조립 레이어

#### `src/nested_learning/model.py`

역할:
- 최상위 `HOPEModel` 정의
- block variant 조립

지원 block variant:
- `hope_attention`
- `hope_hybrid`
- `hope_selfmod`
- `transformer`

입력:
- `ModelConfig`
- token tensor
- optional teach signal / fast state / attention cache

출력:
- logits
- optional caches / block outputs

side effect:
- teach signal이 주어지면 block 내부 memory update 발생 가능

#### `src/nested_learning/transformer.py`

역할:
- baseline transformer block

입력:
- hidden states

출력:
- transformed hidden states

side effect:
- 없음

#### `src/nested_learning/backbones.py`

역할:
- shared self-attention backbone

입력:
- hidden states
- optional KV cache

출력:
- attended hidden states
- optional updated KV cache

side effect:
- 없음

### 4. 메모리/업데이트 메커니즘 레이어

#### `src/nested_learning/cms.py`

역할:
- multi-level CMS block chain

입력:
- hidden states

출력:
- transformed hidden states
- optional intermediates

side effect:
- forward 자체는 없음
- update는 상위 block이 관리

#### `src/nested_learning/hope/block.py`

역할:
- HOPE 계열 block 구현

구성:
- attention + CMS
- attention + selfmod + CMS
- attention + TITAN + CMS

입력:
- hidden states
- optional teach signal
- optional fast state
- optional surprise value

출력:
- hidden states
- optional attention cache

side effect:
- teach signal이 주어지면 CMS/TITAN/selfmod update 가능

#### `src/nested_learning/hope/self_mod.py`

역할:
- TITAN/HOPE update에 사용되는 self modifier 유틸리티 제공

#### `src/nested_learning/titan/memory.py`

역할:
- TITAN associative memory 구현

입력:
- query 또는 update용 key/value/error

출력:
- memory output

side effect:
- `update()`는 메모리 파라미터 직접 갱신

#### `src/nested_learning/titan/model.py`

역할:
- TITAN-only 모델 변형

입력:
- token tensor
- optional teach signal / fast state

출력:
- logits

side effect:
- teach signal이 있으면 TITAN memory update 가능

#### `src/nested_learning/titan/self_modifying.py`

역할:
- Self-Modifying TITANs 메커니즘 구현

입력:
- hidden states
- fast state

출력:
- transformed hidden states
- updated fast state

side effect:
- self-modifying memory state update

### 5. 상태/함수형 호출/스케줄 레이어

#### `src/nested_learning/fast_state.py`

역할:
- online update용 fast state 정의

입력:
- module들
- level spec

출력:
- block/model fast state
- attention cache container

side effect:
- 없음

#### `src/nested_learning/functional.py`

역할:
- param dict / delta dict 기반 함수형 호출 유틸리티

사용처:
- fast state 경로
- differentiable / delta-based update 경로

#### `src/nested_learning/levels.py`

역할:
- update frequency를 정의하는 level spec / clock 제공

입력:
- `LevelSpec` 목록

출력:
- 현재 step에서 update 여부

#### `src/nested_learning/assoc_memory.py`

역할:
- associative memory protocol / base class

### 6. Optimizer 레이어

#### `src/nested_learning/optim/deep.py`

역할:
- 내부 memory update용 `DeepMomentum`

주의:
- 이것은 outer training optimizer와 다르다.
- HOPE/TITAN/CMS 내부 업데이트에 사용된다.

#### `src/nested_learning/optim/factory.py`

역할:
- 내부 `DeepMomentum` 생성 팩토리

사용처:
- `optim/manager.py`

#### `src/nested_learning/optim/manager.py`

역할:
- level clock에 따라 내부 update optimizer 적용

입력:
- level spec
- grads / context

출력:
- updated params 혹은 in-place module update

#### `src/nested_learning/optim/m3.py`

역할:
- outer optimizer로 사용할 수 있는 M3 구현

## 호출 흐름

### Python API 학습

1. `compose_config()` 또는 직접 config 준비
2. `build_model_from_cfg()`
3. `build_optimizer()`
4. `train_step()` 또는 `run_training_loop()`
5. 선택적으로 `save_checkpoint()`

### CLI 학습

1. `nl train`
2. `cli.py`가 config compose
3. `trainer.run_training_loop()`
4. `factory.py`에서 model/optimizer/dataloader 생성
5. `trainer.py`가 outer gradient step + teach signal update 수행

### CLI 추론

1. `nl infer`
2. `cli.py`가 config compose
3. `factory.build_model_from_cfg()`
4. optional `checkpoint.load_checkpoint()`
5. `inference.generate()`

### 루트 스크립트 학습

1. `python train.py --config-name ...`
2. `train.py`가 Hydra wrapper 역할 수행
3. 내부적으로 `trainer.run_training_loop()` 호출

## 모듈 의존 관계

### 공개 API -> 실행 레이어

- `__init__.py` -> `api.py`, `factory.py`, `trainer.py`, `inference.py`, `checkpoint.py`
- `api.py` -> `factory.py`, `trainer.py`, `inference.py`, `checkpoint.py`
- `cli.py` -> `config_utils.py`, `factory.py`, `trainer.py`, `inference.py`, `checkpoint.py`

### 실행 레이어 -> 모델 조립 레이어

- `factory.py` -> `model.py`, `titan/model.py`
- `trainer.py` -> `factory.py`

### 모델 조립 레이어 -> 메커니즘 레이어

- `model.py` -> `hope/block.py`, `transformer.py`, `fast_state.py`
- `titan/model.py` -> `titan/memory.py`, `hope/self_mod.py`, `fast_state.py`, `optim/manager.py`
- `hope/block.py` -> `cms.py`, `backbones.py`, `titan/memory.py`, `titan/self_modifying.py`, `optim/manager.py`, `fast_state.py`

### 메커니즘 레이어 -> 스케줄/상태/optimizer 레이어

- `fast_state.py` -> `optim/manager.py`
- `optim/manager.py` -> `levels.py`, `optim/factory.py`
- `optim/factory.py` -> `optim/deep.py`

## 현재 설정 표면

### 유지되는 config

- `configs/pilot.yaml`
- `configs/pilot_smoke.yaml`
- `configs/hope/pilot.yaml`
- `configs/hope/pilot_attention.yaml`
- `configs/hope/pilot_selfmod.yaml`
- `configs/hope/pilot_transformer.yaml`

### 실제 실행에서 핵심적으로 소비되는 필드

모델 생성:
- `model.*`

outer optimizer:
- `optim.type`
- `optim.lr`
- `optim.weight_decay`
- `optim.betas`
- `optim.fused`
- `optim.beta1/beta2/beta3/alpha/eps/ns_steps/slow_chunk` (`m3`일 때)
- `optim.momentum` (`muon`일 때)

데이터:
- `data.source`
- `data.vocab_size`
- `data.seq_len`
- `data.dataset_size`
- `data.batch_size`
- `data.num_workers`
- `data.shards_dir`

학습:
- `train.steps`
- `train.log_interval`
- `train.device`
- `train.save_path`

### 현재 남아 있지만 최소 train loop에서 직접 사용하지 않는 legacy field

예:
- `train.algorithm_mode`
- `train.strict_streaming_contract`
- `train.online_updates`
- `train.online_chunk_size`
- `train.online_boundary_targets`
- `train.online_carry_attention_cache`
- `train.per_layer_teach_signal`
- `train.mixed_precision`
- `train.compile`
- `train.checkpoint.*`
- `logging.*`

이 필드들은 모델 내부 메커니즘 의미를 담고 있거나 과거 실행 표면의 잔재이므로, 새 기능 추가 전에 필요 여부를 먼저 판단하고 정리한다.

## 테스트 기준

테스트는 모두 `tests/` 아래 모듈 단위로 유지한다.

현재 남은 테스트는 아래 축을 보호한다.

- block/model forward shape와 variant 생성
- CMS update/cadence/flush
- fast state semantics 일부
- self-modifying TITANs 메커니즘
- internal optimizer 동작
- teach signal 계산

새 테스트를 추가할 때는 중복 파일을 만들기보다 해당 모듈의 기존 테스트 파일에 케이스를 추가한다.

## 변경 원칙

새 기능 추가 또는 리팩터링 시 순서는 다음을 따른다.

1. 먼저 모듈 경계와 공개 인터페이스를 정한다.
2. 이 문서의 모듈 설명과 의존 관계를 업데이트한다.
3. 구현을 수정한다.
4. 기존 테스트 파일에 인터페이스 기준 테스트를 추가/수정한다.

이 문서는 작업 로그가 아니라 최신 구조 명세다. 구조가 바뀌면 덧붙이지 말고 현재 상태 기준으로 갱신한다.
