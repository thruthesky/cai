# JAI (Job AI)

**JAI(Job AI)**는 실제 채용 공고 데이터를 기반으로, 구인 특화 LLM을 처음부터 학습하여 요약·정리·추천형 응답을 생성하는 프로젝트입니다.

- **공식 명칭**: JAI
- **기술적 분류 명칭**: JobGPT, Job AI, Job GPT, Job AI LLM, Job-specialized LLM
- **전체적 의미**: 구인 도메인에 특화된 LLM 기반 AI

## 프로젝트 목표

- **100% 직접 구현**: 파인튜닝이 아닌, 토크나이저부터 GPT 모델까지 처음부터 학습
- **요약/정리형 출력**: 체크리스트, 구인/구직 정보, 상세 설명을 구조화된 형식으로 생성
- **Mac M4 최적화**: MPS(Metal Performance Shaders) GPU 가속 지원

## 단계별 가이드

| 문서 | 설명 |
|------|------|
| [step-by-step.md](step-by-step.md) | 프로젝트 생성부터 텍스트 생성까지 9단계 가이드 |
| [faq.md](faq.md) | 자주 묻는 질문 (토큰화, 임베딩, 학습 데이터 등) |

## 빠른 시작

```bash
# 1. 의존성 설치 (uv 사용)
uv add torch tokenizers tqdm numpy

# 2. 순차 실행
uv run python scripts/prepare_samples.py      # 데이터 전처리
uv run python scripts/train_tokenizer.py      # 토크나이저 학습
uv run python scripts/build_bin_dataset.py    # 바이너리 데이터셋 생성
uv run python scripts/train_gpt.py            # GPT 모델 학습
uv run python scripts/generate.py             # 텍스트 생성
```

## 프로젝트 구조

```
jai/
├── data/                    # 데이터 디렉토리
│   ├── raw.txt              # 원본 데이터
│   ├── samples.txt          # 전처리된 학습 샘플
│   ├── tokenizer.json       # BPE 토크나이저
│   └── train.bin, val.bin   # 바이너리 데이터셋
├── scripts/                 # 실행 스크립트
│   ├── prepare_samples.py   # 데이터 전처리
│   ├── train_tokenizer.py   # 토크나이저 학습
│   ├── build_bin_dataset.py # 바이너리 변환
│   ├── train_gpt.py         # GPT 학습
│   └── generate.py          # 텍스트 생성
├── checkpoints/             # 모델 체크포인트
├── docs/                    # 상세 기술 문서 (00~08)
├── faq/                     # FAQ 문서 (개념별 분리)
├── faq.md                   # FAQ 목차 및 빠른 참조
├── CLAUDE.md                # AI 어시스턴트 가이드
└── pyproject.toml           # uv 프로젝트 설정
```

## 문서 목록

### 기술 문서 (docs/)

상세한 학습 자료는 `docs/` 폴더에서 확인하세요.

| 문서 | 내용 |
|------|------|
| [00-overview.md](docs/00-overview.md) | 프로젝트 개요, 목표, 출력 형식 예시 |
| [01-environment-setup.md](docs/01-environment-setup.md) | Python 환경, PyTorch, MPS 설정 |
| [02-project-structure.md](docs/02-project-structure.md) | 폴더 구조, 파일 설명, 실행 순서 |
| [03-data-preparation.md](docs/03-data-preparation.md) | 데이터 수집, 전처리, 학습 형식 |
| [04-tokenizer.md](docs/04-tokenizer.md) | BPE 토크나이저 학습 |
| [05-model-architecture.md](docs/05-model-architecture.md) | GPT 모델 아키텍처, 코드 |
| [06-training.md](docs/06-training.md) | 학습 루프, 체크포인트 |
| [07-generation.md](docs/07-generation.md) | 텍스트 생성, 샘플링 파라미터 |
| [08-server.md](docs/08-server.md) | JAI LLM 서버, API, 데몬 실행 |

### FAQ 문서 (faq/)

핵심 개념을 쉽게 설명합니다. 전체 목차는 [faq.md](faq.md)를 참조하세요.

| 문서 | 내용 |
|------|------|
| [data-flow.md](faq/data-flow.md) | 데이터 흐름 파이프라인 |
| [why-job-llm.md](faq/why-job-llm.md) | 구인 정보 LLM 필요성 |
| [sample-data.md](faq/sample-data.md) | 샘플 데이터 요구사항 |
| [sample-data-strategy.md](faq/sample-data-strategy.md) | 샘플 데이터 구성 전략 |
| [why-tokenize.md](faq/why-tokenize.md) | 토큰화 이유 |
| [how-to-tokenize.md](faq/how-to-tokenize.md) | 토큰화 방법 (외부 라이브러리) |
| [vocab-size.md](faq/vocab-size.md) | vocab_size 설명 |
| [after-tokenize.md](faq/after-tokenize.md) | 토큰화 다음 단계 |
| [embedding-and-prediction.md](faq/embedding-and-prediction.md) | 임베딩과 다음 토큰 예측 |
| [core-concepts.md](faq/core-concepts.md) | 핵심 개념 9가지 |
| [troubleshooting.md](faq/troubleshooting.md) | 트러블슈팅 |

## 핵심 개념

1. **Tokenizer**: 텍스트 → 정수 토큰 변환
2. **Embedding**: 토큰 ID → 고차원 벡터
3. **Self-Attention**: 문맥 내 단어 간 관계 학습
4. **Next-token Prediction**: GPT의 유일한 학습 목표
5. **데이터 포맷 = 모델 능력**: 구인/구직 데이터로 학습하면 구인/구직 정보 출력

## 권장 하이퍼파라미터 (M4 기준)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| vocab_size | 24,000 | 토크나이저 어휘 크기 |
| block_size | 256 | 컨텍스트 길이 |
| n_layer | 6 | Transformer 블록 수 |
| n_head | 6 | Attention Head 수 |
| n_embd | 384 | 임베딩 차원 |
| batch_size | 16 | 배치 크기 |

## 출력 예시

```
[ANSWER]
요약:
- 미국 실리콘밸리 소프트웨어 엔지니어 채용 정보
- 지원 마감: 2024-12-31

체크리스트:
- 지원 자격:
  - (1) CS 학위 또는 관련 경력 3년 이상
  - (2) Python, JavaScript 능숙
- 준비물:
  - (1) 이력서 (영문)
  - (2) 포트폴리오

구인 정보:
- Google Inc.
  - 포지션: Senior Software Engineer
  - 연봉: $150,000 - $200,000
  - 위치: Mountain View, CA
  - WEB: https://careers.google.com/

상세 설명:
...
[/ANSWER]
```

## 참고 자료

- [build-nanogpt](https://github.com/karpathy/build-nanogpt) - Karpathy의 GPT 구현 튜토리얼
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/) - BPE 토크나이저 문서
- [PyTorch MPS](https://pytorch.org/docs/stable/notes/mps.html) - Mac GPU 가속 문서
