# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 언어

모든 응답은 한국어로 작성합니다. 코드 주석도 한국어로 작성합니다.

## 프로젝트 개요

JAI (Job AI)는 전 세계 구인/구직 정보를 제공하는 Job LLM을 처음부터(from scratch) 구현하는 학습 프로젝트입니다. 파인튜닝이 아닌, 토크나이저부터 GPT 모델까지 직접 구현합니다.

## 프로젝트 구조

```
jai/
├── data/
│   ├── raw.txt              # 원본 데이터
│   ├── samples.txt          # 전처리된 학습 샘플
│   ├── tokenizer.json       # BPE 토크나이저
│   └── train.bin, val.bin   # 바이너리 데이터셋
├── scripts/
│   ├── prepare_samples.py   # 데이터 전처리
│   └── train_tokenizer.py   # 토크나이저 학습
├── checkpoints/
│   └── ckpt.pt              # 모델 체크포인트
├── docs/                    # 상세 기술 문서
│   ├── 00-overview.md
│   ├── 01-environment-setup.md
│   ├── 02-project-structure.md
│   ├── 03-data-preparation.md
│   ├── 04-tokenizer.md
│   ├── 05-model-architecture.md
│   ├── 06-training.md
│   ├── 07-generation.md
│   └── 08-server.md
├── faq/                     # FAQ 문서 (개념별 분리)
│   ├── data-flow.md
│   ├── why-job-llm.md
│   ├── sample-data.md
│   ├── why-tokenize.md
│   ├── how-to-tokenize.md
│   ├── vocab-size.md
│   ├── after-tokenize.md
│   ├── core-concepts.md
│   └── troubleshooting.md
├── faq.md                   # FAQ 목차 및 빠른 참조
├── CLAUDE.md                # AI 어시스턴트 가이드
├── README.md                # 프로젝트 소개
└── pyproject.toml           # uv 프로젝트 설정
```

## 실행 명령어

```bash
# 의존성 설치 (uv 사용)
uv add torch tokenizers tqdm numpy

# 순차 실행 (순서 중요)
uv run python scripts/prepare_samples.py      # 데이터 전처리 → data/samples.txt
uv run python scripts/train_tokenizer.py      # BPE 토크나이저 → data/tokenizer.json
uv run python scripts/build_bin_dataset.py    # 바이너리 변환 → data/train.bin, data/val.bin
uv run python scripts/train_gpt.py            # GPT 학습 → checkpoints/ckpt.pt
uv run python scripts/generate.py             # 텍스트 생성
```

## 아키텍처

### 데이터 흐름
```
data/raw.txt → scripts/prepare_samples.py → samples.txt → 토큰화 → train.bin/val.bin → GPT 학습 → 생성
```

### GPT 모델 구조 (Decoder-only Transformer)
- `CausalSelfAttention`: Q, K, V를 한번에 계산, causal mask로 미래 토큰 차단
- `MLP`: 2층 Feed Forward (n_embd → 4*n_embd → n_embd), GELU 활성화
- `Block`: Pre-LayerNorm + Residual Connection
- `GPT`: 토큰/위치 임베딩 → N개 Block → 출력 헤드

### 학습 데이터 형식
```
[QUESTION]
질문 내용
[/QUESTION]

[DOC]
원문 내용
[/DOC]

[ANSWER]
요약:
- ...
체크리스트:
- ...
구인 정보:
- 회사: ...
- 포지션: ...
- 연봉: ...
상세 설명:
...
[/ANSWER]
```

## 하이퍼파라미터 (M4 기준)

| 파라미터 | 기본값 |
|----------|--------|
| vocab_size | 24,000 |
| block_size | 256 |
| n_layer | 6 |
| n_head | 6 |
| n_embd | 384 |
| batch_size | 16 |
| learning_rate | 3e-4 |

## 문서 참조

### 기술 문서 (docs/)

| 문서 | 설명 |
|------|------|
| `docs/00-overview.md` | 프로젝트 개요 |
| `docs/01-environment-setup.md` | 환경 설정 |
| `docs/02-project-structure.md` | 프로젝트 구조 |
| `docs/03-data-preparation.md` | 데이터 준비 |
| `docs/04-tokenizer.md` | 토크나이저 |
| `docs/05-model-architecture.md` | 모델 구조 |
| `docs/06-training.md` | 학습 코드 |
| `docs/07-generation.md` | 텍스트 생성 |
| `docs/08-server.md` | 서버 구축 |

### FAQ 문서 (faq/)

| 문서 | 설명 |
|------|------|
| `faq/data-flow.md` | 데이터 흐름 파이프라인 |
| `faq/why-job-llm.md` | 구인 정보 LLM 필요성 |
| `faq/sample-data.md` | 샘플 데이터 요구사항 |
| `faq/why-tokenize.md` | 토큰화 이유 |
| `faq/how-to-tokenize.md` | 토큰화 방법 (외부 라이브러리) |
| `faq/vocab-size.md` | vocab_size 설명 |
| `faq/after-tokenize.md` | 토큰화 다음 단계 |
| `faq/core-concepts.md` | 핵심 개념 9가지 |
| `faq/troubleshooting.md` | 트러블슈팅 |

## FAQ 문서 유지 관리

### 문서 구조

FAQ 문서는 두 가지로 구성됩니다:
- **`faq.md`**: 목차 및 빠른 참조 (인덱스 역할)
- **`faq/*.md`**: 개별 주제별 상세 문서

### 새 FAQ 추가 방법

1. **파일 생성**: `faq/` 폴더에 새 마크다운 파일 생성
   - 파일명: 숫자 없이, 케밥 케이스 사용 (예: `new-topic.md`)

2. **문서 작성**: 아래 템플릿 사용
   ```markdown
   # 제목

   ## 한 줄 요약 또는 결론

   ---

   ## 본문 섹션들
   ...

   ---

   ## 요약

   | 질문 | 답변 |
   |------|------|
   | ... | ... |

   ---

   ## 관련 문서

   - [문서명](파일명.md)
   ```

3. **목차 업데이트**: `faq.md`의 문서 목록에 새 문서 추가
   - 적절한 카테고리(프로젝트 개요/토큰화/심화 학습)에 배치

4. **상호 링크**: 관련 문서들의 "관련 문서" 섹션에 새 문서 링크 추가

### 문서 수정 시 주의사항

- **파일명 변경 시**: 모든 관련 문서의 링크도 함께 업데이트
- **내부 링크**: 상대 경로 사용 (예: `[문서](other-doc.md)`)
- **앵커 링크**: 소문자, 하이픈 사용 (예: `#1-tokenizer-토크나이저`)

### 카테고리 분류

| 카테고리 | 포함 내용 |
|----------|----------|
| 프로젝트 개요 | 데이터 흐름, 프로젝트 목적, 데이터 요구사항 |
| 토큰화 | 토큰화 이유/방법, vocab_size, 다음 단계 |
| 심화 학습 | 핵심 개념, 트러블슈팅 |
