# JAI FAQ (자주 묻는 질문)

JAI(Job AI) 프로젝트의 핵심 개념을 쉽게 설명합니다.

---

## 문서 목록

### 프로젝트 개요

| 문서 | 설명 |
|------|------|
| [data-flow.md](faq/data-flow.md) | JAI 데이터 흐름 (전체 파이프라인) |
| [why-job-llm.md](faq/why-job-llm.md) | 구인 정보 LLM이 필요한가? |
| [sample-data.md](faq/sample-data.md) | 샘플 데이터는 실제 구인 정보여야 하는가? |
| [sample-data-strategy.md](faq/sample-data-strategy.md) | 샘플 데이터 구성 전략 (DB 필드, 변환 규칙, 코드) |

### 토큰화

| 문서 | 설명 |
|------|------|
| [why-tokenize.md](faq/why-tokenize.md) | 왜 토큰화를 해야 하는가? |
| [how-to-tokenize.md](faq/how-to-tokenize.md) | 토큰화를 하는 방법 (외부 라이브러리) |
| [vocab-size.md](faq/vocab-size.md) | vocab_size란 무엇인가? |
| [after-tokenize.md](faq/after-tokenize.md) | 토큰화 다음에는 무엇을 해야 하는가? |

### 심화 학습

| 문서 | 설명 |
|------|------|
| [embedding-and-prediction.md](faq/embedding-and-prediction.md) | 토큰화 → 임베딩 → 다음 토큰 예측 (벡터 검색과의 차이) |
| [core-concepts.md](faq/core-concepts.md) | 핵심 개념 9가지 |
| [troubleshooting.md](faq/troubleshooting.md) | 트러블슈팅 |

---

## 빠른 참조

### JAI 파이프라인

```
raw.txt → prepare_samples.py → samples.txt
                                    ↓
                            train_tokenizer.py → tokenizer.json
                                    ↓
                            build_bin_dataset.py → train.bin / val.bin
                                    ↓
                            train_gpt.py → ckpt.pt
                                    ↓
                            generate.py → "서울에서 React 개발자를..."
```

### 실행 명령어

```bash
uv run python scripts/prepare_samples.py      # 데이터 전처리
uv run python scripts/train_tokenizer.py      # 토크나이저 학습
uv run python scripts/build_bin_dataset.py    # 바이너리 변환
uv run python scripts/train_gpt.py            # GPT 학습
uv run python scripts/generate.py             # 텍스트 생성
```

### 핵심 개념 한눈에 보기

| 개념 | 한 줄 설명 |
|------|-----------|
| Tokenizer | 텍스트 → 숫자 ID 변환 |
| Embedding | 숫자 ID → 고차원 벡터 변환 |
| Positional Encoding | 토큰 순서 정보 추가 |
| Self-Attention | 어떤 단어가 어디에 주목할지 학습 |
| FFN | 정보를 비선형 변환 |
| Residual + LayerNorm | 학습 안정화 |
| Next-token Prediction | GPT의 학습 목표 (다음 토큰 예측) |
| Sampling | 확률 분포에서 토큰 선택 |

---

## 참고 문서

- [모델 구조](docs/05-model-architecture.md)
- [학습 코드](docs/06-training.md)
- [서버 구축](docs/08-server.md)

## 참고 자료

- [nanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathy의 미니 GPT 구현
- [PyTorch 공식 문서](https://pytorch.org/docs/)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/)
