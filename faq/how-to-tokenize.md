# 토큰화를 하는 방법

## 개요

JAI는 **Hugging Face의 `tokenizers` 라이브러리**를 사용하여 BPE(Byte Pair Encoding) 토크나이저를 학습합니다.

---

## 직접 토큰화 로직을 작성할 필요가 있는가?

**아니요, 직접 작성할 필요가 없습니다.** `tokenizers` 라이브러리를 사용합니다.

| 구분 | 직접 구현 | tokenizers 라이브러리 |
|------|----------|----------------------|
| 코드량 | 수백~수천 줄 | 약 20줄 |
| 성능 | Python 속도 (느림) | Rust 기반 (매우 빠름) |
| 안정성 | 버그 가능성 | 검증된 코드 |
| 유지보수 | 직접 해야 함 | 커뮤니티 지원 |

---

## 사용하는 외부 패키지

### tokenizers (Hugging Face)

| 항목 | 내용 |
|------|------|
| **언어** | Rust로 작성, Python 바인딩 제공 |
| **설치** | `uv add tokenizers` 또는 `pip install tokenizers` |
| **특징** | 병렬 처리 지원, 메모리 효율적 |
| **사용처** | GPT-2, BERT 등 유명 모델들이 사용 |

### 성능 비교

| 항목 | tokenizers (Rust) | 순수 Python 구현 |
|------|-------------------|------------------|
| 1GB 텍스트 학습 | ~1분 | ~30분+ |
| 토큰화 속도 | ~1M tokens/초 | ~10K tokens/초 |
| 멀티스레딩 | 자동 지원 | 직접 구현 필요 |

**성능 차이가 100배 이상**이므로 실제 프로젝트에서는 `tokenizers` 라이브러리 사용을 강력 권장합니다.

---

## BPE 토큰화 원리

BPE(Byte Pair Encoding)는 다음과 같이 동작합니다:

```
1. 초기: 모든 문자를 개별 토큰으로 시작
   "안녕하세요" → ["안", "녕", "하", "세", "요"]

2. 반복: 가장 자주 등장하는 연속 쌍을 병합
   ("안", "녕") 빈도 높음 → ["안녕", "하", "세", "요"]
   ("하", "세") 빈도 높음 → ["안녕", "하세", "요"]

3. 종료: vocab_size에 도달할 때까지 반복
```

---

## JAI 코드 흐름 (train_tokenizer.py)

```python
# 1. BPE 모델 생성
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 2. Pre-tokenizer 설정 (공백으로 1차 분리)
tokenizer.pre_tokenizer = Whitespace()

# 3. 트레이너 설정
trainer = BpeTrainer(
    vocab_size=24000,  # 어휘 크기
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
)

# 4. 학습 실행
tokenizer.train([IN_PATH], trainer=trainer)

# 5. 저장 (JSON 형식)
tokenizer.save(OUT_PATH)
```

### 각 단계 설명

| 단계 | 코드 | 설명 |
|------|------|------|
| 1 | `Tokenizer(BPE(...))` | BPE 알고리즘 기반 토크나이저 생성 |
| 2 | `pre_tokenizer = Whitespace()` | 공백 기준으로 1차 분리 후 BPE 적용 |
| 3 | `BpeTrainer(...)` | 어휘 크기, 특수 토큰 설정 |
| 4 | `tokenizer.train(...)` | 실제 학습 (빈도 기반 병합 규칙 생성) |
| 5 | `tokenizer.save(...)` | JSON 형식으로 저장 |

---

## 실행 방법

```bash
# 토크나이저 학습
uv run python scripts/train_tokenizer.py

# 결과물
# → data/tokenizer.json (어휘 사전 + 병합 규칙)
```

---

## 학습 목적으로 직접 구현한다면?

교육 목적으로 BPE를 직접 구현하고 싶다면 핵심 로직은 이렇습니다:

```python
def train_bpe(text, vocab_size):
    # 1. 문자 단위로 초기 어휘 생성
    vocab = set(text)

    # 2. vocab_size까지 병합 반복
    while len(vocab) < vocab_size:
        # 2-1. 모든 연속 쌍의 빈도 계산
        pairs = count_pairs(text)

        # 2-2. 가장 빈번한 쌍 선택
        best_pair = max(pairs, key=pairs.get)

        # 2-3. 해당 쌍을 새 토큰으로 병합
        new_token = best_pair[0] + best_pair[1]
        vocab.add(new_token)
        text = merge_pair(text, best_pair, new_token)

    return vocab
```

하지만 **실제 프로젝트에서는 `tokenizers` 라이브러리 사용을 강력 권장**합니다.

---

## 요약

| 질문 | 답변 |
|------|------|
| 직접 구현 필요? | ❌ 필요 없음 (라이브러리 사용) |
| 사용 라이브러리 | `tokenizers` (Hugging Face, Rust 기반) |
| 토큰화 방식 | BPE (Byte Pair Encoding) |
| 성능 | 매우 빠름 (Rust 기반, 병렬 처리) |
| 설치 방법 | `uv add tokenizers` |

---

## 관련 문서

- [왜 토큰화를 해야 하는가?](why-tokenize.md)
- [vocab_size란?](vocab-size.md)
- [토큰화 다음 단계](after-tokenize.md)
