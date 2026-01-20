# 핵심 개념 9가지

LLM을 이해하기 위해 반드시 알아야 하는 핵심 개념입니다.

---

## 목차

1. [Tokenizer (토크나이저)](#1-tokenizer-토크나이저)
2. [Embedding (임베딩)](#2-embedding-임베딩)
3. [Positional Encoding (위치 인코딩)](#3-positional-encoding-위치-인코딩)
4. [Self-Attention (자기 어텐션)](#4-self-attention-자기-어텐션)
5. [Feed Forward Network (FFN)](#5-feed-forward-network-ffn)
6. [Residual Connection + LayerNorm](#6-residual-connection--layernorm)
7. [Next-token Prediction (다음 토큰 예측)](#7-next-token-prediction-다음-토큰-예측)
8. [Sampling (샘플링)](#8-sampling-샘플링)
9. [데이터 포맷이 곧 모델 능력](#9-데이터-포맷이-곧-모델-능력)

---

## 1. Tokenizer (토크나이저)

텍스트를 정수 토큰 ID로 변환하는 컴포넌트입니다.

```
"안녕하세요" → [1234, 567, 89] → 모델 입력
```

### vocab_size 설정

| 값 | 결과 |
|----|------|
| 너무 작음 | 한국어가 깨짐 (글자 단위로 쪼개짐) |
| 너무 큼 | 학습이 어려워짐 (희소한 토큰 많음) |
| **권장** | 16,000 ~ 32,000 |

### 코드 예시

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")
ids = tokenizer.encode("안녕하세요").ids  # [1234, 567]
text = tokenizer.decode(ids)  # "안녕하세요"
```

---

## 2. Embedding (임베딩)

토큰 ID를 고차원 벡터로 변환하는 테이블입니다.

```
토큰 ID 1234 → [0.1, -0.3, 0.7, ..., 0.2]  (n_embd 차원)
```

### 왜 필요한가?

- 정수 ID는 의미 정보가 없음
- 벡터 공간에서는 의미적 유사성 표현 가능
- 예: "개"와 "강아지"의 벡터가 가까워짐

### 코드 예시

```python
embedding = nn.Embedding(vocab_size=24000, embedding_dim=384)
vector = embedding(torch.tensor([1234]))  # shape: (1, 384)
```

---

## 3. Positional Encoding (위치 인코딩)

토큰의 순서 정보를 모델에 전달합니다. **Transformer는 기본적으로 순서를 모릅니다.**

```
"고양이가 개를 쫓았다" vs "개가 고양이를 쫓았다"
→ 위치 정보 없이는 같은 토큰들로 인식
```

### GPT의 위치 임베딩

GPT는 **학습 가능한 위치 임베딩**을 사용합니다:

```python
pos_embedding = nn.Embedding(block_size, n_embd)
x = token_embedding + pos_embedding  # 최종 입력
```

---

## 4. Self-Attention (자기 어텐션)

문장 내에서 **"어떤 단어가 어떤 단어에 주목해야 하는지"**를 학습합니다.

```
"그 고양이는 매우 귀여웠다. 그것은 집에서 잤다."
→ "그것"은 "고양이"에 강하게 어텐션
```

### 수학적 표현

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Q (Query): "내가 찾고 싶은 것"
K (Key): "나를 설명하는 키워드"
V (Value): "내가 가진 정보"
```

### Causal Self-Attention

GPT는 **Causal Self-Attention**을 사용하여 **미래 토큰을 볼 수 없습니다.**

```
입력: [A, B, C, D]

A가 볼 수 있는 토큰: [A]
B가 볼 수 있는 토큰: [A, B]
C가 볼 수 있는 토큰: [A, B, C]
D가 볼 수 있는 토큰: [A, B, C, D]
```

---

## 5. Feed Forward Network (FFN)

Attention 후에 정보를 비선형 변환하는 2층 MLP입니다.

```
입력 (n_embd) → Linear (4*n_embd) → GELU → Linear (n_embd) → 출력
```

### 역할 분담

| 컴포넌트 | 역할 |
|---------|------|
| Attention | "어디를 볼지" 결정 |
| FFN | "본 정보를 어떻게 변환할지" 결정 |

---

## 6. Residual Connection + LayerNorm

### Residual Connection (잔차 연결)

```python
출력 = 입력 + 레이어(입력)
```

- 깊은 네트워크에서 **gradient 소실 방지**
- 학습 초기에도 정보가 잘 흐름

### LayerNorm (레이어 정규화)

각 샘플 내에서 **평균 0, 분산 1**로 정규화합니다.

- 학습 안정화
- 배치 크기에 무관

### Pre-LayerNorm (JAI 사용 방식)

GPT-2 스타일의 Pre-LayerNorm을 사용합니다:

```python
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

---

## 7. Next-token Prediction (다음 토큰 예측)

**GPT의 유일한 학습 목표**입니다.

```
입력: "나는 밥을"
목표: "먹었다" 예측
```

### 왜 이것만으로 충분한가?

- 라벨링 없이 **텍스트 자체가 학습 데이터**
- 다음 단어를 예측하려면:
  - 문법 이해 필요
  - 문맥 이해 필요
  - 세상 지식 필요
- 자연스럽게 **언어 능력을 습득**

---

## 8. Sampling (샘플링)

모델 출력(확률 분포)에서 **다음 토큰을 선택**하는 방법입니다.

| 방법 | 설명 |
|------|------|
| **Greedy** | 가장 확률 높은 토큰 선택 (결정적) |
| **Temperature** | 확률 분포 조절 (<1.0: 보수적, >1.0: 다양) |
| **Top-K** | 상위 K개 토큰만 고려 |
| **Top-P** | 누적 확률 P까지의 토큰만 고려 |

### Temperature 예시

```python
# temperature = 0.7 (보수적)
logits = logits / 0.7  # 확률 분포가 뾰족해짐

# temperature = 1.5 (다양)
logits = logits / 1.5  # 확률 분포가 평평해짐
```

---

## 9. 데이터 포맷이 곧 모델 능력

> **모델은 학습 데이터의 패턴을 따라합니다**

| 학습 데이터 | 모델 능력 |
|------------|----------|
| Q&A 형식 | 질문에 답변 |
| 구인/구직 정보 | 구인/구직 형식 출력 |
| 대화 형식 | 대화 능력 |
| 코드 | 코드 생성 |

---

## 개념 간 관계도

```
텍스트
    ↓ [Tokenizer]
토큰 ID
    ↓ [Embedding + Position]
벡터
    ↓ [Transformer Block ×N]
        ├─ Self-Attention
        ├─ FFN
        ├─ Residual + LayerNorm
    ↓ [Linear Head]
확률 분포
    ↓ [Sampling]
다음 토큰
    ↓ [Tokenizer.decode]
텍스트
```

---

## 관련 문서

- [토큰화 이유](why-tokenize.md)
- [토큰화 다음 단계](after-tokenize.md)
- [트러블슈팅](troubleshooting.md)
