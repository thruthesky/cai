# 토큰화 다음에는 무엇을 해야 하는가?

## 순서: 토크나이저 학습 → 바이너리 변환 → GPT 학습 → 생성 테스트

---

## 단계 1: 토크나이저 학습

```bash
uv run python scripts/train_tokenizer.py
```

samples.txt를 분석해서 **"어떻게 쪼개면 효율적인지"** 규칙(BPE)을 만듭니다.

**결과물:** `tokenizer.json` (어휘 사전 + 병합 규칙)

---

## 단계 2: 바이너리 데이터셋 생성

```bash
uv run python scripts/build_bin_dataset.py
```

텍스트를 토큰 ID로 변환하고, 빠르게 읽을 수 있는 바이너리로 저장합니다.

### 학습 데이터 구조

```
토큰: [12, 45, 98, 23, 67, 89, ...]
         ↓ block_size=4로 자르기
입력 X: [12, 45, 98, 23]
정답 Y: [45, 98, 23, 67]  ← 한 칸 시프트
```

GPT는 **"12 다음엔 45, 45 다음엔 98..."** 이런 식으로 다음 토큰을 예측하며 학습합니다.

**결과물:** `train.bin`, `val.bin`

---

## 단계 3: GPT 학습

```bash
uv run python scripts/train_gpt.py
```

### 학습 과정

```
1. 토큰 ID → 임베딩(벡터 변환)
2. Transformer 블록 통과 (Self-Attention)
3. 다음 토큰 확률 출력
4. 정답과 비교해서 손실 계산
5. 가중치 업데이트
6. 반복
```

**결과물:** `checkpoints/ckpt.pt`

---

## 단계 4: 생성 테스트

```bash
uv run python scripts/generate.py
```

학습된 모델로 텍스트를 생성해봅니다. 잘 학습됐다면 구인 정보 형태로 출력합니다.

---

## 하이퍼파라미터 (M4 기준)

| 파라미터 | 값 | 의미 |
|----------|-----|------|
| vocab_size | 24,000 | 어휘 크기 |
| block_size | 256 | 한 번에 처리하는 토큰 수 |
| n_layer | 6 | Transformer 블록 수 |
| n_head | 6 | Attention 헤드 수 |
| n_embd | 384 | 임베딩 차원 |
| batch_size | 16 | 배치 크기 |
| learning_rate | 3e-4 | 학습률 |

---

## 추가 질문

### Q: 임베딩을 별도로 해야 하나요?

**아니요.** GPT 모델 내부에 임베딩 레이어가 포함되어 있어서 자동으로 처리됩니다.

```python
# GPT 모델 내부
self.tok_emb = nn.Embedding(vocab_size, n_embd)  # 토큰 임베딩
self.pos_emb = nn.Embedding(block_size, n_embd)  # 위치 임베딩
```

### Q: 학습이 잘 되는지 어떻게 알 수 있나요?

**loss 값**이 점점 낮아지면 잘 되고 있는 것입니다.

- train loss는 줄어드는데 val loss가 올라가면 → **과적합**
- 둘 다 안 줄어들면 → 학습률 조정 필요

### Q: 데이터가 커지면?

samples.txt를 여러 파일로 나누고 스크립트를 수정하면 됩니다.

```python
# train_tokenizer.py 수정 예시
tokenizer.train(["data/samples1.txt", "data/samples2.txt"], trainer=trainer)
```

---

## 전체 파이프라인 요약

```
samples.txt
    ↓ train_tokenizer.py
tokenizer.json
    ↓ build_bin_dataset.py
train.bin / val.bin
    ↓ train_gpt.py
ckpt.pt
    ↓ generate.py
"서울에서 React 개발자를..."
```

---

## 관련 문서

- [토큰화 이유](why-tokenize.md)
- [토큰화 방법](how-to-tokenize.md)
- [핵심 개념](core-concepts.md)
- [트러블슈팅](troubleshooting.md)
