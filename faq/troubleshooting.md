# 트러블슈팅

JAI 개발 중 자주 발생하는 문제와 해결 방법입니다.

---

## 목차

1. [환경 관련 문제](#환경-관련-문제)
2. [토크나이저 문제](#토크나이저-문제)
3. [학습 관련 문제](#학습-관련-문제)
4. [생성 관련 문제](#생성-관련-문제)
5. [데이터 관련 문제](#데이터-관련-문제)

---

## 환경 관련 문제

### MPS 메모리 오류

```
RuntimeError: MPS backend out of memory
```

**해결:** 배치 크기 줄이기

```python
# train_gpt.py
BATCH_SIZE = 16  # → 8 → 4로 줄이기
```

### MPS 연산 미지원

일부 PyTorch 연산이 MPS에서 지원되지 않을 때:

```python
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # CPU로 폴백
```

### CUDA 메모리 부족 (GPU)

```
CUDA out of memory
```

**해결:**
1. 배치 크기 줄이기
2. block_size 줄이기 (256 → 128)
3. `torch.cuda.empty_cache()` 호출

---

## 토크나이저 문제

| 문제 | 원인 | 해결 |
|------|------|------|
| 한국어가 글자 단위로 쪼개짐 | vocab_size가 너무 작음 | vocab_size 늘리기 (24000 권장) |
| [UNK] 토큰이 많이 나옴 | 학습 데이터에 없는 표현 | 학습 데이터에 다양한 표현 추가 |
| 토크나이저 로드 실패 | tokenizer.json이 없음 | `uv run python scripts/train_tokenizer.py` 실행 |

### 토크나이저 테스트 방법

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("data/tokenizer.json")
text = "테스트 문장입니다."
encoded = tokenizer.encode(text)

print(f"토큰: {encoded.tokens}")
print(f"ID: {encoded.ids}")

# [UNK]가 많으면 문제
if "[UNK]" in encoded.tokens:
    print("⚠️ 학습 데이터에 없는 표현이 있습니다.")
```

---

## 학습 관련 문제

### Loss가 줄어들지 않음

**원인:** 학습률이 적절하지 않음

**해결:**
```python
LEARNING_RATE = 3e-4  # 기본값
# 안 줄어들면: 1e-4 ~ 5e-4 범위에서 조정
```

### Loss가 NaN 또는 Inf

**원인:**
- 학습률이 너무 높음
- Gradient 폭발

**해결:**
```python
# 1. 학습률 낮추기
LEARNING_RATE = 1e-4

# 2. Gradient Clipping 추가
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 체크포인트 로드 실패

**원인:** 하이퍼파라미터 불일치

**해결:** 저장 시와 동일한 파라미터 사용

```python
# 저장된 모델과 동일해야 함
n_layer = 6
n_head = 6
n_embd = 384
vocab_size = 24000
block_size = 256
```

### 학습이 너무 느림

| 해결 방법 | 설명 |
|----------|------|
| block_size 줄이기 | 256 → 128 (메모리/속도 개선) |
| n_layer 줄이기 | 6 → 4 (모델 축소) |
| 데이터 줄이기 | 샘플 수 감소 |
| Mixed Precision | `torch.amp` 사용 |

---

## 생성 관련 문제

| 문제 | 원인 | 해결 |
|------|------|------|
| 텍스트가 반복됨 | temperature/top_k가 낮음 | temperature=1.0, top_k=80 |
| 텍스트가 엉뚱함 | temperature가 너무 높음 | temperature=0.7, top_k=30 |
| [ANSWER] 태그가 닫히지 않음 | max_new_tokens가 부족 | max_new_tokens=600 |
| 특수 토큰만 반복 | 학습 부족 | 더 많은 epoch 학습 |

### 권장 생성 파라미터

```python
# 안정적인 생성
output = model.generate(
    prompt_ids,
    max_new_tokens=512,
    temperature=0.8,
    top_k=50,
)

# 창의적인 생성
output = model.generate(
    prompt_ids,
    max_new_tokens=512,
    temperature=1.2,
    top_k=100,
)
```

---

## 데이터 관련 문제

### 과적합 (train loss ↓, val loss ↑)

**증상:** 학습 데이터는 잘 맞추지만 새로운 데이터에서 실패

**해결:**
1. **데이터 더 추가** (가장 효과적)
2. **Dropout 높이기**
   ```python
   dropout = 0.2  # 기본 0.1에서 증가
   ```
3. **모델 크기 줄이기**
   ```python
   n_layer = 4  # 6에서 감소
   n_embd = 256  # 384에서 감소
   ```

### 과소적합 (train loss도 안 줄어듦)

**증상:** 학습 자체가 잘 안 됨

**해결:**
1. 모델 크기 늘리기
2. 학습률 조정
3. 데이터 품질 확인

### 데이터 형식 오류

**체크리스트:**
- [ ] UTF-8 인코딩인가?
- [ ] [QUESTION], [DOC], [ANSWER] 태그가 올바른가?
- [ ] 태그가 닫혀있는가?
- [ ] 빈 줄이 너무 많지 않은가?

```bash
# 인코딩 확인
file -I data/samples.txt
# 출력: data/samples.txt: text/plain; charset=utf-8
```

---

## 빠른 진단 체크리스트

```
□ tokenizer.json이 존재하는가?
□ train.bin, val.bin이 존재하는가?
□ GPU/MPS가 사용되고 있는가?
□ 배치 크기가 적절한가?
□ 학습률이 적절한가? (1e-4 ~ 5e-4)
□ loss가 줄어들고 있는가?
□ val loss도 줄어들고 있는가?
```

---

## 관련 문서

- [데이터 흐름](data-flow.md)
- [핵심 개념](core-concepts.md)
- [토큰화 다음 단계](after-tokenize.md)
