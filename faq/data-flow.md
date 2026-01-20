# JAI 데이터 흐름

JAI는 **"텍스트 → 숫자로 변환 → AI 훈련 → 텍스트 생성"** 과정입니다.

---

## 전체 파이프라인

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

---

## 단계별 설명

| 단계 | 스크립트 | 하는 일 |
|------|----------|---------|
| 1 | prepare_samples.py | 원본 텍스트를 학습 형식으로 정리 |
| 2 | train_tokenizer.py | 텍스트를 숫자로 바꾸는 규칙 만들기 |
| 3 | build_bin_dataset.py | 토큰 ID를 바이너리 파일로 저장 |
| 4 | train_gpt.py | GPT 모델 학습 |
| 5 | generate.py | 텍스트 생성 |

---

## 실행 순서

```bash
# 의존성 설치 (uv 사용)
uv add torch tokenizers tqdm numpy

# 순차 실행 (순서 중요!)
uv run python scripts/prepare_samples.py      # 데이터 전처리 → data/samples.txt
uv run python scripts/train_tokenizer.py      # BPE 토크나이저 → data/tokenizer.json
uv run python scripts/build_bin_dataset.py    # 바이너리 변환 → data/train.bin, data/val.bin
uv run python scripts/train_gpt.py            # GPT 학습 → checkpoints/ckpt.pt
uv run python scripts/generate.py             # 텍스트 생성
```

---

## 각 단계의 입출력

### 1단계: 데이터 전처리
- **입력**: `data/raw.txt` (원본 텍스트)
- **출력**: `data/samples.txt` (정제된 학습 데이터)
- **역할**: 원본 데이터를 [QUESTION], [DOC], [ANSWER] 형식으로 정리

### 2단계: 토크나이저 학습
- **입력**: `data/samples.txt`
- **출력**: `data/tokenizer.json` (BPE 어휘 사전 + 병합 규칙)
- **역할**: 텍스트를 숫자 ID로 변환하는 규칙 생성

### 3단계: 바이너리 변환
- **입력**: `data/samples.txt` + `data/tokenizer.json`
- **출력**: `data/train.bin`, `data/val.bin`
- **역할**: 토큰 ID를 빠르게 읽을 수 있는 바이너리로 저장

### 4단계: GPT 학습
- **입력**: `data/train.bin`, `data/val.bin`, `data/tokenizer.json`
- **출력**: `checkpoints/ckpt.pt` (모델 가중치)
- **역할**: Transformer 모델 학습

### 5단계: 텍스트 생성
- **입력**: `checkpoints/ckpt.pt`, `data/tokenizer.json`
- **출력**: 생성된 텍스트
- **역할**: 학습된 모델로 새로운 텍스트 생성

---

## 관련 문서

- [토큰화 이유](why-tokenize.md)
- [토큰화 다음 단계](after-tokenize.md)
- [핵심 개념](core-concepts.md)
