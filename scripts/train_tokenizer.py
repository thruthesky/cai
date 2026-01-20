
# scripts/train_tokenizer.py
# 설명: BPE 토크나이저를 내 데이터로 직접 학습
# 입력: data/samples.txt
# 출력: data/tokenizer.json

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# ============================================
# 설정
# ============================================
IN_PATH = "data/samples.txt"      # 입력 파일 (전처리된 학습 데이터)
OUT_PATH = "data/tokenizer.json"  # 출력 파일 (토크나이저)

VOCAB_SIZE = 24000  # 어휘 크기 (16,000 ~ 32,000 권장)

# ============================================
# 토크나이저 초기화
# ============================================
# BPE 모델 생성 (미지의 토큰은 [UNK]로 처리)
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Pre-tokenizer: 공백 기준으로 1차 분리
# (이후 BPE가 더 세밀하게 분리)
tokenizer.pre_tokenizer = Whitespace()

# ============================================
# 트레이너 설정
# ============================================
trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    show_progress=True,  # 진행률 표시
)

# ============================================
# 학습 실행
# ============================================
print(f"토크나이저 학습 시작...")
print(f"입력 파일: {IN_PATH}")
print(f"어휘 크기: {VOCAB_SIZE}")

tokenizer.train([IN_PATH], trainer=trainer)

# ============================================
# 저장
# ============================================
tokenizer.save(OUT_PATH)
print(f"토크나이저 저장 완료: {OUT_PATH}")

# ============================================
# 테스트
# ============================================
print("\n--- 토크나이저 테스트 ---")

test_texts = [
    "안녕하세요. 연락처 정보입니다.",
    "TEL: 02-1234-5678",
    "주미 대한민국 대사관",
    "[QUESTION] 연락처를 알려주세요 [/QUESTION]",
]

for text in test_texts:
    encoded = tokenizer.encode(text)
    print(f"원본: {text}")
    print(f"토큰: {encoded.tokens}")
    print(f"ID: {encoded.ids}")
    print()