# 08. JAI LLM 서버

JAI LLM을 데몬(서비스)으로 실행하여 여러 클라이언트가 동시에 접속하고 프롬프트를 전송하면 응답을 받을 수 있는 서버를 구축합니다.

---

## 1. 개요

### 아키텍처

```
┌─────────────┐     HTTP/REST      ┌─────────────────┐
│  Client 1   │ ──────────────────▶│                 │
└─────────────┘                    │                 │
                                   │   JAI Server    │
┌─────────────┐     HTTP/REST      │   (FastAPI)     │──▶ JAI LLM Model
│  Client 2   │ ──────────────────▶│                 │      (ckpt.pt)
└─────────────┘                    │                 │
                                   └─────────────────┘
┌─────────────┐     HTTP/REST             │
│  Client N   │ ──────────────────────────┘
└─────────────┘
```

### 주요 특징

- **FastAPI 기반**: 비동기 처리로 동시 요청 처리
- **모델 싱글톤**: 서버 시작 시 모델 1회 로드, 메모리 효율화
- **요청 큐**: 동시 요청을 순차 처리하여 GPU 메모리 보호
- **데몬 실행**: systemd 또는 nohup으로 백그라운드 실행

---

## 2. 의존성 설치

```bash
uv add fastapi uvicorn
```

---

## 3. 서버 코드

### `scripts/server.py`

```python
"""
JAI LLM 서버
- FastAPI 기반 REST API 서버
- 동시 접속 클라이언트 처리
- 프롬프트 전송 → JAI LLM 응답
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tokenizers import Tokenizer

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent

# ===== 모델 설정 (train_gpt.py와 동일하게 유지) =====
VOCAB_SIZE = 24000
BLOCK_SIZE = 256
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384

# ===== 경로 설정 =====
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "ckpt.pt"
TOKENIZER_PATH = PROJECT_ROOT / "data" / "tokenizer.json"


# ===== GPT 모델 정의 (train_gpt.py에서 가져옴) =====
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Causal Self-Attention: 미래 토큰을 볼 수 없도록 마스킹"""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        # causal mask: 미래 토큰 차단
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    """2층 Feed Forward Network"""

    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    """Transformer 블록: LayerNorm + Attention + LayerNorm + MLP"""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT 모델"""

    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        """텍스트 생성"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :]
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# ===== 전역 변수 (싱글톤 패턴) =====
model = None
tokenizer = None
device = None
# 동시 요청 제어를 위한 세마포어 (GPU 메모리 보호)
inference_semaphore = asyncio.Semaphore(1)


def load_model():
    """모델과 토크나이저 로드"""
    global model, tokenizer, device

    # 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"디바이스: {device}")

    # 토크나이저 로드
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"토크나이저를 찾을 수 없습니다: {TOKENIZER_PATH}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    print(f"토크나이저 로드 완료: {TOKENIZER_PATH}")

    # 모델 생성 및 체크포인트 로드
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {CHECKPOINT_PATH}")

    model = GPT(VOCAB_SIZE, BLOCK_SIZE, N_LAYER, N_HEAD, N_EMBD)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    print(f"모델 로드 완료: {CHECKPOINT_PATH}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 실행되는 lifecycle 관리"""
    # 시작 시: 모델 로드
    print("JAI LLM 서버 시작 중...")
    load_model()
    print("JAI LLM 서버 준비 완료!")
    yield
    # 종료 시: 정리 작업
    print("JAI LLM 서버 종료 중...")


# ===== FastAPI 앱 =====
app = FastAPI(
    title="JAI LLM Server",
    description="구인 정보 특화 LLM API 서버",
    version="1.0.0",
    lifespan=lifespan,
)


# ===== 요청/응답 스키마 =====
class GenerateRequest(BaseModel):
    """텍스트 생성 요청"""
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50


class GenerateResponse(BaseModel):
    """텍스트 생성 응답"""
    prompt: str
    generated: str
    full_text: str


# ===== API 엔드포인트 =====
@app.get("/")
async def root():
    """서버 상태 확인"""
    return {"status": "ok", "message": "JAI LLM Server is running"}


@app.get("/health")
async def health():
    """헬스 체크"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    텍스트 생성 API

    - prompt: 입력 프롬프트
    - max_tokens: 생성할 최대 토큰 수
    - temperature: 샘플링 온도 (높을수록 다양한 출력)
    - top_k: top-k 샘플링
    """
    if model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    # 세마포어로 동시 추론 제어 (GPU 메모리 보호)
    async with inference_semaphore:
        try:
            # 토큰화
            encoded = tokenizer.encode(request.prompt)
            input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

            # 생성
            output_ids = model.generate(
                input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
            )

            # 디코딩
            full_text = tokenizer.decode(output_ids[0].tolist())
            generated = tokenizer.decode(output_ids[0, len(encoded.ids):].tolist())

            return GenerateResponse(
                prompt=request.prompt,
                generated=generated,
                full_text=full_text,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: GenerateRequest):
    """
    대화형 API (QUESTION/ANSWER 형식)

    자동으로 [QUESTION]...[/QUESTION] 형식으로 감싸서 전송
    """
    # QUESTION 형식으로 감싸기
    formatted_prompt = f"[QUESTION]\n{request.prompt}\n[/QUESTION]\n\n[ANSWER]\n"

    # generate 엔드포인트 재사용
    request.prompt = formatted_prompt
    return await generate(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 4. 서버 실행 방법

### 4.1 직접 실행

```bash
# 개발 모드 (자동 리로드)
uv run uvicorn scripts.server:app --reload --host 0.0.0.0 --port 8000

# 프로덕션 모드
uv run uvicorn scripts.server:app --host 0.0.0.0 --port 8000 --workers 1
```

### 4.2 백그라운드 실행 (nohup)

```bash
# 백그라운드 실행
nohup uv run uvicorn scripts.server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

# 프로세스 확인
ps aux | grep uvicorn

# 종료
pkill -f "uvicorn scripts.server:app"
```

### 4.3 systemd 데몬 등록 (Linux)

`/etc/systemd/system/jai-server.service` 파일 생성:

```ini
[Unit]
Description=JAI LLM Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/jai
ExecStart=/home/ubuntu/.local/bin/uv run uvicorn scripts.server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# 데몬 등록 및 시작
sudo systemctl daemon-reload
sudo systemctl enable jai-server
sudo systemctl start jai-server

# 상태 확인
sudo systemctl status jai-server

# 로그 확인
sudo journalctl -u jai-server -f
```

### 4.4 launchd 데몬 등록 (macOS)

`~/Library/LaunchAgents/com.jai.server.plist` 파일 생성:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.jai.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/사용자명/.local/bin/uv</string>
        <string>run</string>
        <string>uvicorn</string>
        <string>scripts.server:app</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>8000</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/사용자명/apps/jai</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/jai-server.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/jai-server.error.log</string>
</dict>
</plist>
```

```bash
# 데몬 등록 및 시작
launchctl load ~/Library/LaunchAgents/com.jai.server.plist

# 상태 확인
launchctl list | grep jai

# 종료
launchctl unload ~/Library/LaunchAgents/com.jai.server.plist
```

---

## 5. API 사용법

### 5.1 curl 예시

```bash
# 헬스 체크
curl http://localhost:8000/health

# 텍스트 생성
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "[QUESTION]\n서울에서 React 개발자 채용 있어?\n[/QUESTION]\n\n[ANSWER]\n",
    "max_tokens": 256,
    "temperature": 0.8
  }'

# 대화형 API (자동 포맷팅)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "판교에서 백엔드 개발자 채용하는 곳 알려줘",
    "max_tokens": 256
  }'
```

### 5.2 Python 클라이언트

```python
"""JAI LLM 클라이언트 예시"""
import requests

BASE_URL = "http://localhost:8000"


def chat(prompt: str, max_tokens: int = 256) -> str:
    """대화형 API 호출"""
    response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.8,
        },
    )
    response.raise_for_status()
    return response.json()["generated"]


def generate(prompt: str, max_tokens: int = 256) -> str:
    """텍스트 생성 API 호출"""
    response = requests.post(
        f"{BASE_URL}/generate",
        json={
            "prompt": prompt,
            "max_tokens": max_tokens,
        },
    )
    response.raise_for_status()
    return response.json()["full_text"]


if __name__ == "__main__":
    # 대화형 API 사용
    answer = chat("서울에서 프론트엔드 개발자 채용 있어?")
    print(answer)
```

---

## 6. 동시 접속 처리

### 동작 방식

```
Client A ─────┐
              │
Client B ─────┼──▶ FastAPI (비동기) ──▶ Semaphore(1) ──▶ GPU 추론
              │         │
Client C ─────┘         │
                        ▼
                   요청 큐잉 (대기)
```

1. **FastAPI 비동기 처리**: 여러 클라이언트의 HTTP 요청을 동시에 받음
2. **Semaphore(1)**: GPU 추론은 한 번에 하나씩만 실행 (메모리 보호)
3. **요청 큐잉**: 추론 중인 요청이 있으면 다른 요청은 대기

### 동시성 조절

```python
# 동시 추론 수 조절 (GPU 메모리에 따라)
inference_semaphore = asyncio.Semaphore(2)  # 동시 2개 추론
```

---

## 7. 성능 최적화

### 7.1 배치 처리 (선택사항)

여러 요청을 모아서 한 번에 처리:

```python
from collections import deque
import asyncio

request_queue = deque()
BATCH_SIZE = 4
BATCH_TIMEOUT = 0.1  # 100ms


async def batch_inference():
    """배치 추론 루프"""
    while True:
        if len(request_queue) >= BATCH_SIZE:
            batch = [request_queue.popleft() for _ in range(BATCH_SIZE)]
            # 배치 추론 수행
            ...
        await asyncio.sleep(BATCH_TIMEOUT)
```

### 7.2 모델 양자화

```python
# INT8 양자화 (메모리 절약)
model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

---

## 8. 모니터링

### Swagger UI

서버 실행 후 브라우저에서 접속:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 로깅 추가

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("jai-server")

@app.post("/generate")
async def generate(request: GenerateRequest):
    logger.info(f"요청 수신: {request.prompt[:50]}...")
    # ...
    logger.info(f"응답 완료: {len(generated)} 토큰 생성")
```

---

## 참고

- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [Uvicorn 공식 문서](https://www.uvicorn.org/)
- [systemd 서비스 작성](https://www.freedesktop.org/software/systemd/man/systemd.service.html)
