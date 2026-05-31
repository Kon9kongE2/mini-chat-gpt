# Tiny GPT 정리

## 0. notebook01~06의 공통된 흐름 정리

```
1. 텍스트 데이터를 모델이 학습할 수 있는 형태로 바꾸기
2. 다음 문자를 예측하는 모델 만들기
3. 모델을 학습
4. 학습된 모델로 문장을 생성하기
```

---

## 1. 데이터 준비

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

if not Path("input.txt").exists():
    !wget -q https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

text = open("input.txt", "r", encoding="utf-8").read()
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
```

```
- 코드 실행에 필요한 패키지 불러오기
- input.txt 파일 읽기
- 읽어온 파일에서 텍스트의 중복된 것을 제외, 알파벳 순으로 정렬하여 리스트 형식으로 저장
- 이를 바탕으로 stoi, itos 딕셔너리 생성; 이때 vocab_size는 문자의 종류를 나타냄
- data는 text에 있는 모든 문자를 모두 숫자로 변환한 형태로 저장됨
```

```python
class NextTokenDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    def __len__(self):
        return len(self.data) - self.block_size
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y
```

```
** NextTokenDataset 클래스 정의 **
- dunder method __init__으로 dataset 생성시 필요한 값들을 저장하게 해줌
- `len(self.data)- self.block_size`는 idx가 너무 뒤로 가면 y를 생성할 수 없으므로 학습 가능한 데이터를 이런 식으로 지정하였음
- x는 지정한 idx ~ idx + block_size(문제지), y는 x의 양쪽 값에서 1씩 더한 값들의 집합(답지)이다.
```

---

## 2. 모델에 필요한 개념

### 1. Multi- head attention

Multi -head attention: 현재의 토큰에서 과거의 토큰을 바탕으로 미래를 예측. 이때 과거의 토큰을 살펴보는 방법(HEAD)이 여러 개 있는데, 그 여러 개의 Head가 각자 다른 방식으로 문맥을 보고, 그 결과를 합치는 구조로 이해하였음.

```python
class Head(nn.Module):
    def __init__(self, emb_dim, head_size, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
```

```
** HEAD 클래스 정의 파트 1 **
- query: 현재의 토큰이 던지는 질문 or 현재의 토큰이 찾고 싶은 정보의 조건
- key: 각 토큰이 내걸고 있는 자기소개표(조건)
- value: query와 key를 비교해서, 실제로 가져오는 것은 value
- 투입값: x = (B, T, emb_dim) / 출력값: (B, T head_size)
- 마스크를 위해서 원소가 1인 하삼각행렬을 먼저 정의
```

```python
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out
```

```
** HEAD 클래스 정의 파트 2 **
- 입력 x를 넣으면 실행되는 부분
- 각 x의 선형변환을 통해 k,q,v를 지정
- attention 점수 계산: q*k가 가중치
  이유: q와 k는 앞에서 말했듯이 찾고자 하는 정보의 조건과 토큰의 자기소개표
  이를 내적했을 때 값이 크다는 것은 벡터의 방향이 비슷하다는 것, 관련이 있다는 것
  따라서 해당 토큰의 value를 더 많이 가져오게 됨
- 미래 토큰 가리기 -> 앞의 하삼각행렬(wei)에서 0인 부분을 True로 변환, 최종적으로 -inf로 가림
- softmax로 wei 행렬를의 원소를 확률화시킴
- dropout은 과적합 방지를 위한 랜덤으로 attention 연결을 의도적으로 줄임
- 최종적으로 wei = (B, T, T), v = (B, T, head_size)를 곱하여 out = (B, T, head_size)를 출력
```

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        head_size = emb_dim // num_heads
        self.heads = nn.ModuleList([Head(emb_dim, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
```

```
** MultiHeadAttention 클래스 정의 파트 1 **
- head_size는 embedding 차원을 Head들이 분담하여 처리함에 따라 결정된다.
- self.heads: 여러개의 Head들을 생성하는 코드
- self.proj: 여러개의 heade들의 출력을 이어붙여 선형변환하는 코드
- self.dropout: 과적합 방지용 최종 출력에 일부 값을 dropout
```

```python
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
```

```
** MultiHeadAttention 클래스 정의 파트 2 **
- 입력값 x를 넣으면 실행되는 부분
- self.heads: HEAD가 여러개이므로 각각의 HEAD에 대해 똑같은 x가 입력값으로 들어감
  각 HEAD 출력의 차원은 (B, T, head_size)
  그러고 나서 dim(-1), 즉 head_size에 대해 합쳐짐 -> (B, T, emb_dim)
- 이후 self.proj에 따라 이어붙인 HEAD들의 결과를 다시 섞음
- 그러고 나서 과적합 방지용 dropout 적용, 그 결과를 출력
```

### 2. Feedforward

Attention이 각 토큰이 다른 토큰을 얼마나 참고할지를 계산했다면, FeedForward는 각 토큰 벡터를 한 번 더 비선형적으로 가공하는 파트

```python
class FeedForward(nn.Module):
    def __init__(self, emb_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )
```

```
** Feedforward 클래스 정의 파트 1 **
- nn.Sequential에 따라 밑의 코드들을 순서대로 실행
- 첫 번째 nn.Linear: 선형변환 행렬이 (emb_dim, 4 * emb_dim)
  즉 이 행렬을 곱한다면 출력 차원이 4배 늘어나게 만듦
- ReLU: 동강에서 나온 함수. 음수는 0 양수는 그대로 두는 활성화 함수
- 두 번째 nn.Linear: 선형변환 행렬이 (4 * emb_dim, emb_dim)
  즉 이 행렬을 곱한다면 출력 차원을 다시 줄어들게 만듦
```

```python
    def forward(self, x):
        return self.net(x)
```

```
** Feedforward 클래스 정의 파트 2 **
- 입력값 x를 넣으면 실행되는 부분
- 즉 파트 1의 단계를 실행한다는 것
```

### 3. Block

Multi - Head Attention으로 각 토큰의 벡터를 변환시키고, FeedForward로 그 토큰 벡터를 다시 가공하고, Residual connection으로 기존 정보를 유지

```python
class Block(nn.Module):
    def __init__(self, emb_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.sa = MultiHeadAttention(emb_dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ffwd = FeedForward(emb_dim, dropout)
```

```
** Block 클래스 정의 파트 1 **
- self.ln1: 첫 번째 LayerNorm
  즉 각 토큰의 embedding 벡터에 대해서, 값이 들쑥날쑥하므로
  정규화시켜서 값을 보기 좋게 정돈시킨다는 것
- self.sa: Multi-Head Attention 실행
  즉 입력값의 각 토큰이 자기 '앞'의 토큰들만을 참고해서 새 벡터로 변환됨
- self. ln2: 두 번째 LayerNorm
- self.ffwd: Feedforward 실행
  즉 Attention을 거친 각 토큰 벡터를 다시 한 번 가공하는 과정
```

```python
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

```
** Block 클래스 정의 파트 2 **
- 입력 x
→ 첫 번째 LayerNorm
→ Multi-Head Attention
→ 원래 x와 더함

→ 두 번째 LayerNorm
→ FeedForward
→ 원래 x와 더함

- 출력
```
---

## 3. 모델(Tiny GPT)

```
TinyGPT
 ├─ token_embedding
 ├─ position_embedding
 ├─ Block × 4
 │   ├─ MultiHeadAttention
 │   │   ├─ Head
 │   │   ├─ Head
 │   │   ├─ Head
 │   │   └─ Head
 │   └─ FeedForward
 ├─ LayerNorm
 └─ Linear
```

```python
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, emb_dim=128, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(block_size, emb_dim)
        self.blocks = nn.Sequential(*[
            Block(emb_dim, num_heads, block_size, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)
```

```
** Tiny GPT 클래스 정의 파트 1 **
- token_embedding: 각 숫자 토큰에 대해 그것들의 정보를 벡터로 변환시킴
  x.shape = (B, T)에 대해 이후 tok 정의 및 tok.shape = (B, T, emb_dim)
  텍스트 데이터로 예시를 들어 이해하면, B개의 문장 / 각 문장에는 T개의 글자 / 각 글자는 emb_dim차원 벡터
- pos_embedding: 각 위치 정보를 벡터로 변환시킴
  `pos = torch.arrange(t)`에 대해 `pos = self.position
- self.blocks: 앞에서 정의한 Block을 num_layers개 만큼 사용함
- self.ln_f: Block을 지난 각 토큰의 벡터 원소들을 정규화시켜 알아보기 쉽게 정리함
- self.lm_head: 최종적으로 선형변환 행렬 emb_dim * vocab_size를 곱함
  이를 통해 그 다음 어떤 글자가 나올지 모든 글자 종류에 대해 점수를 매김
```

```python
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        tok = self.token_embedding(x)
        pos = self.position_embedding(pos)[None]
        h = tok + pos
        h = self.blocks(h)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits
```

```
** Tiny GPT 클래스 정의 파트 2 **
- 입력값 x = (B, T)에 대해
- 토큰 수 T(글자 수 T)에 대해 위치벡터(pos)로 변환시킴
  그 pos를 position embedding시켜서 각 위치에 대해 위치정보벡터(최종 pos)를 생성
- 또한 각 T(token)에 대해 정보를 나타내는 벡터를 embedding 시킴(tok)
- h는 tok + pos로, 각 토큰에 대한 정보와 위치정보를 모두 담고 있음
- 그 h를 self.blocks(h)에 통과시킴
  즉 num_layer = 4 번의 Block들을 통과시킨다는 것
- 이후 위에서 말한 정규화, 다음 글자 종류에 대해 점수를 매김
```

---

## 4. 학습

```python
def sequence_cross_entropy(logits, targets):
    return F.cross_entropy(logits.transpose(1, 2), targets)
```

```
- cross_entropy 계산하는 코드 정의
```

```python
def train_one_epoch(model, loader, optimizer, device, max_steps=None):
    model.train()
    total_loss, total_count = 0.0, 0
```

```
- 학습 모드로 전환
- 초기 로스와 초기 학습 횟수를 0으로 정의
```

```python
    for step, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = sequence_cross_entropy(logits, yb)
```

```
- xb 데이터는 batch 단위로 뽑은 문장들
  이것을 tiny gpt model에 투입, 다음 글자들에 대한 점수를 출력
- 답지 yb 데이터와 loss를 계산
```

```python
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)
        if max_steps is not None and step + 1 >= max_steps:
            break
    return total_loss / total_count
```

```
- 계산한 loss를 바탕으로 그레디언트를 계산, 가중치를 개선시키고 업데이트 시킴
- 현재는 max_step이 None이지만, 만약 설정한 max_step을 넘어선다면 중단
- 총 loss에 대한 평균을 출력
```

---

## 5. 생성

```python
@torch.no_grad()
def sample_gpt(model, block_size, stoi, itos, device, start_text="ROMEO:", max_new_tokens=400):
    model.eval()
    context = torch.zeros((1, block_size), dtype=torch.long, device=device)
```

```
- 학습용이 아니므로 grad 계산을 하지 않음 + 생성/평가 모드로 전환
- 초기 context를 0 * block_size 길이의 벡터로 설정
  0은 .를 의미함
```

```python
    for ch in start_text:
        if ch in stoi:
            ix = torch.tensor([[stoi[ch]]], device=device)
            context = torch.cat([context[:, 1:], ix], dim=1)
    out = list(start_text)
```

```
- start_text가 제시된다면, 그것을 숫자로 변환(ix)
- 그 ix를 바탕으로 context를 업데이트를 시킴
```

```python
    for _ in range(max_new_tokens):
        logits = model(context)
        logits = logits[:, -1, :]
```

```
- context를 모델에 입력시킴
  context.shape = (1,32) -> logits.shape = (1, 32, vocab_size)
- 모델은 block_size 위치 각각에 대해 다음 글자를 예측함
  우리가 필요한건 맨 마지막 글자 다음에 올 글자
  따라서 마지막 위치만 가져오게 함
```

```python
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)
        out.append(itos[ix.item()])
        context = torch.cat([context[:, 1:], ix], dim=1)
    return "".join(out)
```

```
- 이후 그것을 확률화 시키고 다항분포 때려서 랜덤으로 다음 글자가 정해지게 함
- 그리고 그것을 out에다가 append시키고, context도 업데이트 시킴
- 출력은 out을 하나로 모음
```

---

# 한국어 텍스트 적용해보기

preview 자꾸 오류가 떠서 여기다가 출력 결과를 정리했습니다.

## 학습 데이터

심훈의 소설 『상록수』 텍스트

출처: 위키문헌(https://ko.wikisource.org/wiki/%EC%9C%84%ED%82%A4%EB%AC%B8%ED%97%8C:%EB%8C%80%EB%AC%B8)
 
- 불필요한 공백, 과도한 줄바꿈, 다운로드 과정에서 섞인 특수기호를 정리
- 다만 원문의 문체와 문장 구조는 최대한 유지  
- 최종적으로 문자 단위 학습이 가능하도록 UTF-8 형식의 `.txt` 파일로 저장

---

## 학습 손실 변화

```python
def sequence_cross_entropy(logits, targets):
    return F.cross_entropy(logits.transpose(1, 2), targets)

def train_one_epoch(model, loader, optimizer, device, max_steps=None):
    model.train()
    total_loss, total_count = 0.0, 0
    for step, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = sequence_cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)
        if max_steps is not None and step + 1 >= max_steps:
            break
    return total_loss / total_count

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyGPT(vocab_size, block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(50):
    train_loss = train_one_epoch(model, loader, optimizer, device, max_steps=300)
    print(f"epoch {epoch:2d} | train loss {train_loss:.4f}")
```

```text
epoch  0 | train loss 4.3846
epoch  1 | train loss 3.6311
epoch  2 | train loss 3.3752
epoch  3 | train loss 3.1881
epoch  4 | train loss 3.0623
epoch  5 | train loss 2.9515
epoch  6 | train loss 2.8710
epoch  7 | train loss 2.7969
epoch  8 | train loss 2.7350
epoch  9 | train loss 2.6797
epoch 10 | train loss 2.6294
epoch 11 | train loss 2.5816
epoch 12 | train loss 2.5413
epoch 13 | train loss 2.5040
epoch 14 | train loss 2.4670
epoch 15 | train loss 2.4323
epoch 16 | train loss 2.3990
epoch 17 | train loss 2.3716
epoch 18 | train loss 2.3420
epoch 19 | train loss 2.3126
epoch 20 | train loss 2.2931
epoch 21 | train loss 2.2671
epoch 22 | train loss 2.2408
epoch 23 | train loss 2.2198
epoch 24 | train loss 2.1959
epoch 25 | train loss 2.1819
epoch 26 | train loss 2.1575
epoch 27 | train loss 2.1335
epoch 28 | train loss 2.1181
epoch 29 | train loss 2.1014
epoch 30 | train loss 2.0839
epoch 31 | train loss 2.0684
epoch 32 | train loss 2.0500
epoch 33 | train loss 2.0303
epoch 34 | train loss 2.0185
epoch 35 | train loss 2.0046
epoch 36 | train loss 1.9877
epoch 37 | train loss 1.9747
epoch 38 | train loss 1.9611
epoch 39 | train loss 1.9476
epoch 40 | train loss 1.9366
epoch 41 | train loss 1.9256
epoch 42 | train loss 1.9092
epoch 43 | train loss 1.9015
epoch 44 | train loss 1.8892
epoch 45 | train loss 1.8779
epoch 46 | train loss 1.8672
epoch 47 | train loss 1.8551
epoch 48 | train loss 1.8464
epoch 49 | train loss 1.8356
```

loss가 지속적으로 감소하는 것을 볼 수 있다.

colab 런타임 연결이 끊겨서 학습을 100번에서 50번으로 줄였음.

---

## 생성 결과

```python
@torch.no_grad()
def sample_gpt(model, block_size, stoi, itos, device, start_text="상록수", max_new_tokens=400):
    model.eval()
    context = torch.zeros((1, block_size), dtype=torch.long, device=device)
    for ch in start_text:
        if ch in stoi:
            ix = torch.tensor([[stoi[ch]]], device=device)
            context = torch.cat([context[:, 1:], ix], dim=1)
    out = list(start_text)
    for _ in range(max_new_tokens):
        logits = model(context)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)
        out.append(itos[ix.item()])
        context = torch.cat([context[:, 1:], ix], dim=1)
    return "".join(out)

print(sample_gpt(model, block_size, stoi, itos, device, start_text="상록수", max_new_tokens=500))
```

```text
상록수가?" 동혁의 고요히 묻는다. 동혁은,
"안 들었습니다." 하고 명령이라는 말일세만 무어라고 비단 말이니 이 집의 산촌, 솜털 같은 눈에서 속아지 않고 생각에 잠겼다.
"안녕히 인제는 떠날 때면 와서 기천이가 마주 끓는 것버텀 걸어오는 것까지 끝까지 다투는 대로 갈피를 생활한다. 그는 백씨는 며느리로 일을 몇십시다! 조선 같은 낫살이 부르짖고 숭늉병을 들고 나와서 얼굴을 다시다가,
"난 그래야 돈은 한 다 안 되구 보기가 어렵어요?" 하고 제 것을 진지뿌렁거리고 양심에 소개를 불렀다. 저의 여간 지금의 진찰을 같이 멈추고 소개하는 흥분을 바라보고 말욱 심상한다. 한 목소리이 후련진 핑 아래로 영신의 앞으로 고개를 꼬느고 들어온다. 그는 푸숭지기와 같이 헛간에 이름을 뛰고 앉았던 눈치다. 동혁은 발딱 일어서며,
"저 자 고만둬라구 해두 좋은 일이나 사정을 헐 것 없이 없네." 하고 막중을 벌린 것 같은 사실이었다. 기천이는 시골서 동혁이가 그 뒤로 나갔다. 그는 마누라지 말을 끄르고 
```


