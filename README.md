# notebook06 정리

## notebook01~06의 공통된 흐름 정리

```
1. 텍스트 데이터를 모델이 학습할 수 있는 형태로 바꾸기
2. 다음 문자를 예측하는 모델 만들기
3. 모델을 학습
4. 학습된 모델로 문장을 생성하기
```

---

## 1. 데이터 준비

```
- 코드 실행에 필요한 패키지 불러오기
- input.txt 파일 읽기
- 읽어온 파일에서 텍스트의 중복된 것을 제외, 알파벳 순으로 정렬하여 리스트 형식으로 저장
- 이를 바탕으로 stoi, itos 딕셔너리 생성; 이때 vocab_size는 문자의 종류를 나타냄
- data는 text에 있는 모든 문자를 모두 숫자로 변환한 형태로 저장됨
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

```
** HEAD 클래스 정의 파트 1 **
- query: 현재의 토큰이 던지는 질문 or 현재의 토큰이 찾고 싶은 정보의 조건
- key: 각 토큰이 내걸고 있는 자기소개표(조건)
- value: query와 key를 비교해서, 실제로 가져오는 것은 value
- 투입값: x = (B, T, emb_dim) / 출력값: (B, T head_size)
- 마스크를 위해서 원소가 1인 하삼각행렬을 먼저 정의
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

```
** MultiHeadAttention 클래스 정의 파트 1 **
- head_size는 embedding 차원을 Head들이 분담하여 처리함에 따라 결정된다.
- self.heads: 여러개의 Head들을 생성하는 코드
- self.proj: 여러개의 heade들의 출력을 이어붙여 선형변환하는 코드
- self.dropout: 과적합 방지용 최종 출력에 일부 값을 dropout
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

```
** Feedforward 클래스 정의 파트 1 **
- nn.Sequential에 따라 밑의 코드들을 순서대로 실행
- 첫 번째 nn.Linear: 선형변환 행렬이 (emb_dim, 4 * emb_dim)
  즉 이 행렬을 곱한다면 출력 차원이 4배 늘어나게 만듦
- ReLU: 동강에서 나온 함수. 음수는 0 양수는 그대로 두는 활성화 함수
- 두 번째 nn.Linear: 선형변환 행렬이 (4 * emb_dim, emb_dim)
  즉 이 행렬을 곱한다면 출력 차원을 다시 줄어들게 만듦
```

```
** Feedforward 클래스 정의 파트 2 **
- 입력값 x를 넣으면 실행되는 부분
- 즉 파트 1의 단계를 실행한다는 것
```

### 3. Block

Multi - Head Attention으로 각 토큰의 벡터를 변환시키고, FeedForward로 그 토큰 벡터를 다시 가공하고, Residual connection으로 기존 정보를 유지

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

```
- cross_entropy 계산하는 코드 정의
```

```
- 학습 모드로 전환
- 초기 로스와 초기 학습 횟수를 0으로 정의
```

```
- xb 데이터는 batch 단위로 뽑은 문장들
  이것을 tiny gpt model에 투입, 다음 글자들에 대한 점수를 출력
- 답지 yb 데이터와 loss를 계산
```

```
- 계산한 loss를 바탕으로 그레디언트를 계산, 가중치를 개선시키고 업데이트 시킴
- 현재는 max_step이 None이지만, 만약 설정한 max_step을 넘어선다면 중단
- 총 loss에 대한 평균을 출력
```

---

## 5. 생성

```
- 학습용이 아니므로 grad 계산을 하지 않음 + 생성/평가 모드로 전환
- 초기 context를 0 * block_size 길이의 벡터로 설정
  0은 .를 의미함
```

```
- start_text가 제시된다면, 그것을 숫자로 변환(ix)
- 그 ix를 바탕으로 context를 업데이트를 시킴
```

```
- context를 모델에 입력시킴
  context.shape = (1,32) -> logits.shape = (1, 32, vocab_size)
- 모델은 block_size 위치 각각에 대해 다음 글자를 예측함
  우리가 필요한건 맨 마지막 글자 다음에 올 글자
  따라서 마지막 위치만 가져오게 함
```

```
- 이후 그것을 확률화 시키고 다항분포 때려서 랜덤으로 다음 글자가 정해지게 함
- 그리고 그것을 out에다가 append시키고, context도 업데이트 시킴
- 출력은 out을 하나로 모음
```







