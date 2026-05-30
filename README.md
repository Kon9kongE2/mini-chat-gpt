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
- NextTokenDataset 클래스 정의
- dunder method __init__으로 dataset 생성시 필요한 값들을 저장하게 해줌
- `len(self.data)- self.block_size`는 idx가 너무 뒤로 가면 y를 생성할 수 없으므로 학습 가능한 데이터를 이런 식으로 지정하였음
- x는 지정한 idx ~ idx + block_size(문제지), y는 x의 양쪽 값에서 1씩 더한 값들의 집합(답지)이다.
```

---

## 2. 모델에 필요한 개념

### 1. Multi- head attention

Multi -head attention: 현재의 토큰에서 과거의 토큰을 바탕으로 미래를 예측. 이때 과거의 토큰을 살펴보는 방법(HEAD)이 여러 개 있는데, 그 여러 개의 Head가 각자 다른 방식으로 문맥을 보고, 그 결과를 합치는 구조로 이해하였음.

```
- HEAD 클래스 정의 파트 1
- query: 현재의 토큰이 던지는 질문 or 현재의 토큰이 찾고 싶은 정보의 조건
- key: 각 토큰이 내걸고 있는 자기소개표(조건)
- value: query와 key를 비교해서, 실제로 가져오는 것은 value
- 투입값: x = (B, T, emb_dim) / 출력값: (B, T head_size)

```

```
- HEAD 클래스 정의 파트 2
- 


### 2. Feedforward
### 3. Block

---

## 3. 모델(Tiny GPT)

---

## 4. 학습

