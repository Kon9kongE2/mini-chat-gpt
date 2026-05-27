# Notebook 06 학습 정리: Tiny GPT 구현

## 1. 학습 목표

Notebook 06에서는 이전 노트북에서 배운 언어 모델 구조를 바탕으로, 더 GPT에 가까운 작은 모델을 직접 구현했다.  
이번 노트북에서 새로 다룬 핵심 내용은 multi-head attention, feedforward network, residual connection, layer normalization, block stacking이다.

처음에는 GPT라는 모델이 굉장히 복잡하게 느껴졌지만, 코드를 따라가면서 결국 기본 구조는 “이전 문맥을 보고 다음 문자를 예측하는 모델”이라는 점을 이해할 수 있었다.  
이번 실습의 목적은 성능이 좋은 AI 모델을 만드는 것보다는, GPT가 어떤 부품들로 구성되고 각 부품이 어떤 역할을 하는지 확인하는 데 있었다.

## 2. 전체 흐름

Notebook 06은 이전까지 만든 모델들을 바탕으로 최종적으로 작은 GPT 구조를 완성하는 흐름이다.

전체 흐름은 다음과 같다.

1. Bigram 모델
2. 이름 데이터에 대한 MLP
3. Shakespeare 텍스트에 대한 MLP
4. GPT-style dataset과 최소한의 sequence model
5. Single-head masked self-attention
6. Tiny GPT

즉, 처음부터 복잡한 GPT를 바로 만드는 것이 아니라, 단순한 모델에서 시작해서 점점 GPT에 필요한 구조를 추가하는 방식으로 진행된다.

## 3. Multi-head Attention

이번 노트북에서 가장 중요한 부분은 multi-head attention이다.  
Attention은 현재 위치의 토큰이 이전 토큰들 중 어떤 정보를 참고할지 계산하는 구조이다.

코드에서는 하나의 attention head를 먼저 만들고, 여러 개의 head를 묶어서 multi-head attention을 만든다.  
하나의 head만 사용하면 문맥을 한 가지 방식으로만 보게 되지만, 여러 head를 사용하면 문맥을 여러 관점에서 볼 수 있다.

나는 이 부분을 다음과 같이 이해했다.

- 하나의 head: 문장을 한 가지 관점에서 보는 것
- 여러 개의 head: 문장을 여러 관점에서 동시에 보는 것
- attention score: 어떤 토큰을 더 중요하게 볼지 정하는 값
- softmax: attention score를 확률처럼 바꾸는 과정

이 구조 덕분에 모델은 단순히 바로 앞 문자만 보는 것이 아니라, 이전 문맥의 여러 위치를 참고하면서 다음 문자를 예측할 수 있다.

## 4. Masked Self-Attention

GPT는 문장을 왼쪽에서 오른쪽으로 생성한다.  
따라서 현재 위치에서 미래의 토큰을 보면 안 된다.

이를 막기 위해 masked self-attention을 사용한다.  
코드에서는 lower triangular matrix를 만들어서 현재 위치 이후의 값들을 가린다.

이 부분의 의미는 모델이 정답을 미리 보지 못하게 막는 것이다.  
예를 들어 세 번째 글자를 예측할 때 네 번째, 다섯 번째 글자를 보면 학습이 의미가 없어지기 때문에, 미래 정보를 가리는 과정이 필요하다.

결국 masked self-attention은 GPT가 실제 텍스트 생성 상황과 같은 조건에서 학습하도록 만드는 장치라고 볼 수 있다.

## 5. Feedforward Network

Attention을 통과한 뒤에는 feedforward network를 거친다.  
Feedforward network는 각 위치의 표현을 한 번 더 변환하는 역할을 한다.

Notebook 06에서는 중간 차원을 늘렸다가 다시 줄이는 구조를 사용한다.  
이는 모델이 더 복잡한 패턴을 학습할 수 있도록 해준다.

내가 이해한 방식으로 정리하면, attention은 “어떤 정보를 참고할지”를 정하는 부분이고, feedforward network는 “참고한 정보를 바탕으로 표현을 다시 가공하는 부분”이다.

## 6. Residual Connection

Residual connection은 기존 입력을 출력에 다시 더해주는 구조이다.  
코드에서는 다음과 같은 흐름으로 사용된다.

- attention 결과를 기존 입력에 더함
- feedforward 결과를 기존 입력에 더함

이렇게 하는 이유는 모델이 깊어질수록 기존 정보가 사라지는 것을 막기 위해서이다.  
즉, 새로운 정보를 학습하면서도 원래 입력 정보를 계속 유지할 수 있게 해준다.

처음에는 단순히 x를 다시 더하는 코드처럼 보였지만, 실제로는 깊은 신경망을 안정적으로 학습시키는 데 중요한 역할을 한다.

## 7. Layer Normalization

Layer normalization은 각 층의 입력을 안정적으로 만들어주는 역할을 한다.  
Notebook 06에서는 attention과 feedforward network에 들어가기 전에 layer normalization을 적용한다.

이 구조를 통해 학습 과정이 더 안정적으로 진행될 수 있다.  
모델이 깊어질수록 값의 크기가 불안정해질 수 있는데, layer normalization은 이를 완화하는 역할을 한다.

## 8. Transformer Block

Notebook 06에서는 attention, feedforward network, residual connection, layer normalization을 합쳐 하나의 block으로 만든다.

Transformer block의 흐름은 다음과 같이 이해할 수 있다.

1. 입력을 layer normalization에 통과시킨다.
2. masked multi-head attention을 적용한다.
3. 기존 입력과 attention 결과를 더한다.
4. 다시 layer normalization을 적용한다.
5. feedforward network를 통과시킨다.
6. 기존 값과 feedforward 결과를 더한다.

이 block을 여러 개 쌓으면 더 깊은 GPT 구조가 된다.  
즉, GPT는 하나의 복잡한 함수라기보다는 비슷한 구조의 block을 반복해서 쌓은 모델이라고 이해할 수 있다.

## 9. TinyGPT 모델 구조

최종적으로 구현한 모델은 TinyGPT이다.  
이 모델은 문자 입력을 받아 다음 문자를 예측하는 작은 GPT 모델이다.

전체 구조는 다음과 같다.

```text
문자 입력
→ 숫자로 변환
→ token embedding
→ position embedding
→ transformer block
→ layer normalization
→ linear layer
→ 다음 문자 예측
```

Token embedding은 각 문자를 벡터로 바꾸는 역할을 한다.  
Position embedding은 각 문자가 어느 위치에 있는지를 알려주는 역할을 한다.

GPT는 단순히 어떤 문자가 등장했는지만 보는 것이 아니라, 그 문자가 문장 안에서 어느 위치에 있는지도 함께 사용한다.  
이 점이 단순한 문자 예측 모델보다 더 발전된 부분이라고 볼 수 있다.

## 10. 학습 과정

학습에서는 모델이 예측한 다음 문자와 실제 다음 문자를 비교한다.  
이때 cross entropy loss를 사용한다.

모델은 처음에는 거의 무작위에 가까운 예측을 하지만, 학습이 진행되면서 loss를 줄이는 방향으로 파라미터를 업데이트한다.  
Notebook 06에서는 optimizer로 AdamW를 사용한다.

내가 이해한 학습 과정은 다음과 같다.

1. 텍스트 일부를 입력으로 넣는다.
2. 모델이 각 위치에서 다음 문자를 예측한다.
3. 실제 다음 문자와 비교해서 loss를 계산한다.
4. loss가 줄어드는 방향으로 모델의 파라미터를 수정한다.
5. 이 과정을 반복한다.

즉, 모델은 문장을 통째로 외우는 것이 아니라, 주어진 문맥 다음에 어떤 문자가 올 가능성이 높은지를 반복적으로 학습한다.

## 11. Sampling

학습이 끝난 뒤에는 sample_gpt 함수를 사용해서 새로운 텍스트를 생성한다.  
모델은 시작 문장을 입력받고, 그 뒤에 올 문자를 하나씩 예측한다.

여기서 중요한 점은 항상 가장 확률이 높은 문자만 고르는 것이 아니라, 확률분포에서 샘플링한다는 것이다.  
그래서 같은 시작 문장을 넣어도 매번 결과가 조금씩 달라질 수 있다.

이 부분을 통해 텍스트 생성은 단순한 복사가 아니라, 학습된 확률분포를 바탕으로 새로운 문자를 이어 붙이는 과정이라는 점을 알 수 있었다.

## 12. 한국어 텍스트 적용

기본 notebook에서는 기존 input.txt를 사용하지만, 추가 적용으로 한국어 텍스트를 넣어 같은 구조로 학습을 시도할 수 있다.  
예를 들어 국문 소설, 한국어 설명문, 코딩 관련 한국어 문서 등을 텍스트 파일로 정리한 뒤 input.txt 대신 사용할 수 있다.

한국어 텍스트를 사용할 때의 흐름은 다음과 같다.

```text
한국어 텍스트 파일 준비
→ UTF-8 형식으로 저장
→ 텍스트 읽기
→ 문자 단위 vocabulary 생성
→ 문자를 숫자로 변환
→ TinyGPT 모델 학습
→ 시작 문장을 넣고 생성 결과 확인
```

영어 텍스트와 비교했을 때 한국어는 문자 종류가 더 많기 때문에 vocabulary size가 커질 수 있다.  
또한 데이터가 충분하지 않으면 생성 결과가 자연스럽지 않을 수 있다.

따라서 한국어 적용의 목적은 완성도 높은 문장을 만드는 것보다는, 같은 모델 구조를 다른 텍스트 데이터에도 적용할 수 있다는 점을 확인하는 데 있다고 생각한다.

## 13. 직접 적용하면서 느낀 점

한국어 자료를 직접 넣어보면, 모델이 텍스트의 의미를 완전히 이해한다기보다는 자주 등장하는 글자 조합과 문체를 따라 하려는 모습을 보인다.  
데이터가 짧거나 학습 시간이 부족하면 문장이 어색하게 나오지만, 그래도 원본 텍스트의 분위기나 반복되는 표현이 일부 반영될 수 있다.

이를 통해 AI 모델의 결과가 데이터의 양과 품질에 크게 의존한다는 점을 알 수 있었다.  
같은 코드라도 어떤 텍스트를 넣는지에 따라 생성 결과가 달라지기 때문에, 데이터 준비 과정도 모델 구현만큼 중요하다고 느꼈다.

## 14. 배운 점

이번 notebook 06을 통해 다음 내용을 배웠다.

- GPT는 다음 토큰을 예측하는 방식으로 학습된다.
- Attention은 이전 문맥 중 중요한 정보를 참고하는 구조이다.
- Masking은 미래 정보를 보지 못하게 막는 역할을 한다.
- Multi-head attention은 여러 관점에서 문맥을 보는 방식이다.
- Feedforward network는 attention 이후의 표현을 다시 변환한다.
- Residual connection은 기존 정보를 유지하는 데 도움을 준다.
- Layer normalization은 학습을 안정적으로 만든다.
- GPT는 token embedding과 position embedding을 함께 사용한다.
- 학습된 모델은 확률적으로 다음 문자를 선택하면서 텍스트를 생성한다.

## 15. 한계점

이번 모델은 실제 GPT와 비교하면 매우 작은 모델이다.  
또한 단어 단위가 아니라 문자 단위로 학습하기 때문에 긴 문맥을 자연스럽게 이해하는 데 한계가 있다.

한국어 텍스트를 학습할 때도 한계가 있다.  
한국어는 문자 종류가 많고 문장 구조가 복잡하기 때문에, 짧은 데이터만으로는 자연스러운 결과를 얻기 어렵다.  
또한 학습 시간이 짧으면 원본 데이터의 문체를 충분히 반영하지 못한다.

따라서 이번 실습은 실용적인 AI 서비스를 만드는 과정이라기보다는, GPT의 기본 구조를 작은 코드로 이해하는 실습이라고 보는 것이 적절하다.

## 16. 정리

Notebook 06은 작은 GPT 모델을 직접 구현하면서 GPT의 기본 구조를 이해하는 실습이었다.  
처음에는 attention, residual connection, layer normalization 같은 용어가 낯설었지만, 코드를 따라가면서 각각이 모델 안에서 어떤 역할을 하는지 확인할 수 있었다.

결론적으로 TinyGPT는 다음과 같은 방식으로 작동한다.

```text
텍스트 입력
→ 문자 단위 숫자 변환
→ embedding
→ transformer block
→ 다음 문자 예측
→ sampling을 통한 텍스트 생성
```

이번 실습을 통해 GPT가 단순히 문장을 외우는 모델이 아니라, 입력된 문맥을 바탕으로 다음에 올 가능성이 높은 문자를 계산하는 모델이라는 점을 이해할 수 있었다.
