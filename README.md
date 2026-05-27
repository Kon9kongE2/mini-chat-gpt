# Notebook 06 학습 정리: Toward a Tiny GPT

## 1. 학습 목표

이번 notebook 06에서는 이전 노트북들에서 다룬 기본적인 언어 모델 구조를 확장해서, 아주 작은 형태의 GPT 모델을 직접 구현해보았다.  
처음에는 단순히 다음 글자를 예측하는 수준에서 시작했지만, 이번에는 GPT 구조에서 핵심적으로 사용되는 여러 요소들을 하나씩 추가했다.

주요 학습 내용은 다음과 같다.

- Multi-head attention
- Feedforward network
- Residual connection
- Layer normalization
- Block stacking
- Character-level text generation

AI 모델을 처음부터 완전히 이해하기는 어려웠지만, 이번 노트북을 통해 GPT가 단순히 “문장을 외워서 출력하는 모델”이 아니라, 이전 문맥을 바탕으로 다음 토큰을 확률적으로 예측하는 구조라는 점을 이해할 수 있었다.

---

## 2. 데이터 처리 방식

처음에는 `input.txt` 파일을 불러와서 전체 텍스트를 하나의 문자열로 읽는다.  
그다음 텍스트에 등장하는 모든 문자를 모아 vocabulary를 만들고, 각 문자를 숫자로 변환한다.

```python
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
