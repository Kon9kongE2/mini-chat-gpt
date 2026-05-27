# 실행 방법

1. 리포지터리 폴더로 이동한다.

cd /workspaces/mini-chat-gpt

2. 가상환경을 활성화한다.

이미 .venv를 만들었다면:
source .venv/bin/activate

만약 .venv가 없다고 나오면:
python -m venv .venv
source .venv/bin/activate

3. 필요한 패키지를 설치한다.

pip install -r setup/requirements.txt

4. 노트북을 연다.

code notebooks/lucky_day_learning.ipynb

5. 셀을 위에서부터 순서대로 실행한다.

6. 학습 데이터 파일은 아래 경로를 사용한다.

data/lucky_day.txt
