## 개발 환경 및 라이브러리 버전
- Python 3.12.4
- Ubuntu 22.04.4 LTS

- 라이브러리 버전은 requirements.txt 참조
- conda install python==3.12.4 으로 파이썬 설치 후
- pip install -r pre_requirements.txt, pip install -r requirements 이후에 pip install -e . 로 본 프로젝트 설치 (setup.py가 있는 디렉토리에서 실행)


## 실행 방법
inferece.py 실행


## 데이터 
- data/test 폴더 안의 파일들을 data/test/raw 폴더로 옮긴 후
- preprocess/denoise.py, preprocess/diarize.py, preprocess/bound.py 를 실행하면 데이터 전처리가 진행됨 
- 이후 새로운 경로에 맞게 test_denoised.csv 생성
- 기존 data 폴더 대신 위치


## 모델 (weight)
- 모델은 https://drive.google.com/file/d/19tb2Yry6X83Ylw9antHP_0BC_pfWUjqV/view?usp=sharing 에서 다운로드 가능
- data, deep_fake_voice_detection 폴더와 같은 계층에 압축해제 (model 폴더)


## 사전 학습 모델
- 오디오 처리 백본: openai/whisper-small.en (Apache-2.0 License)
- Denoise: resembler-enhance (파이썬 라이브러리, MIT License)
- Diaraize: pyannote-audio (파이썬 라이브러리, MIT License)
