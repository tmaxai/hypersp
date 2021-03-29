# Hypersp - hyper signal processing

## Hypersp
Hypersp는 AI 음성처리를 위한 툴킷으로 개발되었습니다. 현재 제공되는 task는 음성인식으로, 음성인식 이란 음성을 텍스트를 변환해주는 기술입니다. Hypersp는 pytorch 기반으로 학습,평가,추론 및 데모 서버를 제공합니다.

## Usage
docker image 사용이 권장됩니다
4월 15일 공개됩니다
<!-- 
* image 다운로드 및 container 접속
  * docker pull {IMAGE}
  * docker run --gpus all --shm-size 8G -it -d -p 15000-15005:15000-15005 {IMAGE} bash
  * docker exec -it {CONTAINER_ID} bash -->

## Supported Dataset & Pretrained Checkpoint Loadmap
* ~ 2021.04.15
  * klecspeech
  * kconfspeech
  * krespspeech
  * ktelspeech
* ~ 2021.04.30
  * ksponspeech
  * librispeech
* from May every 2 weeks, new dataset will be updated
  
## AI Model Loadmap
- [x] jasper
- [ ] transformer (21.2Q)
- [ ] conformer (21.2Q)
- [ ] quartznet (21.2Q)


### 한국어 강의 데이터
```
* 한국어 강의 데이터 학습
  * 학습데이터셋을 적절한 경로에 다운받습니다
  * 폴더를 이동합니다
    * cd /hypersp/tasks/SpeechRecognition/klecspeech
  * scripts/preprocess_klecspeech.sh에서 데이터를 다운받은 경로로 수정합니다
  * 전처리를 실행합니다
    * ./scripts/preprocess_klecspeech.sh
  * scripts/train.sh에서 전처리된 파일 경로 등 파라미터를 적절히 수정합니다
  * 학습을 시작합니다
    * ./scripts/train.sh
  
* 한국어 강의 데이터 평가
  * 학습데이터셋 및 모델 checkpoint를 적절한 경로에 다운받습니다
  * 폴더를 이동합니다
    * cd /hypersp/tasks/SpeechRecognition/klecspeech
  * scripts/preprocess_klecspeech.sh에서 데이터를 다운받은 경로로 수정합니다
  * 전처리를 실행합니다
    * ./scripts/preprocess_klecspeech.sh
  * scripts/evaluation.sh에서 전처리된 파일 경로, checkpoint 경로 등 파라미터를 적절히 수정합니다
  * 학습을 시작합니다
    * ./scripts/evaluation.sh
```

### 회의 음성 데이터
```
* 회의 음성 데이터 학습
  * 학습데이터셋을 적절한 경로에 다운받습니다
  * 폴더를 이동합니다
    * cd /hypersp/tasks/SpeechRecognition/kconfspeech
  * scripts/preprocess_kconfspeech.sh에서 데이터를 다운받은 경로로 수정합니다
  * 전처리를 실행합니다
    * ./scripts/preprocess_kconfspeech.sh
  * scripts/train.sh에서 전처리된 파일 경로 등 파라미터를 적절히 수정합니다
  * 학습을 시작합니다
    * ./scripts/train.sh
  
* 회의 음성 데이터 평가
  * 학습데이터셋 및 모델 checkpoint를 적절한 경로에 다운받습니다
  * 폴더를 이동합니다
    * cd /hypersp/tasks/SpeechRecognition/kconfspeech
  * scripts/preprocess_kconfspeech.sh에서 데이터를 다운받은 경로로 수정합니다
  * 전처리를 실행합니다
    * ./scripts/preprocess_kconfspeech.sh
  * scripts/evaluation.sh에서 전처리된 파일 경로, checkpoint 경로 등 파라미터를 적절히 수정합니다
  * 학습을 시작합니다
    * ./scripts/evaluation.sh
```

### 고객 응대 데이터
```
* 고객 응대 데이터 학습
  * 학습데이터셋을 적절한 경로에 다운받습니다
  * 폴더를 이동합니다
    * cd /hypersp/tasks/SpeechRecognition/krespspeech
  * scripts/preprocess_krespspeech.sh에서 데이터를 다운받은 경로로 수정합니다
  * 전처리를 실행합니다
    * ./scripts/preprocess_krespspeech.sh
  * scripts/train.sh에서 전처리된 파일 경로 등 파라미터를 적절히 수정합니다
  * 학습을 시작합니다
    * ./scripts/train.sh
  
* 고객 응대 데이터 평가
  * 학습데이터셋 및 모델 checkpoint를 적절한 경로에 다운받습니다
  * 폴더를 이동합니다
    * cd /hypersp/tasks/SpeechRecognition/krespspeech
  * scripts/preprocess_krespspeech.sh에서 데이터를 다운받은 경로로 수정합니다
  * 전처리를 실행합니다
    * ./scripts/preprocess_krespspeech.sh
  * scripts/evaluation.sh에서 전처리된 파일 경로, checkpoint 경로 등 파라미터를 적절히 수정합니다
  * 학습을 시작합니다
    * ./scripts/evaluation.sh
```

### 상담 음성 데이터
```
* 상담 음성 데이터 학습
  * 학습데이터셋을 적절한 경로에 다운받습니다
  * 폴더를 이동합니다
    * cd /hypersp/tasks/SpeechRecognition/ktelspeech
  * scripts/preprocess_ktelspeech.sh에서 데이터를 다운받은 경로로 수정합니다
  * 전처리를 실행합니다
    * ./scripts/preprocess_ktelspeech.sh
  * scripts/train.sh에서 전처리된 파일 경로 등 파라미터를 적절히 수정합니다
  * 학습을 시작합니다
    * ./scripts/train.sh
  
* 상담 음성 데이터 평가
  * 학습데이터셋 및 모델 checkpoint를 적절한 경로에 다운받습니다
  * 폴더를 이동합니다
    * cd /hypersp/tasks/SpeechRecognition/ktelspeech
  * scripts/preprocess_ktelspeech.sh에서 데이터를 다운받은 경로로 수정합니다
  * 전처리를 실행합니다
    * ./scripts/preprocess_ktelspeech.sh
  * scripts/evaluation.sh에서 전처리된 파일 경로, checkpoint 경로 등 파라미터를 적절히 수정합니다
  * 학습을 시작합니다
    * ./scripts/evaluation.sh
```

## DEMO
* demo 폴더에서 5가지의 데모 서버를 제공합니다
  * 한국어 강의 자막 생성 서비스 실행방법
    * cd /hypersp
    * python demo/subtitles_demo.py
    * port: 15001
  * 발화 단위 대화록 생성 서비스 실행방법
    * cd /hypersp
    * python demo/minutes_demo.py
    * port: 15002
  * 고객응대용 실시간 음성인식 서비스 실행방법
    * cd /hypersp
    * python demo/online_resp_demo.py
    * port: 15003
  * 고객상담용(8k) 실시간 음성인식 서비스 실행방법
    * cd /hypersp
    * python demo/online_tel_demo.py
    * port: 15004
  * 고객상담용(8k) 대화록 생성 서비스 실행방법
    * cd /hypersp
    * python demo/offline_tel_demo.py
    * port: 15005

* NOTICE
  * 데모용으로 production-level로 제공되지 않습니다. production-level의 사용을 원하실 경우 별도로 문의주시기 바랍니다
  * local이 아닌 곳에서 데모를 띄울 경우 아래 파일에서 ip 주소를 알맞게 변경해주셔야 합니다
    * 한국어 강의 자막 생성 서비스: demo/templates/index_subtitles.html
    * 발화 단위 대화록 생성 서비스: demo/templates/index_minutes.html
    * 고객응대용 실시간 음성인식 서비스: demo/static/websocket_resp.js
    * 고객상담용(8k) 실시간 음성인식 서비스: demo/static/websocket_resp.js
    * 고객상담용(8k) 대화록 생성 서비스: demo/templates/index_tel_offline.html
  * 실시간 음성인식에서 brower를 통한 마이크 음성 입력은 반드시 https여야 합니다.
    * https://{{IP_ADDRESS}:{PORT}로 접속하시기 바랍니다
    * 데모용으로 임의의 ssl certificatie가 만들어져 있습니다.

## Request for offline ASR
API url (temporary) - http://{IP_ADDRESS}:{PORT}/recognize
### JSON

| Key  | Type  | Description              |
| ---- | ----- | ------------------------ |
| uid  | str   | user id                  |
| sid  | str   | session id               |
| data | Array | list of PCM16 audio data |
<br>

### example
```
{
    uid: "demo",
    sid: "000001",
    data: [0, 0, 0, ... 0]              
}
```

## Response for offline ASR
### JSON
| Key  | Type | Description     |
| ---- | ---- | --------------- |
| text | str  | recognized text |
<br>

### example
```
{
    text: "안녕하세요",
    final: True             
}
```

## Request for online ASR
API url (temporary) - wss://{IP_ADDRESS}:{PORT}/recognize
### JSON

| Key   | Type  | Description                    |
| ----- | ----- | ------------------------------ |
| state | int   | 음성 상태 flag. 0-묵음, 1-음성 |
| data  | Array | list of PCM16 audio data       |
<br>

### example
```

{
    state: 1
    data: [0, 0, 0, ... 0]              
}
```

## Response for online ASR
### JSON
| Key   | Type | Description                                     |
| ----- | ---- | ----------------------------------------------- |
| final | bool | 음성인식 종료 여부, true 인식 끝, false 인식 중 |
| text  | str  | recognized text                                 |
<br>

### example
```
{
    text: "안녕",
    final: False             
}

...

{
    text: "안녕하세요",
    final: True             
}
```



## reference
* https://github.com/NVIDIA/NeMo
* https://github.com/espnet/espnet
* https://github.com/pytorch/fairseq
* https://github.com/sooftware/KoSpeech