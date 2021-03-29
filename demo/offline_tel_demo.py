import os
from flask_cors import CORS, cross_origin
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import librosa
import soundfile as sf
import ffmpeg
import base64
import uuid
import json
from recognizer import Recognizer

app = Flask(__name__)
CORS(app)

recognizer = Recognizer(output_dir='logs',
                        model_cfg='tasks/SpeechRecognition/ktelspeech/configs/jasper10x5dr_sp_offline_specaugment.yaml',
                        ckpt='tasks/SpeechRecognition/ktelspeech/checkpoints/Jasper_epoch80_checkpoint.pt',
                        task_path="tasks.SpeechRecognition.ktelspeech.local.manifest",
                        vocab="tasks/SpeechRecognition/ktelspeech/data/KtelSpeech/vocab")
recognizer.load_model()


@app.route('/')
def index():
    return render_template('index_tel_offline.html')


@app.route('/recognize', methods=['POST'])
def file_upload():
    data = request.get_json()
    print("uid: ", data["uid"])
    print("sid: ", data["sid"])
    dec_data = base64.b64decode(data["data"])

    os.makedirs('./logs', exist_ok=True)
    file_name = os.path.join('./logs', secure_filename(str(uuid.uuid4())))
    wav_file_name = file_name + '.wav'

    with open(file_name, mode='wb') as fd:
        fd.write(dec_data)
    _ = (ffmpeg.input(file_name)
               .output(wav_file_name, format='wav', acodec='pcm_s16le', ac=1, ar=16000)
               .overwrite_output()
               .global_args('-hide_banner')
               .global_args('-loglevel', 'error')
               .run())

    text = recognizer.transcribe(wav_file_name, option=1)

    os.remove(wav_file_name)
    os.remove(file_name)

    return jsonify({"text": text})


if __name__ == "__main__":
    app.run(port=15005)
