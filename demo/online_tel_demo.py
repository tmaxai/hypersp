from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
import uvicorn
import json
import numpy as np
import soundfile as sf
from recognizer import Recognizer

recognizer = Recognizer(output_dir='logs',
                        model_cfg='tasks/SpeechRecognition/ktelspeech/configs/jasper10x5dr_sp_offline_specaugment.yaml',
                        ckpt='tasks/SpeechRecognition/ktelspeech/checkpoints/Jasper_epoch80_checkpoint.pt',
                        task_path="tasks.SpeechRecognition.ktelspeech.local.manifest",
                        vocab="tasks/SpeechRecognition/ktelspeech/data/KtelSpeech/vocab")
recognizer.load_model()


app = FastAPI()
app.mount("/static", StaticFiles(directory="demo/static"), name="static")

app.stream = dict()

templates = Jinja2Templates(directory="demo/templates")


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(json.dumps(message))

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index_tel.html", {"request": request})


@app.websocket("/recognize")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            data = json.loads(data)

            state = data["state"]
            frame = data["data"]

            if state == 2:
                del app.stream[str(websocket)]
                await manager.send_personal_message({"final": True, "text": ""}, websocket)
                await websocket.close()
                break

            else:
                try:
                    app.stream[str(websocket)].extend(data["data"])
                except KeyError:
                    app.stream[str(websocket)] = data["data"]

                if state == 1:
                    audio = np.array(
                        app.stream[str(websocket)], dtype=np.float32)

                    sf.write("test.wav", audio, samplerate=44100)
                    text = recognizer.transcribe("test.wav", option=1)

                    await manager.send_personal_message({"final": False, "text": text}, websocket)
                elif state == 0:
                    app.stream[str(websocket)].clear()

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("disconnected")


if __name__ == "__main__":
    uvicorn.run("online_tel_demo:app", host="0.0.0.0", port=15004, log_level="info",
                ssl_keyfile="demo/cert/future.key", ssl_certfile="demo/cert/future.crt")
