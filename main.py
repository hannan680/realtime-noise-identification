
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from fastapi.responses import JSONResponse
import librosa
from pydub import AudioSegment
import socketio
import uvicorn


sio = socketio.AsyncServer(async_mode='asgi')
socket_app = socketio.ASGIApp(sio)


app = FastAPI()



origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = tf.lite.Interpreter(model_path="./model/model.tflite")

classes = ["airplane",  "Baby Crying",  "bell",
           "construction",  "engine",  "helicoptor",  "horn",]


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


async def convert_audio(file: UploadFile):
    wav_path = ""
    temp_file_path = f"./mp3/{file.filename}"
    with open(temp_file_path, "wb") as temp_file:
        content = await file.read()
        temp_file.write(content)

    if (file.filename.split(".")[1] != 'wav'):
        audio = AudioSegment.from_file(
            temp_file_path, format=file.filename.split(".")[1])
        wav_path = temp_file_path.replace(".mp4", ".wav")
        audio.export(wav_path, format="wav")
        os.remove(temp_file_path);

    else:
        wav_path = temp_file_path
    return wav_path


async def emit_prediction_event(prediction):
    await sio.emit('prediction', prediction)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    print(file.filename)
    wav_file_path = await convert_audio(file)

    print(wav_file_path)
    waveform, sr = librosa.load(wav_file_path, sr=16000)
    if waveform.shape[0] % 16000 != 0:
        waveform = np.concatenate([waveform, np.zeros(16000)])

    input_details = model.get_input_details()
    output_details = model.get_output_details()

    model.resize_tensor_input(input_details[0]['index'], (1, len(waveform)))
    model.allocate_tensors()

    model.set_tensor(input_details[0]['index'],
                     waveform[None].astype('float32'))
    model.invoke()

    class_scores = model.get_tensor(output_details[0]['index'])
    class_scores_list = class_scores.tolist()
    print(" ")
    print(" ")
    print("class_scores", class_scores.tolist())
    print(" ")
    print(" ")
    print("Class : ", classes[class_scores.argmax()])
    os.remove(wav_file_path);

    response_data = {
        "class_scores":  class_scores_list,
        "classes": classes[class_scores.argmax()]
    }
    await emit_prediction_event(response_data)

    return JSONResponse(response_data)




app.mount("/", socket_app)


@sio.event
async def connect(sid, environ):
    print('Connected:', sid)


@sio.event
async def disconnect(sid):
    print('Disconnected:', sid)


@sio.event
async def my_event(sid, data):
    print('Received data:', data)
    await sio.emit('response_event', 'This is the response', room=sid)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    uvicorn.run("main:app", host='0.0.0.0', port=port)
