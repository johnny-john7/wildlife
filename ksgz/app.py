import asyncio
import websockets
import cv2
import csv
import datetime
import time
import queue
import threading
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
import simpleaudio as sa
import requests
from collections import deque
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# Global flags, queues, and alert indicator
running_video = False
running_audio = False
q_audio = queue.Queue()
human_alert = False  # Set to True when a human is detected

# Load YOLO model
yolo_model = YOLO('kaggle 5/working/runs/detect/train3/weights/best.pt')

# Load Bird Audio Model
audio_model = tf.keras.models.load_model("best_model.h5", compile=False)
audio_model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(reduction="sum_over_batch_size"),
    optimizer="adam",
    metrics=["accuracy"]
)

# Audio settings
SAMPLE_RATE = 22050
DURATION = 2
BUFFER_SIZE = int(SAMPLE_RATE * DURATION)

# CSV File Setup
csv_filename_objects = "logs_objects.csv"
csv_filename_birds = "logs_birds.csv"

# Log queues with FIFO (max 20 logs)
log_queue_objects = deque(maxlen=20)
log_queue_birds = deque(maxlen=20)

# Detection threshold for YOLO
threshold = 0.3

# WebSocket-based Audio Handler
async def audio_handler(websocket, path):
    async for audio_chunk in websocket:
        print("Received audio data:", len(audio_chunk))

start_server = websockets.serve(audio_handler, "0.0.0.0", 5002)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_video', methods=['POST'])
def start_video():
    global running_video
    running_video = True
    print("▶ Video started.")
    return jsonify({"status": "started"})

@app.route('/stop_video', methods=['POST'])
def stop_video():
    global running_video
    running_video = False
    print("⏸ Video stopped.")
    return jsonify({"status": "stopped"})

@app.route('/start_audio', methods=['POST'])
def start_audio():
    global running_audio
    running_audio = True
    audio_thread = threading.Thread(target=predict_bird, daemon=True)
    audio_thread.start()
    print("▶ Audio started.")
    return jsonify({"status": "audio_started"})

@app.route('/stop_audio', methods=['POST'])
def stop_audio():
    global running_audio
    running_audio = False
    print("⏸ Audio stopped.")
    return jsonify({"status": "audio_stopped"})

@app.route('/video_feed')
def video_feed():
    return Response(requests.get("http://<RPI_IP>:5001/video_feed").content, mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
