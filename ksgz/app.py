import cv2
import csv
import datetime
import time
import queue
import threading
import numpy as np
import librosa
import requests
import sounddevice as sd
import tensorflow as tf
import simpleaudio as sa
from collections import deque
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import asyncio
import websockets

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

import asyncio
import websockets

async def audio_handler(websocket, path):
    async for audio_chunk in websocket:
        # Process audio here (e.g., save to file or feed to ML model)
        print("Received audio data:", len(audio_chunk))

start_server = websockets.serve(audio_handler, "0.0.0.0", 5002)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()


def generate_frames():
    """Handles video streaming and object detection."""
    global running_video, human_alert

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    time.sleep(2)  # Allow camera to initialize

    while True:
        if not running_video:
            # When video is stopped, send a blank frame
            blank_frame = np.zeros((300, 500, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame.")
            continue

        # Initialize an object count dictionary for the current frame
        object_counts = {}

        # Run YOLO detection on the frame
        results = yolo_model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                label = result.names[int(box.cls[0])]

                if confidence > threshold:
                    # Update count for this label
                    object_counts[label] = object_counts.get(label, 0) + 1

                    # If the detection is a person, use red for the bounding box; otherwise, use green.
                    if label.lower() == "human":
                        color = (0, 0, 255)  # Red
                        human_alert = True
                    else:
                        color = (0, 255, 0)  # Green

                    # Draw detection on the frame with count in the text
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame,
                                f"{label}: {object_counts[label]} ({confidence:.2f})",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2)

                    # Create a log entry for every detection
                    now = datetime.datetime.now()
                    log_entry = [now.strftime("%Y-%m-%d %H:%M:%S"), f"{label} detected", object_counts[label], confidence]

                    log_queue_objects.append(log_entry)
                    # Also log to CSV (optional)
                    with open(csv_filename_objects, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(log_entry)

        # Play beep sound if a human is detected
        if human_alert:
            try:
                sa.WaveObject.from_wave_file("alert.wav").play()
            except Exception as e:
                print("Error playing beep sound:", e)
            # Reset the flag for the next frame
            human_alert = False

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    q_audio.put(indata.copy())


def preprocess_audio(audio_data):
    y = audio_data.flatten()
    mel_spec = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=48, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Resize to 128x48 and add a channel dimension
    mel_spec_db = np.expand_dims(cv2.resize(mel_spec_db, (128, 48)), axis=-1)
    return np.expand_dims(mel_spec_db, axis=0)


def predict_bird():
    """Handles real-time bird sound detection."""
    global running_audio
    print("🎙 Listening for bird sounds...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
        while running_audio:
            audio_data = q_audio.get()
            features = preprocess_audio(audio_data)
            prediction = audio_model.predict(features)
            predicted_class = int(np.argmax(prediction))

            now = datetime.datetime.now()
            log_entry = [now.strftime("%Y-%m-%d %H:%M:%S"), f"Bird-{predicted_class}"]
            log_queue_birds.append(log_entry)
            # Also log to CSV (optional)
            with open(csv_filename_birds, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(log_entry)
            print(f"🦜 Detected Bird Class: {predicted_class}")


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
    # Start the bird prediction in a background thread so it doesn't block the main thread.
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
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    return Response(requests.get("http://192.168.1.9:5001/video_feed").content, mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def get_logs():
    """
    Returns a JSON object with:
      - object_logs: FIFO list of object detection logs.
      - bird_logs: FIFO list of bird sound detection logs.
      - alert: a flag that is True if a human was detected since the last poll.
    After sending, the alert flag is reset.
    """
    global human_alert
    response = {
        "object_logs": list(log_queue_objects),
        "bird_logs": list(log_queue_birds),
        "alert": human_alert
    }
    # Reset the alert flag after sending
    human_alert = False
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
