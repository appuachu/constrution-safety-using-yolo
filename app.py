from flask import Flask, render_template, request, redirect, url_for, Response
from flask_socketio import SocketIO
import cv2
import os
import math
import requests
from ultralytics import YOLO
import cvzone
import time
import subprocess
import winsound  # For beep sound on Windows systems
import pygame  # Import pygame for audio playback

app = Flask(__name__)
socketio = SocketIO(app)

# Simple authentication (use a proper security measure in production)
USERNAME = 'admin'
PASSWORD = 'password'

# Load YOLO model
model = YOLO("cons_safety.pt")
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

# Global variables
selected_camera = 0
cap = None
is_authenticated = False

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = ''
TELEGRAM_CHAT_ID = ''

# To track last alert sent time for each class
last_alert_time = {}
alert_interval = 15  # seconds

# Ngrok configuration
NGROK_API_URL = 'http://localhost:4040/api/tunnels'

# Initialize Pygame mixer for sound playback
pygame.mixer.init()

# Load your MP3 sound file (make sure the path is correct)
non_compliance_sound = 'alert1.mp3'  # Replace with your actual file path
pygame.mixer.music.load(non_compliance_sound)

def start_ngrok():
    """Starts ngrok tunnel and retrieves the public URL."""
    ngrok_path = r''  # Adjust path to ngrok
    ngrok_process = subprocess.Popen([ngrok_path, 'http', '5000'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3)  # Allow ngrok time to initialize

    # Get the public URL from the ngrok API
    try:
        response = requests.get(NGROK_API_URL)
        if response.status_code == 200:
            tunnels = response.json()['tunnels']
            for tunnel in tunnels:
                if tunnel['proto'] == 'https':
                    return tunnel['public_url']
        else:
            print("Error: Could not get the ngrok tunnel information.")
    except Exception as e:
        print(f"Exception occurred while retrieving ngrok URL: {str(e)}")

    return None

def send_ngrok_link():
    """Send the ngrok live camera and detection feed links to Telegram."""
    ngrok_url = start_ngrok()
    if ngrok_url:
        # Links for both raw feed and detection-enabled feed
        live_feed_url = f"{ngrok_url}/raw_feed"
        video_feed_url = f"{ngrok_url}/video_feed"

        # Message with both feeds
        message = f"Live Camera Feed (Raw): {live_feed_url}\nLive Video Feed (With Detection): {video_feed_url}"

        # Send to Telegram
        response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": message}
        )
        print(f"Live camera links sent to Telegram: {response.json()}")  # For debugging
    else:
        print("Error: Could not start ngrok or retrieve the public URL.")

def send_alert(image, class_names):
    """Send alert to Telegram with the person's image for all detected violations."""
    global last_alert_time

    # Check if alerts need to be sent based on time interval
    current_time = time.time()
    message = "Alert: A person is not wearing the required safety gear:\n"
    should_send = False

    for class_name in class_names:
        if class_name not in last_alert_time or (current_time - last_alert_time[class_name]) >= alert_interval:
            message += f"- {class_name}\n"
            last_alert_time[class_name] = current_time
            should_send = True

    if should_send:
        temp_file_path = 'temp_alert_image.jpg'
        cv2.imwrite(temp_file_path, image)

        with open(temp_file_path, 'rb') as photo:
            response = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": message},
                files={"photo": photo}
            )

        print(response.json())  # Print response for debugging
        os.remove(temp_file_path)

def gen_frames():
    """Generates processed frames with YOLO detection."""
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(selected_camera)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

    non_compliance_detected = False  # Track if non-compliance is detected

    while True:
        success, frame = cap.read()  # Read the camera feed
        if not success:
            break
        else:
            # Process the frame (detection)
            img_resized = cv2.resize(frame, (640, 480))
            results = model(img_resized, stream=True)

            detected_classes = []  # List to track detected non-compliance classes

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    currentClass = classNames[cls]

                    # Set bounding box colors based on compliance/non-compliance
                    if conf > 0.5:
                        if currentClass.startswith('NO-'):  # Non-compliance
                            myColor = (0, 0, 255)  # Red for non-compliance
                            detected_classes.append(currentClass)  # Add to detected classes
                            non_compliance_detected = True  # Set flag for non-compliance

                        elif currentClass in ['Hardhat', 'Safety Vest', 'Mask']:
                            myColor = (0, 255, 0)  # Green for compliance
                            non_compliance_detected = False  # Reset flag for compliance

                        else:
                            myColor = (255, 0, 0)  # Blue for other objects

                        cvzone.putTextRect(frame, f'{classNames[cls]} {conf}',
                                           (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                           colorT=(255, 255, 255), colorR=myColor, offset=5)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), myColor, 3)

            # If any non-compliance classes were detected, send alert
            if detected_classes:
                send_alert(frame, detected_classes)

            # Play sound if non-compliance is detected
            if non_compliance_detected:
                if not pygame.mixer.music.get_busy():  # If sound is not already playing
                    pygame.mixer.music.play(-1)  # Play sound indefinitely
            else:
                pygame.mixer.music.stop()  # Stop sound if compliance is detected

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Display frame in JPEG format

def gen_raw_frames():
    """Generates raw frames without any processing."""
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(selected_camera)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
    while True:
        success, frame = cap.read()  # Read the camera feed
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Send the raw frame

# Route for raw live feed without detection
@app.route('/raw_feed')
def raw_feed():
    return Response(gen_raw_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for processed live feed with YOLO detection
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    global is_authenticated
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == USERNAME and password == PASSWORD:
            is_authenticated = True
            send_ngrok_link()  # Send Ngrok link after successful login
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid username or password.")
    return render_template('login.html')


@app.route('/home')
def home():
    global is_authenticated
    if not is_authenticated:
        return redirect(url_for('index'))
    return render_template('home.html')


@app.route('/select_camera', methods=['POST'])
def select_camera():
    global selected_camera, cap
    selected_camera = int(request.form['camera'])
    cap = cv2.VideoCapture(selected_camera)
    return redirect(url_for('video_feed'))


if __name__ == "__main__":
    send_ngrok_link()  # Start ngrok automatically before the server runs
    app.run(debug=True, use_reloader=False)
