from vidstream import AudioSender, AudioReceiver, ScreenShareClient, CameraClient, StreamingServer
import tkinter as tk
import threading
import socket
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np


local_ip_address = socket.gethostbyname(socket.gethostname())
print("Local ip address: ", local_ip_address)

server = StreamingServer(local_ip_address, 9999)
receiver = AudioReceiver(local_ip_address, 8888)


# Hand Capture

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(max_num_hands=1)
classes = ['A','B','C','D','E']
model = load_model("src\keras_model.h5")
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


# Functions

def start_listening():
    t1 = threading.Thread(target=server.start_server)
    t2 = threading.Thread(target=receiver.start_server)
    t1.start()
    t2.start()

def start_camera_stream():
    camera_client = CameraClient(text_target_ip.get(1.0,'end-1c'), 7777)
    t3 = threading.Thread(target=camera_client.start_stream)
    t3.start()
    while True:
        success, img = cap.read()
        frameRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = hands.process(frameRGB)
        handsPoints = results.multi_hand_landmarks
        h, w, _ = img.shape
        if handsPoints != None:
            for hand in handsPoints:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in hand.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                cv2.rectangle(img, (x_min-50, y_min-50), (x_max+50, y_max+50), (0, 255, 0), 2)
                try:
                    imgCrop = img[y_min-50:y_max+50,x_min-50:x_max+50]
                    imgCrop = cv2.resize(imgCrop,(224,224))
                    imgArray = np.asarray(imgCrop)
                    normalized_image_array = (imgArray.astype(np.float32) / 127.0) - 1
                    data[0] = normalized_image_array
                    prediction = model.predict(data)
                    indexVal = np.argmax(prediction)
                    cv2.putText(img,classes[indexVal],(x_min-50,y_min-65),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),5)
                except:
                    continue
        cv2.imshow('Imagem',img)
        cv2.waitKey(1)

def start_screen_stream():
    screen_client = ScreenShareClient(text_target_ip.get(1.0,'end-1c'), 7777)
    t4 = threading.Thread(target=screen_client.start_stream)
    t4.start()

def start_audio_stream():
    audio_client = AudioSender(text_target_ip.get(1.0,'end-1c'), 6666)
    t5= threading.Thread(target=audio_client.start_stream)
    t5.start()


# GUI

window = tk.Tk()
window.title("Libraly v0.0.1 Alpha")
window.geometry("300x200")

label_target_ip = tk.Label(window, text="Target IP: ")
label_target_ip.pack()

text_target_ip = tk.Text(window, height=1)
text_target_ip.pack()

btn_listen = tk.Button(window, text="Start Listening", width=50, command=start_listening)
btn_listen.pack(anchor=tk.CENTER, expand=True)

btn_camera = tk.Button(window, text="Start Camera Stream", width=50, command=start_camera_stream)
btn_camera.pack(anchor=tk.CENTER, expand=True)

btn_screen = tk.Button(window, text="Start Screen Sharing", width=50, command=start_screen_stream)
btn_screen.pack(anchor=tk.CENTER, expand=True)

btn_audio = tk.Button(window, text="Start Audio Stream", width=50, command=start_audio_stream)
btn_audio.pack(anchor=tk.CENTER, expand=True)

ip_label = tk.Label(window, text=f"Your Ip: {local_ip_address}", bd=1, relief=tk.SUNKEN, anchor=tk.W)
ip_label.pack(side=tk.BOTTOM, fill=tk.X)


window.mainloop()