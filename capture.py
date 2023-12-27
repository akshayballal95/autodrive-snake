import cv2
from PIL import ImageGrab
import numpy as np
import keyboard
import os
from datetime import datetime

current_key = "1"
buffer = []


isExist = os.path.exists("snake_captures")

if isExist:
    dir = 'snake_captures'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

else:

    os.mkdir("snake_captures")

def keyboardCallBack(key: keyboard.KeyboardEvent):
    global current_key

    if key.event_type == "down" and key.name not in buffer:
        buffer.append(key.name)
    
    if key.event_type == "up":
        buffer.remove(key.name)

    buffer.sort()
    current_key = " ".join(buffer)

keyboard.hook(callback=keyboardCallBack)
i=0

while(not keyboard.is_pressed("c")):
    
    ###uncomment for dino game
    # image = cv2.cvtColor(np.array(ImageGrab.grab(bbox = (620,220,1280,360))), cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(np.array(ImageGrab.grab(bbox = (685,350,1235,840))), cv2.COLOR_RGB2BGR)
    if len(buffer)!=0:
        cv2.imwrite("snake_captures/" + str(datetime.now()).replace("-","_").replace(":","_").replace(" ", "_")+" "+ current_key +".png", image)
    else:
         cv2.imwrite("snake_captures/" + str(datetime.now()).replace("-","_").replace(":","_").replace(" ", "_") + " n" +".png", image)
    i= i+1
