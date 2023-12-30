# Import the required modules
import base64
import io
import cv2
from PIL import Image
import numpy as np
import keyboard
import os
from datetime import datetime
from selenium import webdriver

from selenium.webdriver.common.by import By

# Check if the captures folder exists and delete any existing files in it
isExist = os.path.exists("captures")

if isExist:
    dir = "captures"
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

else:
    os.mkdir("captures")


current_key = "1"
buffer = []


# Define a function to record the keyboard inputs
def keyboardCallBack(key: keyboard.KeyboardEvent):
    global current_key

    # If a key is pressed and not in the buffer, add it to the buffer
    if key.event_type == "down" and key.name not in buffer:
        buffer.append(key.name)

    # If a key is released, remove it from the buffer
    if key.event_type == "up":
        buffer.remove(key.name)

    # Sort the buffer and join the keys with spaces
    buffer.sort()
    current_key = " ".join(buffer)


# Hook the function to the keyboard events
keyboard.hook(callback=keyboardCallBack)

# Create a webdriver instance using Firefox
driver = webdriver.Firefox()
# Navigate to the Google Snake game website
driver.get("https://www.google.com/fbx?fbx=snake_arcade")

# Loop indefinitely
while True:
    # Find the canvas element on the webpage
    canvas = driver.find_element(By.CSS_SELECTOR, "canvas")

    # Get the base64 encoded image data from the canvas
    canvas_base64 = driver.execute_script(
        "return arguments[0].toDataURL('image/png').substring(21);", canvas
    )
    # Decode the base64 data to get the PNG image
    canvas_png = base64.b64decode(canvas_base64)

    # Convert the PNG image to a grayscale numpy array
    image = cv2.cvtColor(
        np.array(Image.open(io.BytesIO(canvas_png))), cv2.COLOR_BGR2RGB
    )

    # Save the image to the captures folder with the current timestamp and keyboard inputs as the file name
    if len(buffer) != 0:
        cv2.imwrite(
            "captures/"
            + str(datetime.now()).replace("-", "_").replace(":", "_").replace(" ", "_")
            + " "
            + current_key
            + ".png",
            image,
        )
    else:
        cv2.imwrite(
            "captures/"
            + str(datetime.now()).replace("-", "_").replace(":", "_").replace(" ", "_")
            + " n"
            + ".png",
            image,
        )
