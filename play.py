import base64
import torch
import cv2
import keyboard
from PIL import Image
import numpy as np
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    Grayscale,
)
from torch import nn
from collections import deque
from torchvision.models.video import r3d_18
from selenium import webdriver
from selenium.webdriver.common.by import By

label_keys = {0: "", 1: "left", 2: "up", 3: "right", 4: "down"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = r3d_18(weights=None)
model.fc = nn.Linear(in_features=512, out_features=5, bias=True)

model.load_state_dict(torch.load("model_mc3.pth"))
model.to(device)
model.eval()

transformer = Compose(
    [
        Resize((84, 84), antialias=True),
        CenterCrop(84),
        ToTensor(),
        Normalize(mean=[-0.7138, -2.9883, 1.5832], std=[0.2253, 0.2192, 0.2149]),
    ]
)

# Create a webdriver instance using Firefox
driver = webdriver.Firefox()
# Navigate to the Google Snake game website
driver.get("https://www.google.com/fbx?fbx=snake_arcade")

frame_stack = deque(maxlen=4)

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

    frame_stack.append(transformer(image))
    input = torch.stack([*frame_stack], dim=1).to(device).squeeze().unsqueeze(0)

    if len(frame_stack) == 4:
        with torch.inference_mode():
            outputs = model(input).to(device)
            preds = torch.softmax(outputs, dim=1).argmax(dim=1)

            if preds.item() != 0:
                keyboard.press_and_release(label_keys[preds.item()])
