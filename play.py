import torch
import cv2
import keyboard
from PIL import Image, ImageGrab
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize,  Grayscale
from tqdm import tqdm
from torch import nn
from collections import deque
from torchvision.models.video import mc3_18
from matplotlib import pyplot as plt

class SnakeModel(nn.Module):
    def __init__(self, in_channels=4, out_channels=32, mlp_size=256, n_actions=5, dropout = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp_size = mlp_size
        self.n_classes = n_actions
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4,2),
            nn.ReLU()
        )
        self.classifier =nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=out_channels*9*9, out_features=mlp_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=mlp_size, out_features=n_actions)
        )

    def forward(self,x):
        x = self.classifier(self.convblock(x).permute(0,2,3,1))
        return x

label_keys= {
    0: "",
    1 :"left",
    2: "up",
    3: "right",
    4: "down"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mc3_18(weights = None)
model.fc = nn.Linear(in_features=512, out_features=5, bias=True)

model.load_state_dict(torch.load("model_mc3.pth"))
model.to(device)
model.eval()

transformer = Compose([
    Resize((84,84), antialias=True),
    CenterCrop(84),
    ToTensor(),
    Normalize(mean =[ 0.8725,  1.8742, -0.2931], std =[0.3376, 0.3561, 0.3825] )
])

def generator():
    while(not keyboard.is_pressed("esc")):
      yield

frame_stack = deque(maxlen=4)

for _ in tqdm(generator()):
    image = ImageGrab.grab(bbox = (685,350,1235,840)) 

    frame_stack.append(transformer(image))
    input = torch.stack([*frame_stack],dim = 1).to(device).squeeze().unsqueeze(0)

    if len(frame_stack) == 4:
        with torch.inference_mode():
            outputs = model(input).to(device)
            preds = torch.softmax(outputs, dim=1).argmax(dim = 1)

            if preds.item() != 0:
                keyboard.press_and_release(label_keys[preds.item()])