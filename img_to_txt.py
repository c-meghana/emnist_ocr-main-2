import cv2
import torch
import numpy as np
from segment import binarize, segment_characters, predict_class
from nets import LeNet_NClasses_batchnorm

# Load the image
img_path = "test_img_2.jpeg"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Load model
model = LeNet_NClasses_batchnorm(nclasses=26)
model.load_state_dict(torch.load("lenet_caps_parameters.pth", map_location=torch.device("cpu")))
model.eval()

# Class labels
labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Step 1: Binarize and Segment Characters
binary = binarize(img_rgb)
segments = segment_characters(binary)

# Step 2: Sort Characters Left-to-Right by bbox x-coordinate
segments.sort(key=lambda x: x[0][0])  # x[0] is bbox, x[0][0] is x position

# Step 3: Predict Each Character
output_text = ""
for bbox, subimg in segments:
    char_index, prob = predict_class(model, subimg)
    if 0 <= char_index < len(labels):
        output_text += labels[char_index]

# Step 4: Save to Text File
with open("ocr_output.txt", "w") as f:
    f.write(output_text)

print("✅ OCR Output Saved to ocr_output.txt:")
print(output_text)
