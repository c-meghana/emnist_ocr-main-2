import cv2
import torch
import numpy as np
import pyttsx3
from segment import binarize, segment_characters, predict_class
from nets import LeNet_NClasses_batchnorm

# Load the image
img_path = "meg.jpeg"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_display = img_rgb.copy()

# Load model
model = LeNet_NClasses_batchnorm(nclasses=26)
model.load_state_dict(torch.load("lenet_caps_parameters.pth", map_location=torch.device("cpu")))
model.eval()

# Class labels
labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]


binary = binarize(img_rgb)
segments = segment_characters(binary)


segments.sort(key=lambda x: x[0][0])  # x[0] is bbox, x[0][0] is x position


output_text = ""
for bbox, subimg in segments:
    x1, y1, x2, y2 = bbox
    char_index, prob = predict_class(model, subimg)
    if 0 <= char_index < len(labels):
        predicted_char = labels[char_index]
        output_text += predicted_char
        # Drawing the bounding box and label
        cv2.rectangle(img_display, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_display, predicted_char, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


with open("ocr_output.txt", "w") as f:
    f.write(output_text)


output_img_path = "ocr_output_image.png"
img_bgr = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_img_path, img_bgr)


print("OCR Output Saved to ocr_output.txt and image file!")
print("Speaking out loud:", output_text)

engine = pyttsx3.init()
engine.say(output_text)
engine.runAndWait()
