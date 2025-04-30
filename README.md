
# Assistive OCR Reader using LeNet and Text-to-Speech

This project is an offline Optical Character Recognition (OCR) system designed to assist visually impaired users in reading handwritten characters. The system captures text from an image, processes it using image segmentation and a trained LeNet CNN model, and reads the text aloud using a Python-based text-to-speech engine.

After training and evaluating three CNN models — LeNet, SimpleCNN, and ResNet18 — on the EMNIST dataset, **LeNet** was selected as the final model for deployment due to its fast training, low inference time, and small size, making it ideal for lightweight applications.

---
## How to Run the Final OCR + Speech System

### 1. Install all the required dependencies
### 2. Run the models.py file first to define all the 3 CNN models used
### 3. Train the models by executing the file train_custom_model.py
### 4. While training the models, execute it one by one
### 5. After executing each model, run the evaluate_all_models.py file individually for each model
### 6. The accuracies of the models are displayed in the terminal
### 7. LeNet is chosen for the project because of it's light weight structure
### 8. Finally run the text_to_speech.py file where the LeNet model is used to read the input given meg.jpg, store the read text as output in both .png and .txt files and once ran, the output is also read out loud.




