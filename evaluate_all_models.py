import torch
import time
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import LeNet, SimpleCNN, ResNetClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 62

models_to_test = {
    "LeNet": LeNet(num_classes=NUM_CLASSES),
    "SimpleCNN": SimpleCNN(num_classes=NUM_CLASSES),
    "ResNetClassifier": ResNetClassifier(num_classes=NUM_CLASSES)
}

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for model_name, model in models_to_test.items():
    print(f"\nEvaluating {model_name}...")
    model_path = f"{model_name}_emnist.pth"
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found! Skipping.")
        continue

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    inference_times = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            start = time.time()
            outputs = model(images)
            end = time.time()

            inference_times.append(end - start)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_infer_time = (sum(inference_times) / len(inference_times)) * 1000  # ms
    model_size = os.path.getsize(model_path) / (1024 * 1024)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Inference Time (per batch): {avg_infer_time:.2f} ms")
    print(f"Model Size: {model_size:.2f} MB")
