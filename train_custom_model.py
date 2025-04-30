import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import LeNet, SimpleCNN, ResNetClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001
NUM_CLASSES = 62

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# The model has to be chosen here:
#Uncomment and train each model and then evaluate
#model = LeNet(num_classes=NUM_CLASSES)
#model = SimpleCNN(num_classes=NUM_CLASSES)
model = ResNetClassifier(num_classes=NUM_CLASSES)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# TRAINING LOOP
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# SAVE MODEL
model_path = f"{model.__class__.__name__}_emnist.pth"
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to: {model_path}")