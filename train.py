# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from data_loader import get_data_loaders

print('Training started....')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader, class_names = get_data_loaders('fer2013', batch_size=64)

model = resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adapt for grayscale

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)  # 7 emotion classes
model = model.to(device)
# Example class weights (adjust based on dataset)
weights = torch.tensor([1.5, 3.0, 2.0, 1.0, 1.0, 1.2, 1.5])
criterion = nn.CrossEntropyLoss(weight=weights.to(device))

optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, test_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}, Loss: {running_loss:.3f}, Train Accuracy: {train_acc:.2f}%")

        evaluate_model(model, test_loader)

def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


train_model(model, train_loader, test_loader, epochs=15)
torch.save(model.state_dict(), "resnet_emotion.pt")

print("Training successfully Done....")
