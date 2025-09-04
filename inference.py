import torch
from torchvision import datasets, transforms

from train import SimpleNN  # используем определение модели из train.py

# Загружаем модель
model = SimpleNN()
model.load_state_dict(torch.load("mnist_model.pth", map_location="cpu"))
model.eval()

# Берем пример данных
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

image, label = test_dataset[0]
with torch.no_grad():
    output = model(image.unsqueeze(0))
    pred = output.argmax(dim=1, keepdim=True)

print(f"Истинная метка: {label}, предсказание: {pred.item()}")
