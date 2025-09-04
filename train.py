import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import config_script
# Простая нейросеть
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.fc(x)

# Загрузка данных MNIST
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config_script.batch_size, shuffle=True)

# Модель, оптимизатор, функция потерь
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Обучение (1 эпоха для примера)
model.train()
for epoch in range(config_script.epoch):
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete. Loss: {loss.item()}")

# Сохраняем модель
torch.save(model.state_dict(), "mnist_model.pth")
print("Модель сохранена в mnist_model.pth")


s3_client = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=temp_access_key,
        aws_secret_access_key=temp_secret_key,
        config=Config(signature_version="s3v4")
    )
s3_client.upload_file("example.txt", bucket_name, "example.txt")
print("Файл успешно загружен через boto3")