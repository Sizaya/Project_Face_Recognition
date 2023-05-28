import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim


# Визначення трансформації, яка буде застосована до кожного зображення
transform = transforms.Compose([
    transforms.Resize((64, 64)),                            # Змінити розмір зображення до стандартного
    transforms.ToTensor(),                                  # Перетворити зображення в тензор
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормалізація пікселів зображення
])

# Завантаження набору даних
dataset = torchvision.datasets.ImageFolder(root='./data/presidents', transform=transform)

# Визначення завантажувача даних для завантеження пакетами
data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

# Визначення моделі нейронної мережі
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Створити екземпляр моделі нейронної мережі
net = Net()

# Визначити функцію втрат і оптимізатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Навчання нейронної мережі
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Кінець тренування')

# Збереження навченої моделі
PATH = './president_net.pth'
torch.save(net.state_dict(), PATH)
