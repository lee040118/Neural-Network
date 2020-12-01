import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = datasets.CIFAR100('CIFAR100_data',download=False, train=True, transform=transform)
val_data = datasets.CIFAR100('CIFAR100_data',download=False, train=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=4, shuffle=True) #4@ 3*32*32
val_loader = DataLoader(val_data, batch_size=4, shuffle=True)

model = models.resnet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

epochs = 5
total_step = len(train_loader)

for epoch in range(epochs):
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        # get the inputs
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if (i + 1) % 2000 == 0:
            with torch.no_grad():
                val_loss = 0.0
                for j, val in enumerate(val_loader):
                    val_x, val_label = val[0].to(device), val[1].to(device)
                    val_output = model(val_x)
                    v_loss = criterion(val_output, val_label)
                    val_loss += v_loss
            print('Epoch [{}/{}] | Step [{}/{}] | Loss: {:.4f} | val loss: {:.4f}'
                  .format(epoch + 1, epochs, i + 1, total_step, running_loss / 2000, val_loss / len(val_loader)))
            running_loss = 0.0