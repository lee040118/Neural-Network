import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 각 체널별 이미지 영역 평균값과 표준편차

train_data = datasets.CIFAR10('CIFAR10_data',download=True, train=True, transform=transform)
val_data = datasets.CIFAR10('CIFAR10_data',download=True, train=False, transform=transform)

"""class =('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck') -> 10개"""
train_loader = DataLoader(train_data, batch_size=4, shuffle=True) #4@ 3*32*32
val_loader = DataLoader(val_data, batch_size=4, shuffle=True)
# for x,y in train_loader:
#     print(x.size())
class CNN_Filter(nn.Module):
    def __init__(self):
        super(CNN_Filter,self).__init__()
        self.con1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1) #4*6*32*32
        #activate function Relu
        self.pool = nn.MaxPool2d(2) # 4*6*16*16
        self.con2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1) #4*16*16
        self.con3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) #4*8*8

        self.ful1 = nn.Linear(32*4*4, 120)
        self.ful2 = nn.Linear(120, 84)
        self.dropout1 = nn.Dropout(0.5)
        self.ful3 = nn.Linear(84, 10)

    def forward(self, out):
        out = self.pool(f.relu(self.con1(out)))
        out = self.pool(f.relu(self.con2(out)))
        out = self.pool(f.relu(self.con3(out)))
        #out = self.pool(out)
        dim=1
        for i in out.size()[1:]:
            dim *= i
        out = out.view(-1, dim)

        out = f.relu(self.ful1(out))
        out = f.relu(self.ful2(out))
        out = self.dropout1(out)
        out = self.ful3(out)
        return out

model = CNN_Filter().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1E-3, momentum=0.9)

num_epoch = 10
total_step = len(train_loader)
for epoch in range(num_epoch):
    trn_loss = 0.0
    for i, data in enumerate(train_loader):
        x, label = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()
        trn_loss+=loss.item()

        if (i + 1) % 2000 == 0:
            with torch.no_grad():  # very very very very important!!!
                val_loss = 0.0
                for j, val in enumerate(val_loader):
                    val_x, val_label = val[0].to(device), val[1].to(device)
                    val_output = model(val_x)
                    v_loss = criterion(val_output, val_label)
                    val_loss += v_loss
            print('Epoch [{}/{}] | Step [{}/{}] | Loss: {:.4f} | val loss: {:.4f}'
                  .format(epoch + 1, num_epoch, i + 1, total_step, trn_loss/2000,val_loss / len(val_loader)))

            trn_loss=0.0