import datetime
import sys

import torch
import torchvision
from torch import nn, flatten
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
train_data = torchvision.datasets.MNIST(root="./dataset", train=True,
                                        transform=ToTensor(), download=True)
test_data = torchvision.datasets.MNIST(root="./dataset", train=False,
                                       transform=ToTensor(), download=True)
batch_size = 60000
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, drop_last=True)


class Logger(object):
    def __init__(self, filename='bgd_model_log.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# 将控制台输出保存到log.txt中
sys.stdout = Logger(stream=sys.stdout)


# 小批量梯度下降
class BGDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear1 = nn.Linear(in_features=784, out_features=256)
        self.Linear2 = nn.Linear(in_features=256, out_features=10)
        self.x = torch.zeros(1)
        self.y = torch.zeros(1)
        self.y_ = torch.zeros(1)
        self.z = torch.zeros(1)
        self.z_ = torch.zeros(1)

    def forward(self, model_input):
        x = flatten(model_input)
        self.x = x
        x = self.Linear1(x)
        self.y = x
        x = torch.sigmoid(x)
        self.y_ = x
        x = self.Linear2(x)
        self.z = x
        x = torch.sigmoid(x)
        self.z_ = x
        return x


print(f'{datetime.datetime.now()}')
criterion = nn.CrossEntropyLoss()
epochs = 10
lr = 1
print(f'Parameters: Learning rate = {lr}, Epoch = {epochs}')
# MNIST集训练用cpu反而更快，一般用gpu
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
with torch.no_grad():
    print('Training Starting...')
    gd_model = BGDModel().to(device)
    for epoch in range(epochs):
        time1 = datetime.datetime.now()
        total_loss = 0
        for data in train_dataloader:
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            sum_dl_dw1 = sum_dl_db1 = sum_dl_dw2 = sum_dl_db2 = 0
            for i in range(batch_size):
                img = imgs[i][0]
                label = labels[i]
                output = gd_model(img)
                loss = criterion(output, label)
                total_loss += loss.item()
                gd_model.forward(img)
                # 开始计算梯度
                dl_dz_ = torch.exp(gd_model.z_).to(device) / \
                          torch.exp(gd_model.z_).sum().to(device) - \
                          torch.eye(10)[label].to(device)
                dz__dz = torch.sigmoid(gd_model.z.data) * (
                        1 - torch.sigmoid(gd_model.z.data))
                dl_dz = dl_dz_ * dz__dz
                delta2 = dl_dz
                dl_dw2 = delta2.unsqueeze(1) * gd_model.y_.data.unsqueeze(0)
                dl_db2 = delta2
                dy__dy = torch.sigmoid(gd_model.y.data) * (
                        1 - torch.sigmoid(gd_model.y.data))
                delta1 = torch.matmul(delta2, gd_model.Linear2.weight.data) * dy__dy
                dl_dw1 = delta1.unsqueeze(1) * gd_model.x.data.unsqueeze(0)
                dl_db1 = delta1
                sum_dl_dw1 += dl_dw1
                sum_dl_db1 += dl_db1
                sum_dl_dw2 += dl_dw2
                sum_dl_db2 += dl_db2
            gd_model.Linear1.weight -= lr * sum_dl_dw1 / batch_size
            gd_model.Linear1.bias -= lr * sum_dl_db1 / batch_size
            gd_model.Linear2.weight -= lr * sum_dl_dw2 / batch_size
            gd_model.Linear2.bias -= lr * sum_dl_db2 / batch_size
        num_correct = 0
        num_all = 0
        # 测试
        for data in test_data:
            img, label = data
            img = img.to(device)
            label = torch.tensor(label).to(device)
            output = gd_model(img).to(device)
            _, predicted = torch.max(output.data, dim=0)
            num_correct += predicted.item() == label.item()
            num_all += 1
        # 输出
        time2 = datetime.datetime.now()
        print(f'Epoch {epoch + 1:d}, Average Loss: {total_loss / num_all:.4f}'
              f', Accuracy: {100 * num_correct / num_all:.2f}%, '
              f'Time used: {time2 - time1}')
