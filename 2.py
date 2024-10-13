import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # 导入 tqdm
from torch.utils.tensorboard import SummaryWriter

# 数据增强与归一化
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.RandomHorizontalFlip(),     # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 加载数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# 定义卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层 + 批量归一化
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 第1层
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 第2层
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # 第3层
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)  # 第4层
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)  # 池化层

        # 全连接层
        self.fc1 = nn.Linear(256 * 2 * 2, 1024)  # 通过动态计算后，调整这里的展平维度
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

        # Dropout 防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 卷积 + 激活 + 批量归一化 + 池化
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # 打印张量形状以确认展平维度
        # print(x.shape)  # 添加调试代码来查看张量的形状
        x = x.view(x.size(0), -1)  # 动态展平

        # 全连接层 + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 实例化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 训练模型
def train_model(model, trainloader, device, epoch):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    # 用 tqdm 包装 trainloader，显示进度条
    progress_bar = tqdm(trainloader, desc="Training", leave=False)

    for i, data in enumerate(progress_bar, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # 清零梯度

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        # 更新进度条的描述
        progress_bar.set_postfix(loss=running_loss / (i + 1), accuracy=100 * correct / total)
        writer.add_scalar('Loss/train', running_loss / len(trainloader), epoch)

# 测试模型
best_accuracy = 0.0  # 用于记录最佳准确率

def test_model(model, testloader, device, epoch):
    global best_accuracy
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        # 用 tqdm 包装 testloader，显示进度条
        progress_bar = tqdm(testloader, desc="Testing", leave=False)
        for data in progress_bar:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    writer.add_scalar('Accuracy/test', accuracy, epoch)
    print(f'Accuracy on the test set: {accuracy:.2f}%')

    # 如果当前的精度超过之前记录的最佳精度，保存模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'model/model_2/model_best.pth')  # 保存最佳模型
        with open('model/model_2/log.txt', 'w') as f:
            f.write(f'Epoch {epoch+1}: Best test accuracy: {best_accuracy:.2f}%\n')

# 主程序
if __name__ == '__main__':
    start_time = time.time()
    num_epochs = 100
    os.makedirs(f'logs/logs_2_{num_epochs}', exist_ok=True)
    os.makedirs('model/model_2', exist_ok=True)  # 创建模型保存的目录
    writer = SummaryWriter(f'logs/logs_2_{num_epochs}')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_model(model, trainloader, device, epoch)
        scheduler.step()  # 调整学习率
        test_model(model, testloader, device, epoch)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f'Training time: {elapsed_time:.2f} seconds')  # 打印运行时间
