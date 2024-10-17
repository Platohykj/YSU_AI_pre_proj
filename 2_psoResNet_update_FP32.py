import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# 动态定义ResNet类，支持粒子群优化得到的超参数
class BasicBlockDynamic(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, downsample=None):
        super(BasicBlockDynamic, self).__init__()
        padding = kernel_size // 2  # 保证卷积后特征图尺寸不变
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

        # 如果 stride 不等于 1，或者输入和输出通道不一致，使用 1x1 卷积下采样
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # 对 identity 进行下采样

        out += identity  # 将 out 和 identity 相加
        out = self.relu(out)

        return out

class ResNetDynamic(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNetDynamic, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)  # stride = 2 进行下采样
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, kernel_size=3, stride=stride, downsample=downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, kernel_size=3))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# PSO算法中的粒子类，包含学习率、动量、层数等
class Particle:
    def __init__(self):
        # 粒子维度：学习率、动量、层数
        self.position = np.array([
            np.random.uniform(0.0001, 0.1),  # 学习率
            np.random.uniform(0.0001, 0.1),  # 动量
            np.random.uniform(4, 20)         # 层数
        ])
        self.velocity = np.random.rand(3) * 0.1  # 初始速度
        self.best_position = self.position.copy()
        self.best_fitness = 0

def fitness_function(lr, momentum, layers):
    layers = int(layers)
    model = ResNetDynamic(BasicBlockDynamic, [layers, layers, layers]).to('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    model.train()
    running_loss = 0.0
    for epoch in range(5):
        print('fitness_epoch: ', epoch)
        # 适应度评估只训练1个epoch
        for inputs, labels in trainloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    return 1 / (running_loss / len(trainloader))  # 适应度为损失的倒数

# PSO算法
def pso(num_particles=20, num_iterations=10):
    particles = [Particle() for _ in range(num_particles)]
    global_best_position = None
    global_best_fitness = float('-inf')

    for _ in range(num_iterations):
        print('iteration: ', _)
        for particle in particles:
            # 计算适应度，加入新的超参数
            fitness = fitness_function(particle.position[0], particle.position[1], particle.position[2])

            # 更新粒子最佳位置
            if fitness > particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()

            # 更新全局最佳位置
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.best_position.copy()
            print(f"particlenumber: {particle}")

        # 更新粒子位置和速度
        for particle in particles:
            w = 0.5  # 惯性权重
            c1 = 1.0  # 个人学习因子
            c2 = 0.4  # Cg（社会学习因子）
            r1 = np.random.rand(3)
            r2 = np.random.rand(3)

            # 更新速度
            particle.velocity = (w * particle.velocity +
                                 c1 * r1 * (particle.best_position - particle.position) +
                                 c2 * r2 * (global_best_position - particle.position))

            # 更新位置
            particle.position += particle.velocity

            # 限制学习率、动量、层数范围
            particle.position[0:2] = np.clip(particle.position[0:2], 0.0001, 0.1)  # 学习率、动量
            particle.position[2] = np.clip(particle.position[2], 4, 20)
            print(f"particlenumber: {particle}")

    return global_best_position, global_best_fitness

# 模型测试函数
def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total  # 返回准确率

if __name__ == '__main__':
    # 运行PSO进行超参数优化
    best_position, best_fitness = pso(num_particles=20, num_iterations=10)
    best_lr, best_momentum, best_layers = best_position

    print(f'Best Learning Rate: {best_lr}, Best Momentum: {best_momentum}, Best Layers: {best_layers}, Best Fitness: {best_fitness}')

    # 使用最佳超参数训练最终模型
    best_layers = int(best_layers)

    model = ResNetDynamic(BasicBlockDynamic, [best_layers, best_layers, best_layers]).to('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=best_lr, momentum=best_momentum)
    num_epochs = 1000
    os.makedirs(f'logs/logs_psoResNet_{num_epochs}_num_particles=20,num_iterations=10', exist_ok=True)
    writer = SummaryWriter(f'logs/logs_psoResNet_{num_epochs}_num_particles=20,num_iterations=10')

    accuracy_best = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 测试集准确率
        accuracy = evaluate_model(model, testloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%')
        writer.add_scalar('Loss/train', running_loss / len(trainloader), epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)

        if accuracy > accuracy_best:
            accuracy_best = accuracy
            torch.save(model.state_dict(), f'./model/model_psoResNet_best.pth')
            with open(f'./model/model_psoResNet_log.txt', 'w') as f:
                f.write(f'Best Learning Rate: {best_lr}, Best Momentum: {best_momentum}, Best Layers: {best_layers}, Best Fitness: {best_fitness}\n')
                f.write(f'Accuracy of the model on the test set: {accuracy:.2f}%\n')
                f.write(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}')
