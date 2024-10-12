import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import densenet121
from torch.utils.tensorboard import SummaryWriter


def main():


    # 数据预处理与增强
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 加载CIFAR-10数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # 定义DenseNet模型
    model = densenet121(num_classes=10)  # CIFAR-10有10个类别
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 添加Dropout层和Batch Normalization
    for layer in model.features:
        if isinstance(layer, nn.BatchNorm2d):
            layer.train()  # 确保在训练模式下
        elif isinstance(layer, nn.Linear):
            layer.dropout = nn.Dropout(0.5)  # 在全连接层之前添加Dropout

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2正则化

    # 训练模型
    num_epochs = 100
    os.makedirs(f'./logs/logs_2d_{num_epochs}', exist_ok=True)
    writer = SummaryWriter(f'logs/logs_2d_{num_epochs}')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}')
        writer.add_scalar('Loss/train', running_loss / len(trainloader), epoch)
        writer.add_scalar('Accuracy/test', 100 * correct / total, epoch)

if __name__ == '__main__':
    main()
