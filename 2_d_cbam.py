import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import densenet121
from torch.utils.tensorboard import SummaryWriter
# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # 全局池化操作：AdaptiveAvgPool2d会将输入特征图的大小调整为1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 全连接层将通道数减少到in_channels // reduction_ratio
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局池化
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# CBAM模块
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)  # 通道注意力
        self.sa = SpatialAttention(kernel_size)  # 空间注意力

    def forward(self, x):
        # 应用通道注意力
        x = x * self.ca(x)
        # 应用空间注意力
        x = x * self.sa(x)
        return x

# 自定义带CBAM的DenseNet模型
class DenseNetWithCBAM(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNetWithCBAM, self).__init__()
        # 更新为使用weights参数，不使用预训练权重
        self.densenet = densenet121(weights=None)
        self.cbam_modules = nn.ModuleList()

        # 遍历DenseNet的特征部分，插入CBAM模块
        for name, module in self.densenet.features.named_children():
            if isinstance(module, nn.Conv2d):
                cbam_module = CBAM(module.out_channels)  # 每层卷积后添加CBAM
                self.cbam_modules.append(cbam_module)
            else:
                self.cbam_modules.append(None)  # 对于非卷积层，不添加CBAM

        # 修改最后的分类层
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)


    def forward(self, x):
        for idx, layer in enumerate(self.densenet.features):
            x = layer(x)
            if self.cbam_modules[idx] is not None:  # 如果该层有CBAM模块
                x = self.cbam_modules[idx](x)
        x = nn.functional.relu(x, inplace=True)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.densenet.classifier(x)
        return x


def main():
    accuracy_best = 0.0

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

    # 定义DenseNet模型并加入CBAM
    model = DenseNetWithCBAM(num_classes=10)  # CIFAR-10有10个类别
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2正则化

    # 训练模型
    num_epochs = 400
    os.makedirs(f'./logs/logs_2d_cbam_{num_epochs}', exist_ok=True)
    writer = SummaryWriter(f'logs/logs_2d_cbam_{num_epochs}')

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

        # 评估模型
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

        # 打印当前精度和损失
        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test set: {accuracy:.2f}%')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}')

        # 记录到TensorBoard
        writer.add_scalar('Loss/train', running_loss / len(trainloader), epoch)
        writer.add_scalar('Accuracy/test', accuracy, epoch)

        # 保存最好的模型
        if accuracy > accuracy_best:
            accuracy_best = accuracy
            torch.save(model.state_dict(), f'./model/model_2d_cbam/model_best.pth')
            with open(f'./model/model_2d_cbam/log.txt', 'w') as f:
                f.write(f'Accuracy of the model on the test set: {accuracy:.2f}%\n')
                f.write(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}')


if __name__ == '__main__':
    main()
