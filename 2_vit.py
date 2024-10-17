import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据增强和预处理
transform_train = transforms.Compose([
    transforms.Resize(224),                # 调整图像大小为 224x224
    transforms.RandomCrop(224, padding=4), # 随机裁剪 (可以保持在224)
    transforms.RandomHorizontalFlip(),     # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),                # 调整图像大小为 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# 使用 timm 加载预训练的 ViT 模型
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

# 学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 训练函数
def train(epoch):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 记录损失和准确率
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        total_correct += predicted.eq(labels).sum().item()


        print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 100:.3f}, Accuracy: {100.*total_correct/total:.2f}%')
        running_loss = 0.0

# 测试函数
def test():
    model.eval()
    total_correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            total_correct += predicted.eq(labels).sum().item()
    accuracy = 100. * total_correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# 训练循环
if __name__ == '__main__':
    # 加载CIFAR-10数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    num_epochs = 20
    best_acc = 0.0
    for epoch in range(num_epochs):
        train(epoch)
        test_acc = test()

        # 如果准确率提升，保存模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), './model/model_2_vit/model_2_vit.pth')

        # 调整学习率
        scheduler.step()

    print(f'Best Accuracy: {best_acc:.2f}%')
