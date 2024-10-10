import torchvision
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义卷积神经网络（确保这个类定义和之前训练模型时一致）
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层 + 批量归一化
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(256 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 加载模型结构
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载保存的整个模型
model = torch.load('./model_2/model_25.pth', weights_only=False)  # 直接加载整个模型

# 设置为评估模式
model.eval()

# 预处理 plane.png 图像，确保与训练数据的格式一致
image = Image.open('./data/plane.png')  # 打开图片
image = image.convert('RGB')  # 转换为RGB三通道

# 使用与训练数据相同的变换
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),  # 调整大小为32x32
    torchvision.transforms.ToTensor(),  # 转换为Tensor
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

image = transform(image)  # 应用变换
image = image.unsqueeze(0)  # 添加 batch 维度，变为 [1, 3, 32, 32]
image = image.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 进行推理
with torch.no_grad():
    output = model(image)
    _, predicted = output.max(1)

# CIFAR-10数据集的类标签
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 输出预测结果
print(f'Predicted class: {classes[predicted.item()]}')
