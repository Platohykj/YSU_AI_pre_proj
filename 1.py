import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载波士顿房价数据集
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# 数据处理：提取特征和目标
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
print("Data shape: {}".format(data.shape))

# 特征标准化
X = MinMaxScaler().fit_transform(data)

# 多项式特征构造
X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
print("X shape: {}".format(X.shape))

# 划分数据集：前300条数据作为训练集，后206条数据作为测试集
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=206, random_state=1)

# 转换为 PyTorch 张量并移动到设备
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# 定义BP神经网络模型
class BPNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(BPNeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)  # 加入BatchNorm
        self.hidden2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)  # 加入BatchNorm
        self.output = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.5)  # 加入Dropout
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.bn1(self.hidden1(x)))  # 隐藏层1 + BatchNorm + ReLU
        x = self.dropout(x)  # Dropout
        x = self.activation(self.bn2(self.hidden2(x)))  # 隐藏层2 + BatchNorm + ReLU
        x = self.dropout(x)  # Dropout
        x = self.output(x)
        return x

# 实例化模型并移动到设备
input_size = X_train.shape[1]
model = BPNeuralNetwork(input_size).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 调低学习率，加入权重衰减

# 提前停止的相关设置
best_loss = float('inf')
patience = 1000  # 如果损失在1000个epoch中没有下降，则停止训练
trigger_times = 0

# 训练模型
num_epochs = 9000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()  # 清零梯度
    outputs = model(X_train_tensor).squeeze()  # 进行前向传播
    loss = criterion(outputs, y_train_tensor)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重

    # 每100个epoch打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 提前停止逻辑：检查验证集误差是否有改进
    model.eval()
    with torch.no_grad():
        y_pred_val = model(X_test_tensor).squeeze()
        val_loss = criterion(y_pred_val, y_test_tensor)

    if val_loss < best_loss:
        best_loss = val_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

# 进行预测
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor).squeeze()  # 进行预测

# 转换为numpy数组
y_pred = y_pred_tensor.cpu().numpy()

# 计算预测误差（均方误差）
mse = mean_squared_error(y_test, y_pred)
print(f"均方误差（Mean Squared Error）: {mse}")
