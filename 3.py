import os
# 在运行 TensorFlow 之前设置环境变量
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report
import jieba
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool
from torch.utils.tensorboard import SummaryWriter


# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据加载
def load_data(file_path):
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            label, text = line.strip().split('\t', 1)  # 假设标签和文本用tab分割
            texts.append(text)
            labels.append(label)
    return texts, labels

# 定义分词函数
def tokenize(text):
    return list(jieba.cut(text))

# 使用多进程进行分词
def parallel_tokenize(texts):
    with Pool() as pool:
        return list(tqdm(pool.imap(tokenize, texts), total=len(texts)))

# 定义主函数
def main():

    # 加载数据
    train_texts, train_labels = load_data('./data/THUCNews/cnews.train.txt')
    test_texts, test_labels = load_data('./data/THUCNews/cnews.test.txt')

    # 标签转化为数字
    label2idx = {label: idx for idx, label in enumerate(set(train_labels))}
    train_labels = [label2idx[label] for label in train_labels]
    test_labels = [label2idx[label] for label in test_labels]

    # 分词
    train_tokens = parallel_tokenize(train_texts)
    test_tokens = parallel_tokenize(test_texts)

    # 构建词汇表
    all_tokens = [token for tokens in train_tokens for token in tokens]
    vocab = [word for word, count in Counter(all_tokens).items() if count >= 2]  # 过滤出现次数少于2次的词
    word2idx = {word: idx + 2 for idx, word in enumerate(vocab)}  # 0用于padding, 1用于OOV

    # 序列化文本
    def encode_tokens(tokens, word2idx, max_len):
        return [word2idx.get(token, 1) for token in tokens[:max_len]] + [0] * max(0, max_len - len(tokens))

    max_len = 100  # 每个文本的最大长度
    X_train = [encode_tokens(tokens, word2idx, max_len) for tokens in train_tokens]
    X_test = [encode_tokens(tokens, word2idx, max_len) for tokens in test_tokens]

    # 转换为PyTorch的Tensor
    X_train = torch.tensor(X_train, dtype=torch.long).to(device)
    y_train = torch.tensor(train_labels, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test, dtype=torch.long).to(device)
    y_test = torch.tensor(test_labels, dtype=torch.long).to(device)

    # 创建Dataset和DataLoader
    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts[idx], self.labels[idx]

    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 定义LSTM模型
    class LSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embed_size, hidden_size, num_classes, num_layers=2, dropout=0.5):
            super(LSTMClassifier, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirectional LSTM

        def forward(self, x):
            x = self.embedding(x)
            lstm_out, _ = self.lstm(x)
            out = lstm_out[:, -1, :]  # 取最后一个时间步的输出
            out = self.fc(out)
            return out

    # 模型参数
    vocab_size = len(word2idx) + 2  # 词汇表大小+2（padding和OOV）
    embed_size = 128
    hidden_size = 128
    num_classes = len(label2idx)

    # 实例化模型并移动到设备
    model = LSTMClassifier(vocab_size, embed_size, hidden_size, num_classes).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 10000
    #如果不存在文件夹则创建
    os.makedirs(f'./logs/logs_3_{num_epochs}', exist_ok=True)
    writer = SummaryWriter(f'./logs/logs_3_{num_epochs}')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        torch.save(model.state_dict(), f'./model_3/model_epoch_{epoch + 1}.pth')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')
        writer.add_scalar('Loss/train', total_loss / len(train_loader), epoch)


        # 评估模型
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for texts, labels in test_loader:
                outputs = model(texts)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算准确率
        accuracy = accuracy_score(all_labels, all_preds)
        writer.add_scalar('Accuracy/test', accuracy, epoch)
        print(f'测试集准确率: {accuracy:.4f}')
        print(classification_report(all_labels, all_preds, target_names=label2idx.keys()))

if __name__ == '__main__':
    main()
