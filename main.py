import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# =============== 数据预处理 ===============

def load_and_preprocess_data():
    # 加载数据
    incident_df = pd.read_excel('data/incident.xlsx')
    before_df = pd.read_excel('data/common_before.xlsx')
    after_df = pd.read_excel('data/common_after.xlsx')

    # 提取特征列
    data_cols = [col for col in incident_df.columns if col.startswith('Data')]
    time_features = ['Time_tag', 'week', 'Percentage', 'Median']

    # 所有特征列
    feature_cols = data_cols + time_features

    # 标准化器
    scaler = StandardScaler()

    # 标签设置
    incident_df['label'] = 1  # 异常样本
    before_df['label'] = 0  # 正常样本
    after_df['label'] = 0  # 正常样本

    # 合并数据集
    data_df = pd.concat([incident_df, before_df, after_df], ignore_index=True)
    data_df = data_df.ffill()

    # 标准化特征
    features = scaler.fit_transform(data_df[feature_cols])
    labels = data_df['label'].values

    return features, labels, feature_cols


# =============== 小样本学习数据集 ===============

class FewShotDataset(Dataset):
    def __init__(self, features, labels, seq_len=24):
        self.features = features
        self.labels = labels
        self.seq_len = seq_len

        # 按类别分组样本
        self.normal_indices = np.where(labels == 0)[0]
        self.anomaly_indices = np.where(labels == 1)[0]

        print(f"正常样本数: {len(self.normal_indices)}")
        print(f"异常样本数: {len(self.anomaly_indices)}")

        # 检验样本量是否足够
        assert len(self.normal_indices) >= seq_len, "正常样本不足"
        assert len(self.anomaly_indices) >= seq_len, "异常样本不足"

        # 创建序列
        self._create_sequences()

    def _create_sequences(self):
        # 构建时间序列样本
        self.normal_sequences = []
        self.anomaly_sequences = []

        # 正常样本序列
        for i in range(len(self.normal_indices) // self.seq_len):
            start_idx = i * self.seq_len
            end_idx = start_idx + self.seq_len
            if end_idx <= len(self.normal_indices):
                indices = self.normal_indices[start_idx:end_idx]
                self.normal_sequences.append(self.features[indices])

        # 异常样本序列
        for i in range(len(self.anomaly_indices) // self.seq_len):
            start_idx = i * self.seq_len
            end_idx = start_idx + self.seq_len
            if end_idx <= len(self.anomaly_indices):
                indices = self.anomaly_indices[start_idx:end_idx]
                self.anomaly_sequences.append(self.features[indices])

        print(f"正常序列数: {len(self.normal_sequences)}")
        print(f"异常序列数: {len(self.anomaly_sequences)}")

    def __len__(self):
        # 返回可构建的任务数量，这里设为较大数以支持长时间训练
        return 10000

    def __getitem__(self, idx):
        # 每次采样构建一个N-way K-shot任务
        # 默认为2-way (正常/异常), 5-shot支持, 5-query查询
        n_way = 2  # 正常和异常两类
        k_shot = 5  # 每类5个支持样本
        n_query = 5  # 每类5个查询样本

        # 构建支持集和查询集
        support_x = []
        support_y = []
        query_x = []
        query_y = []

        # 正常类别
        normal_indices = np.random.choice(len(self.normal_sequences), k_shot + n_query, replace=False)
        support_indices = normal_indices[:k_shot]
        query_indices = normal_indices[k_shot:]

        for idx in support_indices:
            support_x.append(self.normal_sequences[idx])
            support_y.append(0)

        for idx in query_indices:
            query_x.append(self.normal_sequences[idx])
            query_y.append(0)

        # 异常类别
        anomaly_indices = np.random.choice(len(self.anomaly_sequences), k_shot + n_query, replace=False)
        support_indices = anomaly_indices[:k_shot]
        query_indices = anomaly_indices[k_shot:]

        for idx in support_indices:
            support_x.append(self.anomaly_sequences[idx])
            support_y.append(1)

        for idx in query_indices:
            query_x.append(self.anomaly_sequences[idx])
            query_y.append(1)

        # 转换为tensor
        support_x = torch.FloatTensor(np.array(support_x))
        support_y = torch.LongTensor(np.array(support_y))
        query_x = torch.FloatTensor(np.array(query_x))
        query_y = torch.LongTensor(np.array(query_y))

        return support_x, support_y, query_x, query_y


# =============== 编码器模型 ===============

class TimeSeriesEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, num_layers=4, nhead=8, dropout=0.1):
        super(TimeSeriesEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # 输入投影
        self.input_projection = nn.Linear(feature_dim, hidden_dim)

        # 位置编码
        self.pos_encoder = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(3)
        ])

        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 自适应层
        self.adaptive_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, feature_dim]

        # 特征投影
        x = self.input_projection(x)

        # 简化的位置编码
        for i, layer in enumerate(self.pos_encoder):
            if i % 2 == 0:
                x = x + layer(x)
            else:
                x = x * torch.sigmoid(layer(x))

        # Transformer编码器
        x = self.transformer_encoder(x)

        # 对序列取平均得到序列级表示
        x = torch.mean(x, dim=1)

        # 自适应层
        x = x + self.adaptive_layer(x)

        return x


# =============== 原型网络模型 ===============

class PrototypicalNetwork(nn.Module):
    def __init__(self, encoder):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = encoder

    def forward(self, support_x, support_y, query_x):
        # 编码支持集和查询集
        support_z = self.encoder(support_x)  # [n_support, embedding_dim]
        query_z = self.encoder(query_x)  # [n_query, embedding_dim]

        # 计算原型
        n_way = len(torch.unique(support_y))
        prototypes = torch.zeros(n_way, support_z.size(1)).to(support_z.device)

        for i in range(n_way):
            mask = (support_y == i).unsqueeze(1).expand_as(support_z)
            prototype = torch.sum(support_z * mask.float(), dim=0) / torch.sum(mask.float(), dim=0)
            prototypes[i] = prototype

        # 计算查询样本到原型的距离
        dists = torch.cdist(query_z, prototypes)  # 欧氏距离

        # 距离越小，相似度越高
        logits = -dists

        return logits


# =============== 训练与评估 ===============

def train_prototypical_network(model, train_loader, val_loader, epochs=50, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    best_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            support_x, support_y, query_x, query_y = [x.to(device) for x in batch]

            # 前向传播
            logits = model(support_x, support_y, query_x)

            # 计算损失
            loss = F.cross_entropy(logits, query_y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算准确率
            _, preds = torch.max(logits, 1)
            acc = (preds == query_y).float().mean().item()

            train_loss += loss.item()
            train_acc += acc

            pbar.set_postfix({'loss': loss.item(), 'acc': acc})

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # 验证
        val_loss, val_acc = evaluate_prototypical_network(model, val_loader, device)

        # 学习率调整
        lr_scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate_prototypical_network(model, data_loader, device):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
        for batch in data_loader:
            support_x, support_y, query_x, query_y = [x.to(device) for x in batch]

            # 前向传播
            logits = model(support_x, support_y, query_x)

            # 计算损失
            loss = F.cross_entropy(logits, query_y)

            # 计算准确率
            _, preds = torch.max(logits, 1)
            acc = (preds == query_y).float().mean().item()

            val_loss += loss.item()
            val_acc += acc

    val_loss /= len(data_loader)
    val_acc /= len(data_loader)

    return val_loss, val_acc


# =============== 主函数 ===============

def main():
    # 加载和预处理数据
    features, labels, feature_cols = load_and_preprocess_data()

    # 创建小样本学习数据集
    dataset = FewShotDataset(features, labels, seq_len=24)

    # 分割训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    # 创建模型
    feature_dim = features.shape[1]
    encoder = TimeSeriesEncoder(feature_dim, hidden_dim=128, num_layers=4, nhead=8)
    model = PrototypicalNetwork(encoder)

    # 训练模型
    trained_model = train_prototypical_network(
        model, train_loader, val_loader, epochs=50
    )

    # 保存模型
    torch.save(trained_model.state_dict(), "traffic_prototypical_network.pth")
    print("模型已保存为: traffic_prototypical_network.pth")


if __name__ == "__main__":
    main()