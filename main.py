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
        # 处理支持集
        batch_size, n_support, seq_len, feat_dim = support_x.shape
        support_x_reshaped = support_x.view(-1, seq_len, feat_dim)
        support_z = self.encoder(support_x_reshaped)
        support_z = support_z.view(batch_size, n_support, -1)

        # 处理查询集
        batch_size, n_query, seq_len, feat_dim = query_x.shape
        query_x_reshaped = query_x.view(-1, seq_len, feat_dim)
        query_z = self.encoder(query_x_reshaped)
        query_z = query_z.view(batch_size, n_query, -1)

        # 计算原型和距离（按批次处理）
        logits = []
        for b in range(batch_size):
            n_way = len(torch.unique(support_y[b]))
            prototypes = torch.zeros(n_way, support_z.size(-1)).to(support_z.device)

            for i in range(n_way):
                mask = (support_y[b] == i).unsqueeze(1).expand(-1, support_z.size(-1))
                prototype = torch.sum(support_z[b] * mask.float(), dim=0) / torch.sum(mask.float(), dim=0)
                prototypes[i] = prototype

            # 计算距离
            batch_dists = torch.cdist(query_z[b], prototypes)
            batch_logits = -batch_dists
            logits.append(batch_logits)

        # 返回重新组织的logits，保持批次结构
        return torch.stack(logits, dim=0).view(batch_size * n_query, -1)


# =============== 训练与评估 ===============
# 加入梯度裁剪和线性学习率预热(warmup)策略
def train_prototypical_network(model, train_loader, val_loader, epochs=50, lr=0.001, warmup_steps=1000, grad_clip_norm=1.0):
    """
    训练原型网络模型

    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        lr: 初始学习率 (预热结束后的目标学习率),可以从 0.0005 或 0.0001 开始尝试
            使用 warmup 时，通常可以将初始学习率 lr 设置得稍大一些（或者保持原来的值，因为 warmup 会平稳地达到它）
        warmup_steps: 学习率线性预热的步数,通常设置为总训练步数的一小部分
                      例如前 500-2000 步
                      你的每个 epoch 有 2000 步 (len(train_loader) = 8000 / 4 = 2000)
                      所以 1000 步大约是半个 epoch。
        grad_clip_norm: 梯度裁剪的最大范数,1.0 是常用值
                        但有时可能需要调整（例如 0.5 或 5.0）
                        设为 0 或负数可以禁用它
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # 使用初始LR初始化优化器
    # 学习率调度器，在预热结束后生效
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_acc = 0.0
    best_model_state = None
    global_step = 0 # 用于跟踪总的优化器步数，配合warmup

    print(f"--- 开始训练 ---")
    print(f"初始学习率 (Initial LR): {lr}")
    print(f"预热步数 (Warmup Steps): {warmup_steps}")
    print(f"梯度裁剪范数 (Grad Clip Norm): {grad_clip_norm}")
    print(f"设备 (Device): {device}")
    print("-" * 20)

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        # 使用 tqdm 显示进度，并在描述中加入 Epoch 信息
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [训练]", unit="batch")
        for batch in pbar:
            global_step += 1 # 更新全局步数
            support_x, support_y, query_x, query_y = [x.to(device) for x in batch]

            # --- 学习率预热逻辑 ---
            current_lr = lr # 默认使用初始学习率
            if global_step < warmup_steps:
                # 计算线性增加的学习率
                lr_scale = float(global_step) / float(warmup_steps)
                current_lr = lr * lr_scale
                # 应用 warmup 学习率到优化器
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            # ----------------------

            # 前向传播
            logits = model(support_x, support_y, query_x)

            # 调整 query_y 的形状以匹配 logits
            query_y = query_y.view(-1)

            # 计算损失
            loss = F.cross_entropy(logits, query_y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # --- 梯度裁剪 ---
            if grad_clip_norm > 0: # 允许设置为0或负数来禁用裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm) # 限制梯度的最大范数，防止梯度爆炸
            # -----------------

            # 更新权重
            optimizer.step()

            # 计算准确率
            _, preds = torch.max(logits, 1)
            acc = (preds == query_y).float().mean().item()

            train_loss += loss.item()
            train_acc += acc

            # 更新进度条显示信息，加入当前学习率
            pbar.set_postfix({'loss': loss.item(), 'acc': acc, 'lr': current_lr})

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # 验证 (调用你之前的 evaluate_prototypical_network 函数)
        val_loss, val_acc = evaluate_prototypical_network(model, val_loader, device)

        # --- 学习率调度器更新 ---
        # 注意：只有在 warmup 结束后，ReduceLROnPlateau 才应该根据 val_loss 调整学习率
        # ReduceLROnPlateau 本身会处理好基于当前 optimizer 中的 lr 进行调整
        lr_scheduler.step(val_loss)
        # 获取调度器调整后的实际学习率 (如果没调整，则保持不变)
        actual_lr_after_schedule = optimizer.param_groups[0]['lr']
        #-------------------------

        print(f"Epoch {epoch + 1}/{epochs} 结束 - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Current LR (End of Epoch): {actual_lr_after_schedule:.6f}")

        # 保存最佳模型 (基于验证准确率)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"*** 找到新的最佳模型 (Val Acc: {best_acc:.4f}), 已暂存 ***")


    print(f"--- 训练结束 ---")
    print(f"最高验证准确率 (Best Val Acc): {best_acc:.4f}")

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("已加载最佳模型权重。")
    else:
        print("警告：未能找到最佳模型状态，模型保持训练结束时的状态。")


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

            # 修复维度不匹配问题
            query_y = query_y.view(-1)

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
    # 注意：由于数据集 __len__ 返回固定值，这里的分割比例可能不精确反映实际任务数
    # 但对于随机抽样来说是可行的
    train_size = int(0.8 * len(dataset)) # 这里的 len(dataset) 是 10000
    val_size = len(dataset) - train_size
    # 为了确保每次运行的可复现性，可以设置一个随机种子
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0) # 调整 num_workers 适应你的环境
    val_loader = DataLoader(val_dataset, batch_size=4, num_workers=0)

    # 创建模型
    feature_dim = features.shape[1]
    encoder = TimeSeriesEncoder(feature_dim, hidden_dim=128, num_layers=4, nhead=8, dropout=0.1) # 可以尝试调整 dropout
    model = PrototypicalNetwork(encoder)

    # 训练模型 (使用修改后的函数)
    initial_learning_rate = 0.0005 # 考虑使用稍小一点的学习率配合 warmup
    warmup_steps_count = 1000 # 大约半个 epoch 的步数
    gradient_clip_value = 1.0 # 常用的梯度裁剪值

    trained_model = train_prototypical_network(
        model,
        train_loader,
        val_loader,
        epochs=50,
        lr=initial_learning_rate,
        warmup_steps=warmup_steps_count,
        grad_clip_norm=gradient_clip_value
    )

    # 保存模型
    torch.save(trained_model.state_dict(), "traffic_prototypical_network_v1.pth")
    print("模型已保存为: traffic_prototypical_network_v1.pth")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print("错误：请确保 'data/incident.xlsx', 'data/common_before.xlsx', 'data/common_after.xlsx' 文件存在于正确路径。")
    except Exception as e:
        print(f"发生错误: {e}")
    main()