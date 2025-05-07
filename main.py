import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
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
def train_prototypical_network(model, train_loader, val_loader,
                               epochs=50, lr=0.001,
                               warmup_steps=1000, grad_clip_norm=1.0,
                               early_stopping_patience=10, min_delta=0):
    """
    训练原型网络模型 (包含学习率预热、梯度裁剪和早停法)
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 最大训练轮数
        lr: 初始学习率 (预热结束后的目标学习率)
        warmup_steps: 学习率线性预热的步数
        grad_clip_norm: 梯度裁剪的最大范数
        early_stopping_patience: 早停法耐心值 (多少个 epoch 验证损失没有改善则停止)
        min_delta: 被视为验证损失改善的最小变化量
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5 # 这个 patience 是降低学习率的耐心
    )

    # 早停法相关变量初始化
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None # 用于存储最佳模型的状态字典

    global_step = 0

    print(f"--- 开始训练 ---")
    print(f"初始学习率 (Initial LR): {lr}")
    print(f"预热步数 (Warmup Steps): {warmup_steps}")
    print(f"梯度裁剪范数 (Grad Clip Norm): {grad_clip_norm}")
    print(f"早停耐心值 (Early Stopping Patience): {early_stopping_patience}")
    print(f"最小改善阈值 (Min Delta): {min_delta}")
    print(f"设备 (Device): {device}")
    print("-" * 20)

    stopped_early = False # 标记是否因为早停而结束
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [训练]", unit="batch")
        for batch in pbar:
            global_step += 1
            support_x, support_y, query_x, query_y = [x.to(device) for x in batch]

            # --- 学习率预热逻辑 ---
            current_lr = lr
            if global_step < warmup_steps:
                lr_scale = float(global_step) / float(warmup_steps)
                current_lr = lr * lr_scale
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            # ----------------------

            logits = model(support_x, support_y, query_x)
            query_y = query_y.view(-1)
            loss = F.cross_entropy(logits, query_y)

            optimizer.zero_grad()
            loss.backward()

            # --- 梯度裁剪 ---
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            # -----------------

            optimizer.step()

            _, preds = torch.max(logits, 1)
            acc = (preds == query_y).float().mean().item()
            train_loss += loss.item()
            train_acc += acc
            pbar.set_postfix({'loss': loss.item(), 'acc': acc, 'lr': current_lr})

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # 验证
        val_loss, val_acc = evaluate_prototypical_network(model, val_loader, device)

        # --- 学习率调度器更新 ---
        lr_scheduler.step(val_loss)
        actual_lr_after_schedule = optimizer.param_groups[0]['lr']
        #-------------------------

        print(f"Epoch {epoch + 1}/{epochs} 结束 - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Current LR: {actual_lr_after_schedule:.6f}")

        # --- 早停法逻辑 ---
        if val_loss < best_val_loss - min_delta:
            # 找到了更好的模型 (基于验证损失)
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy() # 保存最佳模型状态
            print(f"*** 验证损失改善，新的最佳损失: {best_val_loss:.4f}, 已暂存模型 ***")
        else:
            # 验证损失没有改善
            epochs_no_improve += 1
            print(f"验证损失未改善或改善小于 {min_delta}，连续未改善 epoch 数: {epochs_no_improve}/{early_stopping_patience}")
            if epochs_no_improve >= early_stopping_patience:
                print(f"--- 早停触发！连续 {early_stopping_patience} 个 Epoch 验证损失未改善 ---")
                stopped_early = True
                break # 中断训练循环
        # -----------------

    print(f"--- 训练结束 ---")
    if stopped_early:
        print(f"训练因早停而在 Epoch {epoch + 1} 结束。")
    else:
        print(f"训练完成所有 {epochs} 个 Epoch。")

    print(f"最低验证损失 (Best Val Loss): {best_val_loss:.4f}")

    # 加载最佳模型 (对应最低验证损失)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("已加载最佳模型权重 (基于最低验证损失)。")
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


# =============== 评估函数 ===============

def evaluate_model_metrics(model, data_loader, device):
    """全面评估模型性能并返回详细指标"""
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估模型", unit="batch"):
            support_x, support_y, query_x, query_y = [x.to(device) for x in batch]

            # 前向传播
            logits = model(support_x, support_y, query_x)

            # 获取预测结果
            query_y = query_y.view(-1)
            _, preds = torch.max(logits, 1)

            # 收集预测和真实标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(query_y.cpu().numpy())

    # 计算各项指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }


# =============== 主函数 ===============

def main():
    # 记录开始时间
    start_time = time.time()

    # 加载和预处理数据
    features, labels, feature_cols = load_and_preprocess_data()

    # 创建小样本学习数据集
    dataset = FewShotDataset(features, labels, seq_len=24)

    # 分割训练集和验证集
    torch.manual_seed(42) # 为了可复现性
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, num_workers=0)

    # 创建模型
    feature_dim = features.shape[1]
    encoder = TimeSeriesEncoder(feature_dim, hidden_dim=128, num_layers=4, nhead=8, dropout=0.1)
    model = PrototypicalNetwork(encoder)

    # 训练模型 (调用包含早停的版本)
    initial_learning_rate = 0.0005
    warmup_steps_count = 1000
    gradient_clip_value = 1.0
    early_stopping_patience_value = 5 # 设置一个合适的耐心值。如果你的验证损失波动较大，可以设置得更大一些 (如 15 或 20)；
                                       # 如果希望更快停止，可以设小一些 (如 5 或 7)。10 是一个常用的起始值。
    min_delta_value = 1e-5             # 通常设为 0 或一个非常小的值 (如 1e-4, 1e-5)。设为 0 表示任何微小的损失下降都算作改善。

    trained_model = train_prototypical_network(
        model,
        train_loader,
        val_loader,
        epochs=50, # 最大 epoch 数
        lr=initial_learning_rate,
        warmup_steps=warmup_steps_count,
        grad_clip_norm=gradient_clip_value,
        early_stopping_patience=early_stopping_patience_value,
        min_delta=min_delta_value
    )

    # 保存模型 (保存的是验证损失最低时的模型)
    torch.save(trained_model.state_dict(), "traffic_prototypical_network_v2.1.pth")
    print("模型已保存为: traffic_prototypical_network_v2.1.pth")

    # 在最终测试集上评估模型性能
    print("\n=== 最终模型性能评估 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate_model_metrics(trained_model, val_loader, device)

    # 输出评估指标
    print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision): {metrics['precision']:.4f}")
    print(f"召回率 (Recall): {metrics['recall']:.4f}")
    print(f"F1值 (F1 Score): {metrics['f1_score']:.4f}")
    print(f"混淆矩阵 (Confusion Matrix):\n{metrics['confusion_matrix']}")

    # 计算并输出总运行时间
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n=== 程序执行完成 ===")
    print(f"总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print("错误：请确保 'data/incident.xlsx', 'data/common_before.xlsx', 'data/common_after.xlsx' 文件存在于正确路径。")
    except Exception as e:
        print(f"发生错误: {e}")