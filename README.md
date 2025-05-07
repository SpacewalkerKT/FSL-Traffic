
# FSL-Traffic

本项目*目前*基于 **TimeSeriesEncoder + 原型网络（Prototypical Network）** 实现小样本交通流异常检测

---

##  项目结构

```
FSL-Traffic/
├── main.py                     # 主训练脚本
├── model/                      # 存放模型参数（已加入 .gitignore，不上传）
    └── traffic_prototypical_network_v0.0.pth
    └── traffic_prototypical_network_v1.0.pth
    └── traffic_prototypical_network_v2.0.pth
├── data/                       # 存放数据样本
    └── incident.xlsx           # 事故数据
    └── common_before.xlsx      # 事故前一天的正常数据
    └── common_after.xlsx       # 事故后一天的正常数据
├── logs/                       # 训练日志
├── README.md                   # 项目说明文件
├── .gitignore                  # Git 忽略规则文件
```

---

##  快速开始


### 运行训练脚本

```bash
python main.py
```

---

##  模型简介

- 编码器：基于 Transformer 的 `TimeSeriesEncoder`
- 分类器：原型网络（Prototypical Network）
- 任务结构：2-way 5-shot few-shot classification
- 数据源：PEMS 交通流数据，异常数据源自事故前后传感器响应

---

##  实验记录

| 版本 | 更新内容 | 验证损失(Val Loss)  | 验证准确率(Val Acc) | 备注 |
|------|------------|------|------|------|
| v0.0 | 使用特征提取器TimeSeriesEncoder + ProtoNet | 2.1134 | 93.41% | 初始代码实现，但出现运行不稳定且过拟合 |
| v1.0 | 加入梯度裁剪+warmup预热 | 0.0692 | 96.55% | 稳定性大幅提升但仍存在过拟合的风险 |
| v2.0 | 加入早停法 | 0.0693 | 96.22% | 由于早停时间显著缩短，且验证损失达到最小，模型适应性大幅提升 |
| v2.1 | 加入运行时间和评估指标的输出 | \ | \ | 增加可读性 |
| v3.0 | 代码模块化 | \ | \ | \ |
---

##  忽略项说明（`.gitignore`）

- 模型参数（*.pth）
- 中间日志与缓存
- PyCache

---

##  后续工程

- [ ] 代码模块化：以供后续调整
- [ ] 配置管理：使用 YAML 文件来存储每次实验的配置
- [ ] 日志和可视化工具：使用 W&B 和 TensorBoard
- [ ] 符合学术趋势的评估指标：使用N-way K-shot 平均准确率以及置信区间 (Confidence Interval, CI)等
- [ ] few-shot 架构模型对比：一阶MAML( FOMAML)
- [ ] 引入**迁移学习**策略优化模型：如比较 ProtoNet+Pretrain vs FOMAML+Pretrain

---
