
# FSL-Traffic

本项目基于 **TimeSeriesEncoder + 原型网络（Prototypical Network）** 实现小样本交通流异常检测。

---

##  项目结构

```
FSL-Traffic/
├── main.py                     # 主训练脚本
├── model/                      # 存放模型参数（已加入 .gitignore，不上传）
│   └── traffic_prototypical_network_v1.pth
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

| 版本 | 特征提取器 | 精度 | 备注 |
|------|------------|------|------|
| v1.0 | TimeSeriesEncoder + ProtoNet | 93.5% | 初始实现 |
| v2.0 | 加入梯度裁剪+warmup | 96.55% | 稳定性大幅提升 |

---

##  忽略项说明（`.gitignore`）

- 模型参数（*.pth）
- 中间日志与缓存
- PyCache

---

##  TODO

- [ ] 加入模型蒸馏
- [ ] 对比不同 few-shot 架构（如 MAML, GNN）
- [ ] 集成 Streamlit 简易展示平台

---
