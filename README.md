# InfLoRA-Jittor
## 1、项目介绍
本项目是基于 Jittor 的 InfLoRA 复现，是《人工智能实践课（初级）》的第二次复现（第四次汇报）内容。<br>
复现的论文是发表在 CVPR 2024 上的《InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning》，论文的 GitHub 链接为 https://github.com/liangyanshuo/InfLoRA ，论文的arxiv为 2404.00228 ，感谢作者对代码进行了开源，这为我的复现工作提供了非常重要的帮助，再次感谢！！！<br>
同时，这篇文章在 PyTorch 上的复现，请参考我的另一个仓库 https://github.com/lgq-coding/InfLoRA-PyTorch <br>
同时，也感谢《人工智能实践课》带给我的与众不同的体验，感谢在学习与复现过程中各位老师和同学的无私帮助！<br>
以下是仓库中的文件结构及作用
```
# 以下包含的文件主要是仓库中和复现相关的重要文件
project/
├── configs/                         # JSON 配置文件
├── data/                            # 数据集存储（自动创建）
├── logs/                            # 训练日志（包含自复现以来所有的 log ，本次实验的成功结果请参见学习率为0.0001的 log 的20000行处）
├── methods/
│   ├── base.py                      # 基础学习器类
│   └── inflora.py                   # InfLoRA 学习器实现
├── models/
│   ├── sinet_inflora.py             # SiNet（ViT + LoRA）定义
│   └── vit_inflora.py               # Attention_LoRA 模块
├── utils/
│   ├── toolkit.py                   # 辅助函数
|   ├── data_manager.py              # 数据设置
│   ├── data_manager.py              # 数据加载与划分
│   └── factory.py                   # 模型工厂
├── main.py                          # 程序入口
├── trainer.py                       # 训练循环封装
├── jittor_test.py                   # Jittor 配置测试
├── model_weight.py                  # 模型权重下载
├── requirements.txt                 # 依赖包列表
└── README.md                        # 本文档
```
## 2、论文介绍
### 2.1 简要概括
以往的基于已有的PEFT(参数高效微调)的持续学习方法，要么用旧任务的参数适配新任务，要么随机扩展参数，然后适配新任务，这些方法都无法避免新任务对旧任务的干扰。本文引入了防干扰机制，注入一小部分参数，然后重参数化，并证明微调这一部分参数($A_t$)等价于在子空间内($B_t$的行向量的张成空间)微调预训练权重，并且消除了新任务对旧任务的干扰。<br>
### 2.2 具体手段
精心设置了矩阵 $B_t$ : 与以往LoRA方法用高斯分布初始化不同，本文使用新任务梯度空间 $N_t$ 和旧任务梯度空间的正交补空间 $M_t$ ⊥的交集，保证 $B_t$ 落在这个交集内，前者是为了保证学习新任务的能力(塑性)，而后者是为了不干扰旧任务的参数(固性)。<br>
在此基础上，每轮仅更新矩阵 $A_t$ (最开始矩阵A初始为0)，其余的均冻结，目标函数是local CE。<br>
空间的估计方法：①对于新任务，用新任务的输入矩阵估计梯度空间。②而旧任务的梯度空间信息，不能直接获取，通过DualGPM方法保存。③由于维度不一致，所以采取了奇异值分解和top-r奇异值选取的方法。<br>
### 2.3 其他信息
本文的方法同样可以扩展到自监督学习的模型，还能结合类对齐方法。<br>
这篇文章运用线性代数知识，简单而优美，比较直观。
## 3、复现内容
### 3.1 使用的平台
- AutoDL 的 GPU + VSCode
- GPU: RTX 4090 (24GB)
- 相关的包请查看 requirements.txt
### 3.2 数据集
- 本次复现基于 CIFAR-100 开展复现
- 对于CIFAR-100，本数据集可以在运行data-loader的过程中自动下载
## 4、尝试复现本项目
克隆仓库：
``` bash
git clone https://github.com/yourusername/inflora-jittor.git
cd inflora-jittor
pip install -r requirements.txt
```
请下载预训练权重并放置在项目根目录，下载地址：https://www.modelscope.cn/models/google/vit-base-patch16-224-in21k <br>
基本运行命令：
``` bash
python main.py --config configs/cifar100_inflora_debug.json
```
Jittor 默认使用所有可用 GPU，您可以通过环境变量限制使用特定 GPU：
``` bash
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/cifar100_inflora_debug.json
```
