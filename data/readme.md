# MOTIP_Mask 数据目录文档

本目录 `MOTIP_Mask/data` 包含了 MOTIP 模型所需的数据集加载、预处理、采样和增强相关的代码实现。主要支持 DanceTrack, SportsMOT, CrowdHuman, BFT 等数据集的统一加载与混合训练。

## 文件详细功能说明

### 数据集类定义
- **`one_dataset.py`**
  - 定义了基类 `OneDataset`，提供了数据集类的基本框架，包括数据根目录、数据集划分（split）和是否加载注释的初始化参数。

- **`dancetrack.py`**
  - 继承自 `OneDataset`，实现了 **DanceTrack** 数据集的具体加载逻辑。
  - 解析 `seqinfo.ini` 获取序列元数据（宽高、长度）。
  - 加载图像路径和 GT（Ground Truth）注释文件。
  - 作为父类被 `SportsMOT`, `BFT`, `CrowdHuman` 等其他数据集类继承。

- **`sportsmot.py`**
  - 继承自 `DanceTrack`，针对 **SportsMOT** 数据集的特定目录结构（图像命名格式）进行了适配。

- **`crowdhuman.py`**
  - 继承自 `DanceTrack`，针对 **CrowdHuman** 静态数据集进行了适配。
  - 将静态图片伪装成长度为 1 的视频序列，以便统一输入流水线。
  - 实现了针对 CrowdHuman 格式的注释读取。

- **`bft.py`**
  - 继承自 `DanceTrack`，针对 **BFT (Brownian Family Tree)** 数据集进行了适配，主要适配了图像路径的生成规则。

- **`seq_dataset.py`**
  - 定义了 `SeqDataset` 类，这是一个简化的数据集类。
  - 主要用于**推理（Inference）或演示（Demo）**阶段，仅负责加载指定序列的图像并进行基本的预处理（Resize, Normalize），不涉及复杂的 GT 注释处理。

### 核心数据处理与接口
- **`joint_dataset.py`**
  - 定义了核心类 `JointDataset`，它是训练时使用的主要入口。
  - **功能**：
    - 支持同时加载多个不同的数据集（如 DanceTrack + CrowdHuman）进行混合训练。
    - 统一管理所有数据集的元数据 (`sequence_infos`)、图像路径和注释。
    - 实现了 `__getitem__`，负责根据采样信息读取图像和注释，并调用 `transforms` 进行数据增强。
    - `set_sample_details`: 用于在每个 epoch 开始前设定采样策略（如采样长度、间隔等）。
    - 负责将 `is_legal` 属性与注释解耦，以便更高效地进行采样合法性检查。

- **`__init__.py`**
  - 模块对外接口，提供了 `build_dataset` 和 `build_dataloader` 便捷函数，用于根据配置创建 `JointDataset` 实例和 PyTorch `DataLoader`。

### 采样与增强
- **`naive_sampler.py`**
  - 定义了 `NaiveSampler` 类，继承自 `torch.utils.data.sampler.Sampler`。
  - **功能**：
    - 负责生成训练时的采样索引。
    - 支持按各种策略（如随机间隔）从视频序列中采样片段。
    - 能够根据数据集的权重 (`data_weights`) 平衡不同数据集的采样频率。
    - 确保采样出的片段中所有帧的注释都是合法的 (`is_legal`)。

- **`transforms.py`**
  - 包含了一系列用于视频数据的数据增强类，能够同时处理图像和对应的 Bounding Box 注释。
    - 基础增强：`MultiRandomCrop`, `MultiRandomResize`, `MultiColorJitter`, `MultiRandomHorizontalFlip` 等。
    - 视频模拟：`MultiSimulate`（通过移动和裁剪静态图片模拟视频效果）。
  - **MOTIP 特有逻辑**：
    - `GenerateIDLabels`: 将原始注释转换为 MOTIP 模型需要的 ID 标签、掩码（Mask）和时间索引张量。
    - `TurnIntoTrajectoryAndUnknown`: 实现了针对轨迹的复杂增强策略，如**轨迹遮挡模拟** (`aug_trajectory_occlusion_prob`) 和 **ID 切换模拟** (`aug_trajectory_switch_prob`)，以及生成用于监督的新生目标（Newborn）标签。

### 工具函数
- **`util.py`**
  - 包含了数据处理通用工具函数：
    - `is_legal`: 校验单帧注释的合法性（字段完整性、ID 唯一性、Query 数量限制等）。
    - `append_annotation`: 用于向注释结构中添加新的对象记录。
    - `collate_fn`: **DataLoader 的整理函数**。它不仅将数据堆叠为 Batch，还负责处理不同样本间 ID 标签数量不一致的问题（通过 Padding 补齐），并生成适应 DETR 架构的 `NestedTensor`。
