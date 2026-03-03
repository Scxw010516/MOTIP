# Tutorial
# 教程

In this tutorial, we aim to provide simple, easy-to-understand explanations and guidance to help you better understand, migrate, and improve our model.
在本教程中，我们旨在提供简单、易于理解的解释和指导，以帮助您更好地理解、迁移和改进我们的模型。

## Improvements *vs.* [prev-engine branch](https://github.com/MCG-NJU/MOTIP/tree/prev-engine)
## 改进之处 *vs.* [prev-engine 分支](https://github.com/MCG-NJU/MOTIP/tree/prev-engine)

1. We adopt **Accelerate** instead of PyTorch's native DDP framework, making multi-GPU training and mixed-precision training more convenient (though the latter wasn't used in our final experiments).
   我们采用 **Accelerate** 代替 PyTorch 的原生 DDP 框架，使多 GPU 训练和混合精度训练更加方便（尽管后者在我们的最终实验中没有使用）。
2. We implement the trajectory augmentation part in data processing (on CPU, together with other image/video augmentation methods), which significantly **improves GPU utilization** during training and accelerates the speed of each iteration.
   我们在数据处理部分（在 CPU 上，与其他图像/视频增强方法一起）实现了轨迹增强部分，这显着**提高了训练期间的 GPU 利用率**，并加快了每次迭代的速度。
3. We use different ID assignment groups for the same video clip sample, as specified by the parameter `AUG_NUM_GROUPS: 6`. This can significantly improve data utilization and **accelerate model convergence**.
   我们对同一个视频片段样本使用不同的 ID 分配组，由参数 `AUG_NUM_GROUPS: 6` 指定。这可以显着提高数据利用率并**加速模型收敛**。
4. By default, we abandon the Hungarian algorithm in favor of **a simpler runtime ID assignment method** that only selects the highest confidence. We present ablation experiments and explanations in Table 4.
   默认情况下，我们舍弃了匈牙利算法，转而采用**一种更简单的运行时 ID 分配方法**，该方法仅选择最高置信度。我们在表 4 中提供了消融实验和解释。

## Temporal Length (Window)
## 时间长度（窗口）

Like most MOT algorithms, our model only handles target disappearance and re-appearance within a certain tolerance range, which is noted as *T*. If this temporal tolerance is exceeded, even the same target will be assigned a different ID.
与大多数 MOT 算法一样，我们的模型仅处理在一定容差范围（记为 *T*）内的目标消失和重新出现。如果超过此时间容差，即使是同一个目标也会被分配不同的 ID。

Since our model utilizes *long-term sequence training*, *relative temporal position encoding*, and *online inference*, there are some parameters that, together, determine the temporal length. If you need to modify the temporal length we have set, these parameters need to be carefully changed together:
由于我们的模型利用了*长期序列训练*、*相对时间位置编码*和*在线推理*，因此有一些参数共同决定了时间长度。如果您需要修改我们设置的时间长度，需要仔细地共同更改这些参数：

- `SAMPLE_LENGTHS`: The temporal length of the sampled video clip during training.
  `SAMPLE_LENGTHS`: 训练期间采样的视频片段的时间长度。
- `REL_PE_LENGTH`: The max length of the relative temporal position encoding.
  `REL_PE_LENGTH`: 相对时间位置编码的最大长度。
- `MISS_TOLERANCE`: The temporal tolerance of re-appear targets during inference.
  `MISS_TOLERANCE`: 推理期间重新出现目标的容差时间。

**A quick and straightforward setting rule is: `SAMPLE_LENGTHS == REL_PE_LENGTH >= MISS_TOLERANCE`.**
**一个快速而直接的设置规则是：`SAMPLE_LENGTHS == REL_PE_LENGTH >= MISS_TOLERANCE`。**

## Thresholds
## 阈值

Although our method is end-to-end, we still need some thresholds to control the model's behavior during inference (like DETRs need thresholds to select positive targets). As we decouple the object detection and association processes, the thresholds are also divided into two parts.
虽然我们的方法是端到端的，但我们仍然需要一些阈值来控制模型在推理期间的行为（就像 DETR 需要阈值来选择正样本目标一样）。由于我们将目标检测和关联过程解耦，因此阈值也分为两部分。

1. **Object Detection.** DETR outputs numerous detection results, but we do not need to process all of them in tracking, as this would lead to excessive computational overhead. Therefore, we use the following thresholds to control the process in the current frame:
   1. **目标检测**。DETR 输出大量的检测结果，但我们在跟踪中不需要处理所有结果，因为这会导致过度的计算开销。因此，我们使用以下阈值来控制当前帧的处理过程：

   1) `DET_THRESH`: A target will only be selected and fed into the ID Decoder if its confidence exceeds this threshold.
      `DET_THRESH`: 只有当目标的置信度超过此阈值时，才会被选中并送入 ID 解码器。
   2) `NEWBORN_THRESH`: When a target does not match any historical trajectory, it can be marked as a newborn target only if it exceeds this threshold. This is to make the generation of new trajectories as reliable as possible.
      `NEWBORN_THRESH`: 当一个目标与任何历史轨迹都不匹配时，只有当它超过此阈值时，才能被标记为新生目标。这是为了使新轨迹的生成尽可能可靠。

2. **Object Association.** The ID Decoder outputs a probability distribution of a target being assigned to different IDs, so we need a threshold to control the minimum confidence:
   2. **目标关联**。ID 解码器输出一个目标被分配到不同 ID 的概率分布，因此我们需要一个阈值来控制最小置信度：
   
   1) `ID_THRESH`: Only when the confidence assigned to an ID is greater than this threshold can it be regarded as a valid allocation.
      `ID_THRESH`: 只有当分配给某个 ID 的置信度大于此阈值时，才能被视为有效的分配。

  
