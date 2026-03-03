# Get Started
# 开始使用

In this documentation, we will primarily focus on training and inference of our MOTIP model on the relevant MOT benchmarks. All the configurations corresponding to our experiments are stored in the [configs](../configs/) folder. You can also customize the configuration files according to your own requirements.
在本说明文档中，我们将主要关注 MOTIP 模型在相关 MOT 基准测试上的训练和推理。所有与我们实验相对应的配置都存储在 [configs](../configs/) 文件夹中。您也可以根据自己的要求自定义配置文件。

## Pre-training
## 预训练

To expedite the training process, we’ll begin by pre-training the DETR component of the model. Typically, training the DETR model on a specific dataset (like DanceTrack, SportsMOT, etc.) is quite efficient, taking only a few hours.
为了加快训练过程，我们将从预训练模型的 DETR 部分开始。通常，在特定数据集（如 DanceTrack、SportsMOT 等）上训练 DETR 模型非常高效，仅需几个小时。

### COCO Pre-trained Weights
### COCO 预训练权重

:floppy_disk: ​Similar to many other methods (e.g., MOTR and MeMOTR), we also use COCO pre-trained DETR weights for initialization. You can obtain them from the following links:
:floppy_disk: ​与许多其他方法（例如 MOTR 和 MeMOTR）类似，我们也使用 COCO 预训练的 DETR 权重进行初始化。您可以从以下链接获取它们：

- Deformable DETR: [[official repo](https://github.com/fundamentalvision/Deformable-DETR)] [[our repo](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco.pth)]

### Pre-train DETR on Specific Datasets
### 在特定数据集上预训练 DETR

To accelerate the convergence, we will first pre-train DETR on the corresponding dataset (target dataset) to serve as the initialization for subsequent MOTIP training.
为了加速收敛，我们将首先在相应的数据集（目标数据集）上预训练 DETR，作为后续 MOTIP 训练的初始化。

#### Our Pre-trained Weights
#### 我们的预训练权重

:floppy_disk: **We recommend directly using our pre-trained DETR weights, which are stored in the [model zoo](./MODEL_ZOO.md#DETR).** If needed, you can pre-train it yourself using the script provided below.
:floppy_disk: **我们建议直接使用我们的预训练 DETR 权重，这些权重存储在 [模型库](./MODEL_ZOO.md#DETR) 中。** 如果需要，您可以按下方提供的脚本自行进行预训练。

You should put necessary pre-trained weights into `./pretrains/` directory as default.
默认情况下，您应该将必要的预训练权重放入 `./pretrains/` 目录。

#### Pre-training Scripts
#### 预训练脚本

**All our pre-train scripts follows the template script below.** You'll need to fill the `<placeholders>` according to your requirements:
**我们所有的预训练脚本都遵循下面的模板脚本。** 您需要根据您的要求填写 `<placeholders>`：

```bash
accelerate launch --num_processes=8 train.py --data-root <data dir> --exp-name <exp name> --config-path <.yaml config file path>
```

For example, you can pre-train a Deformable-DETR model on DanceTrack as follows:
例如，您可以按如下方式在 DanceTrack 上预训练 Deformable-DETR 模型：

```bash
accelerate launch --num_processes=8 train.py --data-root ./datasets/ --exp-name pretrain_r50_deformable_detr_dancetrack --config-path ./configs/pretrain_r50_deformable_detr_dancetrack.yaml
```

#### Gradient Checkpoint
#### 梯度检查点

Please referring to [here](./GET_STARTED.md#gradient-checkpoint) to get more information.
请参阅 [此处](./GET_STARTED.md#gradient-checkpoint) 了解更多信息。

## Training
## 训练

Once you have the DETR pre-trained weights on the corresponding dataset (target dataset), you can use the following script to train your own MOTIP model.
一旦您拥有了相应数据集（目标数据集）的 DETR 预训练权重，您就可以使用以下脚本训练您自己的 MOTIP 模型。

### Training Scripts
### 训练脚本

**All our training scripts follow the template script below.** You'll need to fill the `<placeholders>` according to your requirements:
**我们所有的训练脚本都遵循下面的模板脚本。** 您需要根据您的要求填写 `<placeholders>`：

```shell
accelerate launch --num_processes=8 train.py --data-root <DATADIR> --exp-name <exp name> --config-path <.yaml config file path>
```

For example, you can the default model on DanceTrack as follows:
例如，您可以按如下方式在 DanceTrack 上训练默认模型：

```shell
accelerate launch --num_processes=8 train.py --data-root ./datasets/ --exp-name r50_deformable_detr_motip_dancetrack --config-path ./configs/r50_deformable_detr_motip_dancetrack.yaml
```

*Using this script, you can achieve 69.5 HOTA on DanceTrack test set. There is a relatively high instability (~ 1.5) which is also encountered in other work (e.g., [OC-SORT](https://github.com/noahcao/OC_SORT), [MOTRv2](https://github.com/megvii-research/MOTRv2/issues/2), [MeMOTR](https://github.com/MCG-NJU/MeMOTR/issues/17)).*
*使用此脚本，您可以在 DanceTrack 测试集上达到 69.5 HOTA。存在相对较高的不稳定性（~ 1.5），这在其他工作中也会遇到（例如 [OC-SORT](https://github.com/noahcao/OC_SORT), [MOTRv2](https://github.com/megvii-research/MOTRv2/issues/2), [MeMOTR](https://github.com/MCG-NJU/MeMOTR/issues/17)）。*

### Gradient Checkpoint
### 梯度检查点

If your GPUs have less than 24GB CUDA memory, we offer the gradient checkpoint technology. You can set `--detr-num-checkpoint-frames` to `2` (< 16GB) or `1` (< 12GB) to reduce the CUDA memory requirements.
如果您的 GPU 显存小于 24GB，我们提供了梯度检查点技术。您可以将 `--detr-num-checkpoint-frames` 设置为 `2`（对应 < 16GB）或 `1`（对应 < 12GB）以降低显存需求。

## Inference
## 推理

We have two different inference modes:
我们有两种不同的推理模式：

1. Without ground truth annotations (e.g. DanceTrack test, SportsMOT test), [submission scripts](#Submission) can generate tracker files for submission.
   1. **没有 ground truth 标注**（例如 DanceTrack test, SportsMOT test），[提交脚本](#Submission) 可以生成用于提交的跟踪器文件。
2. With ground truth annotations, [evaluation scripts](#Evaluation) can produce tracking results and obtain evaluation results.
   2. **有 ground truth 标注**，[评估脚本](#Evaluation) 可以生成跟踪结果并获得评估结果。

:pushpin: **Different inference behaviors are controlled by the runtime parameter `--inference-mode`.**
:pushpin: **不同的推理行为由运行时参数 `--inference-mode` 控制。**

### Submission
### 提交

You can obtain the tracking results (tracker files) using the following **template script**:
您可以使用以下**模板脚本**获取跟踪结果（跟踪器文件）：

```shell
accelerate launch --num_processes=8 submit_and_evaluate.py --data-root <DATADIR> --inference-mode submit --config-path <.yaml config file path> --inference-model <checkpoint path> --outputs-dir <outputs dir> --inference-dataset <dataset name> --inference-split <split name>
```

For example, you can get our default results on the DanceTrack test set as follows:
例如，您可以按如下方式获取我们在 DanceTrack 测试集上的默认结果：

```shell
accelerate launch --num_processes=8 submit_and_evaluate.py --data-root ./datasets/ --inference-mode submit --config-path ./configs/r50_deformable_detr_motip_dancetrack.yaml --inference-model ./outputs/r50_deformable_detr_motip_dancetrack/r50_deformable_detr_motip_dancetrack.pth --outputs-dir ./outputs/r50_deformable_detr_motip_dancetrack/ --inference-dataset DanceTrack --inference-split test
```

:racing_car: You can add `--inference-dtype FP16` to the script to use float16 for inference. This can improve inference speed by over 30% with only a slight impact on tracking performance (about 0.5 HOTA on DanceTrack test).
:racing_car: 您可以在脚本中添加 `--inference-dtype FP16` 以使用 float16 进行推理。这可以提高 30% 以上的推理速度，且对跟踪性能仅有轻微影响（在 DanceTrack test 上约为 0.5 HOTA）。

### Evaluation
### 评估

You can obtain both the tracking results (tracker files) and evaluation results using the following **template script**:
您可以使用以下**模板脚本**同时获取跟踪结果（跟踪器文件）和评估结果：

```shell
accelerate launch --num_processes=8 submit_and_evaluate.py --data-root <DATADIR> --inference-mode evaluate --config-path <.yaml config file path> --inference-model <checkpoint path> --outputs-dir <outputs dir> --inference-dataset <dataset name> --inference-split <split name>
```

For example, you can get the evaluation results on the DanceTrack val set as follows:
例如，您可以按如下方式获取 DanceTrack val 集上的评估结果：

```shell
accelerate launch --num_processes=8 submit_and_evaluate.py --data-root ./datasets/ --inference-mode evaluate --config-path ./configs/r50_deformable_detr_motip_dancetrack.yaml --inference-model ./outputs/r50_deformable_detr_motip_dancetrack/r50_deformable_detr_motip_dancetrack.pth --outputs-dir ./outputs/r50_deformable_detr_motip_dancetrack/ --inference-dataset DanceTrack --inference-split val
```

