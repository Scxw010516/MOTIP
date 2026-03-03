# Model Zoo
# 模型库

## MOTIP

### DanceTrack

| Method <br> 方法 | Extra Data <br> 额外数据 | Traj Aug <br> 轨迹增强 |                          Resources <br> 资源                           | HOTA | DetA | AssA |
| :----: | :--------: | :------: | :----------------------------------------------------------: | :--: | :--: | :--: |
| MOTIP  |  ***no***  |  *yes*   | [pre-train](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_dancetrack.pth) \| [config](../configs/r50_deformable_detr_motip_dancetrack.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_motip_dancetrack.pth) | 69.6 | 80.4 | 60.4 |
| MOTIP  |  ***no***  |   *no*   | [pre-train](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_dancetrack.pth) \| [config](../configs/r50_deformable_detr_motip_dancetrack_without_trajectory_augmentation.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.2/r50_deformable_detr_motip_dancetrack_without_trajectory_augmentation.pth) | 65.2 | 80.4 | 53.1 |

### SportsMOT

| Method <br> 方法 | Extra Data <br> 额外数据 | Traj Aug <br> 轨迹增强 |                          Resources <br> 资源                           | HOTA | DetA | AssA |
| :----: | :--------: | :------: | :----------------------------------------------------------: | :--: | :--: | :--: |
| MOTIP  |  ***no***  |  *yes*   | [pre-train](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_sportsmot.pth) \| [config](../configs/r50_deformable_detr_motip_sportsmot.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_motip_sportsmot.pth) | 72.6 | 83.5 | 63.2 |
| MOTIP  |  ***no***  |   *no*   | [pre-train](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_sportsmot.pth) \| [config](../configs/r50_deformable_detr_motip_sportsmot_without_trajectory_augmentation.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.2/r50_deformable_detr_motip_sportsmot_without_trajectory_augmentation.pth) | 70.9 | 83.7 | 60.1 |

### BFT

| Method <br> 方法 | Extra Data <br> 额外数据 | Traj Aug <br> 轨迹增强 |                          Resources <br> 资源                           | HOTA | DetA | AssA |
| :----: | :--------: | :------: | :----------------------------------------------------------: | :--: | :--: | :--: |
| MOTIP  |  ***no***  |  *yes*   | [pre-train](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_bft.pth) \| [config](../configs/r50_deformable_detr_motip_bft.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_motip_bft.pth) | 70.5 | 69.6 | 71.8 |
| MOTIP  |  ***no***  |   *no*   | [pre-train](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_bft.pth) \| [config](../configs/r50_deformable_detr_motip_bft_without_trajectory_augmentation.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.2/r50_deformable_detr_motip_bft_without_trajectory_augmentation.pth) | 71.3 | 69.2 | 73.7 |

***NOTE:***
***注意：***

1. *Traj Aug* is an abbreviation for *Trajectory Augmentation* in the paper.
   *Traj Aug* 是论文中轨迹增强 (*Trajectory Augmentation*) 的缩写。
2. You could also load previous checkpoints for inference from [prev-engine branch](https://github.com/MCG-NJU/MOTIP/tree/prev-engine), using runtime parameter `--use-previous-checkpoint True`. You may need to pass additional parameters to bridge the difference in the experimental setups. Typically, `--rel-pe-length` and `--miss-tolerance`.
   您也可以使用运行时参数 `--use-previous-checkpoint True` 从 [prev-engine 分支](https://github.com/MCG-NJU/MOTIP/tree/prev-engine) 加载以前的检查点进行推理。您可能需要传递额外的参数来缩小实验设置之间的差异。通常是 `--rel-pe-length` 和 `--miss-tolerance`。
3. We present some experimental results not included in the paper, which we plan to discuss in the extended version of the article :soon:.
   我们展示了一些论文中未包含的实验结果，我们计划在文章的扩展版本中讨论这些结果 :soon:。
4. You could also download our well-trained weights from [Baidu disk :cloud:](https://pan.baidu.com/s/1sy4Vv-inQN4U-GlC5NjISQ?pwd=0042).
   您也可以从 [百度网盘 :cloud:](https://pan.baidu.com/s/1sy4Vv-inQN4U-GlC5NjISQ?pwd=0042) 下载我们训练好的权重。

## DETR
## DETR 模型

You can directly download the pre-trained DETR weights used in our experiment here **(recommended)**. Or you can choose to follow the [guidance](./GET_STARTED.md) to perform pre-training yourself.
您可以直接在这里下载我们实验中使用的预训练 DETR 权重**（推荐）**。或者您可以选择按照 [指南](./GET_STARTED.md) 自行进行预训练。

|             Model Name <br> 模型名称              | Target Dataset <br> 目标数据集 | Extra Data <br> 额外数据 |                          Resources <br> 资源                           |
| :---------------------------------: | :------------: | :--------: | :----------------------------------------------------------: |
|      r50_deformable_detr_coco       |      COCO      |  ***no***  | [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco.pth) |
| r50_deformable_detr_coco_dancetrack |   DanceTrack   |  ***no***  | [config](../configs/pretrain_r50_deformable_detr_dancetrack.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_dancetrack.pth) |
| r50_deformable_detr_coco_sportsmot  |   SportsMOT    |  ***no***  | [config](../configs/pretrain_r50_deformable_detr_sportsmot.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_sportsmot.pth) |
|    r50_deformable_detr_coco_bft     |      BFT       |  ***no***  | [config](../configs/pretrain_r50_deformable_detr_bft.yaml) \| [checkpoint](https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_coco_bft.pth) |




