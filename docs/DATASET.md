# Data Preparation
# 数据准备

:link: For all the datasets we used in our experiments, you can access them from the following public link:
:link: 对于我们在实验中使用的所有数据集，您可以从以下公共链接访问它们：

- [DanceTrack](https://github.com/DanceTrack/DanceTrack)
- [SportsMOT](https://github.com/MCG-NJU/SportsMOT)
- [BFT](https://george-zhuang.github.io/nettrack/)
- [CrowdHuman](https://www.crowdhuman.org/)

## Generate GT files
## 生成 GT 文件

For the BFT and CrowdHuman datasets, you’ll need to use the provided script to convert their ground truth files to the format we require:
对于 BFT 和 CrowdHuman 数据集，您需要使用提供的脚本将它们的 ground truth 文件夹转换为我们要求的格式：

- For BFT: [gen_bft_gts.py](../tools/gen_bft_gts.py)
  对于 BFT：[gen_bft_gts.py](../tools/gen_bft_gts.py)
- For CrowdHuman: [gen_crowdhuman_gts.py](../tools/gen_crowdhuman_gts.py)
  对于 CrowdHuman：[gen_crowdhuman_gts.py](../tools/gen_crowdhuman_gts.py)

:pushpin: You need to modify the paths in the script according to your requirements.
:pushpin: 您需要根据您的要求修改脚本中的路径。

## File Tree
## 文件树

```text
<DATADIR>/
  ├── DanceTrack/
  │ ├── train/
  │ ├── val/
  │ ├── test/
  │ ├── train_seqmap.txt
  │ ├── val_seqmap.txt
  │ └── test_seqmap.txt
  ├── SportsMOT/
  │ ├── train/
  │ ├── val/
  │ ├── test/
  │ ├── train_seqmap.txt
  │ ├── val_seqmap.txt
  │ └── test_seqmap.txt
  ├── BFT/
  │ ├── train/
  │ ├── val/
  │ ├── test/
  │ ├── annotations_mot/    # used for generate gts for BFT
  │ ├── train_seqmap.txt
  │ ├── val_seqmap.txt
  │ └── test_seqmap.txt
  └── CrowdHuman/
    ├── images/
    │ ├── train/     # unzip from CrowdHuman
    │ └── val/       # unzip from CrowdHuman
    └── gts/
      ├── train/     # generate by ./data/gen_crowdhuman_gts.py
      └── val/       # generate by ./data/gen_crowdhuman_gts.py
```

## Q & A
## 问题与解答

- Q: Lack the `val_seqmap.txt` file of SportsMOT? </br>
  问：缺少 SportsMOT 的 `val_seqmap.txt` 文件？ </br>
  A: Refer to [The 'val_seqmap.txt' file of SportsMOT dataset · Issue #13](https://github.com/MCG-NJU/MOTIP/issues/13)
  答：请参阅 [The 'val_seqmap.txt' file of SportsMOT dataset · Issue #13](https://github.com/MCG-NJU/MOTIP/issues/13)
