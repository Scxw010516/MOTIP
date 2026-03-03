# Installation
# 安装

Our codebase is built upon **Python 3.12, PyTorch 2.4.0 (recommended)**. 
我们的代码库基于 **Python 3.12, PyTorch 2.4.0 (推荐)** 构建。

:warning: As far as I know, due to the use of some new language features in our code, Python version 3.10 or higher is required. For PyTorch, because there have been changes in the type requirements for attention masks, PyTorch version 2.0 or higher is needed.
:warning: 据我所知，由于代码中使用了某些新的语言特性，需要 Python 3.10 或更高版本。对于 PyTorch，由于注意掩码 (attention masks) 的类型要求发生了变化，需要 PyTorch 2.0 或更高版本。

:construction: We plan to support lower versions of PyTorch in the future, but the exact timeline is yet to be determined. Currently, we do not have sufficient manpower to address this issue.
:construction: 我们计划在未来支持较低版本的 PyTorch，但具体的时间表尚未确定。目前，我们没有足够的人力来解决这个问题。

## Setup scripts
## 安装脚本

```shell
conda create -n MOTIP python=3.12		# suggest to use virtual envs
conda activate MOTIP
# PyTorch:
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# Other dependencies:
conda install pyyaml tqdm matplotlib scipy pandas
pip install wandb accelerate einops
# Compile the Deformable Attention:
cd models/ops/
sh make.sh
# [Optional] After compiled, you can use following script to test it:
python test.py
```




