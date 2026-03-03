# MISCELLANEOUS
# 杂项

Some other settings that are unrelated to the specific model structure or hyperparameters. Such as logging, checkpoint saving, *etc*.
一些与特定模型结构或超参数无关的其他设置，例如日志记录、检查点（checkpoint）保存等。

## Logging
## 日志记录

:hugs: In our codebase, many logging methods are integrated, including tensorboard/wandb/*etc*.
:hugs: 在我们的代码库中，集成许多日志记录方法，包括 tensorboard/wandb 等。

In our default scripts, we set `use_wandb` to `False` to disable wandb logging, because it requires creating an account and making some additional settings, which increases the user's workload. However, if you believe you need to enable wandb logging (which I think is more elegant), you will need to set it up as follows:
在我们的默认脚本中，我们将 `use_wandb` 设置为 `False` 以禁用 wandb 日志，因为它需要创建帐户并进行一些额外设置，这增加了用户的工作量。但是，如果您认为需要启用 wandb 日志（我认为这更优雅），则需要按如下方式进行设置：

1. add `--use-wandb True` to the script.
   在脚本中添加 `--use-wandb True`。
2. set the `EXP_OWNER` or `--exp-owner`, which is the wandb account name.
   设置 `EXP_OWNER` 或 `--exp-owner`，即 wandb 账户名称。
3. manually record your current git revision number (`--git-version`, for example, `--git-version d293ceee8c6d208bb4a5d7b6ba92a7b5d7ec4bca`) to ensure you can rollback and reproduce the experiment. I know this is not very elegant, but it is easy enough :stuck_out_tongue:.
   手动记录您当前的 git 修订号（例如 `--git-version d293ceee8c6d208bb4a5d7b6ba92a7b5d7ec4bca`），以确保您可以回滚并重现实验。我知道这不够优雅，但它足够简单 :stuck_out_tongue:。


