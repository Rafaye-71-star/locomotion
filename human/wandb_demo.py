# 配置wandb
# 1. 注册 wandb 账号
# 2. 命令行输入：wandb login
# 3. 点开链接：https://wandb.ai/authorize ，将生成的代码复制粘贴到 第 2 步命令行中，回车即可
# 4. 运行后即可在 wandb 页面中看到结果
import wandb

config = dict(
    learning_rate=0.01,
    momentum=0.2,
    architecture="CNN",
    dataset_id="peds-0192",
    infra="AWS",
)

wandb.init(
    project="detect-pedestrians",
    notes="tweak baseline",
    tags=["baseline", "paper1"],
    config=config,
)