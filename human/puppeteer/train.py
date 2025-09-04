import os
# os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer, EnsembleBuffer
from envs import make_env
from tdmpc2 import TDMPC2
from trainer import Trainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	"""
	用于训练人类控制任务的木偶代理脚本。

	最相关的参数：
		`task`: 任务名称（默认：tracking）
		`steps`: 训练/环境 步数 (默认: 10M)
		`seed`: 随机种子 (默认: 1)

	完整的参数列表请查看 config.yaml

	示例用法：
	```
		$ python train.py task=tracking
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	# 如果是底层的 tracking 任务，则使用集成缓冲区，否则使用普通缓冲区
	buffer_cls = EnsembleBuffer if cfg.task == 'tracking' else Buffer
	trainer = Trainer(
		cfg=cfg,
		env=make_env(cfg),
		agent=TDMPC2(cfg),
		buffer=buffer_cls(cfg),
		logger=Logger(cfg),
	)
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
