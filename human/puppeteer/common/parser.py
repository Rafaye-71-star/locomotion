import re
from pathlib import Path

# pip install hydra-core
# 通过结构化的YAML文件定义配置，并通过命令行进行灵活的覆盖和组合，同时自动化地管理实验输出目录
import hydra
from omegaconf import OmegaConf


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
	"""
	解析 Hydra 配置。大多数是为了方便。
	"""

	# Logic
	for k in cfg.keys():
		try:
			v = cfg[k]
			if v in {'None', 'none'}:
				cfg[k] = None
		except:
			pass

	# 代数表达式
	for k in cfg.keys():
		try:
			v = cfg[k]
			if isinstance(v, str):
				match = re.match(r"(\d+)([+\-*/])(\d+)", v)
				if match:
					cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
					if isinstance(cfg[k], float) and cfg[k].is_integer():
						cfg[k] = int(cfg[k])
		except:
			pass

	# Convenience
	# 获取当前工作目录，如果运行方式为 python puppeteer/train.py，则 logs 位于当前目录下
	cfg.work_dir = Path(hydra.utils.get_original_cwd()) / 'logs' / cfg.task / str(cfg.seed) / cfg.exp_name
	cfg.task_title = cfg.task.replace("-", " ").title()
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins-1) # Bin size for discrete regression

	# Task-specific
	if cfg.task == 'tracking': # high variance
		cfg.eval_episodes = 100
	cfg.obs = 'rgb' if 'corridor' in cfg.task else 'state'

	return cfg
